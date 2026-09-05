import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.rollout.queue_policy import should_prefetch_rollout_batches
from miles.utils import object_store
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import checkpoint_artifacts_due, should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

logger = logging.getLogger(__name__)

_RESUME_METRIC_PREFIX = "resume/benchmark/checkpoint"


def _updates_before_training_rollout(args, rollout_id: int) -> int:
    update_disabled = any(
        getattr(args, flag, False) for flag in ("debug_train_only", "debug_rollout_only", "debug_skip_weight_update")
    )
    return int(not update_disabled and rollout_id % args.update_weights_interval == 0)


def _debug_failure_thresholds(args) -> dict[str, int]:
    return {
        "outstanding_groups": getattr(args, "debug_failure_min_outstanding_groups", 0),
        "completed_groups_reused": getattr(args, "debug_failure_min_completed_groups", 0),
        "partial_groups_continued": getattr(args, "debug_failure_min_inflight_groups", 0),
        "partial_response_tokens_continued": getattr(args, "debug_failure_min_inflight_tokens", 0),
        "groups_to_regenerate": getattr(args, "debug_failure_min_regenerate_groups", 0),
    }


def _validate_debug_failure_configuration(args) -> None:
    fail_after = getattr(args, "debug_fail_after_rollout", None)
    if fail_after is None:
        return
    if fail_after <= 0:
        raise ValueError("--debug-fail-after-rollout must be positive")
    if getattr(args, "debug_exit_after_rollout", None) is not None:
        raise ValueError("--debug-fail-after-rollout and --debug-exit-after-rollout are mutually exclusive")
    if not getattr(args, "debug_failure_marker", None):
        raise ValueError("--debug-fail-after-rollout requires --debug-failure-marker")
    invalid = {name: value for name, value in _debug_failure_thresholds(args).items() if value < 0}
    if invalid:
        raise ValueError(f"Debug failure replay thresholds must be non-negative: {invalid}")


def _checkpoint_storage_identity(checkpoint_root: str, rollout_id: int) -> dict[str, object]:
    root = Path(checkpoint_root)
    tracker = root / "latest_checkpointed_iteration.txt"
    tracker_value = tracker.read_text(encoding="utf-8").strip()
    if tracker_value != str(rollout_id):
        raise RuntimeError(
            f"Checkpoint tracker does not name the injected-failure checkpoint: "
            f"tracker={tracker_value!r}, rollout_id={rollout_id}"
        )
    replay_manifest_path = root / "rollout" / f"replay_buffer_{rollout_id}.pt.sha256.json"
    replay_manifest = None
    if replay_manifest_path.is_file():
        replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    return {
        "checkpoint_root": checkpoint_root,
        "tracker_iteration": int(tracker_value),
        "replay_manifest": replay_manifest,
    }


def _write_debug_failure_marker(path: str, payload: dict[str, object]) -> None:
    marker = Path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        descriptor = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _terminate_for_debug_failure() -> None:
    logging.shutdown()
    os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError("SIGKILL unexpectedly returned")


def _maybe_inject_debug_failure(
    args,
    *,
    rollout_id: int,
    checkpoint_committed: bool,
    checkpoint_metrics: dict[str, float] | None,
) -> None:
    fail_after = getattr(args, "debug_fail_after_rollout", None)
    if fail_after is None or rollout_id - args.start_rollout_id + 1 < fail_after:
        return
    if not checkpoint_committed:
        raise RuntimeError(
            f"Injected failure reached rollout {rollout_id} without a committed distributed checkpoint"
        )
    if checkpoint_metrics is None:
        raise RuntimeError("Injected failure requires checkpoint replay/conservation metrics")
    for name, minimum in _debug_failure_thresholds(args).items():
        actual = checkpoint_metrics.get(f"{_RESUME_METRIC_PREFIX}/{name}")
        if actual is None or actual < minimum:
            raise RuntimeError(
                f"Injected failure checkpoint does not contain enough {name}: actual={actual}, minimum={minimum}"
            )
    payload = {
        "schema_version": 1,
        "event": "intentional_whole_job_failure",
        "rollout_id": rollout_id,
        "completed_rollouts_in_process": rollout_id - args.start_rollout_id + 1,
        "pid": os.getpid(),
        "wall_time_ns": time.time_ns(),
        "checkpoint": _checkpoint_storage_identity(args.save, rollout_id),
        "checkpoint_metrics": checkpoint_metrics,
    }
    _write_debug_failure_marker(args.debug_failure_marker, payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    logger.critical(
        "debug_failure_after_rollout=%d reached at rollout_id=%d marker=%s payload=%s",
        fail_after,
        rollout_id,
        args.debug_failure_marker,
        encoded,
    )
    _terminate_for_debug_failure()


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    use_replay_buffer = getattr(args, "use_replay_buffer", False)
    _validate_debug_failure_configuration(args)
    validate_async_off_policy_correction(args)
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.control_server_port:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_manager)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync, *, write_dist=True, write_hf=True):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync, write_dist=write_dist, write_hf=write_hf)
        if args.use_critic and args.offload_train:
            await model.offload()

    # async train loop.
    prefetch_rollout_batches = should_prefetch_rollout_batches(args)
    rollout_data_next_future = rollout_manager.generate.remote(
        args.start_rollout_id,
        updates_before_train=0,
    )
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        checkpoint_committed = False
        checkpoint_metrics = None
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if prefetch_rollout_batches and rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(
                rollout_id + 1,
                updates_before_train=_updates_before_training_rollout(args, rollout_id + 1),
            )
        elif not prefetch_rollout_batches:
            rollout_data_next_future = None

        if args.fully_async:
            # Close the batch's queue-wait window and retain its throughput inputs
            # immediately before the trainer call.
            await rollout_manager.record_batch_consumption.remote(rollout_id)

        actor_trained = False
        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_curr_ref)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)
                actor_trained = True
                if args.offload_train:
                    await actor_model.offload()
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)
            actor_trained = True
        if args.fully_async:
            await rollout_manager.record_batch_trained.remote(
                rollout_id,
                actor_trained=actor_trained,
            )
        if use_replay_buffer:
            await rollout_manager.acknowledge_trained_batch.remote(
                rollout_id,
                rollout_data_curr_ref.get("replay_buffer_batch_token"),
            )
        remove_rollout_data_refs(args, rollout_data_curr_ref)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        write_dist, write_hf = checkpoint_artifacts_due(
            rollout_id,
            save_interval=args.save_interval,
            hf_save_interval=args.hf_save_interval,
            num_rollout_per_epoch=num_rollout_per_epoch,
            num_rollout=args.num_rollout,
            external_save=external_save,
        )
        if write_dist or write_hf:
            if write_dist and use_replay_buffer and rollout_data_next_future is not None:
                # Failure-free execution finishes this prefetched batch before
                # the weight push below. Finish it before the snapshot as well,
                # so resume restores one complete, already-admitted batch rather
                # than completing a partial drain under the next weight version.
                await rollout_data_next_future
            force_sync = (
                external_save
                or rollout_id == args.num_rollout - 1
                # The model tracker is the replay-buffer commit record. Megatron's
                # async save can return before publishing it; pruning replay buffers at
                # that point could remove the state named by the old tracker.
                or (write_dist and use_replay_buffer)
            )
            # The model tracker is the commit record. Publish the matching replay
            # buffer first so a visible model checkpoint can never lack replay state.
            if write_dist and use_replay_buffer:
                checkpoint_metrics = await rollout_manager.save.remote(rollout_id)
            await save_training_model(actor_model, rollout_id, force_sync, write_dist=write_dist, write_hf=write_hf)
            if args.use_critic:
                await save_training_model(
                    critic_model, rollout_id, force_sync, write_dist=write_dist, write_hf=write_hf
                )
            if write_dist:
                if use_replay_buffer:
                    await rollout_manager.mark_replay_buffer_committed.remote(rollout_id)
                else:
                    # Preserve the legacy cursor-only checkpoint order.
                    checkpoint_metrics = await rollout_manager.save.remote(rollout_id)
                checkpoint_committed = True
            if external_save:
                os.remove(args.save_trigger_sentinel)

        _maybe_inject_debug_failure(
            args,
            rollout_id=rollout_id,
            checkpoint_committed=checkpoint_committed,
            checkpoint_metrics=checkpoint_metrics,
        )

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            if prefetch_rollout_batches:
                rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            await actor_model.update_weights(rollout_id=rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break

        if not prefetch_rollout_batches and rollout_id + 1 < args.num_rollout:
            # The persistent rollout worker kept filling its policy queue during
            # training. Ask it to select the next batch only now, when the trainer
            # is ready to consume that batch.
            rollout_data_next_future = rollout_manager.generate.remote(
                rollout_id + 1,
                updates_before_train=0,
            )

    await eval_dispatcher.drain()
    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
