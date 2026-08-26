import asyncio
import logging
import os
import time

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.ray.train.types import TrainResultWithTiming
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.misc import checkpoint_artifacts_due, should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking, log as log_tracking

logger = logging.getLogger(__name__)


async def _train_model(
    model,
    rollout_id: int,
    rollout_data_pack,
    *,
    external_data=None,
    collect_wake_up_time: bool = False,
) -> tuple[object, float]:
    train_kwargs = {} if external_data is None else {"external_data": external_data}
    if collect_wake_up_time:
        train_kwargs["collect_wake_up_time"] = True
    result = await model.train(rollout_id, rollout_data_pack, **train_kwargs)
    if not collect_wake_up_time:
        return result, 0.0
    if not isinstance(result, TrainResultWithTiming):
        raise TypeError("Timed train calls must return TrainResultWithTiming")
    return result.result, result.local_wake_up_time


def _log_colocate_switch_metrics(
    args,
    rollout_id: int,
    *,
    rollout_offload_block_time: float,
    local_wake_up_time: float,
    train_to_rollout_block_time: float,
) -> None:
    """Log colocated switch components on the rollout that incurred them.

    ``rollout_to_train_active_time`` is a component sum, not an end-to-end
    wall time: the wake-up component is the maximum worker-local duration and
    excludes train-group refresh and Ray dispatch latency. With a critic, it
    ends when the first trainer (the critic) is ready.
    ``train_to_rollout_block_time`` starts after checkpoint saves, so any
    earlier per-model offloads are excluded.
    """
    rollout_to_train_active_time = rollout_offload_block_time + local_wake_up_time
    metrics = {
        "perf/colocate/rollout_offload_block_time": rollout_offload_block_time,
        "perf/colocate/rollout_to_train_active_time": rollout_to_train_active_time,
        "perf/colocate/train_to_rollout_block_time": train_to_rollout_block_time,
        "perf/colocate/switch_total_active_time": rollout_to_train_active_time
        + train_to_rollout_block_time,
        "rollout/step": compute_rollout_step(args, rollout_id),
    }
    logger.info("colocate switch %d: %s", rollout_id, metrics)
    log_tracking(args, metrics, step_key="rollout/step")


async def train(args):
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
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

    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await rollout_manager.eval.remote(rollout_id=0)

    async def offload_train():
        if args.use_critic:
            return
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id, force_sync=False, *, write_dist=True, write_hf=True):
        force_sync = force_sync or rollout_id == args.num_rollout - 1

        async def save_training_model(model):
            if args.use_critic and args.offload_train:
                await model.onload()
            await model.save_model(rollout_id, force_sync=force_sync, write_dist=write_dist, write_hf=write_hf)
            if args.use_critic and args.offload_train:
                await model.offload()

        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            await save_training_model(actor_model)
        if args.use_critic:
            await save_training_model(critic_model)
        # Buffer state is only meaningful next to a resumable checkpoint.
        if write_dist:
            await rollout_manager.save.remote(rollout_id)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == args.start_rollout_id and not args.skip_eval_before_train:
            await rollout_manager.eval.remote(rollout_id)

        rollout_data_pack = await rollout_manager.generate.remote(rollout_id)
        rollout_offload_block_time = 0.0

        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            rollout_offload_start = time.monotonic() if args.colocate else None
            await rollout_manager.offload.remote(tags=offload_tags)
            if rollout_offload_start is not None:
                rollout_offload_block_time = time.monotonic() - rollout_offload_start

        local_wake_up_time = 0.0
        if args.use_critic:
            # The critic is the first trainer that blocks the rollout-to-train switch.
            values, local_wake_up_time = await _train_model(
                critic_model,
                rollout_id,
                rollout_data_pack,
                collect_wake_up_time=args.colocate,
            )
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await _train_model(
                    actor_model,
                    rollout_id,
                    rollout_data_pack,
                    external_data=values,
                )
                if args.offload_train:
                    await actor_model.offload()
        else:
            _, local_wake_up_time = await _train_model(
                actor_model,
                rollout_id,
                rollout_data_pack,
                collect_wake_up_time=args.colocate,
            )
        remove_rollout_data_refs(args, rollout_data_pack)

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
            await save(rollout_id, force_sync=external_save, write_dist=write_dist, write_hf=write_hf)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        train_to_rollout_start = time.monotonic() if args.colocate else None
        await offload_train()
        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
        await actor_model.update_weights(rollout_id=rollout_id)
        if args.offload_rollout:
            await rollout_manager.onload_kv.remote()
        if train_to_rollout_start is not None:
            train_to_rollout_block_time = time.monotonic() - train_to_rollout_start
            _log_colocate_switch_metrics(
                args,
                rollout_id,
                rollout_offload_block_time=rollout_offload_block_time,
                local_wake_up_time=local_wake_up_time,
                train_to_rollout_block_time=train_to_rollout_block_time,
            )

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

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

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
