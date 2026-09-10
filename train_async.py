import asyncio
import logging
import os

from miles.ray.placement_group import create_rollout_components, create_training_models, update_weights
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.ray.wiring import launch_worker_manager
from miles.rollout.queue_policy import should_prefetch_rollout_batches
from miles.utils import object_store
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.async_utils import eager_create_task
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.api_server.server import start_api_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import checkpoint_artifacts_due, should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

logger = logging.getLogger(__name__)


def _updates_before_training_rollout(args, rollout_id: int) -> int:
    update_disabled = any(
        getattr(args, flag, False) for flag in ("debug_train_only", "debug_rollout_only", "debug_skip_weight_update")
    )
    return int(not update_disabled and rollout_id % args.update_weights_interval == 0)


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    use_replay_buffer = getattr(args, "use_replay_buffer", False)
    validate_async_off_policy_correction(args)
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    _worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, inference_controller, rollout_executor)

    if args.api_server_port:
        start_api_server(
            args=args,
            actor_model=actor_model,
            inference_controller=inference_controller,
            host=args.api_server_host,
            port=args.api_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights(actor_model, rollout_executor)

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync, *, write_dist=True, write_hf=True):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync, write_dist=write_dist, write_hf=write_hf)
        if args.use_critic and args.offload_train:
            await model.offload()

    async def prepare_and_generate(rollout_id, *, updates_before_train=0):
        await inference_controller.prepare_rollout(rollout_id)
        return await rollout_executor.get.remote(rollout_id, updates_before_train=updates_before_train)

    # async train loop.
    prefetch_rollout_batches = should_prefetch_rollout_batches(args)
    rollout_data_next_future = await eager_create_task(prepare_and_generate(args.start_rollout_id))
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if prefetch_rollout_batches and rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = await eager_create_task(
                prepare_and_generate(
                    rollout_id + 1, updates_before_train=_updates_before_training_rollout(args, rollout_id + 1)
                )
            )
        elif not prefetch_rollout_batches:
            rollout_data_next_future = None

        if args.fully_async:
            # Close the batch's queue-wait window and retain its throughput inputs
            # immediately before the trainer call.
            await rollout_executor.record_batch_consumption.remote(rollout_id)

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
            await rollout_executor.record_batch_trained.remote(
                rollout_id,
                actor_trained=actor_trained,
            )
        if use_replay_buffer:
            await rollout_executor.acknowledge_trained_batch.remote(
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
                await rollout_executor.save.remote(rollout_id)
            await save_training_model(actor_model, rollout_id, force_sync, write_dist=write_dist, write_hf=write_hf)
            if args.use_critic:
                await save_training_model(
                    critic_model, rollout_id, force_sync, write_dist=write_dist, write_hf=write_hf
                )
            if write_dist:
                if use_replay_buffer:
                    await rollout_executor.mark_replay_buffer_committed.remote(rollout_id)
                else:
                    # Preserve the legacy cursor-only checkpoint order.
                    await rollout_executor.save.remote(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            if prefetch_rollout_batches:
                rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            await update_weights(actor_model, rollout_executor, rollout_id=rollout_id)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await inference_controller.prepare_eval()
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
            rollout_data_next_future = await eager_create_task(prepare_and_generate(rollout_id + 1))

    await eval_dispatcher.drain()
    await rollout_executor.dispose.remote()
    await inference_controller.dispose()
    await actor_model.dispose()
    if critic_model is not None:
        await critic_model.dispose()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
