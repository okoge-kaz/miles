import asyncio
import logging
import os

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
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


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    fully_async_checkpoint = getattr(args, "fully_async_rollout_checkpoint", False)
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

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await rollout_manager.eval.remote(0)

    async def save_training_model(model, rollout_id, force_sync, *, write_dist=True, write_hf=True):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync, write_dist=write_dist, write_hf=write_hf)
        if args.use_critic and args.offload_train:
            await model.offload()

    # async train loop.
    prefetch_rollout_batches = should_prefetch_rollout_batches(args)
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if prefetch_rollout_batches and rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
        elif not prefetch_rollout_batches:
            rollout_data_next_future = None

        if args.fully_async:
            # Preserve the immediate prefetch above, then sample the authoritative
            # applied version before the trainer call. No weight update can occur
            # between these operations, so post-selection forward lag stays
            # separate from the historical drain-time staleness metrics.
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
        if fully_async_checkpoint:
            await rollout_manager.acknowledge_trained_batch.remote(
                rollout_id,
                rollout_data_curr_ref.get("fully_async_batch_token"),
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
            if write_dist and fully_async_checkpoint and rollout_data_next_future is not None:
                # Failure-free execution finishes this prefetched batch before
                # the weight push below. Finish it before the snapshot as well,
                # so resume restores one complete, already-admitted batch rather
                # than completing a partial drain under the next weight version.
                await rollout_data_next_future
            force_sync = (
                external_save
                or rollout_id == args.num_rollout - 1
                # The model tracker is the full-replay commit record. Megatron's
                # async save can return before publishing it; pruning sidecars at
                # that point could remove the state named by the old tracker.
                or (write_dist and fully_async_checkpoint)
            )
            # The model tracker is the commit record. Publish the matching rollout
            # sidecar first so a visible model checkpoint can never lack replay state.
            if write_dist and fully_async_checkpoint:
                await rollout_manager.save.remote(rollout_id)
            await save_training_model(actor_model, rollout_id, force_sync, write_dist=write_dist, write_hf=write_hf)
            if args.use_critic:
                await save_training_model(
                    critic_model, rollout_id, force_sync, write_dist=write_dist, write_hf=write_hf
                )
            if write_dist:
                if fully_async_checkpoint:
                    await rollout_manager.mark_checkpoint_published.remote(rollout_id)
                else:
                    # Preserve the legacy cursor-only checkpoint order.
                    await rollout_manager.save.remote(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            if prefetch_rollout_batches:
                rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            await actor_model.update_weights(rollout_id=rollout_id)

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

        if not prefetch_rollout_batches and rollout_id + 1 < args.num_rollout:
            # The persistent rollout worker kept filling its policy queue during
            # training. Ask it to select the next batch only now, when the trainer
            # is ready to consume that batch.
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
