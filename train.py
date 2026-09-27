import asyncio
import logging
import os
import time

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.placement_group import (
    create_rollout_components,
    create_training_models,
    maybe_start_api_server,
    update_weights,
)
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.ray.train.types import TrainResultWithTiming
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.data import remove_rollout_data_refs, remove_train_output_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.lora.utils import lora_rollout_enabled
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.misc import checkpoint_artifacts_due, should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script
from miles.utils.tracking_utils.tracking import log as log_tracking

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
    trainer_offload_block_time: float,
    rollout_weight_onload_block_time: float,
    weight_update_block_time: float,
    rollout_kv_onload_block_time: float,
    train_to_rollout_block_time: float,
    worker_metrics: dict[str, int | float] | None = None,
) -> None:
    """Log colocated switch components on the rollout that incurred them.

    ``rollout_to_train_active_time`` is a component sum, not an end-to-end
    wall time: the wake-up component is the maximum worker-local duration and
    excludes train-group refresh and Ray dispatch latency. With a critic, it
    ends when the first trainer (the critic) is ready.
    ``train_to_rollout_block_time`` starts after checkpoint saves, so any
    earlier per-model offloads are excluded. Its four component timers use the
    same driver clock and retain the existing control flow.
    """
    rollout_to_train_active_time = rollout_offload_block_time + local_wake_up_time
    metrics = {
        "perf/colocate/rollout_offload_block_time": rollout_offload_block_time,
        "perf/colocate/rollout_to_train_active_time": rollout_to_train_active_time,
        "perf/colocate/trainer_offload_block_time": trainer_offload_block_time,
        "perf/colocate/rollout_weight_onload_block_time": rollout_weight_onload_block_time,
        "perf/colocate/weight_update_block_time": weight_update_block_time,
        "perf/colocate/rollout_kv_onload_block_time": rollout_kv_onload_block_time,
        "perf/colocate/train_to_rollout_block_time": train_to_rollout_block_time,
        "perf/colocate/switch_total_active_time": rollout_to_train_active_time + train_to_rollout_block_time,
        "rollout/step": compute_rollout_step(args, rollout_id),
    }
    if worker_metrics:
        metrics.update(worker_metrics)
    logger.info("colocate switch %d: %s", rollout_id, metrics)
    log_tracking(args, metrics, step_key="rollout/step")


def _summarize_colocate_worker_metrics(*results) -> dict[str, int | float]:
    """Summarize opt-in worker scalars already returned by blocking Ray calls."""
    values_by_name: dict[str, list[int | float]] = {}
    stack = list(results)
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for name, metric_value in value.items():
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    values_by_name.setdefault(name, []).append(metric_value)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)

    metrics: dict[str, int | float] = {}
    for name, values in values_by_name.items():
        prefix = f"perf/colocate/{name}"
        metrics[f"{prefix}_max"] = max(values)
        if name.endswith("_bytes"):
            metrics[f"{prefix}_sum"] = sum(values)
    return metrics


async def train(args, *, disposer: Disposer):
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
    _worker_manager = init_orchestration_script(args, disposer=disposer)
    collect_switch_metrics = args.log_colocate_switch_metrics

    if args.colocate_memory_peak_device == "gpu":
        assert (
            args.offload_train and args.offload_rollout
        ), "--colocate-memory-peak-device gpu requires --offload-train and --offload-rollout"
        assert not args.use_critic, "--colocate-memory-peak-device gpu is not wired for the critic path"

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, rollout_executor)
    disposer.add(critic_model, actor_model)

    maybe_start_api_server(args, trainer_models={"actor": actor_model}, inference_controller=inference_controller)
    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights(args, actor_model, rollout_executor, inference_controller)

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.offload_rollout:
        await inference_controller.onload_kv()

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)
    disposer.add(eval_dispatcher.drain)

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def offload_train():
        if args.use_critic:
            return []
        if args.offload_train:
            return await actor_model.offload()
        else:
            return await actor_model.clear_memory()

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
        if write_dist:
            await rollout_executor.save(rollout_id)

    if args.num_rollout > args.start_rollout_id and args.eval_interval is not None and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        if args.start_rollout_id == 0:
            await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)
        else:
            await eval_dispatcher.dispatch(args.start_rollout_id - 1)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        await inference_controller.prepare_rollout(rollout_id)
        rollout_data_pack = await rollout_executor.get(rollout_id)

        rollout_offload_block_time = 0.0
        rollout_offload_start = time.monotonic() if collect_switch_metrics else None
        if args.offload_rollout:
            if args.colocate_memory_peak_device == "gpu":
                await inference_controller.offload_kv()
                await actor_model.onload()
                await inference_controller.offload_weights()
            else:
                offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
                if "kv_cache" in args.offload_rollout_level:
                    offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
                if "weight" in args.offload_rollout_level:
                    offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
                await inference_controller.offload(tags=offload_tags)
        if args.offload_rollout and rollout_offload_start is not None:
            rollout_offload_block_time = time.monotonic() - rollout_offload_start

        local_wake_up_time = 0.0
        if args.use_critic:
            # The critic is the first trainer that blocks the rollout-to-train switch.
            values, local_wake_up_time = await _train_model(
                critic_model,
                rollout_id,
                rollout_data_pack,
                collect_wake_up_time=collect_switch_metrics,
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
            remove_train_output_refs(values)
        else:
            _, local_wake_up_time = await _train_model(
                actor_model,
                rollout_id,
                rollout_data_pack,
                collect_wake_up_time=collect_switch_metrics,
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

        eval_due = should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout)
        if rollout_id + 1 < args.num_rollout or eval_due:
            trainer_offload_block_time = 0.0
            rollout_weight_onload_block_time = 0.0
            weight_update_block_time = 0.0
            rollout_kv_onload_block_time = 0.0

            train_to_rollout_start = time.monotonic() if collect_switch_metrics else None
            trainer_offload_worker_metrics = None
            if args.colocate_memory_peak_device == "gpu":
                await actor_model.clear_memory()
                if lora_rollout_enabled(args):
                    await actor_model.offload_grad_buffer()
                phase_start = time.monotonic() if collect_switch_metrics else None
                await inference_controller.onload_weights()
                if phase_start is not None:
                    rollout_weight_onload_block_time = time.monotonic() - phase_start
                phase_start = time.monotonic() if collect_switch_metrics else None
                trainer_offload_worker_metrics = await offload_train()
                if phase_start is not None:
                    trainer_offload_block_time = time.monotonic() - phase_start
            else:
                phase_start = time.monotonic() if collect_switch_metrics else None
                trainer_offload_worker_metrics = await offload_train()
                if phase_start is not None:
                    trainer_offload_block_time = time.monotonic() - phase_start
                if args.offload_rollout:
                    phase_start = time.monotonic() if collect_switch_metrics else None
                    await inference_controller.onload_weights()
                    if phase_start is not None:
                        rollout_weight_onload_block_time = time.monotonic() - phase_start
            phase_start = time.monotonic() if collect_switch_metrics else None
            await update_weights(args, actor_model, rollout_executor, inference_controller, rollout_id=rollout_id)
            weight_update_worker_metrics = (
                await actor_model.pop_colocate_switch_metrics() if collect_switch_metrics else None
            )
            if phase_start is not None:
                weight_update_block_time = time.monotonic() - phase_start
            if args.offload_rollout:
                phase_start = time.monotonic() if collect_switch_metrics else None
                await inference_controller.onload_kv()
                if phase_start is not None:
                    rollout_kv_onload_block_time = time.monotonic() - phase_start
            if train_to_rollout_start is not None:
                train_to_rollout_block_time = time.monotonic() - train_to_rollout_start
                _log_colocate_switch_metrics(
                    args,
                    rollout_id,
                    rollout_offload_block_time=rollout_offload_block_time,
                    local_wake_up_time=local_wake_up_time,
                    trainer_offload_block_time=trainer_offload_block_time,
                    rollout_weight_onload_block_time=rollout_weight_onload_block_time,
                    weight_update_block_time=weight_update_block_time,
                    rollout_kv_onload_block_time=rollout_kv_onload_block_time,
                    train_to_rollout_block_time=train_to_rollout_block_time,
                    worker_metrics=_summarize_colocate_worker_metrics(
                        trainer_offload_worker_metrics,
                        weight_update_worker_metrics,
                    ),
                )

        if eval_due:
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


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(with_disposer(train, args))
