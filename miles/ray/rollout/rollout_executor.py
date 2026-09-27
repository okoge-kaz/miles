import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.eval_fleet import EvalFleetInfo, RolloutExecutorEvalFleet
from miles.ray.rollout.metrics import (
    log_eval_rollout_data,
    log_eval_skip,
    log_replay_resume_checkpoint,
    log_rollout_batch_consumption,
    log_rollout_data,
    log_rollout_pipeline_throughput,
)
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
    split_train_data_by_dp,
)
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.rollout.queue_telemetry import _iter_samples as _iter_group_samples
from miles.rollout.recycle_compute_metrics import (
    GENERATED_TOKENS_KEY,
    append_final_consumed_records,
    batch_consumption_metrics,
    build_batch_consumption_snapshot,
    finalize_useful_rollout_metrics,
    pipeline_throughput_metrics,
)
from miles.rollout.replay_buffer import (
    ensure_no_replay_buffer,
    load_replay_buffer,
    prune_replay_buffers,
    rollout_batch_token,
    save_replay_buffer,
)
from miles.rollout.replay_resume_metrics import checkpoint_resume_metrics, replay_load_metrics
from miles.utils import object_store
from miles.utils.async_utils import maybe_await, run
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import event_logger_context
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.data import RolloutDataPack
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.function_registry import load_function
from miles.utils.hf_utils.config import is_complete_hf_export
from miles.utils.http_utils import init_http_client
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.weight_version import assert_samples_weight_version_sane, assert_weight_version_is_published
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
ROLLOUT_FN_DEBUG_METADATA_KEY = "rollout_fn_debug"


class RolloutExecutor:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(
        self,
        *,
        args,
        router_providers: Sequence[BaseWorkerProvider],
        session_server_provider: BaseWorkerProvider | None,
        inference_controller_provider: BaseWorkerProvider,
    ):
        self._init_once = InitOnce(type(self).__name__)

        configure_logger(args, source=SimpleProcessIdentity(component="rollout_executor"))

        self.args = args
        # set by the training actor after each weight update, keyed by trainer model id (None for one policy)
        self._weight_versions_of_model_id: dict[str | None, int] = {}
        self._rollouts_since_publish_of_model_id: dict[str | None, int] = defaultdict(int)
        self._train_parallel_configs_of_model_id: dict[str | None, dict[str, Any]] = {}
        self._router_providers = router_providers
        self._session_server_provider = session_server_provider
        self._inference_controller_provider = inference_controller_provider

    @init_once
    async def init(self) -> None:
        args = self.args
        if not args.debug_train_only or args.eval_num_gpus > 0:
            await resolve_router_addrs(args, router_providers=self._router_providers)
            await wait_session_server_ready(args, provider=self._session_server_provider)

        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        object_store.init_instance(args, contribute_segment=False)

        init_http_client(args)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.use_legacy_rollout_v1 = use_legacy_rollout_v1()
        if not self.use_legacy_rollout_v1:
            if self.args.load_debug_rollout_data is not None:
                self.generate_rollout = None
                self.eval_generate_rollout = None
            else:
                input = RolloutFnConstructorInput(args=args, data_source=self.data_source)
                self.generate_rollout = load_rollout_function(input, self.args.rollout_function_path)
                if self.args.eval_function_path == self.args.rollout_function_path:
                    # Reuse the instance so train and eval share one state (and stateful
                    # rollout fns like FullyAsyncRolloutFn are not constructed twice).
                    self.eval_generate_rollout = self.generate_rollout
                else:
                    self.eval_generate_rollout = load_rollout_function(input, self.args.eval_function_path)
        else:
            self.generate_rollout = load_function(self.args.rollout_function_path)
            self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if (x := self.args.custom_reward_post_process_path) is not None:
            self.custom_reward_post_process_func = load_function(x)
        self.custom_convert_samples_to_train_data_func = None
        if (x := self.args.custom_convert_samples_to_train_data_path) is not None:
            self.custom_convert_samples_to_train_data_func = load_function(x)
        if self.generate_rollout is not None:
            logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
            logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        self._resume_benchmark_load_metrics: dict[str, float] = {}
        self.rollout_id = -1
        self._fully_async_consumption_snapshots: dict[int, dict] = {}
        self._fully_async_inflight_training: dict[int, tuple[int | None, int, int | None]] = {}
        self._eval_lock = asyncio.Lock()
        self._eval_fleet: RolloutExecutorEvalFleet | None = None

        self._metric_checker = MetricChecker.maybe_create(args)

    async def get_init_state(self) -> str:
        return self._init_once.state.value

    async def dispose(self) -> None:
        if (shutdown := getattr(self.generate_rollout, "shutdown", None)) is not None:
            await asyncio.to_thread(run, shutdown())
        if not self.use_legacy_rollout_v1 and self.generate_rollout is not None:
            await maybe_await(self.generate_rollout.dispose())
        if (close := getattr(self.data_source, "close", None)) is not None:
            close()
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if isinstance(self.eval_generate_rollout, CheckpointEvalFn):
            await maybe_await(self.eval_generate_rollout.dispose())

    # -------------------------- data generation -----------------------------

    @event_logger_context(
        lambda _self, rollout_id, trainer_model_id=None, updates_before_train=0: dict(rollout_id=rollout_id)
    )
    async def get(
        self, rollout_id: int, trainer_model_id: str | None = None, *, updates_before_train: int = 0
    ) -> RolloutDataPack:
        start_time = time.time()
        self.rollout_id = rollout_id
        self._rollouts_since_publish_of_model_id[trainer_model_id] += 1
        assert_weight_version_is_published(
            self.args, rollouts_since_publish=self._rollouts_since_publish_of_model_id[trainer_model_id]
        )
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        with timer("rollout" if trainer_model_id is None else f"{trainer_model_id}/rollout"):
            data, metadata, metrics, debug_metadata, batch_token = await self._get_rollout_data(
                rollout_id=rollout_id, trainer_model_id=trainer_model_id, updates_before_train=updates_before_train
            )
        dump_metadata = dict(metadata)
        if debug_metadata is not None:
            dump_metadata[ROLLOUT_FN_DEBUG_METADATA_KEY] = debug_metadata
        save_debug_rollout_data(
            self.args,
            data,
            rollout_id=rollout_id,
            evaluation=False,
            metadata=dump_metadata,
            trainer_model_id=trainer_model_id,
        )
        log_rollout_data(
            rollout_id, self.args, data, metrics, time.time() - start_time, trainer_model_id=trainer_model_id
        )
        if self.args.fully_async:
            self._fully_async_consumption_snapshots[rollout_id] = build_batch_consumption_snapshot(
                data,
                optimizer_updates=len(data)
                // int(metadata.get("dynamic_global_batch_size", self.args.global_batch_size)),
                cohort_generated_tokens=(
                    int(metrics[GENERATED_TOKENS_KEY])
                    if metrics is not None and GENERATED_TOKENS_KEY in metrics
                    else None
                ),
                has_custom_converter=self.custom_convert_samples_to_train_data_func is not None,
            )
        metadata["training_step"] = rollout_id
        data = convert_samples_to_train_data(
            self.args,
            data,
            metadata=metadata,
            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
            custom_reward_post_process_func=self.custom_reward_post_process_func,
        )
        sample_indices = data.get("sample_indices")
        if self.args.delay_split_train_data_by_dp:
            data_ref = object_store.get_instance().put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        else:
            data_ref = split_train_data_by_dp(
                self.args, data, self._train_parallel_configs_of_model_id[trainer_model_id]
            )
        return RolloutDataPack(sample_indices=sample_indices, data_ref=data_ref, replay_buffer_batch_token=batch_token)

    async def eval(
        self,
        rollout_id: int,
        hf_dir: str | None = None,
        export_time_seconds: float | None = None,
        require_marker: bool = True,
    ) -> None:
        if self.args.debug_train_only and not self.args.eval_uses_snapshots:
            return

        if self.args.eval_uses_snapshots:
            return await self._eval_checkpoint(rollout_id, hf_dir, export_time_seconds, require_marker)

        with timer("eval_rollout"):
            if not self.use_legacy_rollout_v1:
                result = await asyncio.to_thread(
                    call_rollout_function, self.eval_generate_rollout, RolloutFnEvalInput(rollout_id=rollout_id)
                )
            else:
                result = await asyncio.to_thread(
                    call_rollout_fn,
                    self.eval_generate_rollout,
                    self.args,
                    rollout_id,
                    self.data_source,
                    evaluation=True,
                )
        data = result.data
        save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
        metrics = log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    async def _eval_checkpoint(
        self, rollout_id: int, hf_dir: str | None, export_time_seconds: float | None, require_marker: bool
    ):
        """Evaluate a snapshot through the checkpoint eval fn (fleet or external
        backend) and log at ``rollout_id``. Every failure degrades to a skipped
        point; the lock serializes pins against a single backend."""
        assert hf_dir is not None, "checkpoint eval requires an HF snapshot dir"
        start_time = time.time()
        async with self._eval_lock:
            if require_marker and not is_complete_hf_export(hf_dir):
                logger.warning(f"Eval snapshot {hf_dir} missing or incomplete, skipping eval {rollout_id}")
                return self.report_eval_skip(rollout_id, "ckpt_missing")

            version = str(rollout_id)
            try:
                state = await self._eval_fleet.pin(hf_dir, version) if self._eval_fleet else None
                eval_input = RolloutFnEvalInput(
                    rollout_id=rollout_id, weight_version=version, hf_dir=hf_dir, generate_state=state
                )
                result = await asyncio.to_thread(call_rollout_function, self.eval_generate_rollout, eval_input)
            except EvalSkip as e:
                return self.report_eval_skip(rollout_id, e.reason)

            data = result.data
            save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
            extra_metrics = dict(result.metrics or {})
            extra_metrics["eval/lag_steps"] = max(self.rollout_id - rollout_id, 0)
            extra_metrics["eval/duration_seconds"] = time.time() - start_time
            if export_time_seconds is not None:
                extra_metrics["eval/export_time_seconds"] = export_time_seconds
            metrics = log_eval_rollout_data(rollout_id, self.args, data, extra_metrics)
            if self._metric_checker is not None:
                self._metric_checker.on_eval(metrics)

    def report_eval_skip(self, rollout_id: int, reason: str) -> None:
        log_eval_skip(rollout_id, self.args, reason)
        if self.args.ci_test:
            raise RuntimeError(f"CI eval {rollout_id} skipped: {reason}")

    async def _get_rollout_data(
        self, rollout_id: int, trainer_model_id: str | None = None, *, updates_before_train: int = 0
    ):
        batch_token = None
        replay_buffer_sample_indices = None
        if self.args.load_debug_rollout_data is not None:
            data, metadata = load_debug_rollout_data(self.args, rollout_id=rollout_id)
            metadata = dict(metadata)
            debug_metadata = metadata.pop(ROLLOUT_FN_DEBUG_METADATA_KEY, None)
            metrics = None
        else:
            if not self.use_legacy_rollout_v1:
                data = await asyncio.to_thread(
                    call_rollout_function,
                    self.generate_rollout,
                    RolloutFnTrainInput(
                        rollout_id=rollout_id,
                        weight_version=self._weight_versions_of_model_id.get(trainer_model_id),
                        trainer_model_id=trainer_model_id,
                        updates_before_train=updates_before_train,
                    ),
                )
            else:
                data = await asyncio.to_thread(
                    call_rollout_fn, self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
                )
            metrics = data.metrics
            if self._resume_benchmark_load_metrics:
                metrics = {**(metrics or {}), **self._resume_benchmark_load_metrics}
                self._resume_benchmark_load_metrics = {}
            debug_metadata = data.debug_metadata
            data = data.samples
            if getattr(self.args, "use_replay_buffer", False):
                batch_token = rollout_batch_token(data)
                replay_buffer_sample_indices = [
                    sample.index for group in data for sample in _iter_group_samples(group)
                ]
            data, metadata = postprocess_rollout_data(
                self.args, data, train_parallel_config=self._train_parallel_configs_of_model_id[trainer_model_id]
            )
            assert_samples_weight_version_sane(self.args, samples=data)
            finalize_useful_rollout_metrics(
                data,
                metrics,
                has_custom_converter=self.custom_convert_samples_to_train_data_func is not None,
                zero_loss_on_truncated=getattr(self.args, "zero_loss_on_truncated", False),
            )
            append_final_consumed_records(
                debug_metadata,
                data,
                reference_mode=getattr(self.args, "staleness_reference", "completion"),
                bound=getattr(self.args, "max_weight_staleness", None),
                training_step=rollout_id,
            )
            if replay_buffer_sample_indices is not None:
                postprocessed_indices = [sample.index for sample in data]
                if postprocessed_indices != replay_buffer_sample_indices:
                    raise RuntimeError(
                        "The replay buffer cannot acknowledge a partially consumed prepared batch: "
                        f"prepared_indices={replay_buffer_sample_indices}, trained_indices={postprocessed_indices}. "
                        "Use a batch shape that does not trim generated samples."
                    )
            if RolloutDataInjectionUtil.should_inject(self.args, rollout_id):
                generated_data = data
                data, metadata = RolloutDataInjectionUtil.load(self.args, rollout_id=rollout_id)
                metadata = dict(metadata)
                debug_metadata = metadata.pop(ROLLOUT_FN_DEBUG_METADATA_KEY, None)
                RolloutDataInjectionUtil.assert_matches_generated(
                    self.args, generated=generated_data, injected=data, rollout_id=rollout_id
                )
                metrics = None

        return data, metadata, metrics, debug_metadata, batch_token

    # -------------------------- checkpointing -----------------------------

    # TODO the train and eval rollout functions will become one object, so one save/load is enough here
    async def save(self, rollout_id: int) -> dict[str, float] | None:
        replay_state = None
        replay_capture_seconds = 0.0
        replay_write_seconds = 0.0
        replay_total_seconds = 0.0
        replay_size_bytes = 0
        if getattr(self.args, "use_replay_buffer", False):
            replay_buffer_start = time.monotonic()
            replay_state = await asyncio.to_thread(run, self.generate_rollout.replay_buffer_state(rollout_id))
            replay_capture_seconds = time.monotonic() - replay_buffer_start
            write_start = time.monotonic()
            path, replay_size_bytes = await asyncio.to_thread(
                save_replay_buffer,
                self.args.save,
                rollout_id,
                replay_state,
            )
            replay_write_seconds = time.monotonic() - write_start
            replay_total_seconds = time.monotonic() - replay_buffer_start
            logger.info(
                "Published replay buffer %s "
                "(%d bytes, capture %.3f seconds, write %.3f seconds, total %.3f seconds)",
                path,
                replay_size_bytes,
                replay_capture_seconds,
                replay_write_seconds,
                replay_total_seconds,
            )
        else:
            self.data_source.save(rollout_id)
            if not self.use_legacy_rollout_v1:
                if self.generate_rollout is not None:
                    self.generate_rollout.save(rollout_id)
                if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                    eval_fn.save(rollout_id)
        checkpoint_metrics = None
        if (
            getattr(self.args, "log_replay_resume_metrics", False)
            or getattr(
                self.args,
                "debug_fail_after_rollout",
                None,
            )
            is not None
        ):
            data_source_state = (
                replay_state["data_source"] if replay_state is not None else self.data_source.checkpoint_state()
            )
            checkpoint_metrics = checkpoint_resume_metrics(
                rollout_id=rollout_id,
                rollout_batch_size=self.args.rollout_batch_size,
                n_samples_per_prompt=self.args.n_samples_per_prompt,
                data_source_state=data_source_state,
                replay_state=replay_state,
            )
            checkpoint_metrics.update(
                {
                    "resume/benchmark/checkpoint/replay_size_bytes": float(replay_size_bytes),
                    "resume/benchmark/checkpoint/replay_capture_seconds": replay_capture_seconds,
                    "resume/benchmark/checkpoint/replay_write_seconds": replay_write_seconds,
                    "resume/benchmark/checkpoint/replay_total_seconds": replay_total_seconds,
                }
            )
            if getattr(self.args, "log_replay_resume_metrics", False):
                log_replay_resume_checkpoint(rollout_id, self.args, checkpoint_metrics)
        if not getattr(self.args, "use_replay_buffer", False):
            event_logger_checkpoint.snapshot(self.args, rollout_id)
        return checkpoint_metrics

    async def load(self, rollout_id: int | None = None) -> None:
        if not getattr(self.args, "use_replay_buffer", False):
            ensure_no_replay_buffer(self.args.load, rollout_id)
            load_start = time.monotonic()
            self.data_source.load(rollout_id)
            if not self.use_legacy_rollout_v1:
                if self.generate_rollout is not None:
                    self.generate_rollout.load(rollout_id)
                if (eval_fn := self.eval_generate_rollout) is not None and eval_fn is not self.generate_rollout:
                    eval_fn.load(rollout_id)
            if getattr(self.args, "log_replay_resume_metrics", False) and rollout_id is not None and rollout_id >= 0:
                total_seconds = time.monotonic() - load_start
                self._resume_benchmark_load_metrics = replay_load_metrics(
                    replay_type=None,
                    read_seconds=total_seconds,
                    restore_seconds=0.0,
                    total_seconds=total_seconds,
                )
            return
        if self.args.load is None or rollout_id is None or rollout_id < 0:
            return
        fingerprint = self.generate_rollout.replay_buffer_dataset_fingerprint()
        load_start = time.monotonic()
        state = await asyncio.to_thread(
            load_replay_buffer,
            self.args.load,
            rollout_id,
            expected_fingerprint=fingerprint,
        )
        read_seconds = time.monotonic() - load_start
        restore_start = time.monotonic()
        await asyncio.to_thread(run, self.generate_rollout.restore_replay_buffer_state(state))
        restore_seconds = time.monotonic() - restore_start
        total_seconds = time.monotonic() - load_start
        if getattr(self.args, "log_replay_resume_metrics", False):
            self._resume_benchmark_load_metrics = replay_load_metrics(
                replay_type=self.args.replay_buffer_type,
                read_seconds=read_seconds,
                restore_seconds=restore_seconds,
                total_seconds=total_seconds,
            )
        logger.info(
            "Loaded replay buffer from %s at rollout %d "
            "(read %.3f seconds, restore %.3f seconds, total %.3f seconds)",
            self.args.load,
            rollout_id,
            read_seconds,
            restore_seconds,
            total_seconds,
        )

    async def set_applied_weight_version(self, version: int) -> None:
        """Commit after every rollout engine finalized the same weight update."""
        commit_on_loop = getattr(self.generate_rollout, "commit_applied_weight_version_on_loop", None)
        if commit_on_loop is not None:
            await asyncio.to_thread(run, commit_on_loop(version))
            return
        commit = getattr(self.generate_rollout, "commit_applied_weight_version", None)
        if commit is not None:
            commit(version)

    async def get_current_applied_weight_version(self) -> int:
        current_on_loop = getattr(self.generate_rollout, "current_applied_weight_version", None)
        if current_on_loop is not None:
            return await asyncio.to_thread(run, current_on_loop())
        return self._weight_versions_of_model_id.get(None, 0)

    async def record_batch_consumption(self, rollout_id: int) -> dict[str, float | int]:
        """Close consumption telemetry immediately before training."""
        snapshot = self._fully_async_consumption_snapshots.pop(
            rollout_id,
            {},
        )
        metrics = batch_consumption_metrics(snapshot)
        raw_accepted_tokens = snapshot.get("loss_input_tokens")
        accepted_tokens = int(raw_accepted_tokens) if isinstance(raw_accepted_tokens, int) else None
        raw_generated_tokens = snapshot.get("cohort_generated_tokens")
        cohort_generated_tokens = int(raw_generated_tokens) if isinstance(raw_generated_tokens, int) else None
        self._fully_async_inflight_training[rollout_id] = (
            accepted_tokens,
            int(snapshot.get("optimizer_updates", 1)),
            cohort_generated_tokens,
        )
        return log_rollout_batch_consumption(
            rollout_id,
            self.args,
            extra_metrics=metrics,
        )

    async def record_batch_trained(
        self,
        rollout_id: int,
        *,
        actor_trained: bool,
    ) -> dict[str, float | int] | None:
        """Record completed actor work after training succeeds."""
        if rollout_id not in self._fully_async_inflight_training:
            raise RuntimeError(f"Missing fully-async consumption snapshot for trained rollout {rollout_id}")
        accepted_tokens, optimizer_updates, cohort_generated_tokens = self._fully_async_inflight_training.pop(
            rollout_id
        )
        if not actor_trained:
            return None
        complete_on_loop = getattr(self.generate_rollout, "complete_trained_batch_telemetry_on_loop", None)
        if complete_on_loop is None:
            return None
        pipeline_snapshot = await asyncio.to_thread(
            run,
            complete_on_loop(
                accepted_tokens=accepted_tokens,
                optimizer_updates=optimizer_updates,
            ),
        )
        metrics = pipeline_throughput_metrics(
            pipeline_snapshot,
            cohort_accepted_tokens=accepted_tokens,
            cohort_generated_tokens=cohort_generated_tokens,
        )
        return log_rollout_pipeline_throughput(
            rollout_id,
            self.args,
            metrics,
        )

    async def acknowledge_trained_batch(self, rollout_id: int, token: str | None) -> None:
        if not getattr(self.args, "use_replay_buffer", False):
            return
        if token is None:
            raise RuntimeError(f"Missing replay-buffer batch token for trained rollout {rollout_id}")
        await asyncio.to_thread(run, self.generate_rollout.acknowledge_trained_batch(rollout_id, token))

    async def get_restored_applied_weight_version(self) -> int | None:
        if not getattr(self.args, "use_replay_buffer", False) or self.args.load is None:
            return None
        return await asyncio.to_thread(run, self.generate_rollout.current_applied_weight_version())

    async def mark_replay_buffer_committed(self, rollout_id: int) -> None:
        if not getattr(self.args, "use_replay_buffer", False):
            return
        # This snapshot lives inside iter_N, so write it only after the model
        # saver has durably published that directory and its tracker.
        event_logger_checkpoint.snapshot(self.args, rollout_id)
        await asyncio.to_thread(
            prune_replay_buffers,
            self.args.save,
            current_rollout_id=rollout_id,
            keep_last=self.args.replay_buffer_keep_last,
            archive_interval=self.args.save_retain_interval,
        )

    # -------------------------- misc APIs -----------------------------

    def get_num_rollout_per_epoch(self) -> int:
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def set_weight_version(self, weight_version: int, trainer_model_id: str | None = None) -> None:
        # warning instead of assert when use indep_dp ft
        previous = self._weight_versions_of_model_id.get(trainer_model_id)
        if previous is not None and weight_version < previous:
            message = f"Engine weight version went backwards: {previous} -> {weight_version}"
            assert self.args.indep_dp, message
            logger.warning(message)
        self._weight_versions_of_model_id[trainer_model_id] = weight_version
        self._rollouts_since_publish_of_model_id[trainer_model_id] = 0

    def set_train_parallel_config(self, config: dict[str, Any], trainer_model_id: str | None = None) -> None:
        self._train_parallel_configs_of_model_id[trainer_model_id] = config

    async def set_eval_fleet_info(self, eval_fleet_info: EvalFleetInfo | None) -> None:
        if eval_fleet_info is None:
            self._eval_fleet = None
            return

        self._eval_fleet = RolloutExecutorEvalFleet(
            self.args, info=eval_fleet_info, inference_controller_provider=self._inference_controller_provider
        )
