import asyncio
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.metrics import (
    log_eval_rollout_data,
    log_replay_resume_checkpoint,
    log_rollout_batch_consumption,
    log_rollout_data,
    log_rollout_pipeline_throughput,
)
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers
from miles.ray.rollout.router_manager import start_session_server
from miles.ray.rollout.server_cell import get_cell_indexer_of_id_map
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
    split_train_data_by_dp,
)
from miles.ray.utils import Lock
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
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
from miles.utils.async_utils import run
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.process_identity import RolloutManagerProcessIdentity
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.health_monitor import RolloutHealthMonitor
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger
from miles.utils.metric_checker import MetricChecker
from miles.utils.misc import load_function
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.types import Sample

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

ROLLOUT_FN_DEBUG_METADATA_KEY = "rollout_fn_debug"


def _iter_group_samples(group: list[Sample | list]) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from _iter_group_samples(item)
        else:
            yield item


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        event_logger_checkpoint.restore(args)
        configure_logger(args, source=RolloutManagerProcessIdentity())

        self.pg = pg
        self.args = args
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        object_store.init_instance(args, contribute_segment=False)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.use_experimental_refactor = enable_experimental_rollout_refactor()
        if self.use_experimental_refactor:
            input = RolloutFnConstructorInput(args=args, data_source=self.data_source)
            self.generate_rollout = load_rollout_function(input, self.args.rollout_function_path)
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
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)
            start_session_server(args)
            dashboard_hooks.register_router(args)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self._fully_async_consumption_snapshots: dict[int, dict] = {}
        self._fully_async_inflight_training: dict[int, tuple[int | None, int, int | None]] = {}
        self._resume_benchmark_load_metrics: dict[str, float] = {}

        self._metric_checker = MetricChecker.maybe_create(args)

        # TODO will be replaced by full ft, thus temporarily leave it without modifications
        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test and "rollout" in self.args.ft_components

    # -------------------------- lifecycle -----------------------------
    # TODO: may have a `async def init` here later

    def get_router_address(self) -> tuple[str, int]:
        return self.args.sglang_router_ip, self.args.sglang_router_port

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
        return 0

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

    async def dispose(self):
        if (shutdown := getattr(self.generate_rollout, "shutdown", None)) is not None:
            await asyncio.to_thread(run, shutdown())
        if (close := getattr(self.data_source, "close", None)) is not None:
            close()
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        for monitor in self._health_monitors:
            monitor.stop()

    # -------------------------- data generation -----------------------------

    async def generate(self, rollout_id, *, updates_before_train: int = 0):
        start_time = time.time()
        self.rollout_id = rollout_id
        self._health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        dashboard_hooks.register_engines(self.servers)
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        with timer("rollout"):
            data, metadata, metrics, debug_metadata, batch_token = await self._get_rollout_data(
                rollout_id=rollout_id,
                updates_before_train=updates_before_train,
            )
        dump_metadata = dict(metadata)
        if debug_metadata is not None:
            dump_metadata[ROLLOUT_FN_DEBUG_METADATA_KEY] = debug_metadata
        save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=False, metadata=dump_metadata)
        log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.fully_async:
            self._fully_async_consumption_snapshots[rollout_id] = build_batch_consumption_snapshot(
                data,
                optimizer_updates=(
                    len(data) // int(metadata.get("dynamic_global_batch_size", self.args.global_batch_size))
                ),
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
            data_ref = split_train_data_by_dp(self.args, data, self.train_parallel_config["dp_size"])
        result = dict(sample_indices=sample_indices, data_ref=data_ref)
        if getattr(self.args, "use_replay_buffer", False):
            result["replay_buffer_batch_token"] = batch_token
        return result

    async def acknowledge_trained_batch(self, rollout_id: int, token: str | None) -> None:
        if not getattr(self.args, "use_replay_buffer", False):
            return
        if token is None:
            raise RuntimeError(f"Missing replay-buffer batch token for trained rollout {rollout_id}")
        await asyncio.to_thread(run, self.generate_rollout.acknowledge_trained_batch(rollout_id, token))

    async def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self._health_monitoring_resume()

        with timer("eval_rollout"):
            if self.use_experimental_refactor:
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

    async def _get_rollout_data(self, rollout_id: int, *, updates_before_train: int = 0):
        batch_token = None
        replay_buffer_sample_indices = None
        if self.args.load_debug_rollout_data:
            data, metadata = load_debug_rollout_data(self.args, rollout_id=rollout_id)
            metadata = dict(metadata)
            debug_metadata = metadata.pop(ROLLOUT_FN_DEBUG_METADATA_KEY, None)
            metrics = None
        else:
            if self.use_experimental_refactor:
                data = await asyncio.to_thread(
                    call_rollout_function,
                    self.generate_rollout,
                    RolloutFnTrainInput(
                        rollout_id=rollout_id,
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
                self.args, data, train_parallel_config=self.train_parallel_config
            )
            finalize_useful_rollout_metrics(
                data,
                metrics,
                has_custom_converter=self.custom_convert_samples_to_train_data_func is not None,
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

    # -------------------------- rollout persistence -----------------------

    async def save(self, rollout_id):
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
        elif self.args.rollout_global_dataset:
            self.data_source.save(rollout_id)
        checkpoint_metrics = None
        if getattr(self.args, "log_replay_resume_metrics", False) or getattr(
            self.args,
            "debug_fail_after_rollout",
            None,
        ) is not None:
            data_source_state = (
                replay_state["data_source"]
                if replay_state is not None
                else self.data_source.checkpoint_state()
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

    async def load(self, rollout_id=None):
        if not getattr(self.args, "use_replay_buffer", False):
            ensure_no_replay_buffer(self.args.load, rollout_id)
            load_start = time.monotonic()
            self.data_source.load(rollout_id)
            if (
                getattr(self.args, "log_replay_resume_metrics", False)
                and rollout_id is not None
                and rollout_id >= 0
            ):
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

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    async def offload(self, tags: list[str] | None = None):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    async def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            await srv.onload(tags)

    async def onload_weights(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    # -------------------------- engine management -----------------------------

    async def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        srv = self._get_updatable_server()
        if not srv:
            return EnginesAndLock(
                rollout_engines=[],
                rollout_engine_lock=self.rollout_engine_lock,
                has_new_engines=False,
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
            )

        await srv.wait_all_engines_alive()
        return EnginesAndLock(
            rollout_engines=[e.actor_handle for e in srv.engines],
            rollout_engine_lock=self.rollout_engine_lock,
            has_new_engines=srv.has_new_engines,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
        )

    def clear_updatable_has_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear has_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.clear_has_new_engines()

    async def recover_updatable_engines(self) -> None:
        """Restart any dead rollout engines and update has_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            return

        await srv.recover()

    def _get_updatable_server(self) -> RolloutServer | None:
        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        match updatable:
            case []:
                return None
            case [srv]:
                return srv
            case _:
                raise ValueError(
                    f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
                    f"Only one updatable server is supported."
                )

    # -------------------------- external start/stop -----------------------------

    async def start_cell(self, cell_id: int):
        port_cursors = PortCursors.empty()
        idx = get_cell_indexer_of_id_map(self.servers)[cell_id]
        group = self.servers[idx.srv_key].server_groups[idx.group_index]
        await group.recover(port_cursors=port_cursors, filter_indices=idx.engine_indices)

    async def stop_cell(self, cell_id: int):
        idx = get_cell_indexer_of_id_map(self.servers)[cell_id]
        group = self.servers[idx.srv_key].server_groups[idx.group_index]
        group.stop_engines(engine_indices=idx.engine_indices)

    # -------------------------- misc APIs -----------------------------

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server()
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    # -------------------------- utils -----------------------------

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def _health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    @property
    def _server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    # TODO will be replaced by full ft, thus temporarily leave it without modifications
    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if (
            self._server
            and self._server.server_groups[0].all_engines
            and self._server.server_groups[0].all_engines[0].is_allocated
        ):
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self._server.server_groups[0].all_engines[0].actor_handle.simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")


@dataclass(frozen=True)
class EnginesAndLock:
    rollout_engines: list[ray.actor.ActorHandle]
    rollout_engine_lock: ray.actor.ActorHandle
    has_new_engines: bool
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
