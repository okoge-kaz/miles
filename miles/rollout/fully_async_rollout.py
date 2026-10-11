"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import copy
import logging
from dataclasses import replace

from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.filter_hub.base_types import iter_samples
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    Group,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.replay_data_buffer import ReplayDataBuffer
from miles.rollout.replay_data_buffer_inflight import validate_inflight_group
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils import async_utils
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0


class FullyAsyncRolloutFn(BaseRolloutFn):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else (lambda prompt_group: None)
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None
        self._weight_version: int | None = None
        self._active: dict[asyncio.Task, list[Sample]] = {}
        self._original_prompts: dict[asyncio.Task, list[Sample]] = {}
        self._pending_ready: dict[asyncio.Task, DataBufferInput] = {}
        self._draining = False
        self._replay_data_buffer = ReplayDataBuffer(self) if getattr(self.args, "use_replay_data_buffer", False) else None
        self._track_lifecycle = (
            self._replay_data_buffer is not None
            or getattr(self.args, "log_category_staleness", False)
            or getattr(self.args, "async_staleness_metric", "oldest") in ("submission", "ready", "first-prefill")
        )

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if input.weight_version is not None:
            self.on_weight_version(input.weight_version)
        if self._worker is None:
            self._ensure_buffer()
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        self._draining = True
        try:
            result = await self._drain(input)
            if self._replay_data_buffer is not None:
                result.metrics.update(self._replay_data_buffer.pop_metrics())
            return result
        finally:
            self._draining = False

    def _ensure_buffer(self) -> None:
        if self._output is None:
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or DefaultDataBuffer
            self._output = buffer_cls(DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused))

    def on_weight_version(self, version: int) -> None:
        self._weight_version = version

    @property
    def owns_data_source_checkpoint(self) -> bool:
        return self._replay_data_buffer is not None

    def save(self, rollout_id: int) -> None:
        if self._replay_data_buffer is not None:
            self._replay_data_buffer.save(rollout_id)

    def load(self, rollout_id: int | None) -> None:
        if self._replay_data_buffer is not None:
            self._replay_data_buffer.load(rollout_id)

    def close(self) -> None:
        async_utils.run(self._close())

    async def _close(self) -> None:
        tasks = [*self._active, *([self._worker] if self._worker is not None else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._replay_data_buffer is not None:
            await self._replay_data_buffer.close()

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> tuple[asyncio.Task, list[Sample]]:
        inflight = self._replay_data_buffer.inflight if self._replay_data_buffer is not None else None
        if inflight is not None and inflight.pending:
            entry = inflight.pending.popleft()
            validate_inflight_group(entry.group)
            self._scheduler.on_submit([entry.group])
            task = asyncio.create_task(self._generate_group(entry.group))
            self._original_prompts[task] = entry.prompt_group
            return task, entry.group
        samples = self.data_source.get_samples(1)
        if inflight is not None:
            validate_inflight_group(samples[0])
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        if self._track_lifecycle:
            for sample in prompt_group:
                if sample.metadata:
                    sample.metadata = {
                        key: value for key, value in sample.metadata.items() if key != "policy_provenance"
                    }
            self._stamp_lifecycle(prompt_group, "submitted_weight_version", self._weight_version)
        original = copy.deepcopy(prompt_group) if self._replay_data_buffer is not None else None
        task = asyncio.create_task(self._generate_group(prompt_group))
        if original is not None:
            for sample in original:
                sample.reset_for_retry()
            self._original_prompts[task] = original
        return task, prompt_group

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        if self._track_lifecycle:
            self._stamp_lifecycle(result, "ready_weight_version", self._weight_version)
        return DataBufferInput(prompt_group=prompt_group, group=result)

    @staticmethod
    def _stamp_lifecycle(group, key: str, version: int | None) -> None:
        if version is None:
            return
        for sample in iter_samples(group):
            metadata = dict(sample.metadata or {})
            provenance = dict(metadata.get("policy_provenance") or {})
            provenance[key] = version
            metadata["policy_provenance"] = provenance
            sample.metadata = metadata

    async def _worker_loop(self) -> None:
        if self._replay_data_buffer is not None:
            await self._replay_data_buffer.before_production()
        active = self._active
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                task, prompt_group = self._submit_one_group()
                active[task] = prompt_group
            done, _ = await self._scheduler.wait_for_progress(set(active))
            for task in done:
                entry = self._collect_group_result(task, active[task])
                self._pending_ready[task] = entry
                await self._output.put(entry)
                active.pop(task)
                self._pending_ready.pop(task)
                self._original_prompts.pop(task, None)

    def _collect_group_result(self, task: asyncio.Task, prompt_group: list[Sample]) -> DataBufferInput:
        if not task.cancelled():
            entry = task.result()
            if self._replay_data_buffer is not None:
                entry.prompt_group = self._original_prompts[task]
            return entry

        logger.warning(
            "Rollout group was cancelled; marking samples aborted: indices=%s",
            [sample.index for sample in prompt_group],
        )
        return DataBufferInput(
            prompt_group=prompt_group,
            group=[replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group],
        )

    # -------------------------- consumer --------------------------

    async def _next_group(self, current_version: int | None) -> DataBufferInput:
        if self._replay_data_buffer is not None:
            self._replay_data_buffer.check_restore()
        queue_get = asyncio.create_task(self._output.get(current_version=current_version))
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        do_print = True

        while len(data) < target_data_size:
            entry = await self._next_group(input.weight_version)
            assert len(entry.group) == args.n_samples_per_prompt

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            data.append(entry.group)

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        return RolloutFnTrainOutput(samples=data, metrics=self._output.get_metrics())

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
            if sample.metadata:
                sample.metadata = {key: value for key, value in sample.metadata.items() if key != "policy_provenance"}
        self.data_source.add_samples([prompt_group])
