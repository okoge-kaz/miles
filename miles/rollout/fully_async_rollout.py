"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
worker's output queue. Rollout production and training consumption run in parallel,
so per-iteration wall time moves from ``rollout_time + train_time`` toward
``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which also requires the class-based
rollout API (``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1``).

Evaluation is not served by this function; ``--fully-async`` therefore points
``--eval-function-path`` at the standard inference rollout unless it is set
explicitly.
"""

import asyncio
import logging
import time
from collections import deque

import httpx
import numpy as np

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnInput, RolloutFnOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.queue_policy import LEGACY_QUEUE_POLICY, QUEUE_DROP_POLICY, QUEUE_MAX_POLICY
from miles.rollout.queue_telemetry import (
    Group,
    _distribution_metrics,
    _first_sample,
    _iter_samples,
    _QueueLifecycleRecorder,
    _ResponseLengthMetrics,
    group_first_prefill_weight_version,
    group_oldest_weight_version,
    group_queue_entry_weight_version,
    group_response_tokens,
)
from miles.utils.http_utils import get
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

OUTPUT_QUEUE_MAX_GROUPS = 1000
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0
# Realized lag is a small integer; anything past this goes in one overflow bucket
# so the metric count stays bounded no matter how far behind a run drifts.
#
# 16, not 8: unbounded runs and runs with a parked bound need enough resolution
# in the tail to show how realized staleness maps to downstream score. At 8 the
# tail collapsed into one overflow bucket.
STALENESS_HISTOGRAM_MAX = 16

QueueItem = tuple[list[Sample], Group]


SUBMISSION_VERSION_KEY = "submission_weight_version"
GROUP_READY_VERSION_KEY = "group_ready_weight_version"
QUEUE_PUT_VERSION_KEY = "queue_put_weight_version"
DRAIN_VERSION_KEY = "drain_weight_version"


class AppliedWeightVersionTracker:
    """The version committed after every rollout engine acknowledges finalization."""

    def __init__(self, initial_version: int = 0):
        self._version = initial_version

    def commit(self, version: int) -> None:
        if version < self._version:
            raise ValueError(f"Applied weight version cannot move backwards: current={self._version}, new={version}")
        self._version = version

    def current(self) -> int:
        return self._version


def stamp_group_weight_version(group: Group, key: str, version: int) -> None:
    for sample in _iter_samples(group):
        sample.metadata[key] = version


def group_lifecycle_weight_version(group: Group, key: str) -> int | None:
    versions = [sample.metadata.get(key) for sample in _iter_samples(group)]
    numeric_versions = [version for version in versions if isinstance(version, int)]
    if not numeric_versions:
        return None
    if len(numeric_versions) != len(versions) or len(set(numeric_versions)) != 1:
        raise RuntimeError(f"Inconsistent {key} across rollout group: {versions}")
    return numeric_versions[0]


def group_has_mixed_forward_versions(group: Group) -> bool:
    minimums = [version for sample in _iter_samples(group) for version in sample.min_forward_weight_versions]
    maximums = [version for sample in _iter_samples(group) for version in sample.max_forward_weight_versions]
    return bool(minimums and maximums and min(minimums) != max(maximums))


def validate_prefill_policy_provenance(group: Group) -> None:
    fields = (
        "first_prefill_weight_versions",
        "min_forward_weight_versions",
        "max_forward_weight_versions",
        "last_forward_weight_versions",
    )
    for sample in _iter_samples(group):
        values_by_field = {field: getattr(sample, field) for field in fields}
        lengths = {len(values) for values in values_by_field.values()}
        if lengths == {0} or len(lengths) != 1:
            raise RuntimeError(
                "SGLang response is missing aligned prefill policy provenance for "
                f"sample {sample.index}: {values_by_field}. Use the patched SGLang image."
            )
        invalid = {
            field: values for field, values in values_by_field.items() if any(version < 0 for version in values)
        }
        if invalid:
            raise RuntimeError(
                f"SGLang returned invalid prefill policy provenance for sample {sample.index}: {invalid}"
            )


def stamp_submission_weight_version(group: Group, version: int | None) -> None:
    """Record the version the engines were serving when generation started.

    SGLang builds ``meta_info["weight_version"]`` from ``server_args.weight_version``
    while handling a batch-output message (``tokenizer_manager.py:1982`` in
    0.5.17.dev32+g3fe50ed, the build in the image), and ``Req`` carries no version
    of its own, so there is nothing to snapshot at arrival: ``Sample.weight_versions``
    only ever holds the version a turn *finished* under. Under
    ``--pause-generation-mode in_place`` a request is frozen across a weight
    update and resumed on the same KV cache: one response, no retraction, so
    ``num_retractions`` is zero too and nothing in the reply records that the
    sample spans versions. This stamp is the other end of that interval.

    Survives ``reset_for_retry`` (``types.py:240-249`` keeps ``metadata``), which
    is harmless because every submission re-stamps.
    """
    if version is None:
        return
    for sample in _iter_samples(group):
        sample.metadata[SUBMISSION_VERSION_KEY] = version


def group_submission_weight_version(group: Group) -> int | None:
    """Return the weight version stamped on the group when it was submitted."""
    versions = [v for s in _iter_samples(group) if isinstance(v := s.metadata.get(SUBMISSION_VERSION_KEY), int)]
    return min(versions) if versions else None


# Which name carries weight_version depends on the router, so it is discovered
# rather than assumed. Checked against sglang 0.5.17:
#
#   engine (uvicorn)  /model_info is canonical; /get_model_info delegates to it
#                     with a deprecation notice; /get_weight_version raises
#                     (http_server.py:727-733), so it can never answer
#   sglang_router     has its own /model_info route, which 404s; /get_model_info
#                     is not a route, falls through to the engine, and answers
#   MilesRouter       proxies everything (router.py:71), so both reach the engine
#
# Canonical name first: it is the one that survives, and it is right for
# --use-miles-router and for any router that proxies. The fallback covers
# sglang_router as it stands. A wrong first guess costs one 404 round trip, once,
# because the name that answers is remembered.
WEIGHT_VERSION_ENDPOINTS = ("/model_info", "/get_model_info")


class _CachedWeightVersion:
    """Throttled query of the current engine weight version via the router."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query = float("-inf")
        self._failures = 0
        self._endpoint: str | None = None
        self._lock = asyncio.Lock()

    async def get(self, args) -> int | None:
        # Throttles failures too: the drain queries once per group, and an unreachable
        # router would otherwise cost every one of them the full timeout.
        if (time.monotonic() - self._last_query) < self._ttl:
            return self._value
        async with self._lock:
            # Re-checked after the wait, because the submission side calls this from
            # every in-flight group concurrently: the first fill starts
            # `_max_in_flight_groups` tasks in one batch, and without single-flight
            # each would put its own /model_info request on the router.
            if (time.monotonic() - self._last_query) < self._ttl:
                return self._value
            return await self._query(args)

    async def _query(self, args) -> int | None:
        base = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        # Remembered name first, then the rest: preferring it costs nothing, and
        # keeping the others behind it means a router replaced under a resumed run
        # recovers within the same query instead of after another TTL.
        candidates = WEIGHT_VERSION_ENDPOINTS
        if self._endpoint:
            candidates = (self._endpoint, *(e for e in candidates if e != self._endpoint))
        last_error: Exception | None = None
        try:
            for endpoint in candidates:
                try:
                    data = await asyncio.wait_for(get(f"{base}{endpoint}"), timeout=WEIGHT_VERSION_QUERY_TIMEOUT_SECS)
                    self._value = int(data["weight_version"])
                    self._endpoint = endpoint
                    self._failures = 0
                    return self._value
                except (httpx.HTTPStatusError, KeyError, TypeError, ValueError) as e:
                    # The endpoint answered but is not the right one: a 404 for a name
                    # this build does not serve, or a payload with no usable
                    # weight_version. Both mean "try the next name".
                    last_error = e
                except (httpx.HTTPError, asyncio.TimeoutError) as e:
                    # The router itself is unreachable, so every other name is too.
                    # Probing them would multiply the timeout by the endpoint count on
                    # a path that already runs once per group.
                    last_error = e
                    break
            self._failures += 1
            self._warn_on_failure(f"{base}{{{','.join(candidates)}}}", last_error)
        finally:
            # Stamped on completion, so a router slower than the TTL still gets throttled.
            self._last_query = time.monotonic()
        return self._value

    def _warn_on_failure(self, url: str, error: Exception) -> None:
        """Report missing submission diagnostics without affecting control."""
        if self._value is None:
            logger.warning(
                f"Cannot read the engine weight version from {url} ({error!r}). "
                "submission_weight_version diagnostics will be absent; drain-side "
                "staleness control still uses the committed applied-version tracker."
            )
        elif self._failures & (self._failures - 1) == 0:  # 1, 2, 4, 8, ... consecutive
            logger.warning(f"Weight version query failed {self._failures}x, using cached value: {error!r}")


def _staleness_metrics(values: list[int], bound: int | None) -> dict[str, float]:
    """P(L) reduced to bounded scalars: the logger takes scalars, not histograms.

    Percentiles rather than a mean alone because the tail is the quantity of
    interest -- a mean of 0.4 with a p99 of 12 and a mean of 0.4 with a p99 of 1
    are different training regimes. ``frac_at_bound`` says whether the configured
    cap is binding at all; a plateau in a results table is otherwise easy to read
    as insensitivity to staleness when the run was simply never stale.
    """
    array = np.asarray(values, dtype=float)
    metrics = {
        "mean": float(array.mean()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "frac_zero": float((array <= 0).mean()),
        "num_groups": float(array.size),
    }
    if bound is not None:
        # `>=`, not `>`: a group exactly at the bound is kept. This is "how often
        # did the pipeline reach the cap", not the rejection rate -- for that, see
        # `staleness/bound_exceeded_groups`.
        metrics["frac_at_bound"] = float((array >= bound).mean())

    # The full histogram, not just moments. Realized lag is a small integer, so
    # P(L) fits in a handful of scalars, and the shape is the result: percentiles
    # cannot say whether lag 4 happened twice or two hundred times, and that is
    # what decides whether a bound of 4 is a real constraint or a formality.
    # Counts are observations, not distinct groups -- a recycled group is counted
    # again when it is regenerated, which is the honest denominator for "how often
    # did the pipeline produce a sample this stale".
    for level in range(STALENESS_HISTOGRAM_MAX + 1):
        metrics[f"count_{level}"] = float((array == level).sum())
    metrics[f"count_ge_{STALENESS_HISTOGRAM_MAX + 1}"] = float((array > STALENESS_HISTOGRAM_MAX).sum())
    return metrics


class FullyAsyncRolloutFn:
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Groups whose samples were aborted (e.g. by a
    weight update pausing generation) are recycled back into the data source.
    Age-bound failures are recycled by the legacy policy and discarded by
    queue-max; queue-drop instead discards the oldest completed group when its
    bounded queue overflows.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self._dynamic_filter = load_function(input.args.dynamic_sampling_filter_path)
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._weight_version = _CachedWeightVersion()
        self._applied_weight_version = AppliedWeightVersionTracker()
        self._queue_lifecycle = _QueueLifecycleRecorder(
            enabled=getattr(input.args, "save_debug_rollout_data", None) is not None
        )
        # Producer completions continue while the trainer is busy and therefore
        # cannot live in one drain call's local accumulator. Drain and reset these
        # at the same point as the other per-rollout metrics.
        self._producer_response_lengths = _ResponseLengthMetrics(populations=("generated", "queue_evicted"))
        self._queue_evicted_groups = 0
        self._queue_evicted_tokens = 0
        self._worker: asyncio.Task | None = None
        self._output: asyncio.Queue[QueueItem] | None = None
        self._output_slots: asyncio.Semaphore | None = None
        self._policy_output: deque[QueueItem] | None = None
        self._policy_output_ready: asyncio.Event | None = None

    def commit_applied_weight_version(self, version: int) -> None:
        self._applied_weight_version.commit(version)

    def current_applied_weight_version(self) -> int:
        """Return the last weight version finalized on every rollout engine."""
        return self._applied_weight_version.current()

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            raise ValueError(
                "FullyAsyncRolloutFn does not serve eval; set --eval-function-path to "
                "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
            )
        if self._worker is None:
            if self._queue_policy() == LEGACY_QUEUE_POLICY:
                self._output = asyncio.Queue()
                self._output_slots = asyncio.Semaphore(OUTPUT_QUEUE_MAX_GROUPS)
            else:
                self._policy_output = deque()
                self._policy_output_ready = asyncio.Event()
                if self._queue_policy() == QUEUE_MAX_POLICY:
                    self._output_slots = asyncio.Semaphore(self._queue_capacity_groups())
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info(
                "Started fully-async rollout worker (queue_policy=%s, capacity_groups=%d)",
                self._queue_policy(),
                self._queue_capacity_groups(),
            )
        return await self._drain(input.rollout_id)

    def _queue_policy(self) -> str:
        return getattr(self.args, "fully_async_queue_policy", LEGACY_QUEUE_POLICY)

    def _queue_capacity_groups(self) -> int:
        if self._queue_policy() == QUEUE_DROP_POLICY:
            return getattr(self.args, "fully_async_queue_factor", 1) * self.args.rollout_batch_size
        if self._queue_policy() == QUEUE_MAX_POLICY:
            # queue-max waits for a whole batch before dequeueing. Its safety
            # backpressure limit must therefore never be smaller than that batch.
            return max(OUTPUT_QUEUE_MAX_GROUPS, self.args.rollout_batch_size)
        return OUTPUT_QUEUE_MAX_GROUPS

    def _queue_size(self) -> int:
        if self._queue_policy() == LEGACY_QUEUE_POLICY:
            return self._output.qsize()
        return len(self._policy_output)

    # -------------------------- producer --------------------------

    async def _current_weight_version(self) -> int | None:
        """A cached submission-side diagnostic snapshot from the router.

        Guarded because this is now on the submission path as well as the drain,
        and reading it is instrumentation plus bound enforcement -- neither is
        worth killing generation over. An uncaught error here would reach
        ``_worker_loop``'s ``task.result()`` and take the whole rollout down.
        ``_CachedWeightVersion`` stamps its query time in a ``finally``, so a
        persistent failure costs one log line per TTL, not one per group.
        """
        try:
            return await self._weight_version.get(self.args)
        except Exception as e:  # noqa: BLE001 - degrade, never stop generating
            logger.warning(f"Weight version unreadable, staleness not measured for this group: {e!r}")
            return None

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> asyncio.Task:
        [prompt_group] = self.data_source.get_samples(1)
        return asyncio.create_task(self._generate_group(prompt_group))

    async def _generate_group(self, prompt_group: list[Sample]) -> tuple[list[Sample], Group]:
        """Return the submitted prompt group next to its result.

        A retry has to resubmit the prompt group: a generate function may expand one
        trajectory into several samples, and ``generate_and_rm_group`` does not accept
        that shape back.
        """
        submission_version = await self._current_weight_version()
        lifecycle_record = self._queue_lifecycle.begin_attempt(prompt_group, submission_version)
        stamp_submission_weight_version(prompt_group, submission_version)
        try:
            result = await generate_and_rm_group(
                self.state,
                prompt_group,
                sampling_params=self.state.sampling_params.copy(),
                evaluation=False,
            )
        except BaseException:
            self._queue_lifecycle.cancel_attempt(lifecycle_record)
            raise
        # Stamped again on the result: a generate function may return new Sample
        # objects rather than the ones it was handed.
        stamp_submission_weight_version(result, submission_version)
        ready_version = self._applied_weight_version.current()
        stamp_group_weight_version(
            result,
            GROUP_READY_VERSION_KEY,
            ready_version,
        )
        self._queue_lifecycle.group_ready(lifecycle_record, result, ready_version)
        if not any(sample.status == Sample.Status.ABORTED for sample in _iter_samples(result)):
            self._producer_response_lengths.record("generated", result)
        return prompt_group, result

    async def _worker_loop(self):
        active: set[asyncio.Task] = set()
        while True:
            while len(active) < self._max_in_flight_groups():
                active.add(self._submit_one_group())
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await self._enqueue_completed_group(task.result())

    async def _enqueue_completed_group(self, item: QueueItem) -> None:
        policy = self._queue_policy()
        if policy in (LEGACY_QUEUE_POLICY, QUEUE_MAX_POLICY):
            # These policies control age at selection time. The 1000-group limit
            # is only a safety backpressure bound, not their experimental queue
            # capacity.
            await self._output_slots.acquire()

        depth_before = self._queue_size()
        queue_put_version = self._applied_weight_version.current()
        stamp_group_weight_version(item[1], QUEUE_PUT_VERSION_KEY, queue_put_version)

        if policy == LEGACY_QUEUE_POLICY:
            self._output.put_nowait(item)
        else:
            if policy == QUEUE_DROP_POLICY and depth_before >= self._queue_capacity_groups():
                _, evicted_group = self._policy_output.popleft()
                evicted_tokens = group_response_tokens(evicted_group)
                self._queue_evicted_groups += 1
                self._queue_evicted_tokens += evicted_tokens
                self._producer_response_lengths.record("queue_evicted", evicted_group)
                reference = group_first_prefill_weight_version(evicted_group)
                self._queue_lifecycle.finish(
                    evicted_group,
                    disposition="queue_evicted",
                    decision_version=queue_put_version,
                    rollout_id=None,
                    reference_version=reference,
                    bound_staleness=queue_put_version - reference if reference is not None else None,
                )
            self._policy_output.append(item)
            self._policy_output_ready.set()

        self._queue_lifecycle.enqueued(
            item[1],
            queue_put_version=queue_put_version,
            depth_before=depth_before,
            depth_after=self._queue_size(),
        )

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> tuple[list[Sample], Group]:
        queue_get = asyncio.create_task(self._output.get())
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
                    result = queue_get.result()
                    self._output_slots.release()
                    self._queue_lifecycle.dequeued(result[1], depth_after_observed=self._output.qsize())
                    return result
                logger.warning(
                    f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s (queued: {self._output.qsize()})"
                )
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _take_policy_groups(self, count: int) -> list[tuple[QueueItem, int]]:
        """Wait for and atomically remove ``count`` oldest completed groups."""
        assert self._queue_policy() != LEGACY_QUEUE_POLICY
        # Match the legacy queue's fail-fast contract even when a dead worker
        # left enough completed groups to satisfy this request immediately.
        if self._worker is not None and self._worker.done():
            self._worker.result()
            raise RuntimeError("fully-async rollout worker exited without an exception")
        while len(self._policy_output) < count:
            self._policy_output_ready.clear()
            if len(self._policy_output) >= count:
                break
            queue_ready = asyncio.create_task(self._policy_output_ready.wait())
            try:
                done, _ = await asyncio.wait(
                    {queue_ready, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_ready not in done:
                    logger.warning(
                        "No full rollout batch for %.1fs (queued: %d, needed: %d)",
                        NO_PROGRESS_WARN_SECS,
                        len(self._policy_output),
                        count,
                    )
            finally:
                if not queue_ready.done():
                    queue_ready.cancel()

        selected = []
        for _ in range(count):
            item = self._policy_output.popleft()
            if self._queue_policy() == QUEUE_MAX_POLICY:
                self._output_slots.release()
            depth_after = len(self._policy_output)
            self._queue_lifecycle.dequeued(item[1], depth_after_observed=depth_after)
            selected.append((item, depth_after))
        return selected

    async def _drain(self, rollout_id: int) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        queue_size_start = self._queue_size()
        queue_sizes_after_dequeue: list[int] = []
        candidates: deque[tuple[QueueItem, int]] = deque()
        data: list[Group] = []
        aborted_groups_recycled = 0
        stale_groups_recycled = 0
        stale_groups_dropped = 0
        # Two populations, because they answer different questions and only one of
        # them is the study's variable. ``offered`` (logged as
        # ``staleness/bound/rollout/``) is every group the pipeline handed over,
        # including those the bound then sent back -- that is the *natural* lag of
        # this node ratio. ``trained`` (``staleness/bound/train/``) is what survived into
        # the batch, and is what the loss actually saw. They diverge where the
        # bound bites and where the dynamic filter drops a group, which is where a
        # reader is most likely to be misled.
        # What `--max-weight-staleness` is tested against, in its two populations.
        # This is `current - reference`; the reference is completion, submission,
        # or first prefill according to the configured semantics.
        trained_bound_staleness: list[int] = []
        offered_bound_staleness: list[int] = []
        # The decomposition. Per group, with R the selected bound reference, Q
        # the version the group became trainable under, and C the version at drain:
        #   pre-queue = Q - R   updates crossed before the group became trainable
        #   in-queue  = C - Q   updates crossed while waiting to be trained on
        #   total     = C - R   = pre-queue + in-queue = the bound quantity
        trained_pre_queue: list[int] = []
        trained_in_queue: list[int] = []
        trained_total: list[int] = []
        current_version = self._applied_weight_version.current()
        offered_mixed_versions: list[bool] = []
        trained_mixed_versions: list[bool] = []
        # Generation that was produced and then thrown away. Counted in tokens, not
        # groups, because that is the unit a sample-efficiency claim is made in, and
        # because the three ways to waste generation cost wildly different amounts.
        aborted_tokens = 0
        stale_tokens = 0
        age_cutoff_tokens = 0
        filtered_tokens = 0
        response_length_metrics = _ResponseLengthMetrics()
        metric_gatherer = MetricGatherer()
        do_print = True

        while len(data) < target_data_size:
            if not candidates:
                if self._queue_policy() == LEGACY_QUEUE_POLICY:
                    item = await self._next_group()
                    candidates.append((item, self._queue_size()))
                else:
                    needed = target_data_size - len(data)
                    candidates.extend(await self._take_policy_groups(needed))
            (prompt_group, group), depth_after_dequeue = candidates.popleft()
            queue_sizes_after_dequeue.append(depth_after_dequeue)
            assert len(group) == args.n_samples_per_prompt

            # A weight update paused generation mid-group: return it for re-sampling.
            if any(s.status == Sample.Status.ABORTED for s in _iter_samples(group)):
                response_length_metrics.record("aborted_recycled", group)
                aborted_tokens += group_response_tokens(group)
                self._queue_lifecycle.finish(
                    group,
                    disposition="aborted_recycled",
                    decision_version=self._applied_weight_version.current(),
                    rollout_id=rollout_id,
                )
                self._recycle(prompt_group)
                aborted_groups_recycled += 1
                continue

            response_length_metrics.record("offered", group)

            oldest = group_oldest_weight_version(group)
            submitted = group_submission_weight_version(group)
            ready = group_lifecycle_weight_version(group, GROUP_READY_VERSION_KEY)
            current = self._applied_weight_version.current()
            current_version = current
            stamp_group_weight_version(group, DRAIN_VERSION_KEY, current)

            first_prefill = group_first_prefill_weight_version(group)
            if args.staleness_reference == "prefill":
                validate_prefill_policy_provenance(group)
                first_prefill = group_first_prefill_weight_version(group)
                assert first_prefill is not None

            mixed_version = group_has_mixed_forward_versions(group)
            offered_mixed_versions.append(mixed_version)

            # Which end of generation the bound is measured from. `completion` keeps
            # the historical behaviour -- the gap covers queue residency but not the
            # updates a generation crossed. `submission` bounds the whole request
            # lifetime. `prefill` starts at the first scheduler-authoritative forward.
            if args.staleness_reference == "completion":
                reference = oldest
            elif args.staleness_reference == "submission":
                reference = submitted
            else:
                reference = first_prefill

            group_bound_staleness: int | None = None
            if args.max_weight_staleness is not None:
                if reference is not None:
                    staleness = current - reference
                    if staleness < 0:
                        raise RuntimeError(
                            f"Negative weight staleness: current={current}, reference={reference}, "
                            f"mode={args.staleness_reference}"
                        )
                    group_bound_staleness = staleness
                    offered_bound_staleness.append(staleness)
                    if staleness > args.max_weight_staleness:
                        rejected_tokens = group_response_tokens(group)
                        disposition = "stale_recycled"
                        if self._queue_policy() == QUEUE_MAX_POLICY:
                            disposition = "age_cutoff_dropped"
                            response_length_metrics.record(disposition, group)
                            age_cutoff_tokens += rejected_tokens
                            stale_groups_dropped += 1
                        else:
                            response_length_metrics.record(disposition, group)
                            stale_tokens += rejected_tokens
                            stale_groups_recycled += 1
                        self._queue_lifecycle.finish(
                            group,
                            disposition=disposition,
                            decision_version=current,
                            rollout_id=rollout_id,
                            reference_version=reference,
                            bound_staleness=staleness,
                        )
                        if self._queue_policy() == LEGACY_QUEUE_POLICY:
                            self._recycle(prompt_group)
                        logger.info(
                            f"Rejected stale group ({args.staleness_reference}_version={reference}, "
                            f"current={current}, staleness={staleness} > max={args.max_weight_staleness})"
                        )
                        continue

            # Not gated on the bound: the bound tests one derived quantity, and every
            # arm -- including an unbounded one -- needs the decomposition.
            # The decomposition uses the exact same start as the bound. This keeps
            # `staleness/total` equal to the enforced quantity in every reference mode.
            span_start = reference
            have_span = ready is not None and span_start is not None
            group_pre_queue = ready - span_start if have_span else None
            group_in_queue = current - ready if have_span else None
            group_total = current - span_start if have_span else None
            for name, value in (
                ("pre_queue", group_pre_queue),
                ("in_queue", group_in_queue),
                ("total", group_total),
            ):
                if value is not None and value < 0:
                    raise RuntimeError(
                        f"Negative {name} weight staleness for group: start={span_start}, "
                        f"ready={ready}, drain={current}"
                    )

            filter_output = call_dynamic_filter(self._dynamic_filter, args, group)
            if not filter_output.keep:
                # Dropped, not recycled: no usable gradient signal.
                response_length_metrics.record("dynamic_filter_dropped", group)
                filtered_tokens += group_response_tokens(group)
                self._queue_lifecycle.finish(
                    group,
                    disposition="dynamic_filter_dropped",
                    decision_version=current,
                    rollout_id=rollout_id,
                    reference_version=reference,
                    bound_staleness=group_bound_staleness,
                    detail=filter_output.reason,
                )
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                continue

            if do_print:
                sample = _first_sample(group)
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                    f"label: {sample.label}, reward: {sample.reward}"
                )
                do_print = False

            if group_bound_staleness is not None:
                trained_bound_staleness.append(group_bound_staleness)
            if group_pre_queue is not None:
                trained_pre_queue.append(group_pre_queue)
            if group_in_queue is not None:
                trained_in_queue.append(group_in_queue)
            if group_total is not None:
                trained_total.append(group_total)
            trained_mixed_versions.append(mixed_version)
            response_length_metrics.record("trained", group)
            self._queue_lifecycle.finish(
                group,
                disposition="trained",
                decision_version=current,
                rollout_id=rollout_id,
                reference_version=reference,
                bound_staleness=group_bound_staleness,
            )
            data.append(group)

        sample = _first_sample(data[-1])
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
            f"label: {sample.label}, reward: {sample.reward}"
        )

        data.sort(key=lambda group: _first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        kept_tokens = sum(group_response_tokens(group) for group in data)
        queue_evicted_groups = self._queue_evicted_groups
        queue_evicted_tokens = self._queue_evicted_tokens
        self._queue_evicted_groups = 0
        self._queue_evicted_tokens = 0
        wasted_tokens = aborted_tokens + stale_tokens + age_cutoff_tokens + filtered_tokens + queue_evicted_tokens
        metrics = {
            "rollout/fully_async/queue_size": self._queue_size(),
            "queue/occupancy/start_groups": queue_size_start,
            "queue/occupancy/end_groups": self._queue_size(),
            "queue/occupancy/capacity_groups": self._queue_capacity_groups(),
            "queue/occupancy/max_in_flight_groups": self._max_in_flight_groups(),
            "queue/config/policy_is_queue_recycle": float(self._queue_policy() == LEGACY_QUEUE_POLICY),
            "queue/config/policy_is_queue_max": float(self._queue_policy() == QUEUE_MAX_POLICY),
            "queue/config/policy_is_queue_drop": float(self._queue_policy() == QUEUE_DROP_POLICY),
            "queue/config/factor": getattr(args, "fully_async_queue_factor", 1),
            "rollout/fully_async/aborted_groups_recycled": aborted_groups_recycled,
            "rollout/fully_async/stale_groups_recycled": stale_groups_recycled,
            "rollout/fully_async/stale_groups_dropped": stale_groups_dropped,
            "rollout/fully_async/queue_evicted_groups": queue_evicted_groups,
            "rollout/fully_async/aborted_tokens": aborted_tokens,
            "rollout/fully_async/stale_tokens": stale_tokens,
            "rollout/fully_async/age_cutoff_tokens": age_cutoff_tokens,
            "rollout/fully_async/queue_evicted_tokens": queue_evicted_tokens,
            "rollout/fully_async/dynamic_filter_tokens": filtered_tokens,
            "rollout/fully_async/kept_tokens": kept_tokens,
            "rollout/fully_async/wasted_token_frac": (
                wasted_tokens / (wasted_tokens + kept_tokens) if wasted_tokens + kept_tokens else 0.0
            ),
            **{
                f"queue/occupancy/after_dequeue/{name}": value
                for name, value in _distribution_metrics(queue_sizes_after_dequeue).items()
            },
            **self._producer_response_lengths.collect_and_reset(),
            **response_length_metrics.collect(),
            **metric_gatherer.collect(),
        }
        if current_version is not None:
            # Logged next to the staleness itself: staleness is a difference against
            # this version, and without it a missing staleness metric is impossible to
            # tell apart from a router that never answered.
            metrics["rollout/fully_async/current_weight_version"] = current_version
        if offered_mixed_versions:
            metrics["staleness/mixed_version_frac/rollout"] = sum(offered_mixed_versions) / len(offered_mixed_versions)
        if trained_mixed_versions:
            metrics["staleness/mixed_version_frac/train"] = sum(trained_mixed_versions) / len(trained_mixed_versions)
        # `rollout/fully_async/{avg,max}_staleness` are upstream's and keep
        # upstream's meaning: the lag as *offered*, counted before the bound check.
        # Redefining them to the trained lag would leave two miles runs plotting
        # the same key against different quantities.
        if offered_bound_staleness:
            metrics["rollout/fully_async/avg_staleness"] = sum(offered_bound_staleness) / len(offered_bound_staleness)
            metrics["rollout/fully_async/max_staleness"] = max(offered_bound_staleness)

        # The decomposition, over the trained batch. `total` is the selected bound
        # quantity. The two components say where it came from: `pre_queue` is the
        # selected reference to group-ready span, and `in_queue` is queue waiting.
        # `frac_at_bound` is absent from all three: `--max-weight-staleness` is not
        # applied to any of them, see `staleness/bound/` below.
        for name, values in (
            ("total", trained_total),
            ("pre_queue", trained_pre_queue),
            ("in_queue", trained_in_queue),
        ):
            if values:
                metrics |= {
                    f"staleness/{name}/{key}": value for key, value in _staleness_metrics(values, None).items()
                }

        # What the bound actually tests, which depends on `--staleness-reference`:
        #
        #   completion (default)  current - oldest  = in_queue + the group's internal
        #                         completion-version spread = `total` exactly.
        #   submission            current - S       = `total` exactly.
        #   prefill               current - first prefill = `total` exactly.
        #
        # Kept under its own name because it is the only quantity that explains which
        # groups were recycled, in the two populations the bound separates: `rollout`
        # is every group offered, counted before the check; `train` is what survived.
        if trained_bound_staleness:
            metrics |= {
                f"staleness/bound/train/{name}": value
                for name, value in _staleness_metrics(trained_bound_staleness, args.max_weight_staleness).items()
            }
        if offered_bound_staleness:
            metrics |= {
                f"staleness/bound/rollout/{name}": value
                for name, value in _staleness_metrics(offered_bound_staleness, args.max_weight_staleness).items()
            }

        # Named for the reason rather than the mechanism: `stale_groups_recycled`
        # is what happened, `bound_exceeded` is why. Split from the dynamic-filter
        # drops, which land in the same recycled/dropped bucket if only totals are
        # compared.
        bound_exceeded_groups = stale_groups_recycled + stale_groups_dropped
        bound_exceeded_tokens = stale_tokens + age_cutoff_tokens
        metrics["staleness/bound_exceeded_groups"] = bound_exceeded_groups
        metrics["staleness/bound_exceeded_tokens"] = bound_exceeded_tokens
        # `staleness/bound/*` means a different quantity under each reference, so the
        # choice is logged next to it rather than left to the run config.
        metrics["staleness/bound_reference_is_submission"] = float(args.staleness_reference == "submission")
        metrics["staleness/bound_reference_is_prefill"] = float(args.staleness_reference == "prefill")

        # Rejecting more groups than were kept means the age cap, rather than the
        # natural producer/consumer balance, is determining the batch. This does
        # not deadlock: while drain waits, no training update advances the version,
        # so newly completed groups eventually pass. It can still collapse overlap
        # and waste most generation, which is what this warning reports.
        if bound_exceeded_groups > target_data_size:
            logger.warning(
                f"Rejected {bound_exceeded_groups} groups to keep {target_data_size} at "
                f"--max-weight-staleness {args.max_weight_staleness} "
                f"(--staleness-reference {args.staleness_reference}, queue-policy "
                f"{self._queue_policy()}). If this persists, the age cap is collapsing "
                "rollout/training overlap and discarding most generated groups."
            )

        # How many times a group that reached training had to be regenerated. The
        # per-sample counter survives `reset_for_retry`, so this is cumulative over
        # the group's whole life, not just this rollout.
        retries = [max(sample.retry_count for sample in _iter_samples(group)) for group in data]
        if retries:
            metrics["staleness/retry_count_mean"] = sum(retries) / len(retries)
            metrics["staleness/retry_count_max"] = float(max(retries))
            metrics["staleness/retry_frac_nonzero"] = sum(1 for r in retries if r) / len(retries)

        debug_metadata = self._queue_lifecycle.take_metadata(
            policy=self._queue_policy(),
            capacity_groups=self._queue_capacity_groups(),
        )
        return RolloutFnTrainOutput(samples=data, metrics=metrics, debug_metadata=debug_metadata)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.retry_count += 1
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])
