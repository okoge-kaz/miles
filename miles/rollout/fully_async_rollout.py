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
import copy
import gc
import inspect
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.filter_hub.base_types import call_dynamic_filter
from miles.rollout.fully_async_telemetry import FullyAsyncPipelineTelemetry
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.queue_policy import (
    DEFAULT_TRAINING_BUFFER_QUEUE_SIZE,
    QUEUE_DROP_POLICY,
    QUEUE_MAX_POLICY,
    QUEUE_RECYCLE_POLICY,
    fully_async_queue_capacity_groups,
)
from miles.rollout.queue_telemetry import (
    DEFAULT_RESPONSE_LENGTH_POPULATIONS,
    Group,
    _distribution_metrics,
    _first_sample,
    _iter_samples,
    _QueueLifecycleRecorder,
    _ResponseLengthMetrics,
    group_first_prefill_weight_version,
    group_oldest_weight_version,
    group_response_tokens,
    group_reward_values,
)
from miles.rollout.recycle_compute_metrics import (
    ADMITTED_TOKENS_KEY,
    ATTEMPT_WALL_SECONDS_KEY,
    BOUND_REFERENCE_VERSION_KEY,
    DRAIN_TIME_KEY,
    DRAIN_VERSION_KEY,
    GENERATED_TOKENS_KEY,
    GROUP_GENERATION_COMPLETE_TIME_KEY,
    GROUP_GENERATION_COMPLETE_VERSION_KEY,
    GROUP_READY_TIME_KEY,
    GROUP_READY_VERSION_KEY,
    LIFECYCLE_EXACT_KEY,
    QUEUE_PUT_TIME_KEY,
    QUEUE_PUT_VERSION_KEY,
    RECYCLE_AUX_REASONS,
    RECYCLE_DEBUG_SCHEMA_VERSION,
    RECYCLE_REASONS,
    REWARD_SECONDS_KEY,
    SAMPLE_GENERATION_COMPLETE_TIME_KEY,
    SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
    SUBMISSION_VERSION_KEY,
    TRAIN_VERSION_KEY,
    TRAIN_WEIGHT_VERSION_METRIC,
    TRAJECTORY_START_TIME_KEY,
    TRAJECTORY_START_VERSION_KEY,
    aborted_recycle_reason,
    add_discard_accounting,
    add_selection_population,
    classify_stale_recycle_stage,
    discard_waste_metrics,
    group_generation_completion_version,
    recycle_record,
    reset_attempt_telemetry,
    selection_population_metrics,
    stamp_attempt_wall_seconds,
    stamp_sample_lifecycle_boundary,
    stamp_sample_reference_versions,
    straggler_collateral_indices,
)
from miles.rollout.replay_buffer import (
    REPLAY_BUFFER_INFLIGHT,
    REPLAY_BUFFER_ROLLOUT,
    dataset_fingerprint,
    decode_group,
    prompt_group_id,
    rollout_batch_token,
)
from miles.rollout.replay_buffer_codec import (
    SAMPLE_CODEC_STATE_KEY,
    ReplayBufferPackedFieldCache,
    ReplayBufferSampleEncoder,
    materialize_replay_buffer_state,
)
from miles.utils.http_utils import get, post, router_worker_base_urls
from miles.utils.misc import call_agent_abort_hook, load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

ACKED_BATCH_HISTORY_SIZE = 16
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0
# Realized lag is a small integer; anything past this goes in one overflow bucket
# so the metric count stays bounded no matter how far behind a run drifts.
#
# 32, not 16: the high-staleness cohort sweeps bounds through 28 and needs
# enough resolution in the tail to map realized staleness to downstream score.
STALENESS_HISTOGRAM_MAX = 32
QueueItem = tuple[list[Sample], Group]
_CONTINUATION_METADATA_KEYS = (
    SUBMISSION_VERSION_KEY,
    TRAJECTORY_START_VERSION_KEY,
    TRAJECTORY_START_TIME_KEY,
)
_TERMINAL_CONTINUATION_METADATA_KEYS = (
    SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
    SAMPLE_GENERATION_COMPLETE_TIME_KEY,
    ATTEMPT_WALL_SECONDS_KEY,
    REWARD_SECONDS_KEY,
)


@dataclass
class _DrainProgress:
    rollout_id: int
    updates_before_train: int
    data: list[Group] = field(default_factory=list)
    group_ids: list[int] = field(default_factory=list)
    queue_size_start: int | None = None
    queue_sizes_after_dequeue: list[int] = field(default_factory=list)
    aborted_groups_recycled: int = 0
    stale_groups_recycled: int = 0
    stale_groups_dropped: int = 0
    rollout_staleness: list[int] = field(default_factory=list)
    bound_evaluated_samples: int = 0
    bound_exceeded_samples: int = 0
    trained_pre_queue: list[int] = field(default_factory=list)
    trained_in_queue: list[int] = field(default_factory=list)
    trained_total: list[int] = field(default_factory=list)
    train_version: int | None = None
    offered_mixed_versions: list[bool] = field(default_factory=list)
    trained_mixed_versions: list[bool] = field(default_factory=list)
    aborted_tokens: int = 0
    stale_tokens: int = 0
    age_cutoff_tokens: int = 0
    filtered_tokens: int = 0
    response_sample_lengths: dict[str, list[int]] = field(
        default_factory=lambda: {population: [] for population in DEFAULT_RESPONSE_LENGTH_POPULATIONS}
    )
    response_group_max_lengths: dict[str, list[int]] = field(
        default_factory=lambda: {population: [] for population in DEFAULT_RESPONSE_LENGTH_POPULATIONS}
    )
    dynamic_filter_drop_counts: dict[str, int] = field(default_factory=dict)
    recycle_reason_groups: dict[str, int] = field(default_factory=dict)
    recycle_reason_tokens: dict[str, int] = field(default_factory=dict)
    recycle_aux_groups: dict[str, int] = field(default_factory=dict)
    recycle_aux_tokens: dict[str, int] = field(default_factory=dict)
    waste_by_reason: dict[str, dict[str, float]] = field(default_factory=dict)
    selection_populations: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    discard_records: list[dict[str, Any]] = field(default_factory=list)
    do_print: bool = True


@dataclass(frozen=True)
class _PreparedBatch:
    output: RolloutFnTrainOutput
    group_ids: tuple[int, ...]
    token: str


@dataclass(frozen=True)
class _InflightReplayItem:
    prompt_group: list[Sample]
    generation_group: Group


@dataclass(frozen=True)
class _QueueReplaySnapshot:
    ready_items: list[QueueItem]
    pending_prompts: dict[int, list[Sample]]
    lifecycle_state: dict | None
    response_length_state: dict
    queue_evicted_groups: int
    queue_evicted_tokens: int
    snapshot_evicted_groups: int


def _flat_prompt_group(group: Group) -> list[Sample]:
    if any(isinstance(item, list) for item in group):
        raise RuntimeError("A pending prompt lease must be a flat list of Sample objects")
    return list(group)


def _group_requires_continuation(group: Group) -> bool:
    return any(
        sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED} or sample.reward is None
        for sample in _iter_samples(group)
    )


def _prepare_generation_attempt(
    group: Group,
    *,
    continuation: bool,
) -> list[dict[str, Any]] | None:
    samples = list(_iter_samples(group))
    if not continuation:
        reset_attempt_telemetry(samples)
        return None

    preserved = []
    for sample in samples:
        keys = _CONTINUATION_METADATA_KEYS
        if sample.status in {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}:
            keys += _TERMINAL_CONTINUATION_METADATA_KEYS
        preserved.append({key: sample.metadata[key] for key in keys if key in sample.metadata})
    reset_attempt_telemetry(samples)
    for sample, metadata in zip(samples, preserved, strict=True):
        sample.metadata.update(metadata)
    return preserved


def _restore_continuation_metadata(group: Group, preserved: list[dict[str, Any]] | None) -> None:
    if preserved is None:
        return
    samples = list(_iter_samples(group))
    if len(samples) != len(preserved):
        raise RuntimeError("Inflight continuation changed the number of samples in a group")
    for sample, metadata in zip(samples, preserved, strict=True):
        sample.metadata.update(metadata)


@contextmanager
def _defer_cyclic_gc():
    """Avoid a full-heap cyclic-GC scan in the short, acyclic snapshot path."""
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _encode_ready_item(
    item: tuple[list[Sample], Group],
    queue_put_version: int,
    sample_encoder: ReplayBufferSampleEncoder,
) -> dict[str, Any]:
    prompt_group, result = item
    metadata_updates = None
    if group_lifecycle_weight_version(result, QUEUE_PUT_VERSION_KEY) is None:
        # A completed task may be blocked on queue capacity when the snapshot is
        # taken. Restore promotes it into the reconstructed ready queue, so the
        # durable snapshot boundary is its queue-put version. Do not mutate the
        # live result: failure-free execution may enqueue it under a later version.
        metadata_updates = {QUEUE_PUT_VERSION_KEY: queue_put_version}
    return {
        "prompt_group": sample_encoder.encode_group(prompt_group, use_packed_cache=True),
        "result": sample_encoder.encode_group(
            result,
            metadata_updates=metadata_updates,
            use_packed_cache=True,
        ),
    }


def _decode_ready_item(state: dict[str, Any]) -> tuple[list[Sample], Group]:
    prompt_group = _flat_prompt_group(decode_group(state["prompt_group"]))
    return prompt_group, decode_group(state["result"])


def _encode_inflight_item(
    item: _InflightReplayItem,
    sample_encoder: ReplayBufferSampleEncoder,
) -> dict[str, Any]:
    return {
        "prompt_group": sample_encoder.encode_group(item.prompt_group),
        "generation_group": sample_encoder.encode_group(item.generation_group),
    }


def _decode_inflight_item(state: dict[str, Any]) -> _InflightReplayItem:
    prompt_group = _flat_prompt_group(decode_group(state["prompt_group"]))
    generation_group = decode_group(state["generation_group"])
    if any(isinstance(item, list) for item in generation_group):
        raise RuntimeError("Inflight replay currently requires a flat generation group")
    if not _group_requires_continuation(generation_group):
        raise RuntimeError("Inflight replay item has no unfinished sample")
    return _InflightReplayItem(prompt_group=prompt_group, generation_group=generation_group)


def _materialized_group_ids(
    ready_items: list[tuple[list[Sample], Group]],
    drains: dict[int, _DrainProgress],
    prepared_batches: dict[int, _PreparedBatch],
    inflight_items: list[_InflightReplayItem] | tuple[_InflightReplayItem, ...] = (),
) -> set[int]:
    ready_ids = []
    for prompt_group, result in ready_items:
        prompt_id = prompt_group_id(prompt_group)
        result_ids = {sample.group_index for sample in _iter_samples(result)}
        if result_ids != {prompt_id}:
            raise RuntimeError(f"Ready result identity {result_ids} does not match prompt group {prompt_id}")
        ready_ids.append(prompt_id)

    drained_ids = []
    for progress in drains.values():
        if len(progress.data) != len(progress.group_ids):
            raise RuntimeError(
                f"Partial drain {progress.rollout_id} has {len(progress.data)} groups but "
                f"{len(progress.group_ids)} group ids"
            )
        for group, group_id in zip(progress.data, progress.group_ids, strict=True):
            result_ids = {sample.group_index for sample in _iter_samples(group)}
            if result_ids != {group_id}:
                raise RuntimeError(
                    f"Partial drain {progress.rollout_id} result identity {result_ids} "
                    f"does not match stored group {group_id}"
                )
            drained_ids.append(group_id)

    prepared_ids = []
    for rollout_id, prepared in prepared_batches.items():
        if len(prepared.output.samples) != len(prepared.group_ids):
            raise RuntimeError(
                f"Prepared batch {rollout_id} has {len(prepared.output.samples)} groups but "
                f"{len(prepared.group_ids)} group ids"
            )
        for group, group_id in zip(prepared.output.samples, prepared.group_ids, strict=True):
            result_ids = {sample.group_index for sample in _iter_samples(group)}
            if result_ids != {group_id}:
                raise RuntimeError(
                    f"Prepared batch {rollout_id} result identity {result_ids} "
                    f"does not match stored group {group_id}"
                )
            prepared_ids.append(group_id)

    inflight_ids = []
    for item in inflight_items:
        prompt_id = prompt_group_id(item.prompt_group)
        result_ids = {sample.group_index for sample in _iter_samples(item.generation_group)}
        if result_ids != {prompt_id}:
            raise RuntimeError(f"Inflight result identity {result_ids} does not match prompt group {prompt_id}")
        inflight_ids.append(prompt_id)

    all_ids = ready_ids + drained_ids + prepared_ids + inflight_ids
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("A fully-async prompt group appears in more than one materialized lifecycle state")
    return set(all_ids)


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


def _record_reason(
    progress: _DrainProgress,
    *,
    reason: str,
    tokens: int,
    auxiliary: bool = False,
) -> None:
    group_counts = progress.recycle_aux_groups if auxiliary else progress.recycle_reason_groups
    token_counts = progress.recycle_aux_tokens if auxiliary else progress.recycle_reason_tokens
    group_counts[reason] = group_counts.get(reason, 0) + 1
    token_counts[reason] = token_counts.get(reason, 0) + tokens


def _scheduled_train_version(progress: _DrainProgress, dequeue_version: int) -> int:
    if progress.train_version is None:
        progress.train_version = dequeue_version + progress.updates_before_train
    if dequeue_version > progress.train_version:
        raise RuntimeError(
            f"Dequeue version {dequeue_version} exceeds scheduled train version {progress.train_version} "
            f"for rollout {progress.rollout_id}"
        )
    return progress.train_version


def _staleness_metrics(values: list[int]) -> dict[str, float]:
    """P(L) reduced to bounded scalars: the logger takes scalars, not histograms.

    Percentiles rather than a mean alone because the tail is the quantity of
    interest -- a mean of 0.4 with a p99 of 12 and a mean of 0.4 with a p99 of 1
    are different training regimes.
    """
    array = np.asarray(values, dtype=float)
    metrics = {
        "mean": float(array.mean()),
        "variance": float(array.var()),
        "std": float(array.std()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "frac_zero": float((array <= 0).mean()),
        "num_groups": float(array.size),
    }
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


async def _get_rollout_worker_urls(args) -> list[str]:
    """Return unique SGLang worker base URLs across supported router APIs."""
    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    try:
        response = await get(f"{router}/workers")
        urls = [worker["url"] for worker in response["workers"]]
    except Exception as workers_error:  # noqa: BLE001 - compatibility fallback
        try:
            response = await get(f"{router}/list_workers")
            urls = response["urls"]
        except Exception as legacy_error:  # noqa: BLE001 - retain both failure contexts
            legacy_error.add_note(f"The /workers endpoint also failed: {workers_error!r}")
            raise
    return router_worker_base_urls(urls)


async def _abort_inflight_requests(args) -> None:
    """Materialize partial responses by aborting every active rollout request."""
    urls = await _get_rollout_worker_urls(args)
    logger.info("Interrupting inflight rollout requests on %s for replay-buffer capture", urls)
    await asyncio.gather(*(post(f"{url}/abort_request", {"abort_all": True}, max_retries=3) for url in urls))
    await call_agent_abort_hook(args)


class FullyAsyncRolloutFn:
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Groups whose samples were aborted (e.g. by a
    weight update pausing generation) are recycled back into the data source.
    Age-bound failures are recycled by ``queue-recycle`` and discarded by
    queue-max; queue-drop instead discards the oldest completed group when its
    bounded queue overflows.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self._queue_capacity = fully_async_queue_capacity_groups(input.args)
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
        self._producer_selection_populations: dict[str, dict[str, list[float]]] = {}
        self._queue_evicted_groups = 0
        self._queue_evicted_tokens = 0
        generate_group_parameters = inspect.signature(generate_and_rm_group).parameters
        self._generation_lifecycle_supported = "lifecycle_version_provider" in generate_group_parameters
        self._generation_completion_callback_supported = "on_group_generation_complete" in generate_group_parameters
        self._pipeline_telemetry = FullyAsyncPipelineTelemetry()
        self._worker: asyncio.Task | None = None
        self._output: asyncio.Queue[QueueItem] | None = None
        self._output_slots: asyncio.Semaphore | None = None
        self._policy_output: deque[QueueItem] | None = None
        self._policy_output_ready: asyncio.Event | None = None
        self._active: set[asyncio.Task] = set()
        self._completed_waiting: dict[int, tuple[list[Sample], Group]] = {}
        self._queue_gets: set[asyncio.Task] = set()

        self._replay_buffer_enabled = getattr(input.args, "use_replay_buffer", False)
        self._replay_buffer_type = getattr(input.args, "replay_buffer_type", REPLAY_BUFFER_ROLLOUT)
        if self._replay_buffer_type not in {REPLAY_BUFFER_ROLLOUT, REPLAY_BUFFER_INFLIGHT}:
            raise ValueError(f"Unsupported replay-buffer type: {self._replay_buffer_type!r}")
        self._replay_buffer_packed_fields = ReplayBufferPackedFieldCache() if self._replay_buffer_enabled else None
        self._dataset_fingerprint = (
            dataset_fingerprint(input.args, input.data_source) if self._replay_buffer_enabled else None
        )
        self._pending_prompts: dict[int, list[Sample]] = {}
        self._drain_progress: dict[int, _DrainProgress] = {}
        self._prepared_batches: dict[int, _PreparedBatch] = {}
        self._inflight_replay: deque[_InflightReplayItem] = deque()
        self._acked_batch_tokens: dict[int, str] = {}
        self._resume_metrics: dict[str, float] = {}
        self._capturing_inflight = False

    def commit_applied_weight_version(self, version: int) -> None:
        self._applied_weight_version.commit(version)

    async def commit_applied_weight_version_on_loop(self, version: int) -> None:
        self.commit_applied_weight_version(version)

    async def current_applied_weight_version(self) -> int:
        """Return the last weight version finalized on every rollout engine."""
        return self._applied_weight_version.current()

    async def pipeline_telemetry_snapshot(self) -> dict[str, float]:
        """Sample producer/consumer counters on the rollout event loop."""
        return self._pipeline_telemetry.snapshot(
            active_groups=len(self._active),
            max_active_groups=self._max_in_flight_groups(),
        )

    async def complete_trained_batch_telemetry_on_loop(
        self,
        *,
        accepted_tokens: int | None,
        optimizer_updates: int,
    ) -> dict[str, float]:
        """Close one throughput window after a successful actor train call."""
        self._pipeline_telemetry.add_trained_batch(
            accepted_tokens=accepted_tokens,
            optimizer_updates=optimizer_updates,
        )
        return self._pipeline_telemetry.snapshot(
            active_groups=len(self._active),
            max_active_groups=self._max_in_flight_groups(),
        )

    async def acknowledge_trained_batch(self, rollout_id: int, token: str) -> None:
        """Commit consumption only after the trainer reports a successful update."""
        prepared = self._prepared_batches.get(rollout_id)
        if prepared is None:
            if self._acked_batch_tokens.get(rollout_id) == token:
                return
            raise RuntimeError(f"Cannot acknowledge unknown fully-async rollout batch {rollout_id}")
        if prepared.token != token:
            raise RuntimeError(
                f"Fully-async rollout batch token mismatch for {rollout_id}: "
                f"expected={prepared.token}, received={token}"
            )
        for group_id in prepared.group_ids:
            if self._pending_prompts.pop(group_id, None) is None:
                raise RuntimeError(f"Prepared group {group_id} is absent from the pending prompt ledger")
        del self._prepared_batches[rollout_id]
        self._acked_batch_tokens[rollout_id] = token
        for old_rollout_id in sorted(self._acked_batch_tokens)[:-ACKED_BATCH_HISTORY_SIZE]:
            del self._acked_batch_tokens[old_rollout_id]

    async def replay_buffer_state(self, rollout_id: int) -> dict[str, Any]:
        """Capture one coherent lifecycle snapshot on the rollout event loop."""
        if not self._replay_buffer_enabled:
            raise RuntimeError("Replay buffer is disabled")
        if self._replay_buffer_type == REPLAY_BUFFER_INFLIGHT:
            return await self._capture_inflight_replay_buffer_state(rollout_id)
        with _defer_cyclic_gc():
            return self._capture_replay_buffer_state(rollout_id)

    async def _capture_inflight_replay_buffer_state(self, rollout_id: int) -> dict[str, Any]:
        if self._capturing_inflight:
            raise RuntimeError("An inflight replay-buffer capture is already running")

        worker_was_running = self._worker is not None and not self._worker.done()
        self._capturing_inflight = True
        try:
            await self._stop_worker_for_replay_capture()
            await self._materialize_active_inflight_requests()
            with _defer_cyclic_gc():
                return self._capture_replay_buffer_state(rollout_id)
        finally:
            self.state.reset()
            self._capturing_inflight = False
            if worker_was_running:
                self._ensure_worker()

    async def _stop_worker_for_replay_capture(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if worker.done():
            worker.result()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        self._worker = None

    async def _materialize_active_inflight_requests(self) -> None:
        active_tasks = list(self._active)
        unfinished_tasks = [task for task in active_tasks if not task.done()]
        if unfinished_tasks:
            self.state.aborted = True
            await _abort_inflight_requests(self.args)
            await asyncio.gather(*unfinished_tasks)

        completed = dict(self._completed_waiting)
        for task in active_tasks:
            if task.cancelled():
                raise RuntimeError("An active rollout task was cancelled during replay-buffer capture")
            completed.setdefault(id(task), task.result())

        self._active.clear()
        self._completed_waiting.clear()
        # A restored buffer can contain more inflight items than the active
        # generation limit. Merge the not-yet-submitted tail with requests just
        # interrupted, then recover the original prompt-lease order.
        partial_items = list(self._inflight_replay)
        for task_id, item in completed.items():
            if _group_requires_continuation(item[1]):
                partial_items.append(self._inflight_item_from_result(item))
            else:
                self._completed_waiting[task_id] = item

        pending_order = {group_id: position for position, group_id in enumerate(self._pending_prompts)}
        partial_items.sort(key=lambda item: pending_order[prompt_group_id(item.prompt_group)])
        self._inflight_replay = deque(partial_items)
        self._pipeline_telemetry.set_active_groups(0, self._max_in_flight_groups())

    def _inflight_item_from_result(self, item: QueueItem) -> _InflightReplayItem:
        group_id = prompt_group_id(item[0])
        prompt_group = self._pending_prompts.get(group_id)
        if prompt_group is None:
            raise RuntimeError(f"Inflight prompt group {group_id} is absent from the pending ledger")
        generation_group = item[1]
        if any(isinstance(sample, list) for sample in generation_group):
            raise RuntimeError("Inflight replay currently requires flat generation groups")
        return _InflightReplayItem(
            prompt_group=prompt_group,
            generation_group=generation_group,
        )

    def _capture_replay_buffer_state(self, rollout_id: int) -> dict[str, Any]:
        claimed_items = [task.result() for task in self._queue_gets if task.done() and not task.cancelled()]
        finished_active_items = [task.result() for task in self._active if task.done() and not task.cancelled()]
        queued_items = list(claimed_items)
        if self._queue_policy() == QUEUE_RECYCLE_POLICY:
            queued_items.extend(list(self._output._queue) if self._output is not None else [])
        else:
            queued_items.extend(list(self._policy_output) if self._policy_output is not None else [])
        promoted_items = list(self._completed_waiting.values())
        promoted_items.extend(finished_active_items)
        queue_snapshot = self._build_queue_replay_snapshot(queued_items, promoted_items)
        ready_items = queue_snapshot.ready_items
        inflight_items = list(self._inflight_replay)
        materialized = _materialized_group_ids(
            ready_items,
            self._drain_progress,
            self._prepared_batches,
            inflight_items,
        )
        missing = materialized - queue_snapshot.pending_prompts.keys()
        if missing:
            raise RuntimeError(f"Materialized groups are absent from the pending prompt ledger: {sorted(missing)}")
        regeneration_group_ids = self._regeneration_group_ids(materialized, queue_snapshot.pending_prompts)
        sample_encoder = ReplayBufferSampleEncoder(self._replay_buffer_packed_fields)
        state = {
            "dataset_fingerprint": self._dataset_fingerprint,
            "replay_buffer_type": self._replay_buffer_type,
            "queue_config": self._replay_buffer_queue_config(),
            "data_source": self.data_source.checkpoint_state(),
            "applied_weight_version": self._applied_weight_version.current(),
            "pending_prompts": [
                sample_encoder.encode_group(group) for group in queue_snapshot.pending_prompts.values()
            ],
            "ready_items": [
                _encode_ready_item(item, self._applied_weight_version.current(), sample_encoder)
                for item in ready_items
            ],
            "drain_progress": [
                self._encode_drain_progress(progress, sample_encoder) for progress in self._drain_progress.values()
            ],
            "prepared_batches": [
                self._encode_prepared_batch(batch_rollout_id, prepared, sample_encoder)
                for batch_rollout_id, prepared in self._prepared_batches.items()
            ],
            "inflight_items": [_encode_inflight_item(item, sample_encoder) for item in inflight_items],
            "regeneration_group_ids": regeneration_group_ids,
            "acked_batch_tokens": dict(self._acked_batch_tokens),
            "queue_telemetry": {
                "lifecycle": queue_snapshot.lifecycle_state,
                "producer_response_lengths": queue_snapshot.response_length_state,
                "producer_selection_populations": copy.deepcopy(self._producer_selection_populations),
                "queue_evicted_groups": queue_snapshot.queue_evicted_groups,
                "queue_evicted_tokens": queue_snapshot.queue_evicted_tokens,
            },
            "pipeline_telemetry": self._pipeline_telemetry.checkpoint_state(),
            "snapshot_counts": {
                "pending_groups": len(queue_snapshot.pending_prompts),
                "ready_groups": len(ready_items),
                "active_groups": len(self._active) - len(finished_active_items),
                "finished_active_groups": len(finished_active_items),
                "completed_waiting_groups": len(self._completed_waiting),
                "claimed_groups": len(claimed_items),
                "queue_evicted_groups": queue_snapshot.snapshot_evicted_groups,
                "partial_drains": len(self._drain_progress),
                "prepared_batches": len(self._prepared_batches),
                "inflight_groups": len(inflight_items),
                "inflight_response_tokens": sum(
                    group_response_tokens(item.generation_group) for item in inflight_items
                ),
            },
        }
        state[SAMPLE_CODEC_STATE_KEY] = sample_encoder.finish()
        logger.info(
            "Captured replay buffer %d: counts=%s, pack_cache=%s",
            rollout_id,
            state["snapshot_counts"],
            self._replay_buffer_packed_fields.stats(),
        )
        return state

    def _build_queue_replay_snapshot(
        self,
        queued_items: list[QueueItem],
        promoted_items: list[QueueItem],
    ) -> _QueueReplaySnapshot:
        pending_prompts = dict(self._pending_prompts)
        lifecycle_state = self._queue_lifecycle.checkpoint_state()
        response_length_state = self._producer_response_lengths.checkpoint_state()
        if self._queue_policy() != QUEUE_DROP_POLICY:
            return _QueueReplaySnapshot(
                ready_items=queued_items + promoted_items,
                pending_prompts=pending_prompts,
                lifecycle_state=lifecycle_state,
                response_length_state=response_length_state,
                queue_evicted_groups=self._queue_evicted_groups,
                queue_evicted_tokens=self._queue_evicted_tokens,
                snapshot_evicted_groups=0,
            )

        lifecycle = _QueueLifecycleRecorder(enabled=self._queue_lifecycle.enabled)
        lifecycle.restore_checkpoint_state(lifecycle_state)
        response_lengths = _ResponseLengthMetrics(populations=("generated", "queue_evicted"))
        response_lengths.restore_checkpoint_state(response_length_state)
        ready_items = deque(queued_items)
        evicted_groups = 0
        evicted_tokens = 0
        for item in promoted_items:
            depth_before = len(ready_items)
            queue_put_version = self._applied_weight_version.current()
            if depth_before >= self._queue_capacity_groups():
                evicted_item = ready_items.popleft()
                evicted_tokens += self._record_replay_buffer_queue_eviction(
                    evicted_item,
                    decision_version=queue_put_version,
                    pending_prompts=pending_prompts,
                    lifecycle=lifecycle,
                    response_lengths=response_lengths,
                )
                evicted_groups += 1
            ready_items.append(item)
            lifecycle.restore_queue_admission(
                item[1],
                queue_put_version=queue_put_version,
                depth_before=depth_before,
                depth_after=len(ready_items),
            )

        return _QueueReplaySnapshot(
            ready_items=list(ready_items),
            pending_prompts=pending_prompts,
            lifecycle_state=lifecycle.checkpoint_state(),
            response_length_state=response_lengths.checkpoint_state(),
            queue_evicted_groups=self._queue_evicted_groups + evicted_groups,
            queue_evicted_tokens=self._queue_evicted_tokens + evicted_tokens,
            snapshot_evicted_groups=evicted_groups,
        )

    @staticmethod
    def _record_replay_buffer_queue_eviction(
        item: QueueItem,
        *,
        decision_version: int,
        pending_prompts: dict[int, list[Sample]],
        lifecycle: _QueueLifecycleRecorder,
        response_lengths: _ResponseLengthMetrics,
    ) -> int:
        prompt_group, group = item
        group_id = prompt_group_id(prompt_group)
        if pending_prompts.pop(group_id, None) is None:
            raise RuntimeError(f"Queue-evicted prompt group {group_id} is absent from the pending ledger")
        response_lengths.record("queue_evicted", group)
        reference = group_first_prefill_weight_version(group)
        lifecycle.finish(
            group,
            disposition="queue_evicted",
            decision_version=decision_version,
            rollout_id=None,
            reference_version=reference,
        )
        return group_response_tokens(group)

    def _replay_buffer_queue_config(self) -> dict[str, Any]:
        return {
            "policy": self._queue_policy(),
            "capacity_groups": self._queue_capacity_groups(),
        }

    def _validate_replay_buffer_queue_config(self, state: dict[str, Any]) -> None:
        stored = state.get("queue_config")
        if stored is None:
            raise RuntimeError("Replay-buffer state is missing required queue_config")
        try:
            stored_config = {
                "policy": stored["policy"],
                "capacity_groups": int(stored["capacity_groups"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid replay-buffer queue configuration: {stored!r}") from exc
        current_config = self._replay_buffer_queue_config()
        if stored_config != current_config:
            raise RuntimeError(
                "Replay-buffer queue configuration does not match this run: "
                f"stored={stored_config}, current={current_config}"
            )

    async def restore_replay_buffer_state(self, state: dict[str, Any]) -> None:
        """Restore materialized trajectories and regenerate only active prompt leases."""
        if not self._replay_buffer_enabled:
            raise RuntimeError("Replay buffer is disabled")
        if self._worker is not None:
            raise RuntimeError("Fully-async rollout state must be restored before the worker starts")
        state = materialize_replay_buffer_state(state)
        stored_type = state.get("replay_buffer_type", REPLAY_BUFFER_ROLLOUT)
        if stored_type != self._replay_buffer_type:
            raise RuntimeError(
                "Replay-buffer type does not match this run: "
                f"stored={stored_type!r}, current={self._replay_buffer_type!r}"
            )
        self._validate_replay_buffer_queue_config(state)

        self.data_source.restore_checkpoint_state(state["data_source"])
        telemetry_state = state.get("queue_telemetry", {})
        self._queue_lifecycle.restore_checkpoint_state(telemetry_state.get("lifecycle"))
        self._producer_response_lengths.restore_checkpoint_state(telemetry_state.get("producer_response_lengths"))
        self._producer_selection_populations = copy.deepcopy(telemetry_state.get("producer_selection_populations", {}))
        self._queue_evicted_groups = int(telemetry_state.get("queue_evicted_groups", 0))
        self._queue_evicted_tokens = int(telemetry_state.get("queue_evicted_tokens", 0))
        self._pipeline_telemetry.restore_checkpoint_state(state.get("pipeline_telemetry"))
        applied_version = int(state["applied_weight_version"])
        self._applied_weight_version = AppliedWeightVersionTracker(applied_version)

        pending_groups = [decode_group(group) for group in state["pending_prompts"]]
        self._pending_prompts = {}
        for group in pending_groups:
            prompt_group = _flat_prompt_group(group)
            group_id = prompt_group_id(prompt_group)
            if group_id in self._pending_prompts:
                raise RuntimeError(f"Duplicate pending prompt group {group_id} in replay buffer")
            self._pending_prompts[group_id] = prompt_group

        ready_items = [_decode_ready_item(item) for item in state["ready_items"]]
        for prompt_group, result in ready_items:
            # Decode intentionally recreates prompt/result occurrences as
            # independent Samples. Cache both so a second replay-buffer capture after
            # resume does not move list/string conversion back onto its boundary.
            self._replay_buffer_packed_fields.cache_group(prompt_group)
            self._replay_buffer_packed_fields.cache_group(result)
        self._restore_ready_queue(ready_items)
        self._pipeline_telemetry.set_queue_depth(self._queue_size())

        self._drain_progress = {}
        for progress_state in state["drain_progress"]:
            progress = self._decode_drain_progress(progress_state)
            if progress.rollout_id in self._drain_progress:
                raise RuntimeError(f"Duplicate partial drain for rollout {progress.rollout_id}")
            self._drain_progress[progress.rollout_id] = progress
            for group in progress.data:
                self._replay_buffer_packed_fields.cache_group(group)

        self._prepared_batches = {}
        for prepared_state in state["prepared_batches"]:
            batch_rollout_id, prepared = self._decode_prepared_batch(prepared_state)
            if batch_rollout_id in self._prepared_batches:
                raise RuntimeError(f"Duplicate prepared batch for rollout {batch_rollout_id}")
            self._prepared_batches[batch_rollout_id] = prepared
            for group in prepared.output.samples:
                self._replay_buffer_packed_fields.cache_group(group)

        inflight_items = [_decode_inflight_item(item) for item in state.get("inflight_items", [])]
        self._inflight_replay = deque(inflight_items)
        # Inflight generation mutates response fields and status. The packed
        # cache is intentionally completed-sample-only, so these groups enter it
        # later in ``_generate_group`` after continuation finishes.
        acked_batch_tokens = {
            int(rollout_id): token for rollout_id, token in state.get("acked_batch_tokens", {}).items()
        }
        self._acked_batch_tokens = dict(sorted(acked_batch_tokens.items())[-ACKED_BATCH_HISTORY_SIZE:])

        materialized = _materialized_group_ids(
            ready_items,
            self._drain_progress,
            self._prepared_batches,
            inflight_items,
        )
        missing = materialized - self._pending_prompts.keys()
        if missing:
            raise RuntimeError(f"Materialized groups are absent from the pending prompt ledger: {sorted(missing)}")
        regeneration_group_ids = [int(group_id) for group_id in state["regeneration_group_ids"]]
        expected_regeneration_ids = self._pending_prompts.keys() - materialized
        if len(regeneration_group_ids) != len(set(regeneration_group_ids)) or set(regeneration_group_ids) != set(
            expected_regeneration_ids
        ):
            raise RuntimeError(
                "Fully-async regeneration order does not match pending non-materialized groups: "
                f"stored={regeneration_group_ids}, expected={sorted(expected_regeneration_ids)}"
            )
        regeneration_groups = [self._pending_prompts[group_id] for group_id in regeneration_group_ids]
        # Keep the canonical prompt leases detached from objects handed to the
        # generator. Generation mutates Sample instances in place; sharing these
        # objects would turn a partially generated response into the next
        # replay buffer's supposed original prompt.
        self.data_source.add_samples(copy.deepcopy(regeneration_groups))

        self._resume_metrics = {
            "resume/replay_buffer/pending_groups_restored": float(len(pending_groups)),
            "resume/replay_buffer/ready_groups_restored": float(len(ready_items)),
            "resume/replay_buffer/regenerated_active_groups": float(len(regeneration_groups)),
            "resume/replay_buffer/inflight_groups_restored": float(len(inflight_items)),
            "resume/replay_buffer/inflight_tokens_restored": float(
                sum(group_response_tokens(item.generation_group) for item in inflight_items)
            ),
            "resume/replay_buffer/partial_drains_restored": float(len(self._drain_progress)),
            "resume/replay_buffer/prepared_batches_restored": float(len(self._prepared_batches)),
            "resume/replay_buffer/applied_weight_version_restored": float(applied_version),
        }
        logger.info("Restored replay buffer: %s", self._resume_metrics)

    def _restore_ready_queue(self, ready_items: list[QueueItem]) -> None:
        policy = self._queue_policy()
        if policy == QUEUE_RECYCLE_POLICY:
            self._output = asyncio.Queue()
            for item in ready_items:
                self._output.put_nowait(item)
            self._policy_output = None
            self._policy_output_ready = None
        else:
            self._output = None
            self._policy_output = deque(ready_items)
            self._policy_output_ready = asyncio.Event()
            if ready_items:
                self._policy_output_ready.set()

        for depth_before, item in enumerate(ready_items):
            queue_put_version = group_lifecycle_weight_version(item[1], QUEUE_PUT_VERSION_KEY)
            if queue_put_version is None:
                raise RuntimeError("Restored ready group has no queue-put weight version")
            self._queue_lifecycle.restore_queue_admission(
                item[1],
                queue_put_version=queue_put_version,
                depth_before=depth_before,
                depth_after=depth_before + 1,
            )
        if policy in (QUEUE_RECYCLE_POLICY, QUEUE_MAX_POLICY):
            available_slots = max(0, self._queue_capacity_groups() - len(ready_items))
            self._output_slots = asyncio.Semaphore(available_slots)
        else:
            self._output_slots = None

    def _regeneration_group_ids(
        self,
        materialized: set[int],
        pending_prompts: dict[int, list[Sample]] | None = None,
    ) -> list[int]:
        pending_prompts = self._pending_prompts if pending_prompts is None else pending_prompts
        pending_regeneration_ids = pending_prompts.keys() - materialized
        retry_group_ids = [prompt_group_id(group) for group in self.data_source.checkpoint_retry_buffer_groups()]
        if len(retry_group_ids) != len(set(retry_group_ids)) or not set(retry_group_ids) <= set(
            pending_regeneration_ids
        ):
            raise RuntimeError(
                "Retry-buffer groups do not match pending prompt leases: "
                f"retry={retry_group_ids}, pending={sorted(pending_regeneration_ids)}"
            )
        active_group_ids = [
            group_id
            for group_id in pending_prompts
            if group_id in pending_regeneration_ids and group_id not in retry_group_ids
        ]
        # Requests that were already running at the snapshot regain the in-flight
        # slots they lost with the process. Retry-buffer groups remain FIFO behind
        # them, matching the failure-free state where those retries were waiting
        # while the active requests continued generation.
        return active_group_ids + retry_group_ids

    def restored_applied_weight_version(self) -> int:
        return self._applied_weight_version.current()

    def replay_buffer_dataset_fingerprint(self) -> str:
        if self._dataset_fingerprint is None:
            raise RuntimeError("Replay buffer is disabled")
        return self._dataset_fingerprint

    async def shutdown(self) -> None:
        tasks = [task for task in [self._worker, *self._active] if task is not None]
        tasks.extend(self._queue_gets)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._worker = None
        self._active.clear()
        self._pipeline_telemetry.set_active_groups(0, self._max_in_flight_groups())
        self._queue_gets.clear()

    @staticmethod
    def _encode_drain_progress(
        progress: _DrainProgress,
        sample_encoder: ReplayBufferSampleEncoder,
    ) -> dict[str, Any]:
        state = {key: copy.deepcopy(value) for key, value in progress.__dict__.items() if key != "data"}
        state["data"] = [sample_encoder.encode_group(group, use_packed_cache=True) for group in progress.data]
        return state

    @staticmethod
    def _decode_drain_progress(state: dict[str, Any]) -> _DrainProgress:
        state = copy.deepcopy(state)
        state["data"] = [decode_group(group) for group in state["data"]]
        return _DrainProgress(**state)

    @staticmethod
    def _encode_prepared_batch(
        rollout_id: int,
        prepared: _PreparedBatch,
        sample_encoder: ReplayBufferSampleEncoder,
    ) -> dict[str, Any]:
        return {
            "rollout_id": rollout_id,
            "samples": [
                sample_encoder.encode_group(group, use_packed_cache=True) for group in prepared.output.samples
            ],
            "metrics": copy.deepcopy(prepared.output.metrics),
            "debug_metadata": copy.deepcopy(prepared.output.debug_metadata),
            "group_ids": list(prepared.group_ids),
            "token": prepared.token,
        }

    @staticmethod
    def _decode_prepared_batch(state: dict[str, Any]) -> tuple[int, _PreparedBatch]:
        samples = [decode_group(group) for group in state["samples"]]
        prepared = _PreparedBatch(
            output=RolloutFnTrainOutput(
                samples=samples,
                metrics=copy.deepcopy(state["metrics"]),
                debug_metadata=copy.deepcopy(state.get("debug_metadata")),
            ),
            group_ids=tuple(int(group_id) for group_id in state["group_ids"]),
            token=state["token"],
        )
        if rollout_batch_token(samples) != prepared.token:
            raise RuntimeError(f"Prepared rollout batch {state['rollout_id']} token does not match its samples")
        return int(state["rollout_id"]), prepared

    def _take_resume_metrics(self) -> dict[str, float]:
        metrics, self._resume_metrics = self._resume_metrics, {}
        return metrics

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            raise ValueError(
                "FullyAsyncRolloutFn does not serve eval; set --eval-function-path to "
                "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
            )
        assert isinstance(input, RolloutFnTrainInput)
        if input.updates_before_train < 0:
            raise ValueError("updates_before_train must be non-negative")
        if prepared := self._prepared_batches.get(input.rollout_id):
            prepared.output.metrics = {
                **(prepared.output.metrics or {}),
                "resume/replay_buffer/warm_prepared_batch_hit": 1.0,
                "resume/replay_buffer/current_applied_weight_version": float(self._applied_weight_version.current()),
                **self._take_resume_metrics(),
            }
            logger.info("Reusing prepared fully-async rollout batch %d", input.rollout_id)
            return prepared.output

        self._ensure_worker()
        output = await self._drain(input.rollout_id, input.updates_before_train)
        if self._replay_buffer_enabled:
            group_ids = tuple(prompt_group_id([_first_sample(group)]) for group in output.samples)
            prepared = _PreparedBatch(
                output=output,
                group_ids=group_ids,
                token=rollout_batch_token(output.samples),
            )
            self._prepared_batches[input.rollout_id] = prepared
        return output

    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        if self._queue_policy() == QUEUE_RECYCLE_POLICY:
            if self._output is None:
                self._output = asyncio.Queue()
            if self._output_slots is None:
                available_slots = max(0, self._queue_capacity_groups() - self._output.qsize())
                self._output_slots = asyncio.Semaphore(available_slots)
        else:
            if self._policy_output is None:
                self._policy_output = deque()
            if self._policy_output_ready is None:
                self._policy_output_ready = asyncio.Event()
            if self._queue_policy() == QUEUE_MAX_POLICY and self._output_slots is None:
                available_slots = max(0, self._queue_capacity_groups() - len(self._policy_output))
                self._output_slots = asyncio.Semaphore(available_slots)
        self._pipeline_telemetry.set_queue_depth(self._queue_size())
        self._pipeline_telemetry.set_active_groups(len(self._active), self._max_in_flight_groups())
        self._pipeline_telemetry.reset_window()
        self._worker = asyncio.create_task(self._worker_loop())
        logger.info(
            "Started fully-async rollout worker (queue_type=%s, capacity_groups=%d)",
            self._queue_policy(),
            self._queue_capacity_groups(),
        )

    def _queue_policy(self) -> str:
        return getattr(self.args, "fully_async_queue_type", QUEUE_RECYCLE_POLICY)

    def _queue_capacity_groups(self) -> int:
        return self._queue_capacity

    def _queue_size(self) -> int:
        if self._queue_policy() == QUEUE_RECYCLE_POLICY:
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
        if self._inflight_replay:
            item = self._inflight_replay.popleft()
            return asyncio.create_task(
                self._generate_group(
                    item.prompt_group,
                    generation_group=item.generation_group,
                )
            )
        [prompt_group] = self.data_source.get_samples(1)
        if self._replay_buffer_enabled:
            group_id = prompt_group_id(prompt_group)
            if group_id not in self._pending_prompts:
                self._pending_prompts[group_id] = copy.deepcopy(prompt_group)
        return asyncio.create_task(self._generate_group(prompt_group))

    async def _generate_group(
        self,
        prompt_group: list[Sample],
        *,
        generation_group: Group | None = None,
    ) -> tuple[list[Sample], Group]:
        """Return the submitted prompt group next to its result.

        A retry has to resubmit the prompt group: a generate function may expand one
        trajectory into several samples, and ``generate_and_rm_group`` does not accept
        that shape back.
        """
        attempt_start = time.monotonic()
        continuation = generation_group is not None
        generation_group = prompt_group if generation_group is None else generation_group
        preserved_metadata = _prepare_generation_attempt(
            generation_group,
            continuation=continuation,
        )
        submission_version = await self._current_weight_version()
        lifecycle_submission_version = (
            group_submission_weight_version(generation_group) if preserved_metadata is not None else submission_version
        )
        lifecycle_record = self._queue_lifecycle.begin_attempt(
            generation_group,
            lifecycle_submission_version,
        )
        if preserved_metadata is None:
            stamp_submission_weight_version(generation_group, submission_version)
        trajectory_start_version = self._applied_weight_version.current()
        trajectory_start_time = time.time()
        if preserved_metadata is None:
            stamp_sample_lifecycle_boundary(
                _iter_samples(generation_group),
                version_key=TRAJECTORY_START_VERSION_KEY,
                version=trajectory_start_version,
                time_key=TRAJECTORY_START_TIME_KEY,
                wall_time=trajectory_start_time,
            )
        generate_kwargs = {
            "sampling_params": self.state.sampling_params.copy(),
            "evaluation": False,
        }
        if self._generation_lifecycle_supported:
            generate_kwargs["lifecycle_version_provider"] = self._applied_weight_version.current
        generation_count_recorded = False

        def record_generated_group(samples: list[Sample]) -> None:
            nonlocal generation_count_recorded
            if self._capturing_inflight and _group_requires_continuation(samples):
                return
            self._pipeline_telemetry.add_generated_group(sum(sample.response_length for sample in samples))
            generation_count_recorded = True

        if self._generation_completion_callback_supported:
            generate_kwargs["on_group_generation_complete"] = record_generated_group
        try:
            result = await generate_and_rm_group(self.state, generation_group, **generate_kwargs)
        except BaseException:
            self._queue_lifecycle.cancel_attempt(lifecycle_record)
            raise
        _restore_continuation_metadata(result, preserved_metadata)
        if self._capturing_inflight and _group_requires_continuation(result):
            self._queue_lifecycle.cancel_attempt(lifecycle_record)
            return prompt_group, result
        # Stamped again on the result: a generate function may return new Sample
        # objects rather than the ones it was handed.
        if preserved_metadata is None:
            stamp_submission_weight_version(result, submission_version)
        result_samples = list(_iter_samples(result))
        for sample in result_samples:
            if not isinstance(sample.metadata.get(TRAJECTORY_START_VERSION_KEY), int) or not isinstance(
                sample.metadata.get(TRAJECTORY_START_TIME_KEY), float
            ):
                stamp_sample_lifecycle_boundary(
                    [sample],
                    version_key=TRAJECTORY_START_VERSION_KEY,
                    version=trajectory_start_version,
                    time_key=TRAJECTORY_START_TIME_KEY,
                    wall_time=trajectory_start_time,
                )
        lifecycle_complete = all(
            isinstance(sample.metadata.get(SAMPLE_GENERATION_COMPLETE_VERSION_KEY), int)
            and isinstance(sample.metadata.get(GROUP_GENERATION_COMPLETE_VERSION_KEY), int)
            for sample in result_samples
        )
        if not lifecycle_complete:
            fallback_version = self._applied_weight_version.current()
            fallback_time = time.time()
            stamp_sample_lifecycle_boundary(
                result_samples,
                version_key=SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
                version=fallback_version,
                time_key=SAMPLE_GENERATION_COMPLETE_TIME_KEY,
                wall_time=fallback_time,
            )
            stamp_sample_lifecycle_boundary(
                result_samples,
                version_key=GROUP_GENERATION_COMPLETE_VERSION_KEY,
                version=fallback_version,
                time_key=GROUP_GENERATION_COMPLETE_TIME_KEY,
                wall_time=fallback_time,
            )
            for sample in result_samples:
                sample.metadata[LIFECYCLE_EXACT_KEY] = False
        stamp_attempt_wall_seconds(_iter_samples(result), time.monotonic() - attempt_start)
        # Completed siblings do not generate again while an aborted sibling is
        # continued. Restore their original per-sample generation/reward timing
        # after the group-level continuation attempt stamp above.
        _restore_continuation_metadata(result, preserved_metadata)
        ready_version = self._applied_weight_version.current()
        ready_time = time.time()
        stamp_group_weight_version(
            result,
            GROUP_READY_VERSION_KEY,
            ready_version,
        )
        stamp_sample_lifecycle_boundary(
            result_samples,
            version_key=GROUP_READY_VERSION_KEY,
            version=ready_version,
            time_key=GROUP_READY_TIME_KEY,
            wall_time=ready_time,
        )
        reward_values = None
        if lifecycle_record is not None:
            reward_values = group_reward_values(result, getattr(self.args, "reward_key", None))
        self._queue_lifecycle.group_ready(
            lifecycle_record,
            result,
            ready_version,
            reward_values=reward_values,
        )
        if not any(sample.status == Sample.Status.ABORTED for sample in _iter_samples(result)):
            self._producer_response_lengths.record("generated", result)
        add_selection_population(
            self._producer_selection_populations,
            population_name="generated",
            samples=_iter_samples(result),
        )
        if self._replay_buffer_packed_fields is not None:
            self._replay_buffer_packed_fields.cache_group(result)
        if not generation_count_recorded:
            self._pipeline_telemetry.add_generated_group(group_response_tokens(result))
        return prompt_group, result

    async def _worker_loop(self):
        while True:
            while self._completed_waiting:
                task_id, item = next(iter(self._completed_waiting.items()))
                await self._enqueue_completed_group(item)
                del self._completed_waiting[task_id]
            while len(self._active) < self._max_in_flight_groups():
                self._active.add(self._submit_one_group())
            self._pipeline_telemetry.set_active_groups(len(self._active), self._max_in_flight_groups())
            done, self._active = await asyncio.wait(self._active, return_when=asyncio.FIRST_COMPLETED)
            self._pipeline_telemetry.set_active_groups(len(self._active), self._max_in_flight_groups())
            completed = [(id(task), task.result()) for task in done]
            if self._replay_buffer_enabled:
                self._completed_waiting.update(completed)
            for task_id, item in completed:
                await self._enqueue_completed_group(item)
                if self._replay_buffer_enabled:
                    del self._completed_waiting[task_id]

    async def _enqueue_completed_group(self, item: QueueItem) -> None:
        policy = self._queue_policy()
        if policy in (QUEUE_RECYCLE_POLICY, QUEUE_MAX_POLICY):
            # These policies control age at selection time. The configured
            # completed-group limit is a safety backpressure bound, not their
            # experimental selection capacity.
            backpressure_start = time.monotonic()
            await self._output_slots.acquire()
            self._pipeline_telemetry.add_rollout_backpressure(time.monotonic() - backpressure_start)

        depth_before = self._queue_size()
        queue_put_version = self._applied_weight_version.current()
        queue_put_time = time.time()
        stamp_group_weight_version(item[1], QUEUE_PUT_VERSION_KEY, queue_put_version)
        stamp_sample_lifecycle_boundary(
            _iter_samples(item[1]),
            version_key=QUEUE_PUT_VERSION_KEY,
            version=queue_put_version,
            time_key=QUEUE_PUT_TIME_KEY,
            wall_time=queue_put_time,
        )

        if policy == QUEUE_RECYCLE_POLICY:
            self._output.put_nowait(item)
        else:
            if policy == QUEUE_DROP_POLICY and depth_before >= self._queue_capacity_groups():
                evicted_prompt_group, evicted_group = self._policy_output.popleft()
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
                )
                self._finish_prompt(evicted_prompt_group)
            self._policy_output.append(item)
            self._policy_output_ready.set()

        self._queue_lifecycle.enqueued(
            item[1],
            queue_put_version=queue_put_version,
            depth_before=depth_before,
            depth_after=self._queue_size(),
        )
        self._pipeline_telemetry.set_queue_depth(self._queue_size())

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> tuple[list[Sample], Group]:
        if self._worker.done():
            self._worker.result()
            raise RuntimeError("fully-async rollout worker exited without an exception")
        if not self._output.empty():
            result = self._output.get_nowait()
            self._pipeline_telemetry.set_queue_depth(self._output.qsize())
            self._release_output_slot()
            self._queue_lifecycle.dequeued(result[1], depth_after_observed=self._output.qsize())
            return result
        starvation_start = time.monotonic()
        queue_get = asyncio.create_task(self._output.get())
        if self._replay_buffer_enabled:
            self._queue_gets.add(queue_get)
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
                    self._pipeline_telemetry.add_trainer_starvation(time.monotonic() - starvation_start)
                    self._pipeline_telemetry.set_queue_depth(self._output.qsize())
                    self._release_output_slot()
                    self._queue_lifecycle.dequeued(result[1], depth_after_observed=self._output.qsize())
                    return result
                logger.warning(
                    f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s (queued: {self._output.qsize()})"
                )
        finally:
            if self._replay_buffer_enabled:
                self._queue_gets.discard(queue_get)
            if not queue_get.done():
                queue_get.cancel()

    async def _take_policy_groups(self, count: int) -> list[tuple[QueueItem, int]]:
        """Wait for and atomically remove ``count`` oldest completed groups."""
        assert self._queue_policy() != QUEUE_RECYCLE_POLICY
        # Match queue-recycle's fail-fast contract even when a dead worker
        # left enough completed groups to satisfy this request immediately.
        if self._worker is not None and self._worker.done():
            self._worker.result()
            raise RuntimeError("fully-async rollout worker exited without an exception")
        starvation_start = time.monotonic() if len(self._policy_output) < count else None
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
        if starvation_start is not None:
            self._pipeline_telemetry.add_trainer_starvation(time.monotonic() - starvation_start)

        selected = []
        for _ in range(count):
            item = self._policy_output.popleft()
            if self._queue_policy() == QUEUE_MAX_POLICY:
                self._release_output_slot()
            depth_after = len(self._policy_output)
            self._queue_lifecycle.dequeued(item[1], depth_after_observed=depth_after)
            selected.append((item, depth_after))
        self._pipeline_telemetry.set_queue_depth(len(self._policy_output))
        return selected

    def _release_output_slot(self) -> None:
        """Release capacity only after a restored overfull queue reaches its cap."""
        if self._queue_size() < self._queue_capacity_groups():
            self._output_slots.release()

    async def _drain(self, rollout_id: int, updates_before_train: int) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        if self._replay_buffer_enabled:
            other_drains = set(self._drain_progress) - {rollout_id}
            if other_drains:
                raise RuntimeError(f"Cannot start rollout {rollout_id} with partial drains {sorted(other_drains)}")
            progress = self._drain_progress.setdefault(
                rollout_id,
                _DrainProgress(
                    rollout_id=rollout_id,
                    updates_before_train=updates_before_train,
                ),
            )
        else:
            progress = _DrainProgress(
                rollout_id=rollout_id,
                updates_before_train=updates_before_train,
            )
        data = progress.data
        if progress.queue_size_start is None:
            progress.queue_size_start = self._queue_size()
        queue_size_start = progress.queue_size_start
        queue_sizes_after_dequeue = progress.queue_sizes_after_dequeue
        candidates: deque[tuple[QueueItem, int]] = deque()
        # Would-be train staleness of every group handed to the drain, before the
        # bound check. This is ``train_version - reference``; the reference is
        # completion, submission, or first prefill according to the configured semantics.
        # The accepted population is already recorded by ``trained_total`` below.
        rollout_staleness = progress.rollout_staleness
        # The decomposition. Per group, with R the selected bound reference, Q
        # the version the group became trainable under, and T the train version:
        #   pre-queue = Q - R   updates crossed before the group became trainable
        #   in-queue  = T - Q   updates crossed before training
        #   total     = T - R   = pre-queue + in-queue
        trained_pre_queue = progress.trained_pre_queue
        trained_in_queue = progress.trained_in_queue
        trained_total = progress.trained_total
        offered_mixed_versions = progress.offered_mixed_versions
        trained_mixed_versions = progress.trained_mixed_versions
        # Generation that was produced and then thrown away. Counted in tokens, not
        # groups, because that is the unit a sample-efficiency claim is made in, and
        # because the three ways to waste generation cost wildly different amounts.
        response_length_metrics = _ResponseLengthMetrics(
            sample_lengths=progress.response_sample_lengths,
            group_max_lengths=progress.response_group_max_lengths,
        )
        while len(data) < target_data_size:
            if not candidates:
                if self._queue_policy() == QUEUE_RECYCLE_POLICY:
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
                samples = list(_iter_samples(group))
                tokens = group_response_tokens(group)
                current = self._applied_weight_version.current()
                train_version = _scheduled_train_version(progress, current)
                submitted = group_submission_weight_version(group)
                ready = group_lifecycle_weight_version(group, GROUP_READY_VERSION_KEY)
                queue_put = group_lifecycle_weight_version(group, QUEUE_PUT_VERSION_KEY)
                stamp_sample_lifecycle_boundary(
                    _iter_samples(group),
                    version_key=DRAIN_VERSION_KEY,
                    version=current,
                    time_key=DRAIN_TIME_KEY,
                    wall_time=time.time(),
                )
                stamp_group_weight_version(group, TRAIN_VERSION_KEY, train_version)
                add_selection_population(
                    progress.selection_populations,
                    population_name="recycled",
                    samples=samples,
                )
                reason = aborted_recycle_reason(
                    submission_version=submitted,
                    group_ready_version=ready,
                )
                waste = add_discard_accounting(
                    progress.waste_by_reason,
                    reason=reason,
                    samples=samples,
                )
                _record_reason(progress, reason=reason, tokens=tokens)
                if getattr(args, "save_debug_rollout_data", None) is not None:
                    progress.discard_records.append(
                        recycle_record(
                            samples,
                            disposition="aborted_recycled",
                            reason_code=reason,
                            reference_mode=args.staleness_reference,
                            reference_version=None,
                            generation_completion_version=group_generation_completion_version(samples),
                            group_ready_version=ready,
                            queue_put_version=queue_put,
                            drain_version=current,
                            train_version=train_version,
                            bound=args.max_weight_staleness,
                            waste=waste,
                        )
                    )
                progress.aborted_tokens += tokens
                self._queue_lifecycle.finish(
                    group,
                    disposition="aborted_recycled",
                    decision_version=current,
                    rollout_id=rollout_id,
                    train_version=train_version,
                )
                self._recycle(prompt_group)
                progress.aborted_groups_recycled += 1
                continue

            response_length_metrics.record("offered", group)

            oldest = group_oldest_weight_version(group)
            submitted = group_submission_weight_version(group)
            ready = group_lifecycle_weight_version(group, GROUP_READY_VERSION_KEY)
            queue_put = group_lifecycle_weight_version(group, QUEUE_PUT_VERSION_KEY)
            current = self._applied_weight_version.current()
            train_version = _scheduled_train_version(progress, current)
            stamp_sample_lifecycle_boundary(
                _iter_samples(group),
                version_key=DRAIN_VERSION_KEY,
                version=current,
                time_key=DRAIN_TIME_KEY,
                wall_time=time.time(),
            )
            stamp_group_weight_version(group, TRAIN_VERSION_KEY, train_version)

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
            if reference is not None:
                stamp_group_weight_version(group, BOUND_REFERENCE_VERSION_KEY, reference)
            stamp_sample_reference_versions(_iter_samples(group), args.staleness_reference)

            samples = list(_iter_samples(group))
            group_train_staleness: int | None = None
            if reference is not None:
                dequeue_staleness = current - reference
                train_staleness = train_version - reference
                if dequeue_staleness < 0 or train_staleness < 0:
                    raise RuntimeError(
                        f"Negative weight staleness: dequeue={current}, train={train_version}, "
                        f"reference={reference}, mode={args.staleness_reference}"
                    )
                group_train_staleness = train_staleness
                rollout_staleness.append(train_staleness)
                progress.bound_evaluated_samples += len(samples)
                max_staleness = args.max_weight_staleness
                strict_bound = self._queue_policy() == QUEUE_RECYCLE_POLICY
                exceeds_bound = max_staleness is not None and (
                    dequeue_staleness >= max_staleness if strict_bound else dequeue_staleness > max_staleness
                )
                if exceeds_bound:
                    progress.bound_exceeded_samples += len(samples)
                    tokens = group_response_tokens(group)
                    disposition = (
                        "age_cutoff_dropped" if self._queue_policy() == QUEUE_MAX_POLICY else "stale_recycled"
                    )
                    generation_completion = group_generation_completion_version(samples)
                    reason = classify_stale_recycle_stage(
                        reference_version=reference,
                        generation_completion_version=generation_completion,
                        group_ready_version=ready,
                        queue_put_version=queue_put,
                        drain_version=current,
                        bound=args.max_weight_staleness,
                        strict_bound=strict_bound,
                    )
                    collateral = straggler_collateral_indices(
                        samples,
                        reference_mode=args.staleness_reference,
                        drain_version=current,
                        bound=args.max_weight_staleness,
                        strict_bound=strict_bound,
                    )
                    waste = add_discard_accounting(
                        progress.waste_by_reason,
                        reason=reason,
                        samples=samples,
                    )
                    outcome_population = "dropped" if disposition == "age_cutoff_dropped" else "recycled"
                    add_selection_population(
                        progress.selection_populations,
                        population_name=outcome_population,
                        samples=samples,
                    )
                    _record_reason(progress, reason=reason, tokens=tokens)
                    if collateral:
                        collateral_tokens = sum(
                            sample.response_length for sample in samples if sample.index in collateral
                        )
                        _record_reason(
                            progress,
                            reason="group_straggler_collateral",
                            tokens=collateral_tokens,
                            auxiliary=True,
                        )
                    if getattr(args, "save_debug_rollout_data", None) is not None:
                        progress.discard_records.append(
                            recycle_record(
                                samples,
                                disposition=disposition,
                                reason_code=reason,
                                reference_mode=args.staleness_reference,
                                reference_version=reference,
                                generation_completion_version=generation_completion,
                                group_ready_version=ready,
                                queue_put_version=queue_put,
                                drain_version=current,
                                train_version=train_version,
                                bound=args.max_weight_staleness,
                                waste=waste,
                                collateral_indices=collateral,
                            )
                        )
                    response_length_metrics.record(disposition, group)
                    self._queue_lifecycle.finish(
                        group,
                        disposition=disposition,
                        decision_version=current,
                        rollout_id=rollout_id,
                        reference_version=reference,
                        train_version=train_version,
                        bound_staleness=train_staleness,
                    )
                    if disposition == "age_cutoff_dropped":
                        progress.age_cutoff_tokens += tokens
                        progress.stale_groups_dropped += 1
                        self._finish_prompt(prompt_group)
                    else:
                        progress.stale_tokens += tokens
                        progress.stale_groups_recycled += 1
                        self._recycle(prompt_group)
                    comparison = ">=" if strict_bound else ">"
                    logger.info(
                        f"Rejected stale group ({args.staleness_reference}_version={reference}, "
                        f"dequeue={current}, dequeue_staleness={dequeue_staleness} {comparison} "
                        f"max={max_staleness}, train={train_version}, train_staleness={train_staleness})"
                    )
                    continue

            # Not gated on the bound: the bound tests one derived quantity, and every
            # arm -- including an unbounded one -- needs the decomposition.
            # The decomposition uses the exact same start as the bound and ends at
            # the scheduled train version.
            span_start = reference
            have_span = ready is not None and span_start is not None
            group_pre_queue = ready - span_start if have_span else None
            group_in_queue = train_version - ready if have_span else None
            group_total = train_version - span_start if have_span else None
            for name, value in (
                ("pre_queue", group_pre_queue),
                ("in_queue", group_in_queue),
                ("total", group_total),
            ):
                if value is not None and value < 0:
                    raise RuntimeError(
                        f"Negative {name} weight staleness for group: start={span_start}, "
                        f"ready={ready}, drain={current}, train={train_version}"
                    )

            filter_output = call_dynamic_filter(self._dynamic_filter, args, group)
            if not filter_output.keep:
                # Dropped, not recycled: no usable gradient signal.
                response_length_metrics.record("dynamic_filter_dropped", group)
                samples = list(_iter_samples(group))
                tokens = group_response_tokens(group)
                waste = add_discard_accounting(
                    progress.waste_by_reason,
                    reason="dynamic_filter_dropped",
                    samples=samples,
                )
                add_selection_population(
                    progress.selection_populations,
                    population_name="dropped",
                    samples=samples,
                )
                if getattr(args, "save_debug_rollout_data", None) is not None:
                    progress.discard_records.append(
                        recycle_record(
                            samples,
                            disposition="dynamic_filter_dropped",
                            reason_code="dynamic_filter_dropped",
                            reference_mode=args.staleness_reference,
                            reference_version=reference,
                            generation_completion_version=group_generation_completion_version(samples),
                            group_ready_version=ready,
                            queue_put_version=queue_put,
                            drain_version=current,
                            train_version=train_version,
                            bound=args.max_weight_staleness,
                            waste=waste,
                            detail=filter_output.reason,
                        )
                    )
                progress.filtered_tokens += tokens
                self._queue_lifecycle.finish(
                    group,
                    disposition="dynamic_filter_dropped",
                    decision_version=current,
                    rollout_id=rollout_id,
                    reference_version=reference,
                    train_version=train_version,
                    bound_staleness=group_train_staleness,
                    detail=filter_output.reason,
                )
                if filter_output.reason:
                    progress.dynamic_filter_drop_counts[filter_output.reason] = (
                        progress.dynamic_filter_drop_counts.get(filter_output.reason, 0) + 1
                    )
                self._finish_prompt(prompt_group)
                continue

            if progress.do_print:
                sample = _first_sample(group)
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                    f"label: {sample.label}, reward: {sample.reward}"
                )
                progress.do_print = False

            if group_pre_queue is not None:
                trained_pre_queue.append(group_pre_queue)
            if group_in_queue is not None:
                trained_in_queue.append(group_in_queue)
            if group_total is not None:
                trained_total.append(group_total)
            trained_mixed_versions.append(mixed_version)
            response_length_metrics.record("trained", group)
            accepted_samples = list(_iter_samples(group))
            add_selection_population(
                progress.selection_populations,
                population_name="admitted",
                samples=accepted_samples,
            )
            if getattr(args, "save_debug_rollout_data", None) is not None:
                progress.discard_records.append(
                    recycle_record(
                        accepted_samples,
                        disposition="admitted",
                        reason_code="accepted_for_training",
                        reference_mode=args.staleness_reference,
                        reference_version=reference,
                        generation_completion_version=group_generation_completion_version(accepted_samples),
                        group_ready_version=ready,
                        queue_put_version=queue_put,
                        drain_version=current,
                        train_version=train_version,
                        bound=args.max_weight_staleness,
                        waste={},
                    )
                )
            self._queue_lifecycle.finish(
                group,
                disposition="trained",
                decision_version=current,
                rollout_id=rollout_id,
                reference_version=reference,
                train_version=train_version,
                bound_staleness=group_train_staleness,
            )
            data.append(group)
            progress.group_ids.append(prompt_group_id(prompt_group))

        sample = _first_sample(data[-1])
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
            f"label: {sample.label}, reward: {sample.reward}"
        )

        data.sort(key=lambda group: _first_sample(group).index)

        admitted_tokens_before_sample_filter = sum(group_response_tokens(group) for group in data)
        if self._sample_filter is not None:
            self._sample_filter(args, data)

        kept_tokens = sum(group_response_tokens(group) for group in data)
        queue_evicted_groups = self._queue_evicted_groups
        queue_evicted_tokens = self._queue_evicted_tokens
        self._queue_evicted_groups = 0
        self._queue_evicted_tokens = 0
        wasted_tokens = (
            progress.aborted_tokens
            + progress.stale_tokens
            + progress.age_cutoff_tokens
            + progress.filtered_tokens
            + queue_evicted_tokens
        )
        generated_tokens = wasted_tokens + admitted_tokens_before_sample_filter
        producer_selection_populations = self._producer_selection_populations
        self._producer_selection_populations = {}
        metrics = {
            "rollout/fully_async/queue_size": self._queue_size(),
            "queue/occupancy/start_groups": queue_size_start,
            "queue/occupancy/end_groups": self._queue_size(),
            "queue/occupancy/capacity_groups": self._queue_capacity_groups(),
            "queue/occupancy/max_in_flight_groups": self._max_in_flight_groups(),
            "queue/config/type_is_queue_recycle": float(self._queue_policy() == QUEUE_RECYCLE_POLICY),
            "queue/config/type_is_queue_max": float(self._queue_policy() == QUEUE_MAX_POLICY),
            "queue/config/type_is_queue_drop": float(self._queue_policy() == QUEUE_DROP_POLICY),
            "queue/config/factor": getattr(args, "fully_async_queue_factor", 1),
            "queue/config/training_buffer_queue_size": getattr(
                args,
                "training_buffer_queue_size",
                DEFAULT_TRAINING_BUFFER_QUEUE_SIZE,
            ),
            "rollout/fully_async/aborted_groups_recycled": progress.aborted_groups_recycled,
            "rollout/fully_async/stale_groups_recycled": progress.stale_groups_recycled,
            "rollout/fully_async/stale_groups_dropped": progress.stale_groups_dropped,
            "rollout/fully_async/queue_evicted_groups": queue_evicted_groups,
            "rollout/fully_async/aborted_tokens": progress.aborted_tokens,
            "rollout/fully_async/stale_tokens": progress.stale_tokens,
            "rollout/fully_async/age_cutoff_tokens": progress.age_cutoff_tokens,
            "rollout/fully_async/queue_evicted_tokens": queue_evicted_tokens,
            "rollout/fully_async/dynamic_filter_tokens": progress.filtered_tokens,
            "rollout/fully_async/kept_tokens": kept_tokens,
            "rollout/fully_async/wasted_token_frac": (
                wasted_tokens / (wasted_tokens + kept_tokens) if wasted_tokens + kept_tokens else 0.0
            ),
            GENERATED_TOKENS_KEY: generated_tokens,
            ADMITTED_TOKENS_KEY: admitted_tokens_before_sample_filter,
            **{
                f"queue/occupancy/after_dequeue/{name}": value
                for name, value in _distribution_metrics(queue_sizes_after_dequeue).items()
            },
            **self._producer_response_lengths.collect_and_reset(),
            **response_length_metrics.collect(),
            **{
                f"rollout/dynamic_filter/drop_{reason}": count
                for reason, count in progress.dynamic_filter_drop_counts.items()
            },
            **self._take_resume_metrics(),
            **discard_waste_metrics(progress.waste_by_reason),
            **selection_population_metrics(
                producer_selection_populations,
                population_names=("generated",),
            ),
            **selection_population_metrics(
                progress.selection_populations,
                population_names=("admitted", "recycled", "dropped"),
            ),
        }
        for reason in RECYCLE_REASONS:
            count = progress.recycle_reason_groups.get(reason, 0)
            metrics[f"rollout/fully_async/recycle_reason/{reason}/groups"] = count
            metrics[f"rollout/fully_async/recycle_reason/{reason}/tokens"] = progress.recycle_reason_tokens.get(
                reason, 0
            )
        for reason in RECYCLE_AUX_REASONS:
            count = progress.recycle_aux_groups.get(reason, 0)
            metrics[f"rollout/fully_async/recycle_aux/{reason}/groups"] = count
            metrics[f"rollout/fully_async/recycle_aux/{reason}/tokens"] = progress.recycle_aux_tokens.get(reason, 0)
        if progress.train_version is not None:
            metrics[TRAIN_WEIGHT_VERSION_METRIC] = progress.train_version
        if offered_mixed_versions:
            metrics["staleness/mixed_version_frac/rollout"] = sum(offered_mixed_versions) / len(offered_mixed_versions)
        if trained_mixed_versions:
            metrics["staleness/mixed_version_frac/train"] = sum(trained_mixed_versions) / len(trained_mixed_versions)
        if rollout_staleness:
            metrics |= {
                f"staleness/rollout/{name}": value for name, value in _staleness_metrics(rollout_staleness).items()
            }

        # The decomposition over the trained batch. `pre_queue` is the selected
        # reference-to-ready span, and `in_queue` is ready-to-train staleness.
        for name, values in (
            ("total", trained_total),
            ("pre_queue", trained_pre_queue),
            ("in_queue", trained_in_queue),
        ):
            if values:
                metrics |= {f"staleness/{name}/{key}": value for key, value in _staleness_metrics(values).items()}

        # Named for the reason rather than the mechanism: `stale_groups_recycled`
        # is what happened, `bound_exceeded` is why. Split from the dynamic-filter
        # drops, which land in the same recycled/dropped bucket if only totals are
        # compared.
        bound_exceeded_groups = progress.stale_groups_recycled + progress.stale_groups_dropped
        bound_exceeded_tokens = progress.stale_tokens + progress.age_cutoff_tokens
        metrics["staleness/bound_exceeded_groups"] = bound_exceeded_groups
        metrics["staleness/bound_exceeded_samples"] = progress.bound_exceeded_samples
        metrics["staleness/bound_exceeded_sample_frac"] = (
            progress.bound_exceeded_samples / progress.bound_evaluated_samples
            if progress.bound_evaluated_samples
            else 0.0
        )
        metrics["staleness/bound_exceeded_tokens"] = bound_exceeded_tokens
        # The staleness namespaces mean a different quantity under each reference,
        # so the choice is logged next to them rather than left to the run config.
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
                f"(--staleness-reference {args.staleness_reference}, queue-type "
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
        if getattr(args, "save_debug_rollout_data", None) is not None:
            if debug_metadata is None:
                debug_metadata = {}
            debug_metadata["recycle_compute"] = {
                "schema_version": RECYCLE_DEBUG_SCHEMA_VERSION,
                "records": progress.discard_records,
            }

        self._drain_progress.pop(rollout_id, None)
        return RolloutFnTrainOutput(samples=data, metrics=metrics, debug_metadata=debug_metadata)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.retry_count += 1
            sample.reset_for_retry()
        if self._replay_buffer_enabled:
            self._pending_prompts[prompt_group_id(prompt_group)] = copy.deepcopy(prompt_group)
        self.data_source.add_samples([prompt_group])

    def _finish_prompt(self, prompt_group: list[Sample]) -> None:
        if self._replay_buffer_enabled:
            group_id = prompt_group_id(prompt_group)
            if self._pending_prompts.pop(group_id, None) is None:
                raise RuntimeError(f"Finished prompt group {group_id} is absent from the pending ledger")
