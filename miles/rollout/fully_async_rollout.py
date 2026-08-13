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
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnInput, RolloutFnOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import call_dynamic_filter
from miles.rollout.fully_async_checkpoint import (
    dataset_fingerprint,
    decode_group,
    encode_group,
    prompt_group_id,
    rollout_batch_token,
)
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils.http_utils import get
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

OUTPUT_QUEUE_MAX_GROUPS = 1000
ACKED_BATCH_HISTORY_SIZE = 16
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0
# Realized lag is a small integer; anything past this goes in one overflow bucket
# so the metric count stays bounded no matter how far behind a run drifts.
#
# 16, not 8: unbounded runs and runs with a parked bound need enough resolution
# in the tail to show how realized staleness maps to downstream score. At 8 the
# tail collapsed into one overflow bucket.
STALENESS_HISTOGRAM_MAX = 16

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


@dataclass
class _DrainProgress:
    rollout_id: int
    data: list[Group] = field(default_factory=list)
    group_ids: list[int] = field(default_factory=list)
    aborted_groups_recycled: int = 0
    stale_groups_recycled: int = 0
    trained_bound_staleness: list[int] = field(default_factory=list)
    offered_bound_staleness: list[int] = field(default_factory=list)
    trained_pre_queue: list[int] = field(default_factory=list)
    trained_in_queue: list[int] = field(default_factory=list)
    trained_total: list[int] = field(default_factory=list)
    current_version: int = 0
    offered_mixed_versions: list[bool] = field(default_factory=list)
    trained_mixed_versions: list[bool] = field(default_factory=list)
    aborted_tokens: int = 0
    stale_tokens: int = 0
    filtered_tokens: int = 0
    dynamic_filter_drop_counts: dict[str, int] = field(default_factory=dict)
    do_print: bool = True


@dataclass(frozen=True)
class _PreparedBatch:
    output: RolloutFnTrainOutput
    group_ids: tuple[int, ...]
    token: str


def _iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def _flat_prompt_group(group: Group) -> list[Sample]:
    if any(isinstance(item, list) for item in group):
        raise RuntimeError("A pending prompt lease must be a flat list of Sample objects")
    return list(group)


def _encode_ready_item(item: tuple[list[Sample], Group], queue_put_version: int) -> dict[str, Any]:
    prompt_group, result = item
    result = copy.deepcopy(result)
    if group_lifecycle_weight_version(result, QUEUE_PUT_VERSION_KEY) is None:
        # A completed task may be blocked on queue capacity when the snapshot is
        # taken. Restore promotes it into the reconstructed ready queue, so the
        # durable snapshot boundary is its queue-put version. Do not mutate the
        # live result: failure-free execution may enqueue it under a later version.
        stamp_group_weight_version(result, QUEUE_PUT_VERSION_KEY, queue_put_version)
    return {"prompt_group": encode_group(prompt_group), "result": encode_group(result)}


def _decode_ready_item(state: dict[str, Any]) -> tuple[list[Sample], Group]:
    prompt_group = _flat_prompt_group(decode_group(state["prompt_group"]))
    return prompt_group, decode_group(state["result"])


def _materialized_group_ids(
    ready_items: list[tuple[list[Sample], Group]],
    drains: dict[int, _DrainProgress],
    prepared_batches: dict[int, _PreparedBatch],
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

    all_ids = ready_ids + drained_ids + prepared_ids
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("A fully-async prompt group appears in more than one materialized lifecycle state")
    return set(all_ids)


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
    return min(versions) if versions else None


def group_queue_entry_weight_version(group: Group) -> int | None:
    """Return the version the group became available to the trainer under.

    The **maximum**, not the minimum. A group is one concurrent request per
    sample joined by ``asyncio.gather`` (``inference_rollout_common.py:137-146``),
    so it enters the output queue when its *slowest* sample lands. Taking the
    minimum would charge a straggler that crossed a weight update to in-queue
    staleness, when it is the defining case of pre-queue staleness.
    """
    versions = [v for s in _iter_samples(group) if (v := s.newest_weight_version) is not None]
    return max(versions) if versions else None


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


def group_first_prefill_weight_version(group: Group) -> int | None:
    versions = [version for sample in _iter_samples(group) for version in sample.first_prefill_weight_versions]
    return min(versions) if versions else None


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


def group_response_tokens(group: Group) -> int:
    """Response tokens generated for a group.

    Call it *before* recycling: ``Sample.reset_for_retry`` clears ``tokens`` and
    ``response`` (``types.py:236``), so a count taken afterwards is always zero.
    """
    return sum(sample.response_length for sample in _iter_samples(group))


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
    weight update pausing generation) or whose weights are older than
    ``--max-weight-staleness`` are recycled back into the data source.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self._dynamic_filter = load_function(input.args.dynamic_sampling_filter_path)
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._weight_version = _CachedWeightVersion()
        self._applied_weight_version = AppliedWeightVersionTracker()
        self._worker: asyncio.Task | None = None
        self._output: asyncio.Queue[tuple[list[Sample], Group]] | None = None
        self._output_slots: asyncio.Semaphore | None = None
        self._active: set[asyncio.Task] = set()
        self._completed_waiting: dict[int, tuple[list[Sample], Group]] = {}
        self._queue_gets: set[asyncio.Task] = set()

        self._checkpoint_enabled = getattr(input.args, "fully_async_rollout_checkpoint", False)
        self._dataset_fingerprint = (
            dataset_fingerprint(input.args, input.data_source) if self._checkpoint_enabled else None
        )
        self._pending_prompts: dict[int, list[Sample]] = {}
        self._drain_progress: dict[int, _DrainProgress] = {}
        self._prepared_batches: dict[int, _PreparedBatch] = {}
        self._acked_batch_tokens: dict[int, str] = {}
        self._resume_metrics: dict[str, float] = {}

    def commit_applied_weight_version(self, version: int) -> None:
        self._applied_weight_version.commit(version)

    async def commit_applied_weight_version_on_loop(self, version: int) -> None:
        self.commit_applied_weight_version(version)

    async def current_applied_weight_version(self) -> int:
        return self._applied_weight_version.current()

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

    async def checkpoint_state(self, rollout_id: int) -> dict[str, Any]:
        """Capture one coherent lifecycle snapshot on the rollout event loop."""
        if not self._checkpoint_enabled:
            raise RuntimeError("Fully-async rollout checkpointing is disabled")
        claimed_items = [task.result() for task in self._queue_gets if task.done() and not task.cancelled()]
        finished_active_items = [task.result() for task in self._active if task.done() and not task.cancelled()]
        ready_items = list(claimed_items)
        ready_items.extend(list(self._output._queue) if self._output is not None else [])
        ready_items.extend(self._completed_waiting.values())
        ready_items.extend(finished_active_items)
        materialized = _materialized_group_ids(ready_items, self._drain_progress, self._prepared_batches)
        missing = materialized - self._pending_prompts.keys()
        if missing:
            raise RuntimeError(f"Materialized groups are absent from the pending prompt ledger: {sorted(missing)}")
        regeneration_group_ids = self._regeneration_group_ids(materialized)
        state = {
            "dataset_fingerprint": self._dataset_fingerprint,
            "data_source": self.data_source.checkpoint_state(),
            "applied_weight_version": self._applied_weight_version.current(),
            "pending_prompts": [encode_group(group) for group in self._pending_prompts.values()],
            "ready_items": [_encode_ready_item(item, self._applied_weight_version.current()) for item in ready_items],
            "drain_progress": [self._encode_drain_progress(progress) for progress in self._drain_progress.values()],
            "prepared_batches": [
                self._encode_prepared_batch(batch_rollout_id, prepared)
                for batch_rollout_id, prepared in self._prepared_batches.items()
            ],
            "regeneration_group_ids": regeneration_group_ids,
            "acked_batch_tokens": dict(self._acked_batch_tokens),
            "snapshot_counts": {
                "pending_groups": len(self._pending_prompts),
                "ready_groups": len(ready_items),
                "active_groups": len(self._active) - len(finished_active_items),
                "finished_active_groups": len(finished_active_items),
                "completed_waiting_groups": len(self._completed_waiting),
                "claimed_groups": len(claimed_items),
                "partial_drains": len(self._drain_progress),
                "prepared_batches": len(self._prepared_batches),
            },
        }
        logger.info(
            "Captured fully-async rollout state %d: %s",
            rollout_id,
            state["snapshot_counts"],
        )
        return state

    async def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore materialized trajectories and regenerate only active prompt leases."""
        if not self._checkpoint_enabled:
            raise RuntimeError("Fully-async rollout checkpointing is disabled")
        if self._worker is not None:
            raise RuntimeError("Fully-async rollout state must be restored before the worker starts")

        self.data_source.restore_checkpoint_state(state["data_source"])
        applied_version = int(state["applied_weight_version"])
        self._applied_weight_version = AppliedWeightVersionTracker(applied_version)

        pending_groups = [decode_group(group) for group in state["pending_prompts"]]
        self._pending_prompts = {}
        for group in pending_groups:
            prompt_group = _flat_prompt_group(group)
            group_id = prompt_group_id(prompt_group)
            if group_id in self._pending_prompts:
                raise RuntimeError(f"Duplicate pending prompt group {group_id} in rollout checkpoint")
            self._pending_prompts[group_id] = prompt_group

        ready_items = [_decode_ready_item(item) for item in state["ready_items"]]
        self._output = asyncio.Queue()
        for item in ready_items:
            self._output.put_nowait(item)

        self._drain_progress = {}
        for progress_state in state["drain_progress"]:
            progress = self._decode_drain_progress(progress_state)
            if progress.rollout_id in self._drain_progress:
                raise RuntimeError(f"Duplicate partial drain for rollout {progress.rollout_id}")
            self._drain_progress[progress.rollout_id] = progress

        self._prepared_batches = {}
        for prepared_state in state["prepared_batches"]:
            batch_rollout_id, prepared = self._decode_prepared_batch(prepared_state)
            if batch_rollout_id in self._prepared_batches:
                raise RuntimeError(f"Duplicate prepared batch for rollout {batch_rollout_id}")
            self._prepared_batches[batch_rollout_id] = prepared
        acked_batch_tokens = {
            int(rollout_id): token for rollout_id, token in state.get("acked_batch_tokens", {}).items()
        }
        self._acked_batch_tokens = dict(sorted(acked_batch_tokens.items())[-ACKED_BATCH_HISTORY_SIZE:])

        materialized = _materialized_group_ids(ready_items, self._drain_progress, self._prepared_batches)
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
        # checkpoint's supposed original prompt.
        self.data_source.add_samples(copy.deepcopy(regeneration_groups))
        self._output_slots = asyncio.Semaphore(max(0, OUTPUT_QUEUE_MAX_GROUPS - len(ready_items)))

        self._resume_metrics = {
            "resume/fully_async/pending_groups_restored": float(len(pending_groups)),
            "resume/fully_async/ready_groups_restored": float(len(ready_items)),
            "resume/fully_async/regenerated_active_groups": float(len(regeneration_groups)),
            "resume/fully_async/partial_drains_restored": float(len(self._drain_progress)),
            "resume/fully_async/prepared_batches_restored": float(len(self._prepared_batches)),
            "resume/fully_async/applied_weight_version_restored": float(applied_version),
        }
        logger.info("Restored fully-async rollout state: %s", self._resume_metrics)

    def _regeneration_group_ids(self, materialized: set[int]) -> list[int]:
        pending_regeneration_ids = self._pending_prompts.keys() - materialized
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
            for group_id in self._pending_prompts
            if group_id in pending_regeneration_ids and group_id not in retry_group_ids
        ]
        # Requests that were already running at the snapshot regain the in-flight
        # slots they lost with the process. Retry-buffer groups remain FIFO behind
        # them, matching the failure-free state where those retries were waiting
        # while the active requests continued generation.
        return active_group_ids + retry_group_ids

    def restored_applied_weight_version(self) -> int:
        return self._applied_weight_version.current()

    def checkpoint_dataset_fingerprint(self) -> str:
        if self._dataset_fingerprint is None:
            raise RuntimeError("Fully-async rollout checkpointing is disabled")
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
        self._queue_gets.clear()

    @staticmethod
    def _encode_drain_progress(progress: _DrainProgress) -> dict[str, Any]:
        state = {key: copy.deepcopy(value) for key, value in progress.__dict__.items() if key != "data"}
        state["data"] = [encode_group(group) for group in progress.data]
        return state

    @staticmethod
    def _decode_drain_progress(state: dict[str, Any]) -> _DrainProgress:
        state = copy.deepcopy(state)
        state["data"] = [decode_group(group) for group in state["data"]]
        return _DrainProgress(**state)

    @staticmethod
    def _encode_prepared_batch(rollout_id: int, prepared: _PreparedBatch) -> dict[str, Any]:
        return {
            "rollout_id": rollout_id,
            "samples": [encode_group(group) for group in prepared.output.samples],
            "metrics": copy.deepcopy(prepared.output.metrics),
            "group_ids": list(prepared.group_ids),
            "token": prepared.token,
        }

    @staticmethod
    def _decode_prepared_batch(state: dict[str, Any]) -> tuple[int, _PreparedBatch]:
        samples = [decode_group(group) for group in state["samples"]]
        prepared = _PreparedBatch(
            output=RolloutFnTrainOutput(samples=samples, metrics=copy.deepcopy(state["metrics"])),
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
        if prepared := self._prepared_batches.get(input.rollout_id):
            prepared.output.metrics = {
                **(prepared.output.metrics or {}),
                "resume/fully_async/warm_prepared_batch_hit": 1.0,
                "resume/fully_async/current_applied_weight_version": float(self._applied_weight_version.current()),
                **self._take_resume_metrics(),
            }
            logger.info("Reusing prepared fully-async rollout batch %d", input.rollout_id)
            return prepared.output

        self._ensure_worker()
        output = await self._drain(input.rollout_id)
        if self._checkpoint_enabled:
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
        if self._output is None:
            self._output = asyncio.Queue()
        if self._output_slots is None:
            available_slots = max(0, OUTPUT_QUEUE_MAX_GROUPS - self._output.qsize())
            self._output_slots = asyncio.Semaphore(available_slots)
        self._worker = asyncio.create_task(self._worker_loop())
        logger.info("Started fully-async rollout worker")

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
        if self._checkpoint_enabled:
            group_id = prompt_group_id(prompt_group)
            if group_id not in self._pending_prompts:
                self._pending_prompts[group_id] = copy.deepcopy(prompt_group)
        return asyncio.create_task(self._generate_group(prompt_group))

    async def _generate_group(self, prompt_group: list[Sample]) -> tuple[list[Sample], Group]:
        """Return the submitted prompt group next to its result.

        A retry has to resubmit the prompt group: a generate function may expand one
        trajectory into several samples, and ``generate_and_rm_group`` does not accept
        that shape back.
        """
        submission_version = await self._current_weight_version()
        stamp_submission_weight_version(prompt_group, submission_version)
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
        )
        # Stamped again on the result: a generate function may return new Sample
        # objects rather than the ones it was handed.
        stamp_submission_weight_version(result, submission_version)
        stamp_group_weight_version(
            result,
            GROUP_READY_VERSION_KEY,
            self._applied_weight_version.current(),
        )
        return prompt_group, result

    async def _worker_loop(self):
        while True:
            while len(self._active) < self._max_in_flight_groups():
                self._active.add(self._submit_one_group())
            done, self._active = await asyncio.wait(self._active, return_when=asyncio.FIRST_COMPLETED)
            completed = [(id(task), task.result()) for task in done]
            if self._checkpoint_enabled:
                self._completed_waiting.update(completed)
            for task_id, item in completed:
                # Blocks when the queue is full: training lagging behind rollout
                # production pauses submission instead of growing the queue unboundedly.
                await self._output_slots.acquire()
                stamp_group_weight_version(
                    item[1],
                    QUEUE_PUT_VERSION_KEY,
                    self._applied_weight_version.current(),
                )
                self._output.put_nowait(item)
                if self._checkpoint_enabled:
                    del self._completed_waiting[task_id]

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> tuple[list[Sample], Group]:
        if self._worker.done():
            self._worker.result()
            raise RuntimeError("fully-async rollout worker exited without an exception")
        if not self._output.empty():
            result = self._output.get_nowait()
            self._release_output_slot()
            return result
        queue_get = asyncio.create_task(self._output.get())
        if self._checkpoint_enabled:
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
                    self._release_output_slot()
                    return result
                logger.warning(
                    f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s (queued: {self._output.qsize()})"
                )
        finally:
            if self._checkpoint_enabled:
                self._queue_gets.discard(queue_get)
            if not queue_get.done():
                queue_get.cancel()

    def _release_output_slot(self) -> None:
        """Release capacity only after a restored overfull queue reaches its cap."""
        if self._output.qsize() < OUTPUT_QUEUE_MAX_GROUPS:
            self._output_slots.release()

    async def _drain(self, rollout_id: int) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        if self._checkpoint_enabled:
            other_drains = set(self._drain_progress) - {rollout_id}
            if other_drains:
                raise RuntimeError(f"Cannot start rollout {rollout_id} with partial drains {sorted(other_drains)}")
            progress = self._drain_progress.setdefault(
                rollout_id,
                _DrainProgress(
                    rollout_id=rollout_id,
                    current_version=self._applied_weight_version.current(),
                ),
            )
        else:
            progress = _DrainProgress(
                rollout_id=rollout_id,
                current_version=self._applied_weight_version.current(),
            )
        data = progress.data
        # Two populations, because they answer different questions and only one of
        # them is the study's variable. ``offered`` (logged as
        # ``staleness/bound/rollout/``) is every group the pipeline handed over,
        # including those the bound then sent back -- that is the *natural* lag of
        # this node ratio. ``trained`` (``staleness/bound/train/``) is what survived into
        # the batch, and is what the loss actually saw. They diverge where the
        # bound bites and where the dynamic filter drops a group, which is where a
        # reader is most likely to be misled.
        # What `--max-weight-staleness` is tested against, in its two populations.
        # This is `current - oldest`, which is neither of the two components below:
        # it is in-queue staleness plus however far the group's samples spread
        # across versions among themselves.
        trained_bound_staleness = progress.trained_bound_staleness
        offered_bound_staleness = progress.offered_bound_staleness
        # The decomposition. Per group, with R the selected bound reference, Q
        # the version the group became trainable under, and C the version at drain:
        #   pre-queue = Q - R   updates crossed before the group became trainable
        #   in-queue  = C - Q   updates crossed while waiting to be trained on
        #   total     = C - R   = pre-queue + in-queue = the bound quantity
        trained_pre_queue = progress.trained_pre_queue
        trained_in_queue = progress.trained_in_queue
        trained_total = progress.trained_total
        offered_mixed_versions = progress.offered_mixed_versions
        trained_mixed_versions = progress.trained_mixed_versions
        # Generation that was produced and then thrown away. Counted in tokens, not
        # groups, because that is the unit a sample-efficiency claim is made in, and
        # because the three ways to waste generation cost wildly different amounts.
        while len(data) < target_data_size:
            prompt_group, group = await self._next_group()
            assert len(group) == args.n_samples_per_prompt

            # A weight update paused generation mid-group: return it for re-sampling.
            if any(s.status == Sample.Status.ABORTED for s in _iter_samples(group)):
                progress.aborted_tokens += group_response_tokens(group)
                self._recycle(prompt_group)
                progress.aborted_groups_recycled += 1
                continue

            oldest = group_oldest_weight_version(group)
            submitted = group_submission_weight_version(group)
            ready = group_lifecycle_weight_version(group, GROUP_READY_VERSION_KEY)
            current = self._applied_weight_version.current()
            progress.current_version = current
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
                        progress.stale_tokens += group_response_tokens(group)
                        self._recycle(prompt_group)
                        progress.stale_groups_recycled += 1
                        logger.info(
                            f"Recycled stale group ({args.staleness_reference}_version={reference}, "
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
                progress.filtered_tokens += group_response_tokens(group)
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

            if group_bound_staleness is not None:
                trained_bound_staleness.append(group_bound_staleness)
            if group_pre_queue is not None:
                trained_pre_queue.append(group_pre_queue)
            if group_in_queue is not None:
                trained_in_queue.append(group_in_queue)
            if group_total is not None:
                trained_total.append(group_total)
            trained_mixed_versions.append(mixed_version)
            data.append(group)
            progress.group_ids.append(prompt_group_id(prompt_group))

        sample = _first_sample(data[-1])
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
            f"label: {sample.label}, reward: {sample.reward}"
        )

        data.sort(key=lambda group: _first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        kept_tokens = sum(group_response_tokens(group) for group in data)
        wasted_tokens = progress.aborted_tokens + progress.stale_tokens + progress.filtered_tokens
        metrics = {
            "rollout/fully_async/queue_size": self._output.qsize(),
            "rollout/fully_async/aborted_groups_recycled": progress.aborted_groups_recycled,
            "rollout/fully_async/stale_groups_recycled": progress.stale_groups_recycled,
            "rollout/fully_async/aborted_tokens": progress.aborted_tokens,
            "rollout/fully_async/stale_tokens": progress.stale_tokens,
            "rollout/fully_async/dynamic_filter_tokens": progress.filtered_tokens,
            "rollout/fully_async/kept_tokens": kept_tokens,
            "rollout/fully_async/wasted_token_frac": (
                wasted_tokens / (wasted_tokens + kept_tokens) if wasted_tokens + kept_tokens else 0.0
            ),
            **{
                f"rollout/dynamic_filter/drop_{reason}": count
                for reason, count in progress.dynamic_filter_drop_counts.items()
            },
            **self._take_resume_metrics(),
        }
        if progress.current_version is not None:
            # Logged next to the staleness itself: staleness is a difference against
            # this version, and without it a missing staleness metric is impossible to
            # tell apart from a router that never answered.
            metrics["rollout/fully_async/current_weight_version"] = progress.current_version
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
        metrics["staleness/bound_exceeded_groups"] = progress.stale_groups_recycled
        metrics["staleness/bound_exceeded_tokens"] = progress.stale_tokens
        # `staleness/bound/*` means a different quantity under each reference, so the
        # choice is logged next to it rather than left to the run config.
        metrics["staleness/bound_reference_is_submission"] = float(args.staleness_reference == "submission")
        metrics["staleness/bound_reference_is_prefill"] = float(args.staleness_reference == "prefill")

        # Rejecting more groups than were kept means the bound is not being met by
        # regenerating, which `--staleness-reference submission` makes reachable:
        # the pre-queue part of the lag is the generation itself, so a retry pays
        # it again rather than shrinking it.
        #
        # This does not deadlock. The drain blocks the training step and the
        # training step is what publishes the weight update, so a stalled drain
        # freezes the version; groups submitted after the freeze cross no update
        # and pass. The equilibrium is the *synchronous* one -- everything in
        # flight at each boundary discarded, the batch refilled from cold -- which
        # is a throughput collapse rather than a hang, and is what this warning is
        # for. `retry_count_max` above says how deep it has gone.
        if progress.stale_groups_recycled > target_data_size:
            logger.warning(
                f"Recycled {progress.stale_groups_recycled} groups to keep {target_data_size} at "
                f"--max-weight-staleness {args.max_weight_staleness} "
                f"(--staleness-reference {args.staleness_reference}). If this persists the pipeline "
                "has degenerated to synchronous: the batch is being refilled from cold after every "
                "weight update instead of overlapping with training."
            )

        # How many times a group that reached training had to be regenerated. The
        # per-sample counter survives `reset_for_retry`, so this is cumulative over
        # the group's whole life, not just this rollout.
        retries = [max(sample.retry_count for sample in _iter_samples(group)) for group in data]
        if retries:
            metrics["staleness/retry_count_mean"] = sum(retries) / len(retries)
            metrics["staleness/retry_count_max"] = float(max(retries))
            metrics["staleness/retry_frac_nonzero"] = sum(1 for r in retries if r) / len(retries)

        self._drain_progress.pop(rollout_id, None)
        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.retry_count += 1
            sample.reset_for_retry()
        if self._checkpoint_enabled:
            self._pending_prompts[prompt_group_id(prompt_group)] = copy.deepcopy(prompt_group)
        self.data_source.add_samples([prompt_group])

    def _finish_prompt(self, prompt_group: list[Sample]) -> None:
        if self._checkpoint_enabled:
            group_id = prompt_group_id(prompt_group)
            if self._pending_prompts.pop(group_id, None) is None:
                raise RuntimeError(f"Finished prompt group {group_id} is absent from the pending ledger")
