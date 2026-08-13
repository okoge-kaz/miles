"""Low-overhead response-length and lifecycle telemetry for rollout queues."""

import time
from collections.abc import Iterator

import numpy as np

from miles.utils.types import Sample

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


def _iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [version for sample in _iter_samples(group) if (version := sample.oldest_weight_version) is not None]
    return min(versions) if versions else None


def group_queue_entry_weight_version(group: Group) -> int | None:
    """Return the version the group became available to the trainer under.

    The maximum, not the minimum, represents the slowest sample in the group.
    """
    versions = [version for sample in _iter_samples(group) if (version := sample.newest_weight_version) is not None]
    return max(versions) if versions else None


def group_first_prefill_weight_version(group: Group) -> int | None:
    versions = [version for sample in _iter_samples(group) for version in sample.first_prefill_weight_versions]
    return min(versions) if versions else None


def group_response_tokens(group: Group) -> int:
    """Return response tokens before a possible recycle clears generated fields."""
    return sum(sample.response_length for sample in _iter_samples(group))


def _distribution_metrics(values: list[int]) -> dict[str, float]:
    """Reduce an in-memory integer population to fixed-cardinality scalars."""
    if not values:
        return {"count": 0.0, "sum": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "count": float(array.size),
        "sum": float(array.sum()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


class _ResponseLengthMetrics:
    """Small integer-only view of queue admission outcomes.

    Queue admission is group-valued, while the loss and token cost are
    sample-valued. Keep both views: the slowest response is the natural proxy for
    group completion latency, and the flattened sample lengths describe the data
    distribution seen (or rejected) by training.
    """

    _DEFAULT_POPULATIONS = (
        "offered",
        "trained",
        "stale_recycled",
        "age_cutoff_dropped",
        "dynamic_filter_dropped",
        "aborted_recycled",
    )

    def __init__(self, populations: tuple[str, ...] | None = None) -> None:
        self._populations = populations or self._DEFAULT_POPULATIONS
        self._sample_lengths = {population: [] for population in self._populations}
        self._group_max_lengths = {population: [] for population in self._populations}

    def record(self, population: str, group: Group) -> None:
        lengths = [sample.response_length for sample in _iter_samples(group)]
        if not lengths:
            return
        self._sample_lengths[population].extend(lengths)
        self._group_max_lengths[population].append(max(lengths))

    def collect(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for population, sample_lengths in self._sample_lengths.items():
            for view, values in (
                ("sample_length", sample_lengths),
                ("group_max_length", self._group_max_lengths[population]),
            ):
                metrics |= {
                    f"queue/selection/{population}/{view}/{name}": value
                    for name, value in _distribution_metrics(values).items()
                }
        return metrics

    def collect_and_reset(self) -> dict[str, float]:
        metrics = self.collect()
        self._sample_lengths = {population: [] for population in self._populations}
        self._group_max_lengths = {population: [] for population in self._populations}
        return metrics


class _QueueLifecycleRecorder:
    """Compact queue-attempt records, enabled only alongside rollout dumps.

    The full samples are already serialized by ``save_debug_rollout_data``. This
    recorder adds only primitive identifiers, versions, lengths, queue depths,
    and monotonic timestamps. Terminal records are removed from the live map
    immediately, so rejected groups never remain reachable through diagnostics.
    """

    SCHEMA_VERSION = 1

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._origin_ns = time.monotonic_ns()
        self._next_attempt_id = 0
        self._next_submission_seq = 0
        self._next_enqueue_seq = 0
        self._next_dequeue_seq = 0
        self._records_by_group_id: dict[int, dict] = {}
        self._terminal_records: list[dict] = []

    def _now_ns(self) -> int:
        return time.monotonic_ns() - self._origin_ns

    def begin_attempt(self, prompt_group: list[Sample], submission_version: int | None) -> dict | None:
        if not self.enabled:
            return None
        samples = list(_iter_samples(prompt_group))
        record = {
            "attempt_id": self._next_attempt_id,
            "submission_seq": self._next_submission_seq,
            "group_index": _first_sample(prompt_group).group_index,
            "sample_indices": [sample.index for sample in samples],
            "retry_count": max((sample.retry_count for sample in samples), default=0),
            "submission_version": submission_version,
            "submission_time_ns": self._now_ns(),
        }
        self._next_attempt_id += 1
        self._next_submission_seq += 1
        return record

    def cancel_attempt(self, record: dict | None) -> None:
        if record is not None:
            record.clear()

    def group_ready(self, record: dict | None, group: Group, ready_version: int) -> None:
        if record is None:
            return
        samples = list(_iter_samples(group))
        record.update(
            response_lengths=[sample.response_length for sample in samples],
            statuses=[sample.status.value for sample in samples],
            first_prefill_version=group_first_prefill_weight_version(group),
            completion_version_min=group_oldest_weight_version(group),
            completion_version_max=group_queue_entry_weight_version(group),
            ready_version=ready_version,
            ready_time_ns=self._now_ns(),
        )
        self._records_by_group_id[id(group)] = record

    def enqueued(
        self,
        group: Group,
        *,
        queue_put_version: int,
        depth_before: int,
        depth_after: int,
    ) -> None:
        if (record := self._records_by_group_id.get(id(group))) is None:
            return
        record.update(
            enqueue_seq=self._next_enqueue_seq,
            queue_put_version=queue_put_version,
            enqueue_time_ns=self._now_ns(),
            queue_depth_before_enqueue=depth_before,
            queue_depth_after_enqueue=depth_after,
        )
        self._next_enqueue_seq += 1

    def dequeued(self, group: Group, *, depth_after_observed: int) -> None:
        if (record := self._records_by_group_id.get(id(group))) is None:
            return
        record.update(
            dequeue_seq=self._next_dequeue_seq,
            dequeue_time_ns=self._now_ns(),
            queue_depth_after_dequeue_observed=depth_after_observed,
        )
        self._next_dequeue_seq += 1

    def finish(
        self,
        group: Group,
        *,
        disposition: str,
        decision_version: int,
        rollout_id: int | None,
        reference_version: int | None = None,
        bound_staleness: int | None = None,
        detail: str | None = None,
    ) -> None:
        if (record := self._records_by_group_id.pop(id(group), None)) is None:
            return
        record.update(
            disposition=disposition,
            decision_version=decision_version,
            decision_time_ns=self._now_ns(),
            rollout_id=rollout_id,
            reference_version=reference_version,
            bound_staleness=bound_staleness,
        )
        if detail is not None:
            record["detail"] = detail
        self._terminal_records.append(record)

    def take_metadata(self, *, policy: str, capacity_groups: int) -> dict | None:
        if not self.enabled:
            return None
        records = self._terminal_records
        self._terminal_records = []
        return {
            "schema_version": self.SCHEMA_VERSION,
            "clock": "monotonic_ns_since_rollout_fn_construction",
            "policy": policy,
            "capacity_groups": capacity_groups,
            "decision_version_semantics": "queue_selection",
            "records": records,
        }
