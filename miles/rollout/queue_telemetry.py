"""Low-overhead response-length and lifecycle telemetry for rollout queues."""

import copy
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


def group_reward_values(group: Group, reward_key: str | None) -> list[float | None]:
    """Return scalar rewards aligned with the group's flattened samples.

    Queue diagnostics must never trigger reward computation. The values here are
    copied only after generation and reward evaluation have completed. A custom
    non-scalar reward remains missing rather than making optional telemetry fail
    an otherwise usable rollout.
    """
    values = []
    for sample in _iter_samples(group):
        reward = sample.reward
        if reward is not None and reward_key:
            reward = reward.get(reward_key) if isinstance(reward, dict) else None
        if isinstance(reward, (int, float, np.integer, np.floating)):
            values.append(float(reward))
        else:
            values.append(None)
    return values


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


DEFAULT_RESPONSE_LENGTH_POPULATIONS = (
    "offered",
    "trained",
    "stale_recycled",
    "age_cutoff_dropped",
    "dynamic_filter_dropped",
    "aborted_recycled",
)


class _ResponseLengthMetrics:
    """Small integer-only view of queue admission outcomes.

    Queue admission is group-valued, while the loss and token cost are
    sample-valued. Keep both views: the slowest response is the natural proxy for
    group completion latency, and the flattened sample lengths describe the data
    distribution seen (or rejected) by training.
    """

    def __init__(
        self,
        populations: tuple[str, ...] | None = None,
        *,
        sample_lengths: dict[str, list[int]] | None = None,
        group_max_lengths: dict[str, list[int]] | None = None,
    ) -> None:
        self._populations = populations or DEFAULT_RESPONSE_LENGTH_POPULATIONS
        if (sample_lengths is None) != (group_max_lengths is None):
            raise ValueError("response-length backing stores must be provided together")
        if sample_lengths is None:
            sample_lengths = {population: [] for population in self._populations}
            group_max_lengths = {population: [] for population in self._populations}
        expected = set(self._populations)
        if set(sample_lengths) != expected or set(group_max_lengths) != expected:
            raise ValueError("response-length backing stores do not match their populations")
        self._sample_lengths = sample_lengths
        self._group_max_lengths = group_max_lengths

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

    def checkpoint_state(self) -> dict:
        return {
            "populations": list(self._populations),
            "sample_lengths": copy.deepcopy(self._sample_lengths),
            "group_max_lengths": copy.deepcopy(self._group_max_lengths),
        }

    def restore_checkpoint_state(self, state: dict | None) -> None:
        if state is None:
            return
        populations = tuple(state["populations"])
        if populations != self._populations:
            raise RuntimeError(
                "Response-length checkpoint populations do not match this run: "
                f"stored={populations}, current={self._populations}"
            )
        self._sample_lengths = copy.deepcopy(state["sample_lengths"])
        self._group_max_lengths = copy.deepcopy(state["group_max_lengths"])


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
        self._records_by_group_key: dict[str, dict] = {}
        self._terminal_records: list[dict] = []

    def _now_ns(self) -> int:
        return time.monotonic_ns() - self._origin_ns

    @staticmethod
    def _group_key(group: Group) -> str:
        samples = list(_iter_samples(group))
        group_index = _first_sample(group).group_index
        if group_index is not None:
            return f"group:{group_index}"
        return "samples:" + ",".join(str(sample.index) for sample in samples)

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

    def group_ready(
        self,
        record: dict | None,
        group: Group,
        ready_version: int,
        *,
        reward_values: list[float | None] | None = None,
    ) -> None:
        if record is None:
            return
        samples = list(_iter_samples(group))
        if reward_values is not None and len(reward_values) != len(samples):
            raise ValueError(f"Reward telemetry has {len(reward_values)} values for {len(samples)} flattened samples")
        record.update(
            response_lengths=[sample.response_length for sample in samples],
            statuses=[sample.status.value for sample in samples],
            first_prefill_version=group_first_prefill_weight_version(group),
            completion_version_min=group_oldest_weight_version(group),
            completion_version_max=group_queue_entry_weight_version(group),
            ready_version=ready_version,
            ready_time_ns=self._now_ns(),
        )
        if reward_values is not None:
            record["reward_values"] = list(reward_values)
        key = self._group_key(group)
        if key in self._records_by_group_key:
            raise RuntimeError(f"Duplicate live queue lifecycle identity: {key}")
        self._records_by_group_key[key] = record

    def enqueued(
        self,
        group: Group,
        *,
        queue_put_version: int,
        depth_before: int,
        depth_after: int,
    ) -> None:
        if (record := self._records_by_group_key.get(self._group_key(group))) is None:
            return
        record.update(
            enqueue_seq=self._next_enqueue_seq,
            queue_put_version=queue_put_version,
            enqueue_time_ns=self._now_ns(),
            queue_depth_before_enqueue=depth_before,
            queue_depth_after_enqueue=depth_after,
        )
        self._next_enqueue_seq += 1

    def restore_queue_admission(
        self,
        group: Group,
        *,
        queue_put_version: int,
        depth_before: int,
        depth_after: int,
    ) -> None:
        """Complete admission metadata for work promoted by a checkpoint.

        A snapshot can capture a completed task while it is blocked behind queue
        capacity. The checkpoint promotes that task into its restored ready queue,
        so the snapshot boundary becomes its admission point. Items that had
        already entered the live queue keep their original admission record.
        """
        record = self._records_by_group_key.get(self._group_key(group))
        if record is None or "enqueue_seq" in record:
            return
        self.enqueued(
            group,
            queue_put_version=queue_put_version,
            depth_before=depth_before,
            depth_after=depth_after,
        )

    def dequeued(self, group: Group, *, depth_after_observed: int) -> None:
        if (record := self._records_by_group_key.get(self._group_key(group))) is None:
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
        if (record := self._records_by_group_key.pop(self._group_key(group), None)) is None:
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

    def checkpoint_state(self) -> dict | None:
        if not self.enabled:
            return None
        return {
            "schema_version": self.SCHEMA_VERSION,
            "elapsed_ns": self._now_ns(),
            "next_attempt_id": self._next_attempt_id,
            "next_submission_seq": self._next_submission_seq,
            "next_enqueue_seq": self._next_enqueue_seq,
            "next_dequeue_seq": self._next_dequeue_seq,
            "live_records": copy.deepcopy(self._records_by_group_key),
            "terminal_records": copy.deepcopy(self._terminal_records),
        }

    def restore_checkpoint_state(self, state: dict | None) -> None:
        if not self.enabled or state is None:
            return
        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported queue lifecycle checkpoint schema: {state.get('schema_version')!r}")
        self._origin_ns = time.monotonic_ns() - int(state["elapsed_ns"])
        self._next_attempt_id = int(state["next_attempt_id"])
        self._next_submission_seq = int(state["next_submission_seq"])
        self._next_enqueue_seq = int(state["next_enqueue_seq"])
        self._next_dequeue_seq = int(state["next_dequeue_seq"])
        self._records_by_group_key = copy.deepcopy(state["live_records"])
        self._terminal_records = copy.deepcopy(state["terminal_records"])

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
