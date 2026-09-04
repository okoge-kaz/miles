"""Standardized metrics for replay-buffer resume comparisons."""

from __future__ import annotations

from typing import Any


METRIC_PREFIX = "resume/benchmark"


def checkpoint_resume_metrics(
    *,
    rollout_id: int,
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    data_source_state: dict[str, Any],
    replay_state: dict[str, Any] | None,
) -> dict[str, float]:
    """Describe sample work that a checkpoint can carry across restart."""

    trained_groups = (rollout_id + 1) * rollout_batch_size
    allocated_groups = int(data_source_state["sample_group_index"])
    if allocated_groups < trained_groups:
        raise RuntimeError(
            "Rollout cursor trails trained prompt groups: "
            f"allocated={allocated_groups}, trained={trained_groups}"
        )
    outstanding_groups = allocated_groups - trained_groups

    if replay_state is None:
        replay_type = "none"
        pending_groups = 0
        completed_groups = 0
        partial_groups = 0
        partial_tokens = 0
        regenerated_groups = 0
        carried_groups = 0
        lost_groups = outstanding_groups
    else:
        replay_type = str(replay_state["replay_buffer_type"])
        counts = replay_state["snapshot_counts"]
        pending_groups = int(counts["pending_groups"])
        completed_groups = _completed_materialized_groups(replay_state)
        partial_groups = int(counts["inflight_groups"])
        partial_tokens = int(counts["inflight_response_tokens"])
        regenerated_groups = len(replay_state["regeneration_group_ids"])
        carried_groups = pending_groups
        lost_groups = 0

    preserved_work_groups = completed_groups + partial_groups
    return {
        f"{METRIC_PREFIX}/checkpoint/trained_groups": float(trained_groups),
        f"{METRIC_PREFIX}/checkpoint/allocated_groups": float(allocated_groups),
        f"{METRIC_PREFIX}/checkpoint/outstanding_groups": float(outstanding_groups),
        f"{METRIC_PREFIX}/checkpoint/outstanding_samples": float(
            outstanding_groups * n_samples_per_prompt
        ),
        f"{METRIC_PREFIX}/checkpoint/carried_groups": float(carried_groups),
        f"{METRIC_PREFIX}/checkpoint/carried_samples": float(
            carried_groups * n_samples_per_prompt
        ),
        f"{METRIC_PREFIX}/checkpoint/lost_groups": float(lost_groups),
        f"{METRIC_PREFIX}/checkpoint/lost_samples": float(lost_groups * n_samples_per_prompt),
        f"{METRIC_PREFIX}/checkpoint/completed_groups_reused": float(completed_groups),
        f"{METRIC_PREFIX}/checkpoint/partial_groups_continued": float(partial_groups),
        f"{METRIC_PREFIX}/checkpoint/partial_response_tokens_continued": float(partial_tokens),
        f"{METRIC_PREFIX}/checkpoint/groups_to_regenerate": float(regenerated_groups),
        f"{METRIC_PREFIX}/checkpoint/sample_conservation_fraction": _fraction(
            carried_groups,
            outstanding_groups,
        ),
        f"{METRIC_PREFIX}/checkpoint/generated_work_group_fraction": _fraction(
            preserved_work_groups,
            outstanding_groups,
        ),
        f"{METRIC_PREFIX}/checkpoint/pending_accounting_delta": float(
            pending_groups - outstanding_groups if replay_state is not None else 0
        ),
        f"{METRIC_PREFIX}/mode/no_replay": float(replay_type == "none"),
        f"{METRIC_PREFIX}/mode/rollout": float(replay_type == "rollout"),
        f"{METRIC_PREFIX}/mode/inflight": float(replay_type == "inflight"),
    }


def replay_load_metrics(
    *,
    replay_type: str | None,
    read_seconds: float,
    restore_seconds: float,
    total_seconds: float,
) -> dict[str, float]:
    """Return first-rollout metrics for state loading at process resume."""

    mode = "none" if replay_type is None else replay_type
    return {
        f"{METRIC_PREFIX}/load/read_seconds": read_seconds,
        f"{METRIC_PREFIX}/load/restore_seconds": restore_seconds,
        f"{METRIC_PREFIX}/load/total_seconds": total_seconds,
        f"{METRIC_PREFIX}/mode/no_replay": float(mode == "none"),
        f"{METRIC_PREFIX}/mode/rollout": float(mode == "rollout"),
        f"{METRIC_PREFIX}/mode/inflight": float(mode == "inflight"),
    }


def _completed_materialized_groups(replay_state: dict[str, Any]) -> int:
    ready_groups = len(replay_state["ready_items"])
    partial_drain_groups = sum(len(progress["data"]) for progress in replay_state["drain_progress"])
    prepared_groups = sum(
        len(prepared["group_ids"]) for prepared in replay_state["prepared_batches"]
    )
    return ready_groups + partial_drain_groups + prepared_groups


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0
