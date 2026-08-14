import pytest
import torch
from experiments.src.offpolicy_acceleration.analyze_staleness_telemetry import (
    flatten_record,
    load_policy_diagnostics,
    normalize_queue_eviction_record,
    reconcile_attempt_rows,
    summarize,
)
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])


def _record(disposition: str, attempt: int, indices: list[int]) -> dict:
    size = len(indices)
    return {
        "schema_version": 2,
        "disposition": disposition,
        "reason_code": disposition,
        "group_index": 7,
        "sample_indices": indices,
        "recycle_count_before": attempt,
        "generation_attempt_id": f"7:{attempt}",
        "versions": {
            "reference": 1,
            "generation_completion": 2,
            "group_ready": 3,
            "queue_put": 3,
            "drain": 4,
        },
        "response_lengths": [10 + index for index in range(size)],
        "generation_duration_seconds": [1.0 + index for index in range(size)],
        "rewards": [float(index) for index in range(size)],
        "difficulty": [2.0 + index for index in range(size)],
        "pre_queue_active": [1] * size,
        "pre_queue_group_wait": [1] * size,
        "pre_queue_postprocess": [1] * size,
        "in_queue_staleness": [1] * size,
        "queue_wait_seconds": [0.5] * size,
    }


def _flatten(record: dict) -> list[dict]:
    return flatten_record(
        record,
        rollout_id=3,
        dump_root="run-a",
        source="run-a/rollout_data/3.pt",
    )


def test_reconcile_uses_final_consumed_rows_and_exposes_postprocess_trim() -> None:
    stale = _record("stale_recycle", 0, [10, 11])
    admitted = _record("admitted", 1, [10, 11])
    consumed = _record("consumed", 1, [10])
    consumed["training_step"] = 3
    consumed["loss_input_tokens"] = [8]

    rows = reconcile_attempt_rows(_flatten(stale) + _flatten(admitted) + _flatten(consumed))

    assert len(rows) == 4
    assert sum(row["accepted"] for row in rows) == 1
    assert sum(row["recycled"] for row in rows) == 2
    trimmed = [row for row in rows if row["disposition"] == "postprocess_trimmed"]
    assert len(trimmed) == 1
    assert trimmed[0]["loss_input_tokens"] == 0


def test_summary_compares_generated_consumed_recycled_distributions() -> None:
    stale = _record("stale_recycle", 0, [10, 11])
    admitted = _record("admitted", 1, [10, 11])
    consumed = _record("consumed", 1, [10, 11])
    consumed["training_step"] = 3
    consumed["loss_input_tokens"] = [8, 9]
    rows = reconcile_attempt_rows(_flatten(stale) + _flatten(admitted) + _flatten(consumed))

    summary = summarize(rows, num_bins=2, min_joint_cell=1)

    assert summary["counts"] == {
        "generated": 4,
        "consumed": 2,
        "recycled": 2,
        "dropped": 0,
    }
    assert summary["useful_sample_efficiency"] == 0.5
    assert summary["population_distributions"]["generated"]["response_length"]["count"] == 4
    assert summary["acceptance_by_feature"]["response_length"]
    assert summary["acceptance_by_length_reward_difficulty"]


def test_queue_eviction_uses_canonical_lifecycle_record_without_duplicate_length_schema() -> None:
    record = normalize_queue_eviction_record(
        {
            "attempt_id": 4,
            "group_index": 7,
            "sample_indices": [10, 11],
            "retry_count": 2,
            "submission_version": 1,
            "completion_version_min": 2,
            "completion_version_max": 3,
            "ready_version": 4,
            "queue_put_version": 4,
            "decision_version": 6,
            "enqueue_time_ns": 2_000_000_000,
            "decision_time_ns": 2_500_000_000,
            "response_lengths": [5, 9],
            "reward_values": [0.0, 1.0],
        },
        schema_version=1,
    )

    rows = reconcile_attempt_rows(_flatten(record))

    assert len(rows) == 2
    assert all(row["dropped"] and not row["accepted"] for row in rows)
    assert rows[0]["generation_attempt_id"] == "7:2"
    assert rows[0]["queue_wait_seconds"] == pytest.approx(0.5)
    assert rows[1]["response_length"] == 9


def test_policy_diagnostics_sum_cp_parts_and_ignore_tp_duplicates(tmp_path) -> None:
    dump_dir = tmp_path / "policy_loss_debug"
    dump_dir.mkdir()

    def write(rank: int, *, tp_rank: int, values: dict, optimizer_step_id: int = 0) -> None:
        sample = {
            "training_step": 3,
            "generation_attempt_id": "7:1",
            "sample_index": 10,
            "sample_staleness": 2,
            "optimizer_step_id": optimizer_step_id,
            **values,
        }
        torch.save(
            {
                "parallel": {"tp_rank": tp_rank},
                "samples": [sample],
            },
            dump_dir / f"rank_{rank}_call_0.pt",
        )

    cp_zero = {
        "response_token_count_local": 3.0,
        "pre_loss_token_count_local": 2.0,
        "final_loss_token_count_local": 1.0,
        "ppo_clip_count_local": 1.0,
        "importance_clip_count_local": 0.0,
        "sequence_policy_rollout_log_ratio_local": 0.3,
        "final_pg_loss": torch.tensor([2.0]),
        "final_local_loss_mask": torch.tensor([1.0]),
    }
    write(0, tp_rank=0, values=cp_zero)
    write(1, tp_rank=1, values=cp_zero)
    write(
        2,
        tp_rank=0,
        values={
            "response_token_count_local": 1.0,
            "pre_loss_token_count_local": 1.0,
            "final_loss_token_count_local": 1.0,
            "ppo_clip_count_local": 0.0,
            "importance_clip_count_local": 1.0,
            "sequence_policy_rollout_log_ratio_local": 0.2,
            "final_pg_loss": torch.tensor([3.0]),
            "final_local_loss_mask": torch.tensor([1.0]),
        },
    )

    diagnostics = load_policy_diagnostics([tmp_path])
    [row] = diagnostics.values()

    assert row["ppo_clip_fraction"] == pytest.approx(1 / 3)
    assert row["importance_clip_fraction"] == pytest.approx(1 / 3)
    assert row["mask_fraction"] == pytest.approx(0.5)
    assert row["sequence_policy_rollout_log_ratio"] == pytest.approx(0.5)
    assert row["absolute_pg_contribution"] == pytest.approx(5.0)
    assert row["optimizer_updates_observed"] == 1.0


def test_policy_diagnostics_aggregate_optimizer_steps_without_mixing_cp_parts(tmp_path) -> None:
    dump_dir = tmp_path / "policy_loss_debug"
    dump_dir.mkdir()

    def write(call: int, optimizer_step_id: int, values: dict) -> None:
        torch.save(
            {
                "parallel": {"tp_rank": 0, "cp_rank": call},
                "samples": [
                    {
                        "training_step": 3,
                        "generation_attempt_id": "7:1",
                        "sample_index": 10,
                        "sample_staleness": 2,
                        "optimizer_step_id": optimizer_step_id,
                        **values,
                    }
                ],
            },
            dump_dir / f"rank_{call}_call_0.pt",
        )

    write(
        0,
        4,
        {
            "response_token_count_local": 2.0,
            "pre_loss_token_count_local": 2.0,
            "final_loss_token_count_local": 1.0,
            "ppo_clip_count_local": 1.0,
            "importance_clip_count_local": 0.0,
            "sequence_policy_rollout_log_ratio_local": 0.3,
            "absolute_pg_contribution_local": 2.0,
        },
    )
    write(
        1,
        4,
        {
            "response_token_count_local": 1.0,
            "pre_loss_token_count_local": 1.0,
            "final_loss_token_count_local": 1.0,
            "ppo_clip_count_local": 0.0,
            "importance_clip_count_local": 1.0,
            "sequence_policy_rollout_log_ratio_local": 0.2,
            "absolute_pg_contribution_local": 3.0,
        },
    )
    write(
        2,
        5,
        {
            "response_token_count_local": 2.0,
            "pre_loss_token_count_local": 2.0,
            "final_loss_token_count_local": 2.0,
            "ppo_clip_count_local": 0.0,
            "importance_clip_count_local": 0.0,
            "sequence_policy_rollout_log_ratio_local": 0.4,
            "absolute_pg_contribution_local": 7.0,
        },
    )

    diagnostics = load_policy_diagnostics([tmp_path])
    [row] = diagnostics.values()

    assert row["ppo_clip_fraction"] == pytest.approx(1 / 5)
    assert row["importance_clip_fraction"] == pytest.approx(1 / 5)
    assert row["mask_fraction"] == pytest.approx(1 / 5)
    assert row["sequence_policy_rollout_log_ratio"] == pytest.approx(0.45)
    assert row["absolute_pg_contribution"] == pytest.approx(12.0)
    assert row["optimizer_updates_observed"] == 2.0
