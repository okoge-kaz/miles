import pytest
from experiments.analyze_queue_lifecycle import distribution, queue_drop_prediction, summarize
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def make_record(attempt_id, disposition, lengths, *, rollout_id, first_prefill, ready, decision, rewards=None):
    record = {
        "attempt_id": attempt_id,
        "disposition": disposition,
        "response_lengths": lengths,
        "rollout_id": rollout_id,
        "first_prefill_version": first_prefill,
        "ready_version": ready,
        "decision_version": decision,
    }
    if rewards is not None:
        record["reward_values"] = rewards
    return record


def test_distribution_matches_linear_percentiles():
    result = distribution([1, 2, 3, 4])

    assert result["mean"] == pytest.approx(2.5)
    assert result["std"] == pytest.approx(1.11803398875)
    assert result["p50"] == pytest.approx(2.5)
    assert result["p90"] == pytest.approx(3.7)


def test_queue_drop_summary_and_formula_use_group_batch_shape():
    trained = [
        make_record(
            i,
            "trained",
            [2, 6],
            rollout_id=0,
            first_prefill=1,
            ready=2,
            decision=3,
            rewards=[1, 0],
        )
        for i in range(2)
    ]
    evicted = make_record(
        2,
        "queue_evicted",
        [2, 2],
        rollout_id=None,
        first_prefill=1,
        ready=1,
        decision=2,
        rewards=[0, 0],
    )
    lifecycle = {
        "schema_version": 1,
        "policy": "queue-drop",
        "capacity_groups": 2,
        "records": [*trained, evicted],
    }

    result = summarize([lifecycle], rho=2.0, concurrency_samples=4)

    assert result["inferred_shape"] == {
        "groups_per_batch": 2,
        "samples_per_group": 2,
        "batch_samples": 4,
        "queue_factor": 1.0,
    }
    # Completed sample mean = 20 / 6, completed group-max mean = 14 / 3.
    assert result["length_selection"]["group_tailness_multiplier"] == pytest.approx(1.4)
    assert result["trained"]["reward"]["sample_reward"]["mean"] == pytest.approx(0.5)
    assert result["terminal_completed"]["reward"]["sample_reward"]["mean"] == pytest.approx(1 / 3)
    assert result["reward_selection"]["trained_minus_terminal_completed_sample_mean"] == pytest.approx(1 / 6)
    assert result["dispositions"]["queue_evicted"]["reward"]["all_zero_group_frac"] == 1.0
    assert result["dispositions"]["trained"]["reward"]["sample_length_reward_pearson"] == -1.0
    assert result["trained_selection_staleness"]["total_selection_minus_first_prefill"]["mean"] == 2
    assert result["queue_drop_formula"]["predicted_staleness"] == pytest.approx(
        {"pre_queue": 0.7, "in_queue": 0.75, "total": 1.45}
    )


def test_queue_drop_prediction_rollout_bound_branch():
    assert queue_drop_prediction(
        rho=0.5,
        concurrency_samples=4,
        batch_samples=8,
        queue_factor=1,
        tailness=2,
    ) == pytest.approx({"pre_queue": 1.0, "in_queue": 0.5, "total": 1.5})


def test_queue_drop_prediction_rejects_equal_throughput_boundary():
    with pytest.raises(ValueError, match="boundary"):
        queue_drop_prediction(
            rho=1.0,
            concurrency_samples=64,
            batch_samples=64,
            queue_factor=2,
            tailness=1,
        )


def test_reward_summary_accepts_older_records_without_reward_values():
    record = make_record(0, "trained", [2, 4], rollout_id=0, first_prefill=1, ready=1, decision=1)
    lifecycle = {
        "schema_version": 1,
        "policy": "queue-recycle",
        "capacity_groups": 1000,
        "records": [record],
    }

    result = summarize([lifecycle], rho=None, concurrency_samples=None)

    reward = result["trained"]["reward"]
    assert reward["records_with_reward_values"] == 0
    assert reward["samples_missing_reward"] == 2
    assert reward["sample_reward"] == {"count": 0, "sum": 0.0}
    assert result["reward_selection"]["trained_minus_terminal_completed_sample_mean"] is None


def test_reward_summary_rejects_unaligned_values():
    record = make_record(
        0,
        "trained",
        [2, 4],
        rollout_id=0,
        first_prefill=1,
        ready=1,
        decision=1,
        rewards=[1],
    )
    lifecycle = {
        "schema_version": 1,
        "policy": "queue-recycle",
        "capacity_groups": 1000,
        "records": [record],
    }

    with pytest.raises(ValueError, match="1 rewards do not align with 2 response lengths"):
        summarize([lifecycle], rho=None, concurrency_samples=None)
