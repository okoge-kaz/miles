import pytest
from experiments.analyze_queue_lifecycle import distribution, queue_drop_prediction, summarize
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def make_record(attempt_id, disposition, lengths, *, rollout_id, first_prefill, ready, decision):
    return {
        "attempt_id": attempt_id,
        "disposition": disposition,
        "response_lengths": lengths,
        "rollout_id": rollout_id,
        "first_prefill_version": first_prefill,
        "ready_version": ready,
        "decision_version": decision,
    }


def test_distribution_matches_linear_percentiles():
    result = distribution([1, 2, 3, 4])

    assert result["mean"] == pytest.approx(2.5)
    assert result["std"] == pytest.approx(1.11803398875)
    assert result["p50"] == pytest.approx(2.5)
    assert result["p90"] == pytest.approx(3.7)


def test_queue_drop_summary_and_formula_use_group_batch_shape():
    trained = [make_record(i, "trained", [2, 6], rollout_id=0, first_prefill=1, ready=2, decision=3) for i in range(2)]
    evicted = make_record(2, "queue_evicted", [2, 2], rollout_id=None, first_prefill=1, ready=1, decision=2)
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
