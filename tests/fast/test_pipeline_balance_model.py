import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "experiments/tools/pipeline_balance_model.py"
SPEC = importlib.util.spec_from_file_location("pipeline_balance_model_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


def test_rollout_limited_queue_recycle_predicts_two_version_lower_envelope():
    prediction = MODEL.predict(
        MODEL.ModelInputs(
            train_compute_seconds=120,
            rollout_groups_per_second=0.9,
            batch_groups=192,
            concurrency_groups=192,
            queue_policy="queue-recycle",
            queue_capacity_groups=1000,
            max_weight_staleness=8,
        )
    )

    assert prediction.rho == pytest.approx(0.5625)
    assert prediction.pre_queue_staleness == pytest.approx(1.0)
    assert prediction.in_queue_staleness == 1.0
    assert prediction.steady_staleness_or_cap == pytest.approx(2.0)
    assert prediction.stationary_interval_low == pytest.approx(2.0)
    assert prediction.stationary_interval_high == pytest.approx(3.0)
    assert prediction.regime == "rollout-limited stationary"


def test_overproduced_unbounded_fifo_has_no_stationary_state():
    prediction = MODEL.predict(
        MODEL.ModelInputs(
            train_compute_seconds=240,
            rollout_groups_per_second=1.0,
            batch_groups=192,
            concurrency_groups=192,
            queue_policy="queue-recycle",
        )
    )

    assert prediction.rho == pytest.approx(1.25)
    assert prediction.pre_queue_staleness == pytest.approx(0.8)
    assert prediction.linear_growth_per_training_step == pytest.approx(0.2)
    assert prediction.steady_staleness_or_cap is None
    assert prediction.regime == "unbounded linear growth"


def test_fifo_growth_stops_at_the_smaller_queue_or_staleness_cap():
    prediction = MODEL.predict(
        MODEL.ModelInputs(
            train_compute_seconds=240,
            rollout_groups_per_second=1.0,
            batch_groups=192,
            concurrency_groups=192,
            queue_policy="queue-recycle",
            queue_capacity_groups=1000,
            max_weight_staleness=8,
        )
    )

    assert prediction.queue_capacity_staleness_cap == pytest.approx(0.8 + 1 + 1000 / 192)
    assert prediction.effective_staleness_cap == prediction.queue_capacity_staleness_cap
    assert prediction.steady_staleness_or_cap == prediction.queue_capacity_staleness_cap
    assert prediction.steps_to_cap == pytest.approx(
        (prediction.queue_capacity_staleness_cap - 1.8) / 0.2
    )
    assert MODEL.predicted_trajectory(prediction, 100)[-1]["predicted_training_staleness"] == pytest.approx(
        prediction.queue_capacity_staleness_cap
    )


def test_queue_drop_uses_its_stationary_closed_form():
    prediction = MODEL.predict(
        MODEL.ModelInputs(
            train_compute_seconds=2,
            rollout_groups_per_second=2,
            batch_groups=2,
            concurrency_groups=2.8,
            queue_policy="queue-drop",
            queue_capacity_groups=2,
        )
    )

    assert prediction.rho == 2
    assert prediction.pre_queue_staleness == pytest.approx(0.7)
    assert prediction.in_queue_staleness == pytest.approx(0.75)
    assert prediction.steady_staleness_or_cap == pytest.approx(1.45)
    assert prediction.prediction_kind == "closed-form mean"


def test_inverse_node_fit_recovers_parallel_work_and_fixed_floor():
    fit = MODEL.linear_fit([(1.0, 230.0), (0.5, 120.0), (0.25, 65.0)])

    assert fit.slope == pytest.approx(220.0)
    assert fit.intercept == pytest.approx(10.0)
    assert fit.r_squared == 1.0


def test_history_figures_are_dependency_free_valid_svg(tmp_path):
    fits = {
        "fit_staleness": 8,
        "training_fit": {"observations": 2, "slope": 100.0, "intercept": 10.0, "r_squared": 1.0},
        "rollout_fit": {"observations": 2, "slope": 1.0, "intercept": 1.0, "r_squared": 1.0},
    }
    points = [
        {
            "max_weight_staleness": 8,
            "trainer_nodes": 1,
            "rollout_nodes": 2,
            "train_compute_seconds": 110.0,
            "rollout_groups_per_second": 2 / 3,
            "rollout_group_rate_source": "direct generated-group counter",
            "rollout_rate_capacity_censored": 0,
            "predicted_actual_updates_per_second": 1 / 110,
            "observed_training_staleness": 2.1,
        },
        {
            "max_weight_staleness": 8,
            "trainer_nodes": 2,
            "rollout_nodes": 1,
            "train_compute_seconds": 60.0,
            "rollout_groups_per_second": 0.5,
            "rollout_group_rate_source": "direct generated-group counter",
            "rollout_rate_capacity_censored": 0,
            "predicted_actual_updates_per_second": 1 / 120,
            "observed_training_staleness": 2.0,
        },
    ]
    candidates = MODEL.ratio_candidates(
        fits,
        total_nodes=3,
        trainer_nodes=[1, 2],
        batch_groups=60,
        concurrency_groups=60,
        queue_policy="queue-recycle",
        queue_capacity_groups=300,
        max_weight_staleness=8,
    )

    MODEL._write_history_figures(
        tmp_path,
        points=points,
        fits=fits,
        candidates=candidates,
        trajectory_steps=10,
    )

    figures = sorted((tmp_path / "figures").glob("*.svg"))
    assert len(figures) == 5
    for figure in figures:
        ET.parse(figure)
