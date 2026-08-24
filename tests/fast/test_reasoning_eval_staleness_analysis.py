import importlib.util
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = _load_module(
    "reasoning_eval_analyze_staleness_test_module",
    "experiments/tools/reasoning_eval/analyze_staleness.py",
)
EXPORT = _load_module(
    "reasoning_eval_export_wandb_test_module",
    "experiments/tools/reasoning_eval/export_wandb_history.py",
)
PLOT = _load_module(
    "reasoning_eval_plot_staleness_test_module",
    "experiments/tools/reasoning_eval/plot_staleness_analysis.py",
)


def test_latest_wandb_segment_replaces_replayed_metrics_and_rebuilds_clock():
    old = EXPORT.RunHistory(
        arm="s1-t1r7",
        run_id="old",
        created_at="2026-08-01T00:00:00Z",
        rollout={0: {"_timestamp": 100.0, "perf/step_time": 10.0, "staleness/total/mean": 1.0}},
        train={0: {"_timestamp": 101.0, "train/loss": 0.5}},
    )
    resumed = EXPORT.RunHistory(
        arm="s1-t1r7",
        run_id="new",
        created_at="2026-08-02T00:00:00Z",
        rollout={
            0: {"_timestamp": 200.0, "perf/step_time": 20.0, "staleness/total/mean": 2.0},
            1: {"_timestamp": 230.0, "perf/step_time": 30.0, "staleness/total/mean": 3.0},
        },
        train={
            0: {"_timestamp": 201.0, "train/loss": 0.4},
            1: {"_timestamp": 231.0, "train/loss": 0.3},
        },
    )

    lineage, replacements = EXPORT._merge_lineage([resumed, old])
    rows = EXPORT._history_rows(lineage)

    assert replacements > 0
    assert [row["training_step"] for row in rows] == [1, 2]
    assert rows[0]["staleness/total/mean"] == 2.0
    assert rows[0]["active_wallclock_seconds"] == 20.0
    assert rows[1]["active_wallclock_seconds"] == 50.0
    assert rows[1]["active_wallclock_coverage"] == 1.0
    assert rows[0]["calendar_elapsed_seconds"] == 1.0


def test_score_interval_separates_update_effect_and_throughput():
    base = {
        "arm": "s1-t1r7",
        "completed_tasks": "3",
        "max_weight_staleness": "1",
        "aime24_percent": "50",
        "aime25_percent": "40",
        "aime26_percent": "30",
        "aime_macro_mean_percent": "40",
    }
    aggregates = [
        {**base, "training_step": "10"},
        {
            **base,
            "training_step": "20",
            "aime24_percent": "56",
            "aime25_percent": "43",
            "aime26_percent": "30",
            "aime_macro_mean_percent": "43",
        },
    ]
    history = {
        step: {
            "active_wallclock_seconds": str(100.0 if step == 10 else 3700.0 if step == 20 else 0.0),
            "staleness/total/mean": "2.0",
        }
        for step in range(10, 21)
    }

    intervals = ANALYZE._interval_records(aggregates, {"s1-t1r7": history})

    assert len(intervals) == 1
    assert intervals[0]["delta_macro"] == pytest.approx(3.0)
    assert intervals[0]["active_interval_hours"] == pytest.approx(1.0)
    assert intervals[0]["updates_per_active_hour"] == pytest.approx(10.0)
    assert intervals[0]["macro_points_per_active_hour"] == pytest.approx(3.0)
    assert intervals[0]["staleness/total/mean"] == pytest.approx(2.0)

    decomposition = ANALYZE._wallclock_decomposition(intervals)
    assert len(decomposition) == 1
    assert decomposition[0]["macro_points_per_update"] == pytest.approx(0.3)
    assert decomposition[0]["updates_per_active_hour"] == pytest.approx(10.0)
    assert decomposition[0]["macro_points_per_active_hour"] == pytest.approx(3.0)
    assert decomposition[0]["macro_points_per_update"] * decomposition[0]["updates_per_active_hour"] == pytest.approx(
        decomposition[0]["macro_points_per_active_hour"]
    )


def test_fixed_effect_correlation_centers_same_step_and_ratio():
    records = [
        {
            "arm": f"s{staleness}-t1r7",
            "ratio": "t1r7",
            "end_step": 20,
            "predictor": float(staleness),
            "outcome": float(staleness * 2),
        }
        for staleness in (1, 2, 4, 8)
    ]

    result = ANALYZE._fixed_effect_correlation(
        records,
        predictor="predictor",
        outcome="outcome",
        group_keys=("end_step", "ratio"),
        bootstrap_samples=0,
        seed=0,
    )

    assert result.observations == 4
    assert result.correlation == pytest.approx(1.0)
    assert result.slope == pytest.approx(2.0)


def test_wallclock_decomposition_uses_only_intervals_shared_by_all_arms():
    def interval(arm, start_step, end_step, delta):
        return {
            "arm": arm,
            "ratio": arm.split("-", 1)[1],
            "max_weight_staleness": arm[1],
            "start_step": start_step,
            "end_step": end_step,
            "active_interval_hours": 1.0,
            "delta_aime24": delta,
            "delta_aime25": delta,
            "delta_aime26": delta,
            "delta_macro": delta,
        }

    intervals = [
        interval("s1-t1r7", 10, 20, 2.0),
        interval("s1-t1r7", 20, 30, 100.0),
        interval("s2-t1r7", 10, 20, 1.0),
    ]

    rows = ANALYZE._wallclock_decomposition(intervals)

    assert len(rows) == 2
    assert {row["interval_count"] for row in rows} == {1}
    assert {row["common_start_step"] for row in rows} == {10}
    assert {row["common_end_step"] for row in rows} == {20}
    s1 = next(row for row in rows if row["arm"] == "s1-t1r7")
    assert s1["macro_points_per_update"] == pytest.approx(0.2)


def test_score_panel_svg_is_well_formed():
    rows = [
        PLOT.SeriesRow(
            arm="s1-t1r7",
            training_step=10,
            active_wallclock_hours=1.5,
            scores={"AIME24": 50.0, "AIME25": 40.0, "AIME26": 30.0, "Macro": 40.0},
        ),
        PLOT.SeriesRow(
            arm="s1-t1r7",
            training_step=20,
            active_wallclock_hours=3.0,
            scores={"AIME24": 55.0, "AIME25": 43.0, "AIME26": 32.0, "Macro": 43.3},
        ),
    ]

    svg = PLOT._render_arm_score_panels(
        rows,
        arms=("s1-t1r7",),
        x_attribute="training_step",
        x_label="Training step",
        title="Test",
        subtitle="Test subtitle",
    )

    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "s1-t1r7" in svg
    assert "Macro" in svg


def test_selected_arms_preserve_sweep_order_and_deduplicate():
    groups = [
        {"low_arms": ["s2-t2r6", "s1-t1r7"], "high_arms": ["s8-t4r4"]},
        {"low_arms": ["s1-t1r7"], "high_arms": ["s4-t3r5"]},
    ]

    assert PLOT._selected_arms(groups) == (
        "s1-t1r7",
        "s2-t2r6",
        "s4-t3r5",
        "s8-t4r4",
    )


def test_wallclock_decomposition_svg_is_well_formed():
    rows = [
        PLOT.DecompositionRow(
            arm="s1-t1r7",
            trainer_nodes=1,
            macro_points_per_update=0.2,
            updates_per_active_hour=10.0,
            macro_points_per_active_hour=2.0,
        ),
        PLOT.DecompositionRow(
            arm="s2-t2r6",
            trainer_nodes=2,
            macro_points_per_update=-0.1,
            updates_per_active_hour=12.0,
            macro_points_per_active_hour=-1.2,
        ),
    ]

    svg = PLOT._render_wallclock_decomposition(rows)

    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "dQ/dt" in svg
    assert "s1-t1r7" in svg
