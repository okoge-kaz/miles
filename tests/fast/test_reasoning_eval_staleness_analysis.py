import importlib.util
import math
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

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
DISPLACEMENT = _load_module(
    "reasoning_eval_checkpoint_displacement_test_module",
    "experiments/tools/reasoning_eval/compute_checkpoint_displacements.py",
)
UNPAD = _load_module(
    "reasoning_eval_unpad_vocab_test_module",
    "experiments/tools/reasoning_eval/unpad_vocab.py",
)
EVALUATION_SCRIPT = REPO_ROOT / "experiments/scripts/reasoning_eval/run-evaluation.sbatch"
EVALUATION_LAUNCHER = REPO_ROOT / "experiments/scripts/reasoning_eval/submit-staleness-sweep.sh"
EVALUATION_REFILL = REPO_ROOT / "experiments/scripts/reasoning_eval/refill-snapshot.sbatch"
VLLM_RUNTIME_HOOK = REPO_ROOT / "experiments/tools/reasoning_eval/vllm_runtime_hooks/sitecustomize.py"
PLOT = _load_module(
    "reasoning_eval_plot_staleness_test_module",
    "experiments/tools/reasoning_eval/plot_staleness_analysis.py",
)
TRAINING_STALENESS = _load_module(
    "reasoning_eval_plot_training_staleness_test_module",
    "experiments/tools/reasoning_eval/plot_training_staleness.py",
)
VALIDATE = _load_module(
    "reasoning_eval_validate_checkpoint_test_module",
    "experiments/tools/reasoning_eval/validate_checkpoint.py",
)
GRID = _load_module(
    "reasoning_eval_grid_test_module",
    "experiments/tools/reasoning_eval/grid.py",
)


def test_reasoning_eval_grid_accepts_high_staleness_single_ratio_without_colocated():
    grid = GRID.reasoning_eval_grid_from_environment(
        {
            "STALENESS_LEVELS": "16 20 24 28",
            "RATIOS": "1:7",
            "INCLUDE_COLOCATED": "0",
        }
    )

    assert grid.staleness_levels == (16, 20, 24, 28)
    assert grid.node_ratios == ((1, 7),)
    assert grid.all_arms == (
        "s16-t1r7",
        "s20-t1r7",
        "s24-t1r7",
        "s28-t1r7",
    )


def test_reasoning_eval_plotter_loads_the_high_staleness_grid(monkeypatch):
    monkeypatch.setenv("STALENESS_LEVELS", "16 20 24 28")
    monkeypatch.setenv("RATIOS", "1:7")
    monkeypatch.setenv("INCLUDE_COLOCATED", "0")

    high_staleness_plot = _load_module(
        "reasoning_eval_high_staleness_plot_test_module",
        "experiments/tools/reasoning_eval/plot_results.py",
    )

    assert high_staleness_plot.STALENESS_LEVELS == (16, 20, 24, 28)
    assert high_staleness_plot.NODE_RATIOS == ((1, 7),)
    assert high_staleness_plot.ALL_ARMS == (
        "s16-t1r7",
        "s20-t1r7",
        "s24-t1r7",
        "s28-t1r7",
    )


def test_latest_wandb_segment_replaces_replayed_metrics_and_rebuilds_clock():
    old = EXPORT.RunHistory(
        arm="s1-t1r7",
        run_id="old",
        created_at="2026-08-01T00:00:00Z",
        rollout={
            0: {
                "_timestamp": 100.0,
                "perf/step_time": 10.0,
                "perf/train_time": 6.0,
                "perf/train_wait_time": 4.0,
                "staleness/total/mean": 1.0,
            }
        },
        train={0: {"_timestamp": 101.0, "train/loss": 0.5}},
    )
    resumed = EXPORT.RunHistory(
        arm="s1-t1r7",
        run_id="new",
        created_at="2026-08-02T00:00:00Z",
        rollout={
            0: {
                "_timestamp": 200.0,
                "perf/step_time": 20.0,
                "perf/train_time": 6.0,
                "perf/train_wait_time": 14.0,
                "staleness/total/mean": 2.0,
            },
            1: {
                "_timestamp": 230.0,
                "perf/step_time": 10.0,
                "perf/train_time": 6.0,
                "perf/train_wait_time": 4.0,
                "staleness/total/mean": 3.0,
            },
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
    assert rows[0]["resume_boundary"] == 1
    assert rows[0]["observed_active_wallclock_seconds"] == 20.0
    assert rows[0]["estimated_uninterrupted_wallclock_seconds"] == 10.0
    assert rows[1]["observed_active_wallclock_seconds"] == 30.0
    assert rows[1]["estimated_uninterrupted_wallclock_seconds"] == 20.0
    assert rows[1]["resume_overhead_removed_seconds"] == 10.0
    assert rows[1]["active_wallclock_coverage"] == 1.0
    assert rows[0]["calendar_elapsed_seconds"] == 1.0


def test_switch_only_wandb_row_does_not_replace_step_timestamp_anchor():
    target = {}
    EXPORT._merge_row_metrics(
        target,
        row={
            "rollout/step": 3,
            "_timestamp": 100.0,
            "perf/step_time": 10.0,
        },
        step_key=EXPORT.ROLLOUT_STEP,
        metric_keys=EXPORT.ROLLOUT_METRICS,
        timestamp_anchor_keys=("perf/step_time",),
    )
    EXPORT._merge_row_metrics(
        target,
        row={
            "rollout/step": 3,
            "_timestamp": 120.0,
            "perf/colocate/train_to_rollout_block_time": 4.0,
        },
        step_key=EXPORT.ROLLOUT_STEP,
        metric_keys=EXPORT.ROLLOUT_METRICS,
        timestamp_anchor_keys=("perf/step_time",),
    )

    assert target[3]["_timestamp"] == 100.0
    assert target[3]["perf/colocate/train_to_rollout_block_time"] == 4.0


def test_exporter_normalizes_async_concurrency_tag_but_rejects_partial_colocated():
    namespace = "partial-c4096-s4-test"
    legacy = SimpleNamespace(
        config={"wandb_group": f"s4-t1r7-{namespace}"},
        group="",
        name="",
        id="legacy",
    )
    concurrency_tagged = SimpleNamespace(
        config={"wandb_group": f"s4-t1r7-c4096-{namespace}"},
        group="",
        name="",
        id="concurrency-tagged",
    )
    partial_colocated = SimpleNamespace(
        config={"wandb_group": f"s0-colocated-partial-o256-{namespace}"},
        group="",
        name="",
        id="partial-colocated",
    )
    high_staleness = SimpleNamespace(
        config={"wandb_group": f"s28-t1r7-{namespace}"},
        group="",
        name="",
        id="high-staleness",
    )

    assert EXPORT._arm_from_run(legacy, namespace=namespace) == "s4-t1r7"
    assert EXPORT._arm_from_run(concurrency_tagged, namespace=namespace) == "s4-t1r7"
    assert EXPORT._arm_from_run(high_staleness, namespace=namespace) == "s28-t1r7"
    with pytest.raises(ValueError, match="cannot identify sweep arm"):
        EXPORT._arm_from_run(partial_colocated, namespace=namespace)


def test_evaluation_allows_slow_vllm_startup_without_a_shared_runtime_cache():
    script = EVALUATION_SCRIPT.read_text(encoding="utf-8")

    assert "${EVALUATION_CACHE_ROOT}:/evaluation-cache" in script
    assert "--data-parallel-size 8" in script
    assert 'VLLM_ENGINE_READY_TIMEOUT_SECONDS="${VLLM_ENGINE_READY_TIMEOUT_SECONDS:-2400}"' in script
    assert 'SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-540}"' in script
    assert "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_SECONDS}" in script
    assert "VLLM_CACHE_ROOT=" not in script
    assert "${VLLM_RUNTIME_HOOKS}:/vllm-runtime-hooks:ro" in script
    assert "PYTHONPATH=/vllm-runtime-hooks" in script
    assert "VLLM_RUNTIME_CACHE_ROOT" not in script
    assert "VLLM_PREWARM_LOG" not in script
    assert 'MODEL_MOUNTS="${MODEL_RUNTIME_DIR}:/checkpoint:ro"' in script
    assert 'rm -rf -- "${MODEL_RUNTIME_DIR}" "${TOKENIZER_RUNTIME_DIR}"' in script
    assert 'stage_node_local_image "${VLLM_IMAGE}" "${VLLM_RUNTIME_IMAGE}"' in script
    assert '--container-image="${VLLM_RUNTIME_IMAGE}"' in script


def test_evaluation_retries_transient_evaluator_failures_from_partial_outputs():
    script = EVALUATION_SCRIPT.read_text(encoding="utf-8")

    assert 'EVALUATOR_ATTEMPTS="${EVALUATOR_ATTEMPTS:-3}"' in script
    assert "evaluator_attempt <= EVALUATOR_ATTEMPTS" in script
    assert "attempt_output_records == EXPECTED_OUTPUT_RECORDS" in script
    assert "Evaluator attempt produced incomplete outputs" in script
    assert "retrying from cached partial outputs" in script
    assert '[[ "${evaluator_succeeded}" == true ]]' in script


def test_unpad_runtime_checkpoint_materializes_unchanged_shards(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "runtime"
    source.mkdir()
    (source / "rewritten.safetensors").write_bytes(b"rewrite me")
    (source / "unchanged.safetensors").write_bytes(b"node-local weights")
    (source / "config.json").write_text('{"model_type": "qwen3"}\n', encoding="utf-8")

    UNPAD._copy_checkpoint_files(
        source=source,
        destination=destination,
        rewritten_shard="rewritten.safetensors",
    )

    materialized = destination / "unchanged.safetensors"
    assert materialized.read_bytes() == b"node-local weights"
    assert materialized.is_file()
    assert not materialized.is_symlink()
    assert not (destination / "rewritten.safetensors").exists()
    assert (destination / "config.json").read_text(encoding="utf-8") == '{"model_type": "qwen3"}\n'


def test_vllm_runtime_hook_stabilizes_only_the_timeout_compile_factor():
    source = VLLM_RUNTIME_HOOK.read_text(encoding="utf-8")

    assert 'factors["VLLM_ENGINE_READY_TIMEOUT_S"] = _DEFAULT_ENGINE_READY_TIMEOUT_SECONDS' in source
    assert "_DEFAULT_ENGINE_READY_TIMEOUT_SECONDS = 600" in source


def test_evaluation_queue_queries_tracked_job_ids_without_user_lookup():
    launcher = EVALUATION_LAUNCHER.read_text(encoding="utf-8")
    controller = EVALUATION_REFILL.read_text(encoding="utf-8")

    assert 'squeue --noheader --jobs="${tracked_job_ids}"' in launcher
    assert "active job ids:" in launcher
    assert 'squeue --noheader --jobs="${job_ids}"' in controller
    assert "--user=" not in launcher
    assert "--user=" not in controller


def test_checkpoint_displacement_measures_stored_parameter_change(tmp_path):
    start = tmp_path / "start"
    end = tmp_path / "end"
    start.mkdir()
    end.mkdir()
    save_file({"weight": torch.tensor([1.0, 2.0, 3.0])}, start / "model.safetensors")
    save_file({"weight": torch.tensor([2.0, 1.0, 3.0])}, end / "model.safetensors")
    interval = DISPLACEMENT.CheckpointInterval(
        arm="s1-t1r7",
        study_identity=str(tmp_path),
        namespace="test",
        start_step=10,
        end_step=20,
        start_checkpoint=start,
        end_checkpoint=end,
    )

    result = DISPLACEMENT._measure_interval(interval, chunk_elements=2)

    assert result.parameter_count == 3
    assert result.start_parameter_norm == pytest.approx(math.sqrt(14.0))
    assert result.net_parameter_displacement_norm == pytest.approx(math.sqrt(2.0))
    assert result.net_parameter_displacement_per_update == pytest.approx(math.sqrt(2.0) / 10.0)
    assert result.relative_net_parameter_displacement == pytest.approx(math.sqrt(2.0 / 14.0))


def test_checkpoint_displacement_skips_unevaluated_placeholders(tmp_path):
    aggregates = tmp_path / "aggregate-results.csv"
    aggregates.write_text(
        "arm,training_step,checkpoint_directory,aime_macro_mean_percent\n"
        "s1-t1r7,10,9,40.0\n"
        "s1-t1r7,20,19,42.0\n"
        "s1-t1r7,30,29,\n",
        encoding="utf-8",
    )

    intervals = DISPLACEMENT._read_intervals(
        aggregates,
        study_root=tmp_path,
        study_identity=str(tmp_path),
        namespace="test",
        step_interval=10,
        selected_arm=None,
    )

    assert [(interval.start_step, interval.end_step) for interval in intervals] == [(10, 20)]


def test_checkpoint_displacement_resolves_high_staleness_training_identity(tmp_path):
    checkpoint_root = DISPLACEMENT._checkpoint_root(
        tmp_path,
        arm="s28-t1r7",
        namespace="high-staleness",
        async_max_concurrent_samples=4096,
        training_buffer_queue_size=6000,
    )

    assert checkpoint_root == (
        tmp_path
        / "async/off-policy/max-weight-staleness-28-from-prefill"
        / "s28-t1r7-high-staleness-zero-trunc-rb-inflight-concurrency-4096-tbq6000/hf"
    )


def test_checkpoint_displacement_reuses_only_matching_snapshot_rows(tmp_path):
    def interval(start_step, end_step):
        return DISPLACEMENT.CheckpointInterval(
            arm="s1-t1r7",
            study_identity=str(tmp_path),
            namespace="test",
            start_step=start_step,
            end_step=end_step,
            start_checkpoint=tmp_path / str(start_step),
            end_checkpoint=tmp_path / str(end_step),
        )

    matching = {
        "arm": "s1-t1r7",
        "study_identity": str(tmp_path),
        "namespace": "test",
        "start_step": "10",
        "end_step": "20",
        "optimizer_steps": "10",
        "parameter_count": "1",
        "start_parameter_norm": "1.0",
        "net_parameter_displacement_norm": "0.1",
        "net_parameter_displacement_per_update": "0.01",
        "relative_net_parameter_displacement": "0.1",
        "elapsed_seconds": "1.0",
        "start_checkpoint": str(tmp_path / "10"),
        "end_checkpoint": str(tmp_path / "20"),
    }
    duplicate = {
        **matching,
        "net_parameter_displacement_norm": "2.0",
        "net_parameter_displacement_per_update": "0.2",
        "relative_net_parameter_displacement": "2.0",
    }
    stale = {**matching, "start_step": "20", "end_step": "30"}
    wrong_namespace = {**matching, "namespace": "other"}
    incomplete = {key: value for key, value in matching.items() if key != "parameter_count"}

    rows = DISPLACEMENT._matching_completed(
        [matching, stale, wrong_namespace, incomplete, duplicate],
        [interval(10, 20)],
    )

    assert rows == [duplicate]


def test_checkpoint_displacement_merge_rejects_missing_interval(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    interval = DISPLACEMENT.CheckpointInterval(
        arm="s1-t1r7",
        study_identity=str(tmp_path),
        namespace="test",
        start_step=10,
        end_step=20,
        start_checkpoint=tmp_path / "10",
        end_checkpoint=tmp_path / "20",
    )

    with pytest.raises(ValueError, match="missing 1 interval"):
        DISPLACEMENT._merge_completed_parts(parts, [interval])


def test_checkpoint_displacement_joins_matching_training_interval():
    intervals = [{"arm": "s1-t1r7", "start_step": 10, "end_step": 20}]
    displacements = [
        {
            "arm": "s1-t1r7",
            "start_step": "10",
            "end_step": "20",
            "net_parameter_displacement_per_update": "0.01",
            "relative_net_parameter_displacement": "0.001",
        }
    ]

    joined = ANALYZE._join_checkpoint_displacements(intervals, displacements)

    assert joined[0]["checkpoint/net_parameter_displacement_per_update"] == "0.01"
    assert joined[0]["checkpoint/relative_net_parameter_displacement"] == "0.001"


def test_checkpoint_validator_distinguishes_permission_errors(monkeypatch, capsys, tmp_path):
    def deny_access(_path):
        raise PermissionError(13, "Permission denied", "model.safetensors")

    monkeypatch.setattr(VALIDATE, "validate_checkpoint", deny_access)
    monkeypatch.setattr(sys, "argv", ["validate_checkpoint.py", str(tmp_path)])

    with pytest.raises(SystemExit) as error:
        VALIDATE.main()

    assert error.value.code == 2
    assert "unreadable checkpoint" in capsys.readouterr().err


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
            "estimated_uninterrupted_wallclock_seconds": str(100.0 if step == 10 else 3700.0 if step == 20 else 0.0),
            "staleness/total/mean": "2.0",
            "staleness/token_lag/exact/mean": "4.0",
            "staleness/version_mix/train/forward_version_span/sequence_mean": "1.5",
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
    assert intervals[0]["staleness/token_lag/exact/mean"] == pytest.approx(4.0)
    assert intervals[0]["staleness/version_mix/train/forward_version_span/sequence_mean"] == pytest.approx(1.5)

    decomposition = ANALYZE._wallclock_decomposition(intervals)
    assert len(decomposition) == 1
    assert decomposition[0]["macro_points_per_update"] == pytest.approx(0.3)
    assert decomposition[0]["updates_per_active_hour"] == pytest.approx(10.0)
    assert decomposition[0]["macro_points_per_active_hour"] == pytest.approx(3.0)
    assert decomposition[0]["training_staleness_mean"] == pytest.approx(2.0)
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


def test_selected_downstream_metrics_are_realized_staleness_only():
    def correlation(predictor):
        return ANALYZE.Correlation(
            predictor=predictor,
            outcome="delta_macro",
            observations=40,
            correlation=0.5,
            slope=1.0,
            ci_low=0.1,
            ci_high=0.8,
        )

    selected = ANALYZE._selected_metrics(
        [
            correlation("train/grad_norm_pre_clip"),
            correlation("staleness/total/mean"),
            correlation("staleness/token_lag/exact/mean"),
        ]
    )

    assert [record.predictor for record in selected] == [
        "staleness/total/mean",
        "staleness/token_lag/exact/mean",
    ]


def test_wallclock_decomposition_uses_only_intervals_shared_by_all_arms():
    def interval(arm, start_step, end_step, delta):
        row = {
            "arm": arm,
            "ratio": arm.split("-", 1)[1],
            "max_weight_staleness": arm[1],
            "start_step": start_step,
            "end_step": end_step,
            "active_interval_hours": 1.0,
            "staleness/total/mean": 1.5,
            "delta_aime24": delta,
            "delta_aime25": delta,
            "delta_aime26": delta,
            "delta_macro": delta,
        }
        for label in ANALYZE.SCORE_COLUMNS:
            row[f"start_{label}_score"] = 0.0
            row[f"end_{label}_score"] = delta
        return row

    intervals = [
        interval("s1-t1r7", 10, 20, 2.0),
        interval("s1-t1r7", 20, 30, 100.0),
        interval("s2-t1r7", 10, 20, 1.0),
    ]
    colocated = interval("s0-colocated", 10, 20, 1.5)
    colocated["staleness/total/mean"] = ""
    intervals.append(colocated)

    rows = ANALYZE._wallclock_decomposition(intervals)

    assert len(rows) == 3
    assert {row["interval_count"] for row in rows} == {1}
    assert {row["common_start_step"] for row in rows} == {10}
    assert {row["common_end_step"] for row in rows} == {20}
    s1 = next(row for row in rows if row["arm"] == "s1-t1r7")
    assert s1["macro_points_per_update"] == pytest.approx(0.2)
    colocated_row = next(row for row in rows if row["arm"] == "s0-colocated")
    assert colocated_row["training_staleness_mean"] == 0.0


def test_wallclock_decomposition_fits_all_common_checkpoint_scores():
    scores = (0.0, 100.0, 0.0, 0.0)
    intervals = []
    for start_step, (start_score, end_score) in enumerate(zip(scores[:-1], scores[1:], strict=True)):
        start_step *= 10
        interval = {
            "arm": "s1-t1r7",
            "ratio": "t1r7",
            "max_weight_staleness": 1,
            "start_step": start_step,
            "end_step": start_step + 10,
            "active_interval_hours": 1.0,
            "staleness/total/mean": 1.0,
        }
        for label in ANALYZE.SCORE_COLUMNS:
            interval[f"start_{label}_score"] = start_score
            interval[f"end_{label}_score"] = end_score
            interval[f"delta_{label}"] = end_score - start_score
        intervals.append(interval)

    row = ANALYZE._wallclock_decomposition(intervals)[0]

    assert row["macro_score_change"] == 0.0
    assert row["macro_points_per_update"] == pytest.approx(-1.0)
    assert row["learning_effect_estimator"] == "ols_score_slope"


def test_score_panel_svg_is_well_formed():
    rows = [
        PLOT.SeriesRow(
            arm="s1-t1r7",
            training_step=10,
            active_wallclock_hours=1.5,
            scores={"AIME24": 50.0, "AIME25": 40.0, "AIME26": 30.0, "AIME mean": 40.0},
        ),
        PLOT.SeriesRow(
            arm="s1-t1r7",
            training_step=20,
            active_wallclock_hours=3.0,
            scores={"AIME24": 55.0, "AIME25": 43.0, "AIME26": 32.0, "AIME mean": 43.3},
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
    assert "AIME mean" in svg


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
            training_staleness_mean=0.5,
            common_start_step=10,
            common_end_step=180,
        ),
        PLOT.DecompositionRow(
            arm="s2-t2r6",
            trainer_nodes=2,
            macro_points_per_update=-0.1,
            updates_per_active_hour=12.0,
            macro_points_per_active_hour=-1.2,
            training_staleness_mean=2.0,
        ),
        PLOT.DecompositionRow(
            arm="s0-colocated",
            trainer_nodes=0,
            macro_points_per_update=0.15,
            updates_per_active_hour=8.0,
            macro_points_per_active_hour=1.2,
            training_staleness_mean=0.0,
            common_start_step=10,
            common_end_step=180,
        ),
    ]

    svg = PLOT._render_wallclock_decomposition(rows)

    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "dQ/dt" in svg
    assert "AIME mean OLS trend per update" in svg
    assert "s1-t1-r7" in svg
    assert "s=max weight staleness, t=train nodes, r=rollout nodes" in svg
    assert "Training staleness mean" in svg
    assert "not logged" not in svg
    assert svg.count('class="panel-frame"') == 4
    assert 'class="setting-label"' in svg
    assert ".setting-label{font-size:24px" in svg


def test_training_throughput_svg_is_well_formed():
    rows = [
        PLOT.DecompositionRow(
            arm="s1-t1r7",
            trainer_nodes=1,
            macro_points_per_update=0.2,
            updates_per_active_hour=10.0,
            macro_points_per_active_hour=2.0,
            training_staleness_mean=0.5,
            common_start_step=10,
            common_end_step=180,
        ),
        PLOT.DecompositionRow(
            arm="s0-colocated",
            trainer_nodes=0,
            macro_points_per_update=0.15,
            updates_per_active_hour=8.0,
            macro_points_per_active_hour=1.2,
            training_staleness_mean=0.0,
            common_start_step=10,
            common_end_step=180,
        ),
    ]

    svg = PLOT._render_training_throughput(rows)

    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "Optimizer-update throughput by setting (dU/dt)" in svg
    assert "Optimizer updates per active hour" in svg
    assert "Equivalent active step time" in svg
    assert "shared updates 10–180" in svg
    assert "s1-t1-r7" in svg
    assert "10.00 updates/h · 360 seconds/update" in svg
    assert svg.count('class="throughput-bar"') == len(rows)


def test_positive_factor_bounds_put_zero_at_left_edge():
    lower, upper = PLOT._factor_bounds([0.09, 0.11, 0.15], signed=True)

    assert lower == 0.0
    assert upper >= 0.15


def test_downstream_correlation_bound_uses_tight_padding():
    bound = PLOT._correlation_bound([-0.123, 0.084])

    assert bound == pytest.approx(0.15)


def test_downstream_plot_bounds_only_visible_staleness_predictors():
    rows = [
        PLOT.CorrelationRow(
            predictor="staleness/pre_queue/variance",
            outcome="delta_macro",
            observations=20,
            correlation=0.06,
            ci_low=None,
            ci_high=None,
        ),
        PLOT.CorrelationRow(
            predictor="train/tis_abs",
            outcome="delta_macro",
            observations=20,
            correlation=0.9,
            ci_low=None,
            ci_high=None,
        ),
    ]

    svg = PLOT._render_downstream_correlations(rows)

    assert "pre-queue/variance" in svg
    assert 'class="correlation-label"' in svg
    assert 'x="750.00"' in svg
    assert ">-0.08</text>" in svg
    assert ">0.08</text>" in svg


def test_downstream_plot_includes_token_lag_and_version_span():
    predictors = (
        "staleness/token_lag/exact/mean",
        "staleness/version_mix/train/forward_version_span/sequence_mean",
    )
    rows = [
        PLOT.CorrelationRow(
            predictor=predictor,
            outcome="delta_macro",
            observations=20,
            correlation=0.1,
            ci_low=None,
            ci_high=None,
        )
        for predictor in predictors
    ]

    svg = PLOT._render_downstream_correlations(rows)

    assert "exact-token-lag/mean" in svg
    assert "within-sample/span" in svg


def test_staleness_metric_heatmap_includes_phase_and_version_predictors():
    predictors = (
        "staleness/pre_queue/variance",
        "staleness/in_queue/variance",
        "staleness/token_lag/exact/mean",
        "staleness/version_mix/train/forward_version_span/sequence_mean",
    )
    rows = [
        PLOT.CorrelationRow(
            predictor=predictor,
            outcome="train/policy_rollout_kl",
            observations=20,
            correlation=0.1 * (index + 1),
            ci_low=None,
            ci_high=None,
        )
        for index, predictor in enumerate(predictors)
    ]
    excluded_outcomes = (
        "throughput/cohort_useful_efficiency",
        "rollout/fully_async/wasted_token_frac",
        "perf/step_time",
        "throughput/useful_tokens_per_second",
    )
    rows.extend(
        PLOT.CorrelationRow(
            predictor="staleness/total/mean",
            outcome=outcome,
            observations=20,
            correlation=0.99,
            ci_low=None,
            ci_high=None,
        )
        for outcome in excluded_outcomes
    )
    rows.extend(
        PLOT.CorrelationRow(
            predictor=predictor,
            outcome="checkpoint/net_parameter_displacement_per_update",
            observations=20,
            correlation=0.25,
            ci_low=None,
            ci_high=None,
        )
        for predictor in predictors
    )
    rows.append(
        PLOT.CorrelationRow(
            predictor="staleness/total/mean",
            outcome="train/update_norm",
            observations=0,
            correlation=None,
            ci_low=None,
            ci_high=None,
        )
    )

    svg = PLOT._render_staleness_metric_correlations(rows)

    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "pre-queue" in svg
    assert "in-queue" in svg
    assert "exact token lag" in svg
    assert "version span" in svg
    assert "net parameter displacement / update" in svg
    assert "train/update_norm" not in svg
    for outcome in excluded_outcomes:
        assert outcome not in svg


def test_steady_state_staleness_uses_the_trailing_contiguous_window():
    history = {step: step / 10.0 for step in range(1, 61)}

    rows = TRAINING_STALENESS._steady_state_rows(
        {"s2-t3r5": history},
        steady_window=5,
    )

    assert len(rows) == 1
    assert rows[0].window_start_step == 56
    assert rows[0].window_end_step == 60
    assert rows[0].observations == 5
    assert rows[0].staleness_total_mean == pytest.approx(5.8)
    assert rows[0].slope_per_100_updates == pytest.approx(10.0)


def test_training_staleness_figures_are_well_formed_and_cover_the_grid():
    histories = {
        f"s{staleness}-t{trainer_nodes}r{8 - trainer_nodes}": {
            step: staleness + trainer_nodes / 10.0 + step / 1000.0 for step in range(251, 301)
        }
        for staleness in TRAINING_STALENESS.STALENESS_LEVELS
        for trainer_nodes in range(1, 5)
    }
    steady = TRAINING_STALENESS._steady_state_rows(histories, steady_window=50)
    trajectories = TRAINING_STALENESS._trajectory_rows(histories, rolling_window=10)

    grid = TRAINING_STALENESS._render_steady_grid(steady)
    trajectory = TRAINING_STALENESS._render_trajectories(trajectories, rolling_window=10)

    assert ElementTree.fromstring(grid).tag.endswith("svg")
    assert ElementTree.fromstring(trajectory).tag.endswith("svg")
    assert len(steady) == 16
    assert "train:rollout 1:7" in grid
    assert "steps 251–300 (n=50)" in grid
    assert "train:rollout nodes = 4:4" in trajectory
    assert "s=8" in trajectory


def test_sensitive_metric_selection_keeps_tis_reference_and_threshold(tmp_path):
    correlations = tmp_path / "correlations.csv"
    correlations.write_text(
        "predictor,outcome,observations,correlation,slope,ci_low,ci_high\n"
        "staleness/total/mean,train/tis_abs,100,0.80,0.1,,\n"
        "staleness/total/mean,train/tis_clipfrac,100,0.20,0.1,,\n",
        encoding="utf-8",
    )

    selected = TRAINING_STALENESS._select_sensitive_metrics(correlations)

    assert [metric.metric for metric in selected] == ["train/tis", "train/tis_abs"]
    assert selected[1].absolute_correlation == pytest.approx(0.8)


def test_sensitive_training_metric_figure_is_well_formed():
    metrics = [
        TRAINING_STALENESS.SensitiveMetric(
            metric="train/tis",
            label="TIS signed mean (reference)",
            strongest_predictor="",
            correlation=None,
            absolute_correlation=None,
            observations=None,
            selection_reason="reference",
        ),
        TRAINING_STALENESS.SensitiveMetric(
            metric="train/tis_abs",
            label="TIS absolute deviation",
            strongest_predictor="staleness/total/mean",
            correlation=0.8,
            absolute_correlation=0.8,
            observations=100,
            selection_reason="threshold",
        ),
    ]
    histories = {
        metric.metric: {
            f"s{staleness}-t{trainer_nodes}r{8 - trainer_nodes}": {
                step: 1.0 + staleness / 100.0 + trainer_nodes / 1000.0 + step / 10000.0 for step in range(1, 21)
            }
            for staleness in TRAINING_STALENESS.STALENESS_LEVELS
            for trainer_nodes in range(1, 5)
        }
        for metric in metrics
    }

    svg = TRAINING_STALENESS._render_sensitive_metrics(metrics, histories, rolling_window=10)

    assert ElementTree.fromstring(svg).tag.endswith("svg")
    assert "TIS absolute deviation" in svg
    assert "r=+0.800 vs total staleness mean" in svg
    assert "optimizer update" in svg
    assert "train:rollout nodes = 1:7" in svg
