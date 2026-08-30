from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_current_agentic_recipes_never_select_the_prohibited_checkpoint() -> None:
    recipes = (
        REPO_ROOT
        / "experiments/scripts/tool_call_pivot/async/"
        "nemotron-agentic-conv-tooluse-pivot/qwen3-4b/run.sbatch",
        REPO_ROOT
        / "experiments/scripts/swe/async/"
        "swe-rebench-v2-swe-gym/qwen3-4b/run.sbatch",
    )

    for recipe in recipes:
        source = recipe.read_text(encoding="utf-8")
        assert "Qwen3-4B-Instruct-2507" not in source
        assert "Qwen3-4B-Base-LR2e-5-Step4000" in source
        assert "iter_0004000" in source


def test_submitters_reference_current_agentic_recipes() -> None:
    submitters = {
        REPO_ROOT / "experiments/scripts/tool_call_pivot/submit_effectiveness_when_idle.sh": (
            "nemotron-agentic-conv-tooluse-pivot/qwen3-4b/run.sbatch"
        ),
        REPO_ROOT
        / "experiments/scripts/swe/async/swe-rebench-v2-swe-gym/"
        "qwen3-4b/submit_when_ready.sh": (
            "swe-rebench-v2-swe-gym/qwen3-4b/run.sbatch"
        ),
    }

    for script, expected_recipe in submitters.items():
        source = script.read_text(encoding="utf-8")
        assert expected_recipe in source
        assert "this retired recipe is disabled" not in source


def test_tool_call_pivot_effectiveness_uses_tau_three_evaluation() -> None:
    source = (
        REPO_ROOT / "experiments/scripts/tool_call_pivot/submit_effectiveness_when_idle.sh"
    ).read_text(encoding="utf-8")

    assert source.count("experiments/scripts/tau_bench/evaluate.sbatch") == 2
    assert "experiments/setup/environments/prepare_tau_bench.sbatch" in source
    assert "experiments/scripts/tool_call_pivot/evaluate.sbatch" not in source
