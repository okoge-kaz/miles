from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_submit_training_rejects_the_prohibited_instruct_checkpoint() -> None:
    result = subprocess.run(
        [
            "bash",
            "experiments/submit_training.sh",
            "tool_call/async/nemotron3-agentic-static/qwen3-4b-instruct-2507",
            "must-not-submit",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "Qwen3-4B-Instruct-2507 is prohibited" in result.stderr


def test_legacy_automatic_submitters_fail_before_calling_sbatch() -> None:
    scripts = (
        REPO_ROOT / "experiments/scripts/tool_call/submit_effectiveness_when_idle.sh",
        REPO_ROOT
        / "experiments/scripts/swe/async/r2e-gym-swe-rebench-v2"
        / "qwen3-4b-instruct-2507/submit_when_ready.sh",
    )

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        prohibition = source.index("Qwen3-4B-Base-LR2e-5-Step4000")
        failure = source.index("exit 2", prohibition)
        first_submission = source.index("sbatch")
        assert prohibition < failure < first_submission


def test_retired_direct_sbatch_recipes_fail_before_setup() -> None:
    retired_recipes = (
        REPO_ROOT
        / "experiments/scripts/tool_call/async/nemotron3-agentic-static"
        / "qwen3-4b-instruct-2507/run.sbatch",
        REPO_ROOT
        / "experiments/scripts/swe/async/r2e-gym-swe-rebench-v2"
        / "qwen3-4b-instruct-2507/run.sbatch",
    )

    for recipe_path in retired_recipes:
        recipe = recipe_path.read_text(encoding="utf-8")
        guard = recipe.index("this retired recipe is disabled")
        first_setup = recipe.index('source "')
        assert guard < first_setup
