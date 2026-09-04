from pathlib import Path
from types import SimpleNamespace

from experiments.src.datasets.nemotron.adapters import adapt_math500
from miles.utils.arguments import _resolve_eval_datasets


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_math500_eval_config_uses_canonical_math_artifact_and_reward() -> None:
    args = SimpleNamespace(
        eval_config=str(REPO_ROOT / "experiments/configs/eval_math500.yaml"),
        eval_prompt_data=None,
    )
    datasets = _resolve_eval_datasets(args)

    assert len(datasets) == 1
    assert datasets[0].name == "math500"
    assert datasets[0].path == "/data/math-500/math-500.jsonl"
    assert datasets[0].n_samples_per_eval_prompt == 4
    assert datasets[0].max_response_len == 16384
    assert datasets[0].rm_type == "math"


def test_math500_adapter_emits_heldout_math_row_without_solution() -> None:
    converted = adapt_math500(
        {
            "problem": "What is 6 times 7?",
            "solution": "This private worked solution must not be exported.",
            "answer": "42",
            "subject": "Algebra",
            "level": 1,
            "unique_id": "test/algebra/1.json",
        }
    )

    assert converted is not None
    assert converted["label"] == "42"
    assert "\\boxed" in converted["prompt"][0]["content"]
    assert "private worked solution" not in str(converted)
    assert converted["metadata"] == {
        "source": "math-500",
        "verifier": "math",
        "eval_only": True,
        "level": 1,
        "record_id": "test/algebra/1.json",
        "rm_type": "math",
        "subject": "Algebra",
    }


def test_math500_adapter_rejects_missing_problem_or_answer() -> None:
    assert adapt_math500({"answer": "42"}) is None
    assert adapt_math500({"problem": "question"}) is None


def test_math500_setup_job_writes_the_configured_canonical_artifact() -> None:
    setup_job = (REPO_ROOT / "experiments/setup/datasets/prepare_math500.sbatch").read_text(
        encoding="utf-8"
    )

    assert "--dataset math500" in setup_job
    assert "--output /data/math-500/math-500.jsonl" in setup_job
    assert 'test "$(wc -l < /data/math-500/math-500.jsonl)" -eq 500' in setup_job
