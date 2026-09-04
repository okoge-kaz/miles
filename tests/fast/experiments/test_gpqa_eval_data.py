from pathlib import Path
from types import SimpleNamespace

from experiments.src.datasets.nemotron.adapters import adapt_gpqa
from miles.rollout.rm_hub.gpqa import compute_gpqa_reward
from miles.utils.arguments import _resolve_eval_datasets


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_gpqa_eval_config_uses_canonical_artifacts_and_gpqa_reward() -> None:
    args = SimpleNamespace(
        eval_config=str(REPO_ROOT / "experiments/configs/eval_gpqa.yaml"),
        eval_prompt_data=None,
    )
    datasets = _resolve_eval_datasets(args)

    assert [dataset.name for dataset in datasets] == [
        "gpqa_diamond",
        "gpqa_main",
        "gpqa_extended",
    ]
    assert [dataset.path for dataset in datasets] == [
        "/data/gpqa/gpqa-diamond-miles.jsonl",
        "/data/gpqa/gpqa-main-miles.jsonl",
        "/data/gpqa/gpqa-extended-miles.jsonl",
    ]
    assert [dataset.n_samples_per_eval_prompt for dataset in datasets] == [8, 4, 4]
    assert all(dataset.max_response_len == 16384 for dataset in datasets)
    assert all(dataset.rm_type == "gpqa" for dataset in datasets)


def test_gpqa_adapter_preserves_published_nemo_skills_ordering() -> None:
    row = {
        "problem": "Which option is correct?",
        "A": "alpha",
        "B": "beta",
        "C": "gamma",
        "D": "delta",
        "expected_answer": "C",
        "subset_for_metrics": "Physics",
        "explanation": "published explanation",
        "difficulty": "Hard",
    }

    converted = adapt_gpqa(row)

    assert converted is not None
    assert converted["label"] == "C"
    assert converted["metadata"]["choices"] == ["alpha", "beta", "gamma", "delta"]
    assert converted["metadata"]["valid_letters"] == list("ABCD")
    assert converted["metadata"]["source_format"] == "nemo-skills-preprocessed"
    assert converted["metadata"]["verifier"] == "gpqa"
    assert converted["metadata"]["eval_only"] is True


def test_gpqa_csv_adapter_is_deterministic_and_keeps_the_correct_choice() -> None:
    row = {
        "Question": "A deterministic GPQA question",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong one",
        "Incorrect Answer 2": "wrong two",
        "Incorrect Answer 3": "wrong three",
    }

    first = adapt_gpqa(row)
    second = adapt_gpqa(row)

    assert first == second
    assert first is not None
    assert first["metadata"]["source_format"] == "idavidrein-csv"
    assert first["metadata"]["choices"]["ABCD".index(first["label"])] == "correct"


def test_gpqa_scorer_uses_only_the_post_thinking_final_answer() -> None:
    metadata = {"valid_letters": list("ABCD"), "choices": ["a", "b", "c", "d"]}

    assert compute_gpqa_reward("<think>Answer: A</think>Final answer: C", "C", metadata) == 1.0
    assert compute_gpqa_reward("<think>Answer: C</think>Final answer: B", "C", metadata) == 0.0


def test_gpqa_setup_job_builds_and_audits_every_configured_split() -> None:
    setup_job = (REPO_ROOT / "experiments/setup/datasets/prepare_gpqa.sbatch").read_text(
        encoding="utf-8"
    )

    assert "--dataset gpqa" in setup_job
    for split, records in (("diamond", 198), ("main", 448), ("extended", 546)):
        assert f'gpqa-{split}-miles.jsonl' in setup_job
        assert f"--expected-rows {records} --require-verifiers gpqa" in setup_job
        assert f'--summary "${{GPQA}}/gpqa-${{split}}-conversion-summary.json"' in setup_job
