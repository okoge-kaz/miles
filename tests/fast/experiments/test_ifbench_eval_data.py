from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.datasets.nemotron.adapters import adapt_ifbench
from miles.rollout.rm_hub import ifbench as evaluator
from miles.rollout.rm_hub import async_rm
from miles.utils.arguments import _resolve_eval_datasets
from miles.utils.types import Sample


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_ifbench_eval_config_uses_canonical_data_and_official_reward() -> None:
    args = SimpleNamespace(
        eval_config=str(REPO_ROOT / "experiments/configs/eval_ifbench.yaml"),
        eval_prompt_data=None,
    )

    datasets = _resolve_eval_datasets(args)

    assert len(datasets) == 1
    dataset = datasets[0]
    assert dataset.name == "ifbench"
    assert dataset.path == "/data/ifbench/IFBench_test_miles.jsonl"
    assert dataset.rm_type == "ifbench"
    assert dataset.n_samples_per_eval_prompt == 8
    assert dataset.max_response_len == 16384
    assert dataset.temperature == 0.6
    assert dataset.top_p == 0.95
    assert dataset.top_k == 20


def test_ifbench_adapter_preserves_official_inputs_as_eval_only() -> None:
    row = {
        "key": "17",
        "prompt": " Use the word miles exactly once. ",
        "instruction_id_list": ["count:keywords_multiple"],
        "kwargs": [{"keyword1": "miles"}],
    }

    converted = adapt_ifbench(row)

    assert converted is not None
    assert converted == {
        "prompt": [{"role": "user", "content": "Use the word miles exactly once."}],
        "label": "count:keywords_multiple",
        "metadata": {
            "source": "ifbench",
            "verifier": "ifbench",
            "instruction_id_list": ["count:keywords_multiple"],
            "prompt_text": "Use the word miles exactly once.",
            "kwargs": [{"keyword1": "miles"}],
            "record_id": "17",
            "eval_only": True,
        },
    }


def test_ifbench_reward_routes_to_official_strict_scorer(monkeypatch) -> None:
    class FakeEvaluationLib:
        InputExample = SimpleNamespace

        @staticmethod
        def test_instruction_following_strict(example, prompt_to_response):
            assert example.instruction_id_list == ["keywords:existence"]
            assert example.kwargs == [{"keywords": ["cat", "dog"]}]
            response = prompt_to_response[example.prompt]
            return SimpleNamespace(follow_all_instructions=response == "A cat and a dog.")

    monkeypatch.setattr(evaluator, "_load_evaluation_lib", lambda: FakeEvaluationLib)
    metadata = {
        "rm_type": "ifbench",
        "verifier": "ifbench",
        "eval_only": True,
        "instruction_id_list": ["keywords:existence"],
        "prompt_text": "Include cat and dog.",
        "kwargs": [{"keywords": ["cat", "dog"]}],
        "record_id": 0,
    }
    args = SimpleNamespace(
        custom_rm_path=None,
        rm_type=None,
        zero_reward_on_truncated=False,
    )

    correct = Sample(response="A cat and a dog.", label=None, metadata=metadata)
    wrong = Sample(response="A cat.", label=None, metadata=metadata)

    assert asyncio.run(async_rm(args, correct)) == 1.0
    assert asyncio.run(async_rm(args, wrong)) == 0.0


def test_ifbench_evaluator_fails_closed_without_pinned_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "ifbench"
    deps = tmp_path / "ifbench-deps"
    repo.mkdir()
    deps.mkdir()
    (repo / "evaluation_lib.py").write_text("# staged evaluator\n", encoding="utf-8")
    (deps / "nltk_data").mkdir()
    monkeypatch.setenv("IFBENCH_REPO_PATH", str(repo))
    monkeypatch.setenv("IFBENCH_DEPS_PATH", str(deps))

    with pytest.raises(ImportError, match="pinned IFBench dependencies"):
        evaluator._ensure_ifbench_repo()


def test_ifbench_evaluator_never_clones_or_installs_at_runtime() -> None:
    source = (REPO_ROOT / "miles/rollout/rm_hub/ifbench.py").read_text(encoding="utf-8")

    assert "git clone" not in source
    assert "pip install" not in source
    assert ".write_text(" not in source
    assert "PINNED_IFBENCH_COMMIT" in source
    assert "PINNED_DEPS_MARKER" in source
    assert '"--untracked-files=no"' in source


def test_ifbench_setup_job_pins_and_audits_the_configured_artifact() -> None:
    setup_job = (
        REPO_ROOT / "experiments/setup/environments/prepare_ifbench.sbatch"
    ).read_text(encoding="utf-8")

    assert evaluator.PINNED_IFBENCH_COMMIT in setup_job
    assert evaluator.PINNED_DEPS_MARKER in setup_job
    assert "--dataset ifbench" in setup_job
    assert "--input /data/ifbench/data/IFBench_test.jsonl" in setup_job
    assert "--output /data/ifbench/IFBench_test_miles.jsonl" in setup_job
    assert "official IFBench verifier: correct=1 wrong=0" in setup_job
