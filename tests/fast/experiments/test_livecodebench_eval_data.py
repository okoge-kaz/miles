from __future__ import annotations

import asyncio
import base64
import json
import os
import pickle
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.datasets.nemotron.adapters import adapt_livecodebench
from experiments.src.evaluators import livecodebench as evaluator
from miles.utils.arguments import _resolve_eval_datasets
from miles.utils.types import Sample


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_livecodebench_eval_config_is_generation_only_and_uses_canonical_data() -> None:
    args = SimpleNamespace(
        eval_config=str(REPO_ROOT / "experiments/configs/eval_livecodebench.yaml"),
        eval_prompt_data=None,
    )
    datasets = _resolve_eval_datasets(args)

    assert len(datasets) == 1
    assert datasets[0].name == "livecodebench"
    assert datasets[0].path == "/data/livecodebench-lite/livecodebench-release-v6-miles-eval.jsonl"
    assert datasets[0].n_samples_per_eval_prompt == 1
    assert datasets[0].max_response_len == 16384
    assert datasets[0].rm_type is None


def test_livecodebench_adapter_keeps_official_tests_eval_only() -> None:
    row = {
        "question_id": "task-1",
        "question_content": "Add two integers.",
        "starter_code": "",
        "public_test_cases": json.dumps([{"input": "1 2\n", "output": "3\n"}]),
        "private_test_cases": "encoded-private-tests",
        "metadata": json.dumps({"func_name": None}),
        "platform": "codeforces",
    }

    converted = adapt_livecodebench(row)

    assert converted is not None
    assert converted["label"] == "task-1"
    assert converted["metadata"]["private_test_cases"] == "encoded-private-tests"
    assert converted["metadata"]["verifier"] == "livecodebench"
    assert converted["metadata"]["eval_only"] is True
    assert "private" not in converted["prompt"][0]["content"]


def test_private_test_decoder_rejects_pickle_globals() -> None:
    payload = base64.b64encode(zlib.compress(pickle.dumps(os.system))).decode()

    with pytest.raises(pickle.UnpicklingError, match="pickle global is forbidden"):
        evaluator.decode_private_tests(payload)


def test_livecodebench_reward_requires_explicit_local_execution(monkeypatch) -> None:
    monkeypatch.delenv("LCB_ALLOW_LOCAL_EXECUTION", raising=False)
    sample = Sample(metadata={"eval_only": True, "verifier": "livecodebench"})

    with pytest.raises(RuntimeError, match="LCB_ALLOW_LOCAL_EXECUTION"):
        asyncio.run(evaluator.livecodebench_reward(None, sample))


def test_livecodebench_reward_uses_official_result_shape(monkeypatch) -> None:
    metadata = {
        "eval_only": True,
        "verifier": "livecodebench",
        "public_test_cases": json.dumps([{"input": "1 2\n", "output": "3\n"}]),
        "private_test_cases": json.dumps([{"input": "4 5\n", "output": "9\n"}]),
        "lcb_metadata": json.dumps({"func_name": None}),
    }
    samples = [
        Sample(response="```python\nprint(3)\n```", metadata=metadata),
        Sample(response="```python\nprint(0)\n```", metadata=metadata),
    ]

    def fake_metrics(evaluation_samples, generations, **kwargs):
        assert len(evaluation_samples) == 2
        assert generations == [["print(3)"], ["print(0)"]]
        assert kwargs["k_list"] == [1]
        return {}, {0: [[True, True]], 1: [[True, False]]}, {}

    monkeypatch.setenv("LCB_ALLOW_LOCAL_EXECUTION", "1")
    monkeypatch.setattr(evaluator, "_load_official_metrics", lambda: fake_metrics)

    assert asyncio.run(evaluator.livecodebench_reward(None, samples)) == [1.0, 0.0]


def test_livecodebench_setup_job_writes_and_audits_the_configured_artifact() -> None:
    setup_job = (
        REPO_ROOT / "experiments/setup/datasets/prepare_livecodebench.sbatch"
    ).read_text(encoding="utf-8")

    assert "--dataset livecodebench" in setup_job
    assert 'livecodebench-release-v6-miles-eval.jsonl"' in setup_job
    assert '--summary "${LCB}/miles-conversion-summary.json"' in setup_job
    assert "--require-verifiers livecodebench" in setup_job
    assert "--require-eval-only" in setup_job
