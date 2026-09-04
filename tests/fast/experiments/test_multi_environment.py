from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.reward_sets import math_code_stem
from miles.utils.types import Sample


REPO_ROOT = Path(__file__).resolve().parents[3]


def _reward(sample_or_samples):
    return asyncio.run(math_code_stem.reward(SimpleNamespace(), sample_or_samples))


def test_multi_reward_routes_math_code_and_stem_groups_in_input_order(monkeypatch):
    calls: dict[str, list[Sample]] = {}

    def handler(name):
        async def score(args, samples, **kwargs):
            calls[name] = samples
            return [float(sample.index) for sample in samples]

        return score

    for verifier in math_code_stem.ALLOWED_VERIFIERS:
        monkeypatch.setitem(math_code_stem._HANDLERS, verifier, handler(verifier))
    samples = [
        Sample(index=1, response="", metadata={"verifier": "python_code"}),
        Sample(index=2, response="", metadata={"verifier": "math"}),
        Sample(index=3, response="", metadata={"verifier": "reasoning_gym"}),
        Sample(index=4, response="", metadata={"verifier": "python_code"}),
        Sample(index=5, response="", metadata={"verifier": "mcqa_regex"}),
    ]

    assert _reward(samples) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert calls["python_code"] == [samples[0], samples[3]]
    assert calls["math"] == [samples[1]]
    assert calls["reasoning_gym"] == [samples[2]]
    assert calls["mcqa_regex"] == [samples[4]]


def test_multi_reward_rejects_instruction_following_rows():
    sample = Sample(response="", metadata={"verifier": "ifeval_g"})

    with pytest.raises(ValueError, match="rejects verifier"):
        _reward(sample)


def test_multi_dataset_job_requires_every_supported_verifier() -> None:
    setup_job = (
        REPO_ROOT / "experiments/setup/datasets/prepare_math_code_stem_blend.sbatch"
    ).read_text(encoding="utf-8")

    assert "--expected-rows 32673" in setup_job
    assert setup_job.count("--require-verifiers math python_code mcqa_regex reasoning_gym") == 2
    assert "math-code-stem-balanced-summary.json" in setup_job
    assert "math-code-stem-balanced-audit.json" in setup_job
