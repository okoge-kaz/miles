from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from experiments.src.environments.reasoning_gym.verifier import score_reasoning_gym_sample
from experiments.src.reward_sets import stem
from miles.utils.types import Sample


def _reward(reward_function, sample_or_samples):
    return asyncio.run(reward_function(SimpleNamespace(), sample_or_samples))


def test_stem_reward_scores_mcqa_and_reasoning_rows_in_input_order(monkeypatch):
    monkeypatch.setenv("REASONING_GYM_ALLOW_EXACT_FALLBACK", "1")
    mcqa = Sample(
        response="Selected Option -> B",
        label="B",
        metadata={
            "verifier": "mcqa_regex",
            "output_regex": r"Selected Option\s*->\s*([A-D])",
            "valid_letters": ["A", "B", "C", "D"],
        },
    )
    reasoning = Sample(
        response="Answer: Richard",
        label="Richard",
        metadata={"verifier": "reasoning_gym"},
    )
    wrong = Sample(response="Selected Option -> A", label="B", metadata=mcqa.metadata)

    assert _reward(stem.reward, mcqa) == 1.0
    assert _reward(stem.reward, [mcqa, reasoning, wrong]) == [1.0, 1.0, 0.0]


def test_reasoning_gym_fallback_uses_the_final_answer_not_reasoning(monkeypatch):
    monkeypatch.setenv("REASONING_GYM_ALLOW_EXACT_FALLBACK", "1")
    metadata = {"verifier": "reasoning_gym"}
    correct = Sample(response="work\nAnswer: Richard", label="Richard", metadata=metadata)
    wrong = Sample(
        response="Richard may be relevant.\nAnswer: Alice",
        label="Richard",
        metadata=metadata,
    )

    assert score_reasoning_gym_sample(correct) == 1.0
    assert score_reasoning_gym_sample(wrong) == 0.0


def test_stem_reward_fails_closed_before_dispatch(monkeypatch):
    called = False

    async def unexpected_handler(args, sample_or_samples, **kwargs):
        nonlocal called
        called = True
        return 1.0

    monkeypatch.setitem(stem._HANDLERS, "reasoning_gym", unexpected_handler)
    samples = [
        Sample(response="", metadata={"verifier": "reasoning_gym"}),
        Sample(response="", metadata={"verifier": "python_code"}),
    ]

    with pytest.raises(ValueError, match="rejects verifier"):
        _reward(stem.reward, samples)
    assert called is False
