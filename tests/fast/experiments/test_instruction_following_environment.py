from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from experiments.src.environments.instruction_following import verifier as ifeval
from experiments.src.reward_sets import instruction_following
from miles.utils.types import Sample


def _reward(sample_or_samples):
    return asyncio.run(instruction_following.reward(SimpleNamespace(), sample_or_samples))


def test_ifeval_normalizes_only_count_increment_singleton_keywords(monkeypatch):
    class CountIncrementChecker:
        def __init__(self, instruction_id):
            self.id = instruction_id

        def get_instruction_args_keys(self):
            return ["keyword1", "keyword2"]

        def build_description(self, keyword1, keyword2):
            assert isinstance(keyword1, str)
            assert isinstance(keyword2, str)
            self.keyword1 = keyword1
            self.keyword2 = keyword2

        def check_following(self, response):
            return response.count(self.keyword1) == 1 and response.count(self.keyword2) == 2

    registry = SimpleNamespace(
        INSTRUCTION_DICT={"count:count_increment_word": CountIncrementChecker}
    )
    monkeypatch.setattr(ifeval, "_load_registry", lambda: registry)
    metadata = {
        "instruction_id_list": ["count:count_increment_word"],
        "kwargs": [{"keyword1": ["help"], "keyword2": ["dump"]}],
    }
    sample = Sample(response="help dump dump", metadata=metadata)

    assert ifeval.validate_ifeval_metadata(metadata) == 1
    assert ifeval.score_ifeval_sample(sample) == 1.0

    malformed = dict(
        metadata,
        kwargs=[{"keyword1": ["help", "assist"], "keyword2": ["dump"]}],
    )
    assert ifeval.score_ifeval_sample(Sample(response=sample.response, metadata=malformed)) == 0.0


def test_ifeval_preserves_genuine_list_arguments(monkeypatch):
    class KeywordChecker:
        def __init__(self, instruction_id):
            self.id = instruction_id

        def get_instruction_args_keys(self):
            return ["keywords"]

        def build_description(self, keywords):
            assert keywords == ["cat", "dog"]
            self.keywords = keywords

        def check_following(self, response):
            return all(keyword in response for keyword in self.keywords)

    registry = SimpleNamespace(INSTRUCTION_DICT={"keywords:existence": KeywordChecker})
    monkeypatch.setattr(ifeval, "_load_registry", lambda: registry)
    metadata = {
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["cat", "dog"]}],
    }

    assert ifeval.score_ifeval_sample(Sample(response="cat and dog", metadata=metadata)) == 1.0


def test_ifeval_matches_official_fractional_reward_and_removes_thinking(monkeypatch):
    class ContainsKeyword:
        def __init__(self, instruction_id):
            self.id = instruction_id

        def get_instruction_args_keys(self):
            return ["keyword"]

        def build_description(self, keyword):
            self.keyword = keyword

        def check_following(self, response):
            return self.keyword in response

    class NoComma:
        def __init__(self, instruction_id):
            self.id = instruction_id

        def get_instruction_args_keys(self):
            return []

        def build_description(self):
            return None

        def check_following(self, response):
            return "," not in response

    registry = SimpleNamespace(
        INSTRUCTION_DICT={
            "keywords:frequency": ContainsKeyword,
            "punctuation:no_comma": NoComma,
        }
    )
    monkeypatch.setattr(ifeval, "_load_registry", lambda: registry)
    metadata = {
        "instruction_id_list": ["keywords:frequency", "punctuation:no_comma"],
        "kwargs": [{"keyword": "cat"}, None],
    }
    response = "<|assistant|><think>reasoning, has commas</think><answer>dog</answer>"

    assert ifeval.score_ifeval_sample(Sample(response=response, metadata=metadata)) == 0.5


def test_instruction_following_reward_preserves_scalar_and_batch_contract(monkeypatch):
    async def fake_handler(args, sample_or_samples, **kwargs):
        if isinstance(sample_or_samples, list):
            return [float(index) for index in range(len(sample_or_samples))]
        return 0.25

    monkeypatch.setitem(instruction_following._HANDLERS, "ifeval_g", fake_handler)
    sample = Sample(response="", metadata={"verifier": "ifeval_g"})

    assert _reward(sample) == 0.25
    assert _reward([sample, sample]) == [0.0, 1.0]


def test_instruction_following_reward_fails_closed_before_dispatch(monkeypatch):
    called = False

    async def unexpected_handler(args, sample_or_samples, **kwargs):
        nonlocal called
        called = True
        return 1.0

    monkeypatch.setitem(instruction_following._HANDLERS, "ifeval_g", unexpected_handler)
    samples = [
        Sample(response="", metadata={"verifier": "ifeval_g"}),
        Sample(response="", metadata={"verifier": "calendar_constraints"}),
    ]

    with pytest.raises(ValueError, match="rejects verifier"):
        _reward(samples)
    assert called is False
