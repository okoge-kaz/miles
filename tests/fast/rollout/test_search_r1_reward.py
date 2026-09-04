from types import SimpleNamespace

import pytest

from miles.rollout.rm_hub import async_rm
from miles.rollout.rm_hub.search_r1 import compute_search_r1_reward
from miles.utils.types import Sample


PROMPT = (
    "<|im_start|>user\nUse <answer> final answer </answer>."
    "<|im_end|>\n<|im_start|>assistant\n"
)
LABEL = {"ground_truth": {"target": ["McComb, Mississippi"]}, "style": "rule"}


def test_search_r1_reward_scores_exact_match_and_optional_format_shaping():
    correct = "<think>done</think><answer>McComb Mississippi</answer>"
    wrong = "<think>done</think><answer>Houston</answer>"

    assert compute_search_r1_reward(prompt=PROMPT, response=correct, label=LABEL) == 1.0
    assert compute_search_r1_reward(prompt=PROMPT, response=wrong, label=LABEL) == 0.0
    assert compute_search_r1_reward(prompt=PROMPT, response=wrong, label=LABEL, format_score=0.2) == 0.2


def test_search_r1_reward_accepts_multi_turn_search_and_has_no_retrieval_bonus():
    response = (
        "<think>look it up</think><search>birthplace</search>"
        "<information>McComb, Mississippi</information>"
        "<think>answer</think><answer>Houston</answer>"
    )

    assert compute_search_r1_reward(prompt=PROMPT, response=response, label=LABEL) == 0.0
    assert compute_search_r1_reward(prompt=PROMPT, response=response, label=LABEL, format_score=0.2) == 0.2


def test_search_r1_reward_penalizes_correct_answer_with_invalid_format():
    response = "<answer>McComb, Mississippi</answer>"

    assert compute_search_r1_reward(prompt=PROMPT, response=response, label=LABEL, format_score=0.2) == 0.8


async def test_search_r1_is_a_builtin_reward_for_replay_buffer():
    args = SimpleNamespace(
        custom_rm_path=None,
        rm_type="search_r1",
        search_r1_format_score=0.0,
    )
    sample = Sample(
        prompt=PROMPT,
        response="<think>done</think><answer>McComb, Mississippi</answer>",
        label=LABEL,
    )

    assert await async_rm(args, sample) == 1.0


def test_search_r1_reward_rejects_malformed_labels_and_score():
    with pytest.raises(ValueError, match="ground_truth.target"):
        compute_search_r1_reward(prompt=PROMPT, response="", label={})
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_search_r1_reward(prompt=PROMPT, response="", label=LABEL, format_score=1.1)
