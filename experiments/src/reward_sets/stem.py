"""Reward entry point restricted to the STEM/knowledge training blend."""

from __future__ import annotations

from typing import Any

from experiments.src.environments.reasoning_gym.verifier import score_reasoning_gym_sample
from experiments.src.reward_sets._common import score_gpqa_sample, score_mcqa_regex_sample
from experiments.src.reward_sets._routing import dispatch_restricted_reward, synchronous_handler

ALLOWED_VERIFIERS = frozenset({"gpqa", "mcqa_regex", "reasoning_gym"})
_HANDLERS = {
    "gpqa": synchronous_handler(score_gpqa_sample),
    "mcqa_regex": synchronous_handler(score_mcqa_regex_sample),
    "reasoning_gym": synchronous_handler(score_reasoning_gym_sample),
}


async def reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    return await dispatch_restricted_reward(
        args,
        sample_or_samples,
        handlers=_HANDLERS,
        reward_set="stem",
        **kwargs,
    )
