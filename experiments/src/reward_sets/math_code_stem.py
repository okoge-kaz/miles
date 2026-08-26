"""Reward entry point restricted to the Math + Code + STEM training blend."""

from __future__ import annotations

from typing import Any

from experiments.src.environments.competitive_programming.verifier import code_exec_reward
from experiments.src.environments.reasoning_gym.verifier import score_reasoning_gym_sample
from experiments.src.reward_sets._common import score_math_sample, score_mcqa_regex_sample
from experiments.src.reward_sets._routing import dispatch_restricted_reward, synchronous_handler

ALLOWED_VERIFIERS = frozenset({"math", "mcqa_regex", "python_code", "reasoning_gym"})
_HANDLERS = {
    "math": synchronous_handler(score_math_sample),
    "mcqa_regex": synchronous_handler(score_mcqa_regex_sample),
    "python_code": code_exec_reward,
    "reasoning_gym": synchronous_handler(score_reasoning_gym_sample),
}


async def reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    return await dispatch_restricted_reward(
        args,
        sample_or_samples,
        handlers=_HANDLERS,
        reward_set="math_code_stem",
        **kwargs,
    )
