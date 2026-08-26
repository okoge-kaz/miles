"""Reward entry point restricted to IFEvalG instruction-following rows."""

from __future__ import annotations

from typing import Any

from experiments.src.environments.instruction_following.verifier import ifeval_reward
from experiments.src.reward_sets._routing import dispatch_restricted_reward

ALLOWED_VERIFIERS = frozenset({"ifeval_g"})
_HANDLERS = {"ifeval_g": ifeval_reward}


async def reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    return await dispatch_restricted_reward(
        args,
        sample_or_samples,
        handlers=_HANDLERS,
        reward_set="instruction_following",
        **kwargs,
    )
