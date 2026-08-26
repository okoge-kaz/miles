"""Reward entry point restricted to competitive-programming rows."""

from __future__ import annotations

from typing import Any

from experiments.src.environments.competitive_programming.verifier import code_exec_reward
from experiments.src.reward_sets._routing import dispatch_restricted_reward

ALLOWED_VERIFIERS = frozenset({"python_code"})
_HANDLERS = {"python_code": code_exec_reward}


async def reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    return await dispatch_restricted_reward(
        args,
        sample_or_samples,
        handlers=_HANDLERS,
        reward_set="code",
        **kwargs,
    )
