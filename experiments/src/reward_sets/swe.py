"""Reward entry point restricted to verifier-scored Harbor SWE rollouts."""

from __future__ import annotations

from typing import Any

from experiments.src.environments.swe.result import HarborSWEOutcome
from experiments.src.reward_sets._routing import dispatch_restricted_reward, synchronous_handler

ALLOWED_VERIFIERS = frozenset({"swe_environment"})


def _score_swe_sample(sample: Any) -> float:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    return HarborSWEOutcome.from_mapping(metadata).reward


_HANDLERS = {"swe_environment": synchronous_handler(_score_swe_sample)}


async def reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    return await dispatch_restricted_reward(
        args,
        sample_or_samples,
        handlers=_HANDLERS,
        reward_set="swe",
        **kwargs,
    )
