"""Reward entry point restricted to static exact-match function calls."""

from __future__ import annotations

from typing import Any

from experiments.src.environments.tool_call_pivot.verifier import score_tool_call_sample
from experiments.src.protocols.tool_call_pivot import TOOL_CALL_PIVOT_INTERACTION_MODE
from experiments.src.reward_sets._routing import dispatch_restricted_reward, synchronous_handler

ALLOWED_VERIFIERS = frozenset({"expert_action"})


def _score_pivot_sample(sample: Any) -> float:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    if metadata.get("interaction_mode") != TOOL_CALL_PIVOT_INTERACTION_MODE:
        raise ValueError("tool_call_pivot requires static_single_turn_pivot samples")
    if metadata.get("stateful_environment") is not False:
        raise ValueError("tool_call_pivot rejects stateful environment samples")
    if not str(metadata.get("source") or "").endswith("-pivot"):
        raise ValueError("tool_call_pivot requires a Pivot source")
    return score_tool_call_sample(sample)


_HANDLERS = {"expert_action": synchronous_handler(_score_pivot_sample)}


async def reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    return await dispatch_restricted_reward(
        args,
        sample_or_samples,
        handlers=_HANDLERS,
        reward_set="tool_call_pivot",
        **kwargs,
    )
