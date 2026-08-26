"""Reward entry point for static rows mixed with stateful Tau rollouts.

Tau environment rollouts arrive with ``sample.reward`` already populated by the
environment and therefore bypass this entry point. Only the static Nemotron
function-call rows in the same training file are accepted here.
"""

from __future__ import annotations

from typing import Any

from experiments.src.environments.tool_call.verifier import score_tool_call_sample
from experiments.src.reward_sets._routing import dispatch_restricted_reward, synchronous_handler

ALLOWED_VERIFIERS = frozenset({"expert_action"})
_HANDLERS = {"expert_action": synchronous_handler(score_tool_call_sample)}


async def reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    return await dispatch_restricted_reward(
        args,
        sample_or_samples,
        handlers=_HANDLERS,
        reward_set="tau_static_rows",
        **kwargs,
    )
