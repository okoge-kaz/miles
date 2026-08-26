"""Broad reward registry for converter validation and diagnostics."""

from __future__ import annotations

import json
from typing import Any

from experiments.src.environments.calendar.verifier import build_calendar_solution, score_calendar_response
from experiments.src.environments.competitive_programming.verifier import (
    build_preflight_probes as build_code_probes,
)
from experiments.src.environments.competitive_programming.verifier import code_exec_reward
from experiments.src.environments.instruction_following.verifier import score_ifeval_sample
from experiments.src.environments.reasoning_gym.verifier import score_reasoning_gym_sample
from experiments.src.environments.tool_call.verifier import parse_tool_calls, score_tool_call_sample
from experiments.src.environments.workplace.verifier import score_action_trajectory
from experiments.src.protocols.openai_responses import expected_action_signature
from experiments.src.reward_sets._common import (
    score_gpqa_sample,
    score_math_sample,
    score_mcqa_regex_sample,
)
from experiments.src.reward_sets._routing import dispatch_restricted_reward, synchronous_handler
from experiments.src.reward_sets.structured_output import score_structured_output_sample

__all__ = [
    "blend_reward",
    "build_preflight_probes",
    "mcqa_regex_reward",
    "reasoning_gym_reward",
    "structured_output_reward",
    "tool_call_match_reward",
]


def _score_calendar_sample(sample: Any) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return score_calendar_response(sample.response, metadata.get("expected_calendar_state"))


def _score_workplace_sample(sample: Any) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    expected = metadata.get("expected_actions")
    predicted = parse_tool_calls(sample.response)
    return score_action_trajectory(predicted, expected)


reasoning_gym_reward = synchronous_handler(score_reasoning_gym_sample)
structured_output_reward = synchronous_handler(score_structured_output_sample)
tool_call_match_reward = synchronous_handler(score_tool_call_sample)
mcqa_regex_reward = synchronous_handler(score_mcqa_regex_sample)

_HANDLERS = {
    "calendar_constraints": synchronous_handler(_score_calendar_sample),
    "expert_action": tool_call_match_reward,
    "gpqa": synchronous_handler(score_gpqa_sample),
    "ifeval_g": synchronous_handler(score_ifeval_sample),
    "json_schema": structured_output_reward,
    "math": synchronous_handler(score_math_sample),
    "mcqa_regex": mcqa_regex_reward,
    "python_code": code_exec_reward,
    "reasoning_gym": reasoning_gym_reward,
    "workplace_environment": synchronous_handler(_score_workplace_sample),
}


async def blend_reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    """Dispatch heterogeneous validation rows by their declared verifier."""

    return await dispatch_restricted_reward(
        args,
        sample_or_samples,
        handlers=_HANDLERS,
        reward_set="all_domains_validation",
        verifier_keys=("verifier", "rm_type"),
        **kwargs,
    )


def build_preflight_probes(label: Any, metadata: dict[str, Any]) -> tuple[str, str] | None:
    verifier = str((metadata or {}).get("verifier") or "")
    if verifier in {"math", "gpqa", "mcqa_regex"}:
        pattern = (metadata or {}).get("output_regex")
        if verifier == "math":
            correct = f"Answer: \\boxed{{{label}}}"
        elif verifier == "mcqa_regex" and pattern:
            template_id = str((metadata or {}).get("template_id") or "")
            if template_id.startswith("mcqa_benchmark_001"):
                correct = f"Answer: \\boxed{{{label}}}"
            else:
                return None
        else:
            correct = f"Answer: {label}"
        return correct, "Answer: definitely-wrong"
    if verifier == "reasoning_gym":
        return (f"Answer: {label}", "Answer: definitely-wrong") if label not in (None, "") else None
    if verifier == "python_code":
        return build_code_probes(label, metadata)
    if verifier == "expert_action":
        signature = expected_action_signature((metadata or {}).get("expected_action"))
        if signature is None:
            return None
        if signature["kind"] == "message":
            return "No tool is required.", '<tool_call>{"name":"wrong","arguments":{}}</tool_call>'
        call = json.dumps({"name": signature["name"], "arguments": signature["arguments"]})
        return f"<tool_call>{call}</tool_call>", "No tool is required."
    if verifier == "calendar_constraints":
        expected = (metadata or {}).get("expected_calendar_state")
        if not isinstance(expected, dict) or not expected:
            return None
        return build_calendar_solution(expected), "[]"
    if verifier == "workplace_environment":
        expected = (metadata or {}).get("expected_actions")
        if not isinstance(expected, list):
            return None
        if not expected:
            return "No tool is required.", '<tool_call>{"name":"wrong","arguments":{}}</tool_call>'
        calls = [
            f"<tool_call>{json.dumps(action, ensure_ascii=False, separators=(',', ':'))}</tool_call>"
            for action in expected
        ]
        return "".join(calls), "No tool is required."
    return None
