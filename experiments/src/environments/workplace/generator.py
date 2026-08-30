"""Stateful, single-turn, multi-step Workplace Assistant rollout generation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from typing import Any

from experiments.src.environments.common.observations import append_tool_observation
from experiments.src.environments.workplace.runtime import create_tool_environment, execute_action
from experiments.src.environments.workplace.verifier import score_action_trajectory
from experiments.src.protocols.workplace import WORKPLACE_INTERACTION_MODE
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _parse_calls(parser: Any, response: str) -> list[dict[str, Any]]:
    _, calls = parser.parse_non_stream(response)
    parsed = []
    for call in calls:
        arguments = json.loads(call.parameters or "{}")
        if not isinstance(arguments, dict):
            raise ValueError(f"Workplace arguments for {call.name!r} are not an object")
        parsed.append({"name": call.name, "arguments": arguments})
    return parsed


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Run tools in an isolated local environment and assign state-match reward."""

    args = input.args
    if args.partial_rollout:
        raise ValueError("Workplace Assistant does not support partial rollout")
    sample = deepcopy(input.sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    tools = metadata.get("tools")
    expected = metadata.get("expected_actions")
    if metadata.get("verifier") != "workplace_environment" or not isinstance(tools, list):
        raise ValueError("Workplace generator received a non-Workplace sample")
    if metadata.get("interaction_mode") != WORKPLACE_INTERACTION_MODE:
        raise ValueError("Workplace generator requires single-turn multi-step samples")
    if metadata.get("stateful_environment") is not True:
        raise ValueError("Workplace generator requires environment state")
    if not isinstance(expected, list):
        raise ValueError("Workplace sample has no expected action trajectory")

    environment = create_tool_environment()
    parser = create_tool_call_parser(tools, args.workplace_tool_call_parser)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    executed: list[dict[str, Any]] = []
    model_steps = 0
    sample.metadata = {**metadata, "workplace_executed_actions": executed}

    for step in range(args.workplace_max_steps):
        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            break
        output = await post(url, payload, headers=compute_routing_headers(args, sample))
        model_steps = step + 1
        await update_sample_from_response(args, sample, payload, output, update_loss_mask=True)
        finish_type = output.get("meta_info", {}).get("finish_reason", {}).get("type")
        if finish_type in {"abort", "length"}:
            sample.reward = 0.0
            break
        try:
            calls = _parse_calls(parser, str(output.get("text") or ""))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Workplace tool parse failed at step %d: %s", step + 1, exc)
            sample.status = Sample.Status.FAILED
            sample.reward = 0.0
            break
        if not calls:
            sample.status = Sample.Status.COMPLETED
            break

        started = time.monotonic()
        for call_index, call in enumerate(calls):
            executed.append(call)
            observation = execute_action(environment, call["name"], call["arguments"])
            tool_message = {
                "role": "tool",
                "name": call["name"],
                "content": json.dumps(observation, ensure_ascii=False, default=str),
                "tool_call_id": f"workplace_{step}_{call_index}_{call['name']}",
            }
            if not append_tool_observation(
                sample,
                input.state.tokenizer,
                tool_message,
                args.rollout_max_response_len,
            ):
                sample.reward = 0.0
                break
        sample.non_generation_time += time.monotonic() - started
        sample.metadata["workplace_executed_actions"] = list(executed)
        if sample.status == Sample.Status.TRUNCATED:
            break

    if sample.reward is None:
        sample.reward = score_action_trajectory(executed, expected)
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    sample.metadata["workplace_steps"] = model_steps
    sample.metadata["workplace_model_steps"] = model_steps
    sample.metadata["workplace_tool_calls"] = len(executed)
    sample.metadata["workplace_state_match"] = sample.reward == 1.0
    sample.validate()
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workplace-max-steps",
        type=int,
        default=6,
        help="Maximum model/tool steps in the single user request.",
    )
    parser.add_argument("--workplace-tool-call-parser", type=str, default="qwen25")


generate.add_arguments = _add_arguments
