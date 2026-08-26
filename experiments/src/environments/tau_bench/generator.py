"""Mixed Nemotron single-turn and stateful Tau Bench rollout generation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import deepcopy
from typing import Any

from experiments.src.environments.tau_bench.compat import install_litellm_import_stub
from experiments.src.environments.tau_bench.task_identity import validate_task_identity
from experiments.src.environments.tau_bench.user_simulator import (
    DEFAULT_GEMINI_MODEL,
    STOP_MARKER,
    build_user_system_prompt,
    generate_gemini_user,
    require_gemini_api_key,
)
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as generate_single_turn
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser, tokenize_tool_responses
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

TAU_VERIFIER = "tau_bench_environment"


def _tokenize_user_observation(tokenizer: Any, content: str) -> list[int]:
    base = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "acknowledged"},
    ]
    with_user = tokenizer.apply_chat_template(
        [*base, {"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    without_user = tokenizer.apply_chat_template(
        base,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    if with_user[: len(without_user)] != without_user:
        mismatch = next(
            (
                index
                for index, (with_token, without_token) in enumerate(zip(with_user, without_user, strict=False))
                if with_token != without_token
            ),
            min(len(with_user), len(without_user)),
        )
        start = max(0, mismatch - 8)
        stop = mismatch + 8
        raise RuntimeError(
            "Tau user observation is not append-only under this chat template: "
            f"mismatch={mismatch}, "
            f"with={tokenizer.decode(with_user[start:stop])!r}, "
            f"without={tokenizer.decode(without_user[start:stop])!r}"
        )
    return list(with_user[len(without_user) :])


def _append_loss_masked_tokens(
    sample: Sample,
    tokenizer: Any,
    token_ids: list[int],
    max_response_len: int,
) -> bool:
    remaining = max_response_len - sample.response_length
    if len(token_ids) > remaining:
        sample.status = Sample.Status.TRUNCATED
        return False
    sample.tokens.extend(token_ids)
    sample.response += tokenizer.decode(token_ids)
    sample.response_length += len(token_ids)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask.extend([0] * len(token_ids))
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs.extend([0.0] * len(token_ids))
    return True


def append_user_observation(sample: Sample, tokenizer: Any, content: str, max_response_len: int) -> bool:
    """Append a loss-masked user turn and return whether it fit the budget."""

    token_ids = _tokenize_user_observation(tokenizer, content)
    return _append_loss_masked_tokens(sample, tokenizer, token_ids, max_response_len)


def append_tool_observation(
    sample: Sample,
    tokenizer: Any,
    tool_message: dict[str, Any],
    max_response_len: int,
) -> bool:
    """Append one loss-masked tool result without exceeding the rollout budget."""

    token_ids = tokenize_tool_responses([tool_message], tokenizer=tokenizer)
    return _append_loss_masked_tokens(sample, tokenizer, token_ids, max_response_len)


def _user_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": args.tau_user_max_tokens,
        "temperature": args.tau_user_temperature,
        "top_p": args.tau_user_top_p,
        "top_k": -1,
    }


def _extract_local_user_response(output: dict[str, Any], sample: Sample) -> str:
    finish_type = output.get("meta_info", {}).get("finish_reason", {}).get("type")
    if finish_type == "abort":
        raise RuntimeError("Tau local user generation was aborted")
    response = str(output.get("text") or "").strip()
    if not response:
        raise RuntimeError("Tau local user generated an empty response")
    if finish_type == "length":
        logger.warning("Tau local user reached its token limit; continuing with the non-empty partial message")
        truncations = int(sample.metadata.get("tau_user_length_truncations", 0))
        sample.metadata["tau_user_length_truncations"] = truncations + 1
    return STOP_MARKER if STOP_MARKER in response else response


async def _generate_local_user(
    input: GenerateFnInput,
    url: str,
    messages: list[dict[str, str]],
    sample: Sample,
) -> str:
    tokenizer = input.state.tokenizer
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    payload, halt_status = compute_request_payload(input.args, prompt_ids, _user_sampling_params(input.args))
    if payload is None:
        raise RuntimeError(f"Tau local user prompt exceeded the context limit: {halt_status}")
    payload["return_logprob"] = False
    payload["return_routed_experts"] = False
    payload["return_indexer_topk"] = False
    output = await post(url, payload, headers=compute_routing_headers(input.args, sample))
    return _extract_local_user_response(output, sample)


async def _post_gemini(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    response = await post(url, payload, max_retries=1, headers=headers)
    if not isinstance(response, dict):
        raise RuntimeError("Gemini returned a non-object JSON response")
    return response


def _gemini_seed(input: GenerateFnInput, user_turn: int) -> int:
    sample_index = input.sample.index if input.sample.index is not None else 0
    return int(input.args.rollout_seed) + sample_index * (input.args.tau_max_turns + 1) + user_turn


async def _generate_gemini_user(
    input: GenerateFnInput,
    messages: list[dict[str, str]],
    sample: Sample,
    *,
    user_turn: int,
) -> str:
    result = await generate_gemini_user(
        messages,
        post_json=_post_gemini,
        model=input.args.tau_user_model,
        max_output_tokens=input.args.tau_user_max_tokens,
        temperature=input.args.tau_user_temperature,
        top_p=input.args.tau_user_top_p,
        seed=_gemini_seed(input, user_turn),
        request_timeout=input.args.tau_user_request_timeout,
        max_retries=input.args.tau_user_max_retries,
        retry_backoff=input.args.tau_user_retry_backoff,
    )
    if result.finish_reason == "MAX_TOKENS":
        logger.warning("Tau Gemini user reached its token limit; continuing with the non-empty partial message")
        truncations = int(sample.metadata.get("tau_user_length_truncations", 0))
        sample.metadata["tau_user_length_truncations"] = truncations + 1
    return result.text


async def _generate_user(
    input: GenerateFnInput,
    url: str,
    messages: list[dict[str, str]],
    sample: Sample,
    *,
    user_turn: int,
) -> str:
    if input.args.tau_user_backend == "local-policy":
        return await _generate_local_user(input, url, messages, sample)
    if input.args.tau_user_backend == "gemini":
        return await _generate_gemini_user(input, messages, sample, user_turn=user_turn)
    raise ValueError(f"unsupported Tau user backend {input.args.tau_user_backend!r}")


def _build_action(parser: Any, response: str) -> Any:
    install_litellm_import_stub()
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    normal_text, calls = parser.parse_non_stream(response)
    if not calls:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": normal_text.strip()})
    if len(calls) > 1:
        logger.warning("Tau allows one tool call per turn; using the first of %d", len(calls))
    call = calls[0]
    parameters = json.loads(call.parameters or "{}")
    if not isinstance(parameters, dict):
        raise ValueError(f"Tau tool arguments for {call.name!r} are not an object")
    return Action(name=call.name, kwargs=parameters)


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def _load_environment(metadata: dict[str, Any]) -> Any:
    install_litellm_import_stub()
    from tau_bench.envs import get_env

    env_name = str(metadata.get("tau_env") or "")
    split = str(metadata.get("tau_split") or "")
    task_index = int(metadata.get("tau_task_index"))
    if env_name not in {"retail", "airline"}:
        raise ValueError(f"unsupported Tau environment {env_name!r}")
    env = get_env(
        env_name=env_name,
        user_strategy="human",
        user_model="local-policy",
        task_split=split,
        task_index=task_index,
    )
    if not 0 <= task_index < len(env.tasks):
        raise ValueError(f"Tau task index {task_index} is outside the pinned {split} split")
    validate_task_identity(metadata, env.tasks[task_index])
    if split == "train" and metadata.get("tau_reward_verified") is not True:
        raise ValueError(f"Tau training task {task_index} has not passed reward verification")
    return env


def _initial_agent_messages(env: Any, user_message: str) -> list[dict[str, str]]:
    policy = str(env.wiki)
    if env.rules:
        policy += "\n\nRules:\n" + "\n".join(f"- {rule}" for rule in env.rules)
    return [{"role": "system", "content": policy}, {"role": "user", "content": user_message}]


async def _generate_tau(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    if args.partial_rollout:
        raise ValueError("Tau Bench does not support partial rollout")
    if args.tau_user_backend == "gemini":
        require_gemini_api_key()

    sample = deepcopy(input.sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env = _load_environment(metadata)
    task_index = int(metadata["tau_task_index"])
    env.task_index = task_index
    env.task = env.tasks[task_index]
    env.data = env.data_load_func()
    env.actions = []

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    user_messages = [
        {"role": "system", "content": build_user_system_prompt(env.task.instruction)},
        {"role": "user", "content": "Hi! How can I help you today?"},
    ]
    try:
        initial_user = await _generate_user(input, url, user_messages, sample, user_turn=0)
    except RuntimeError as exc:
        logger.warning("Tau initial %s user failed; returning the sample for retry: %s", args.tau_user_backend, exc)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)
    user_messages.append({"role": "assistant", "content": initial_user})

    agent_messages = _initial_agent_messages(env, initial_user)
    tokenizer = input.state.tokenizer
    prompt_ids = tokenizer.apply_chat_template(
        agent_messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=env.tools_info,
        return_dict=False,
    )
    sample.prompt = agent_messages
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.reward = None
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.status = Sample.Status.PENDING
    sample.metadata = {
        **metadata,
        "messages": list(agent_messages),
        "tau_user_backend": args.tau_user_backend,
        "tau_user_model": args.tau_user_model if args.tau_user_backend == "gemini" else "rollout-policy",
    }

    parser = create_tool_call_parser(env.tools_info, args.tau_tool_call_parser)
    reward_info: dict[str, Any] | None = None
    done = False
    for turn in range(args.tau_max_turns):
        remaining = args.rollout_max_response_len - sample.response_length
        if remaining <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        sampling_params = {**input.sampling_params, "max_new_tokens": remaining}
        payload, halt_status = compute_request_payload(args, sample.tokens, sampling_params)
        if payload is None:
            sample.status = halt_status
            break
        output = await post(url, payload, headers=compute_routing_headers(args, sample))
        await update_sample_from_response(args, sample, payload, output, update_loss_mask=True)
        agent_text = str(output.get("text") or "")
        agent_messages.append({"role": "assistant", "content": agent_text})
        sample.metadata["messages"] = list(agent_messages)
        finish_type = output.get("meta_info", {}).get("finish_reason", {}).get("type")
        if finish_type in {"abort", "length"}:
            sample.reward = 0.0
            break

        try:
            action = _build_action(parser, agent_text)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Tau action parse failed on turn %d: %s", turn + 1, exc)
            sample.status = Sample.Status.FAILED
            sample.reward = 0.0
            break

        step_started = time.monotonic()
        if action.name != "respond":
            env_response = env.step(action)
            observation = str(env_response.observation)
            tool_message = {
                "role": "tool",
                "name": action.name,
                "content": observation,
                "tool_call_id": f"tau_{turn}_{action.name}",
            }
            if append_tool_observation(sample, tokenizer, tool_message, args.rollout_max_response_len):
                agent_messages.append(tool_message)
                sample.metadata["messages"] = list(agent_messages)
                done = bool(env_response.done)
                if done:
                    sample.reward = float(env_response.reward)
                    reward_info = _model_dump(env_response.info)
            else:
                sample.reward = 0.0
                sample.non_generation_time += time.monotonic() - step_started
                break
        else:
            env.actions.append(action)
            user_messages.append({"role": "user", "content": str(action.kwargs.get("content") or "")})
            try:
                observation = await _generate_user(
                    input,
                    url,
                    user_messages,
                    sample,
                    user_turn=turn + 1,
                )
            except RuntimeError as exc:
                logger.warning("Tau %s user failed on turn %d: %s", args.tau_user_backend, turn + 1, exc)
                sample.status = Sample.Status.FAILED
                sample.reward = 0.0
                break
            user_messages.append({"role": "assistant", "content": observation})
            done = STOP_MARKER in observation
            if done:
                official_reward = env.calculate_reward()
                sample.reward = float(official_reward.reward)
                reward_info = _model_dump(official_reward)
            elif append_user_observation(sample, tokenizer, observation, args.rollout_max_response_len):
                agent_messages.append({"role": "user", "content": observation})
                sample.metadata["messages"] = list(agent_messages)
            else:
                sample.reward = 0.0
        sample.non_generation_time += time.monotonic() - step_started
        if done:
            sample.status = Sample.Status.COMPLETED
            break

    if not done and sample.reward is None:
        sample.status = Sample.Status.TRUNCATED
        sample.reward = 0.0
    sample.metadata["tau_turns"] = sum(message["role"] == "assistant" for message in agent_messages)
    sample.metadata["tau_reward_info"] = reward_info
    sample.metadata["tau_done"] = done
    sample.validate()
    return GenerateFnOutput(samples=sample)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Dispatch Tau rows to the environment loop and Nemotron rows to Miles."""

    metadata = input.sample.metadata if isinstance(input.sample.metadata, dict) else {}
    if metadata.get("verifier") == TAU_VERIFIER:
        return await _generate_tau(input)
    return await generate_single_turn(input)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tau-max-turns", type=int, default=30)
    parser.add_argument("--tau-user-backend", choices=("local-policy", "gemini"), default="local-policy")
    parser.add_argument("--tau-user-model", type=str, default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--tau-user-max-tokens", type=int, default=512)
    parser.add_argument("--tau-user-temperature", type=float, default=0.7)
    parser.add_argument("--tau-user-top-p", type=float, default=0.95)
    parser.add_argument("--tau-user-request-timeout", type=float, default=120.0)
    parser.add_argument("--tau-user-max-retries", type=int, default=4)
    parser.add_argument("--tau-user-retry-backoff", type=float, default=1.0)
    parser.add_argument("--tau-tool-call-parser", type=str, default="qwen25")


generate.add_arguments = _add_arguments
