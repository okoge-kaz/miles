"""Evaluate a checkpoint in pinned Tau v1 with a local or Gemini user simulator."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from experiments.src.datasets.common.io import read_rows
from experiments.src.environments.tau_bench.compat import install_litellm_import_stub
from experiments.src.environments.tau_bench.task_identity import validate_task_identity
from experiments.src.environments.tau_bench.user_simulator import (
    DEFAULT_GEMINI_MODEL,
    STOP_MARKER,
    GeminiRequestError,
    build_user_system_prompt,
    generate_gemini_user,
    require_gemini_api_key,
)


@dataclass(frozen=True)
class EvaluationResult:
    row_index: int
    repeat: int
    environment: str
    split: str
    task_index: int
    reward: float
    done: bool
    turns: int
    agent_tokens: int
    user_tokens: int
    error: str | None


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def _tools_for_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            converted.append(tool)
        else:
            converted.append({"type": "function", "function": tool})
    return converted


async def _completion(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = True,
) -> tuple[dict[str, Any], int]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    async with semaphore:
        async with session.post(endpoint, json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
    parsed = json.loads(body)
    message = parsed["choices"][0]["message"]
    completion_tokens = int((parsed.get("usage") or {}).get("completion_tokens") or 0)
    return message, completion_tokens


async def _gemini_post(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    async with semaphore:
        async with session.post(url, json=payload, headers=headers) as response:
            if not 200 <= response.status < 300:
                raise GeminiRequestError("Gemini API returned an HTTP error", status_code=response.status)
            try:
                parsed = await response.json(content_type=None)
            except (aiohttp.ClientError, ValueError) as error:
                raise GeminiRequestError(
                    "Gemini API returned a non-JSON response",
                    status_code=response.status,
                ) from error
    if not isinstance(parsed, dict):
        raise GeminiRequestError("Gemini API returned a non-object JSON response")
    return parsed


async def _user_completion(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    messages: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[str, int]:
    if args.user_backend == "local-policy":
        message, used = await _completion(
            session,
            semaphore,
            endpoint=args.endpoint,
            model=args.model,
            messages=messages,
            max_tokens=args.user_max_tokens,
            seed=seed,
            temperature=args.user_temperature,
            top_p=args.user_top_p,
            enable_thinking=False,
        )
        text = str(message.get("content") or "").strip()
        if not text:
            raise RuntimeError("local user produced an empty message")
        return (STOP_MARKER if STOP_MARKER in text else text), used

    async def post_json(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        return await _gemini_post(session, semaphore, url, payload, headers)

    result = await generate_gemini_user(
        messages,
        post_json=post_json,
        model=args.user_model,
        max_output_tokens=args.user_max_tokens,
        temperature=args.user_temperature,
        top_p=args.user_top_p,
        seed=seed,
        request_timeout=args.user_request_timeout,
        max_retries=args.user_max_retries,
        retry_backoff=args.user_retry_backoff,
    )
    return result.text, result.output_tokens


def _load_environment(metadata: dict[str, Any]) -> Any:
    install_litellm_import_stub()
    from tau_bench.envs import get_env

    env_name = str(metadata["tau_env"])
    split = str(metadata["tau_split"])
    task_index = int(metadata["tau_task_index"])
    environment = get_env(
        env_name=env_name,
        user_strategy="human",
        user_model="local-policy",
        task_split=split,
        task_index=task_index,
    )
    validate_task_identity(metadata, environment.tasks[task_index])
    environment.task_index = task_index
    environment.task = environment.tasks[task_index]
    environment.data = environment.data_load_func()
    environment.actions = []
    return environment


def _agent_messages(environment: Any, user_message: str) -> list[dict[str, Any]]:
    policy = str(environment.wiki)
    if environment.rules:
        policy += "\n\nRules:\n" + "\n".join(f"- {rule}" for rule in environment.rules)
    return [{"role": "system", "content": policy}, {"role": "user", "content": user_message}]


async def _evaluate_one(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    row_index: int,
    repeat: int,
) -> EvaluationResult:
    install_litellm_import_stub()
    from tau_bench.types import Action, RESPOND_ACTION_NAME

    metadata = row.get("metadata") or {}
    env_name = str(metadata.get("tau_env") or "")
    split = str(metadata.get("tau_split") or "")
    task_index = int(metadata.get("tau_task_index") or 0)
    seed = args.seed + row_index * args.repeats + repeat
    turns = 0
    agent_tokens = 0
    user_tokens = 0
    try:
        environment = _load_environment(metadata)
        user_messages = [
            {"role": "system", "content": build_user_system_prompt(environment.task.instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        initial_user, used = await _user_completion(
            args,
            session,
            semaphore,
            user_messages,
            seed=seed,
        )
        user_tokens += used
        user_messages.append({"role": "assistant", "content": initial_user})
        agent_messages = _agent_messages(environment, initial_user)
        tools = _tools_for_openai(environment.tools_info)

        for turn in range(args.max_turns):
            turns = turn + 1
            remaining = args.max_response_tokens - agent_tokens
            if remaining <= 0:
                break
            message, used = await _completion(
                session,
                semaphore,
                endpoint=args.endpoint,
                model=args.model,
                messages=agent_messages,
                max_tokens=remaining,
                seed=seed + 1000 + turn,
                temperature=args.agent_temperature,
                top_p=args.agent_top_p,
                tools=tools,
            )
            agent_tokens += used
            tool_calls = message.get("tool_calls") or []
            content = str(message.get("content") or "")
            agent_messages.append(message)
            if tool_calls:
                call = tool_calls[0]
                function = call.get("function") or {}
                arguments = json.loads(function.get("arguments") or "{}")
                action = Action(name=str(function.get("name") or ""), kwargs=arguments)
                result = environment.step(action)
                agent_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or f"tau-{turn}"),
                        "name": action.name,
                        "content": str(result.observation),
                    }
                )
                if result.done:
                    return EvaluationResult(
                        row_index, repeat, env_name, split, task_index, float(result.reward), True,
                        turns, agent_tokens, user_tokens, None,
                    )
                continue

            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
            environment.actions.append(action)
            user_messages.append({"role": "user", "content": content})
            observation, used = await _user_completion(
                args,
                session,
                semaphore,
                user_messages,
                seed=seed + 2000 + turn,
            )
            user_tokens += used
            user_messages.append({"role": "assistant", "content": observation})
            if STOP_MARKER in observation:
                reward = environment.calculate_reward()
                return EvaluationResult(
                    row_index, repeat, env_name, split, task_index, float(reward.reward), True,
                    turns, agent_tokens, user_tokens, None,
                )
            agent_messages.append({"role": "user", "content": observation})
        return EvaluationResult(
            row_index, repeat, env_name, split, task_index, 0.0, False,
            turns, agent_tokens, user_tokens, "turn or response-token limit reached",
        )
    except (
        aiohttp.ClientError,
        asyncio.TimeoutError,
        KeyError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        return EvaluationResult(
            row_index, repeat, env_name, split, task_index, 0.0, False,
            turns, agent_tokens, user_tokens, f"{type(exc).__name__}: {exc}",
        )


async def evaluate(args: argparse.Namespace) -> None:
    if args.user_backend == "gemini":
        require_gemini_api_key()
    rows = list(read_rows([args.input]))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Tau evaluation input contains no rows")
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _evaluate_one(args, session, semaphore, row, index, repeat)
            for index, row in enumerate(rows)
            for repeat in range(args.repeats)
        ]
        results = await asyncio.gather(*tasks)

    output = args.output
    partial = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(partial, output)

    rewards = [result.reward for result in results]
    done = sum(result.done for result in results)
    error_count = sum(result.error is not None for result in results)
    summary = {
        "task": "tau_bench_v1",
        "input": str(args.input),
        "samples": len(results),
        "prompts": len(rows),
        "repeats": args.repeats,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "terminal_rate": done / len(results) if results else 0.0,
        "errors": error_count,
        "error_rate": error_count / len(results),
        "user_backend": args.user_backend,
        "user_model": args.model if args.user_backend == "local-policy" else args.user_model,
        "external_model_api": args.user_backend == "gemini",
        "max_response_tokens": args.max_response_tokens,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if error_count == len(results):
        raise RuntimeError("all Tau evaluation episodes failed; refusing to mark the evaluation successful")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--user-backend", choices=("local-policy", "gemini"), default="local-policy")
    parser.add_argument("--user-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--user-max-tokens", type=int, default=512)
    parser.add_argument("--agent-temperature", type=float, default=0.6)
    parser.add_argument("--agent-top-p", type=float, default=0.95)
    parser.add_argument("--user-temperature", type=float, default=0.7)
    parser.add_argument("--user-top-p", type=float, default=0.95)
    parser.add_argument("--user-request-timeout", type=float, default=120.0)
    parser.add_argument("--user-max-retries", type=int, default=4)
    parser.add_argument("--user-retry-backoff", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--request-timeout", type=int, default=3600)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
