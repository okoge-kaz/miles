"""Evaluate a checkpoint on pinned Tau three tasks with the official Gym reward."""

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
from experiments.src.environments.tau_bench.runtime import (
    DEFAULT_NVIDIA_MODEL,
    TauSession,
    TauUserConfig,
)
from experiments.src.environments.tau_bench.task_identity import TAU_RELEASE, TAU_VERIFIER


@dataclass(frozen=True)
class EvaluationResult:
    """One complete or failed Tau three episode."""

    row_index: int
    repeat: int
    domain: str
    split: str
    task_id: str
    reward: float
    terminated: bool
    turns: int
    agent_tokens: int
    termination: str
    reward_info: dict[str, Any] | None
    simulation_run: dict[str, Any] | None
    error: str | None


async def _completion(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    *,
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    seed: int,
) -> tuple[dict[str, Any], int]:
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tokens,
        "temperature": args.agent_temperature,
        "top_p": args.agent_top_p,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": args.agent_enable_thinking},
    }
    async with semaphore:
        async with session.post(args.endpoint, json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"agent endpoint returned HTTP {response.status}: {body[:300]}")
    parsed = json.loads(body)
    choice = parsed["choices"][0]
    message = choice["message"]
    message["_miles_finish_reason"] = choice.get("finish_reason")
    used = int((parsed.get("usage") or {}).get("completion_tokens") or 0)
    return message, used


def _tau_action(message: dict[str, Any], turn: int) -> str:
    finish_reason = message.pop("_miles_finish_reason", None)
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        function = call.get("function") or {}
        arguments = json.loads(function.get("arguments") or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("Tau three agent tool arguments must be a JSON object")
        return json.dumps(
            {
                "id": str(call.get("id") or f"tau3-eval-{turn}"),
                "name": str(function.get("name") or ""),
                "arguments": arguments,
                "requestor": "assistant",
            },
            separators=(",", ":"),
        )
    content = str(message.get("content") or "").strip()
    if content:
        return content
    reasoning = bool(message.get("reasoning") or message.get("reasoning_content"))
    detail = f"finish_reason={finish_reason}, reasoning_only={reasoning}"
    raise RuntimeError(f"Tau three agent produced no final text or tool call ({detail})")


async def _agent_action(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    *,
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    remaining_tokens: int,
    seed: int,
    turn: int,
) -> tuple[dict[str, Any], str, int]:
    used_total = 0
    for attempt in range(args.agent_empty_response_retries + 1):
        remaining = remaining_tokens - used_total
        if remaining <= 0:
            break
        message, used = await _completion(
            session,
            semaphore,
            args=args,
            messages=messages,
            tools=tools,
            max_tokens=remaining,
            seed=seed + attempt,
        )
        used_total += used
        try:
            action = _tau_action(message, turn)
        except RuntimeError:
            if attempt == args.agent_empty_response_retries:
                raise
            continue
        return message, action, used_total
    raise RuntimeError("Tau three agent exhausted its response-token budget")


def _user_config(args: argparse.Namespace, seed: int) -> TauUserConfig:
    return TauUserConfig(
        provider=args.user_provider,
        model=args.user_model,
        max_tokens=args.user_max_tokens,
        temperature=args.user_temperature,
        top_p=args.user_top_p,
        timeout=args.user_request_timeout,
        retries=args.user_max_retries,
        seed=seed,
    )


def _result(
    metadata: dict[str, Any],
    *,
    row_index: int,
    repeat: int,
    reward: float = 0.0,
    terminated: bool = False,
    turns: int = 0,
    agent_tokens: int = 0,
    termination: str,
    reward_info: dict[str, Any] | None = None,
    simulation_run: dict[str, Any] | None = None,
    error: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        row_index=row_index,
        repeat=repeat,
        domain=str(metadata.get("tau_domain") or ""),
        split=str(metadata.get("tau_split") or ""),
        task_id=str(metadata.get("tau_task_id") or ""),
        reward=reward,
        terminated=terminated,
        turns=turns,
        agent_tokens=agent_tokens,
        termination=termination,
        reward_info=reward_info,
        simulation_run=simulation_run,
        error=error,
    )


async def _evaluate_one(
    args: argparse.Namespace,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    row_index: int,
    repeat: int,
) -> EvaluationResult:
    metadata = row.get("metadata") or {}
    seed = args.seed + row_index * args.repeats + repeat
    environment: TauSession | None = None
    turns = 0
    agent_tokens = 0
    try:
        if metadata.get("verifier") != TAU_VERIFIER:
            raise ValueError(f"input row does not use {TAU_VERIFIER}")
        if metadata.get("tau_split") != "test" or metadata.get("eval_only") is not True:
            raise ValueError("Tau downstream evaluation accepts only held-out test rows")
        environment = TauSession(metadata, _user_config(args, seed), max_steps=args.max_steps)
        reset = await asyncio.to_thread(environment.reset)
        messages = [{"role": "system", "content": reset.system_prompt}, *reset.messages]

        for turn in range(args.max_turns):
            remaining = args.max_response_tokens - agent_tokens
            if remaining <= 0:
                break
            message, action, used = await _agent_action(
                http_session,
                semaphore,
                args=args,
                messages=messages,
                tools=reset.tools,
                remaining_tokens=remaining,
                seed=seed + 1000 + turn,
                turn=turn,
            )
            turns = turn + 1
            agent_tokens += used
            messages.append(message)
            step = await asyncio.to_thread(environment.step, action)
            messages.extend(step.observations)
            if step.terminated or step.truncated:
                return _result(
                    metadata,
                    row_index=row_index,
                    repeat=repeat,
                    reward=step.reward,
                    terminated=step.terminated,
                    turns=turns,
                    agent_tokens=agent_tokens,
                    termination="environment" if step.terminated else "environment_limit",
                    reward_info=step.reward_info,
                    simulation_run=step.simulation_run,
                )
        return _result(
            metadata,
            row_index=row_index,
            repeat=repeat,
            turns=turns,
            agent_tokens=agent_tokens,
            termination="agent_limit",
        )
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError, RuntimeError) as error:
        return _result(
            metadata,
            row_index=row_index,
            repeat=repeat,
            turns=turns,
            agent_tokens=agent_tokens,
            termination="error",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        if environment is not None:
            try:
                await asyncio.to_thread(environment.close)
            except Exception:
                pass


def _atomic_write_jsonl(path: Path, results: list[EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(partial, path)


def _summary(args: argparse.Namespace, rows: list[dict[str, Any]], results: list[EvaluationResult]) -> dict[str, Any]:
    rewards = [result.reward for result in results]
    successes = sum(reward == 1.0 for reward in rewards)
    return {
        "task": "tau3",
        "tau_release": TAU_RELEASE,
        "input": str(args.input),
        "split": sorted({result.split for result in results}),
        "prompts": len(rows),
        "samples": len(results),
        "repeats": args.repeats,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "terminal_rate": sum(result.terminated for result in results) / len(results) if results else 0.0,
        "errors": sum(result.error is not None for result in results),
        "user_provider": args.user_provider,
        "user_model": args.user_model,
        "agent_enable_thinking": args.agent_enable_thinking,
        "evaluation_policy": "held-out Tau three test only; Tau train/base are prohibited",
    }


async def evaluate(args: argparse.Namespace) -> None:
    rows = list(read_rows([args.input]))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Tau three evaluation input contains no rows")

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as http_session:
        results = await asyncio.gather(
            *(
                _evaluate_one(args, http_session, semaphore, row, index, repeat)
                for index, row in enumerate(rows)
                for repeat in range(args.repeats)
            )
        )
    _atomic_write_jsonl(args.output, results)
    summary = _summary(args, rows, results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    partial = args.summary.with_suffix(args.summary.suffix + ".partial")
    partial.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument("--agent-top-p", type=float, default=1.0)
    parser.add_argument("--agent-empty-response-retries", type=int, default=2)
    parser.add_argument(
        "--agent-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--user-provider", choices=("gemini", "nvidia"), default="nvidia")
    parser.add_argument("--user-model", default=DEFAULT_NVIDIA_MODEL)
    parser.add_argument("--user-max-tokens", type=int, default=512)
    parser.add_argument("--user-temperature", type=float, default=0.7)
    parser.add_argument("--user-top-p", type=float, default=0.95)
    parser.add_argument("--user-request-timeout", type=float, default=120.0)
    parser.add_argument("--user-max-retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    main()
