"""Evaluate exact next-action tool calls against a held-out Nemotron split."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from experiments.src.datasets.common.io import read_rows
from experiments.src.environments.tool_call.verifier import (
    arguments_match,
    normalize_arguments,
    parse_tool_calls,
)
from experiments.src.protocols.openai_responses import expected_action_signature


@dataclass(frozen=True)
class EvaluationResult:
    row_index: int
    source: str
    fingerprint: str
    exact: bool
    name_correct: bool
    arguments_correct: bool
    call_count: int
    expected_name: str
    actual_names: list[str]
    completion_tokens: int
    error: str | None


def _calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function") or {}
        if function.get("name"):
            calls.append(
                {
                    "name": str(function["name"]),
                    "arguments": normalize_arguments(function.get("arguments")),
                }
            )
    if calls:
        return calls
    return parse_tool_calls(str(message.get("content") or ""))


async def _completion(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], int]:
    payload = {
        "model": args.model,
        "messages": row["prompt"],
        "tools": row["tools"],
        "tool_choice": "auto",
        "max_tokens": args.max_response_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    last_error = "request was not attempted"
    for attempt in range(args.retries + 1):
        try:
            async with semaphore:
                async with session.post(args.endpoint, json=payload) as response:
                    body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
            parsed = json.loads(body)
            message = parsed["choices"][0]["message"]
            tokens = int((parsed.get("usage") or {}).get("completion_tokens") or 0)
            return message, tokens
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(last_error)


async def _evaluate_one(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    row_index: int,
) -> EvaluationResult:
    metadata = row.get("metadata") or {}
    signature = expected_action_signature(metadata.get("expected_action"))
    if signature is None or signature.get("kind") != "function_call":
        raise ValueError(f"evaluation row {row_index} is not an exact function-call action")
    expected_name = str(signature["name"])
    try:
        message, completion_tokens = await _completion(
            args, session, semaphore, row, args.seed + row_index
        )
        calls = _calls_from_message(message)
        name_correct = len(calls) == 1 and calls[0]["name"] == expected_name
        arguments_correct = name_correct and arguments_match(
            signature["arguments"], calls[0]["arguments"]
        )
        return EvaluationResult(
            row_index=row_index,
            source=str(metadata.get("source") or "unknown"),
            fingerprint=str(metadata.get("split_fingerprint") or ""),
            exact=arguments_correct,
            name_correct=name_correct,
            arguments_correct=arguments_correct,
            call_count=len(calls),
            expected_name=expected_name,
            actual_names=[call["name"] for call in calls],
            completion_tokens=completion_tokens,
            error=None,
        )
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError, RuntimeError) as exc:
        return EvaluationResult(
            row_index=row_index,
            source=str(metadata.get("source") or "unknown"),
            fingerprint=str(metadata.get("split_fingerprint") or ""),
            exact=False,
            name_correct=False,
            arguments_correct=False,
            call_count=0,
            expected_name=expected_name,
            actual_names=[],
            completion_tokens=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _metrics(results: list[EvaluationResult]) -> dict[str, Any]:
    def aggregate(items: list[EvaluationResult]) -> dict[str, Any]:
        total = len(items)
        return {
            "samples": total,
            "exact_action_accuracy": sum(item.exact for item in items) / total if total else 0.0,
            "tool_name_accuracy": sum(item.name_correct for item in items) / total if total else 0.0,
            "arguments_accuracy": sum(item.arguments_correct for item in items) / total if total else 0.0,
            "single_call_rate": sum(item.call_count == 1 for item in items) / total if total else 0.0,
            "no_call_rate": sum(item.call_count == 0 for item in items) / total if total else 0.0,
            "error_rate": sum(item.error is not None for item in items) / total if total else 0.0,
            "completion_tokens": sum(item.completion_tokens for item in items),
        }

    by_source: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        by_source[result.source].append(result)
    return {
        "task": "nemotron_exact_next_tool_action",
        "verifier": "exact tool name + exact argument keys/values + exactly one call",
        "overall": aggregate(results),
        "by_source": {source: aggregate(items) for source, items in sorted(by_source.items())},
    }


async def evaluate(args: argparse.Namespace) -> None:
    rows = list(read_rows([args.input]))
    if args.limit is not None:
        rows = rows[: args.limit]
    fingerprints = [str((row.get("metadata") or {}).get("split_fingerprint") or "") for row in rows]
    if not all(fingerprints) or len(fingerprints) != len(set(fingerprints)):
        raise ValueError("evaluation rows require unique non-empty split_fingerprint values")
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *(
                _evaluate_one(args, session, semaphore, row, row_index)
                for row_index, row in enumerate(rows)
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(partial, args.output)
    summary = {
        **_metrics(results),
        "input": str(args.input),
        "model": args.model,
        "external_model_api": False,
        "max_response_tokens": args.max_response_tokens,
        "enable_thinking": args.enable_thinking,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
