#!/usr/bin/env python3
"""Generate and score the non-NeMo portion of the reasoning evaluation suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp

from experiments.src.datasets.common.io import read_rows
from experiments.src.evaluators.livecodebench import livecodebench_reward
from experiments.src.reward_sets._common import score_gpqa_sample
from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.types import Sample


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def _request_completion(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    *,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    row_index: int,
    repeat: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_retries: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": row["prompt"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": 1234 + repeat,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    async with semaphore:
        error = ""
        for attempt in range(1, max_retries + 1):
            try:
                async with session.post(endpoint, json=payload) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
                    parsed = json.loads(body)
                    choice = parsed["choices"][0]["message"]
                    content = str(choice.get("content") or "").strip()
                    reasoning = str(
                        choice.get("reasoning_content")
                        or choice.get("reasoning")
                        or ""
                    )
                    return {
                        "row_index": row_index,
                        "repeat": repeat,
                        "response": content,
                        "finish_reason": parsed["choices"][0].get("finish_reason"),
                        "usage": parsed.get("usage") or {},
                        "generation_status": "ok" if content else "empty_final_content",
                        "reasoning_length": len(reasoning),
                    }
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
                error = f"attempt {attempt}: {exc}"
                if attempt < max_retries:
                    await asyncio.sleep(min(2**attempt, 30))
        raise RuntimeError(f"row={row_index} repeat={repeat}: {error}")


async def generate(args: argparse.Namespace) -> None:
    output = args.output
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        print(f"already complete: {output}")
        return
    completed_rows = _read_jsonl(partial)
    completed = {(int(row["row_index"]), int(row["repeat"])) for row in completed_rows}
    input_rows = list(read_rows([args.input]))
    if args.limit is not None:
        input_rows = input_rows[: args.limit]
    expected = len(input_rows) * args.repeats
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a", encoding="utf-8") as handle:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for start in range(0, len(input_rows), args.concurrency):
                tasks = []
                for row_index in range(start, min(start + args.concurrency, len(input_rows))):
                    row = input_rows[row_index]
                    for repeat in range(args.repeats):
                        if (row_index, repeat) in completed:
                            continue
                        tasks.append(
                            _request_completion(
                                session,
                                semaphore,
                                endpoint=args.endpoint,
                                model=args.model,
                                row=row,
                                row_index=row_index,
                                repeat=repeat,
                                max_tokens=args.max_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                max_retries=args.max_retries,
                                enable_thinking=args.enable_thinking,
                            )
                        )
                if not tasks:
                    continue
                for future in asyncio.as_completed(tasks):
                    result = await future
                    handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    completed.add((int(result["row_index"]), int(result["repeat"])))
                    handle.flush()
    records = _read_jsonl(partial)
    keys = {(int(row["row_index"]), int(row["repeat"])) for row in records}
    if len(records) != expected or len(keys) != expected:
        raise RuntimeError(f"expected {expected} unique generations, found {len(records)} rows/{len(keys)} keys")
    os.replace(partial, output)
    empty_count = sum(
        not str(record.get("response") or "").strip() for record in records
    )
    print(
        f"generated {expected} responses ({empty_count} empty final responses) "
        f"-> {output}"
    )


async def _score_batch(task: str, samples: list[Sample]) -> list[float]:
    if task in {"aime24", "aime25", "aime26", "math500"}:
        rewards = [float(grade_answer_verl(sample.response, sample.label)) for sample in samples]
    elif task == "livecodebench":
        rewards = await livecodebench_reward(SimpleNamespace(), samples)
    elif task.startswith("gpqa_"):
        rewards = [score_gpqa_sample(sample) for sample in samples]
    elif task == "ifbench":
        from miles.rollout.rm_hub.ifbench import compute_ifbench_reward

        rewards = [
            compute_ifbench_reward(sample.response, sample.label, metadata=sample.metadata)
            for sample in samples
        ]
    else:
        raise ValueError(f"unsupported scoring task: {task}")
    return [float(reward) for reward in rewards]


async def score(args: argparse.Namespace) -> None:
    output = args.output
    partial = output.with_name(output.name + ".partial")
    summary_path = args.summary
    candidates = _read_jsonl(args.candidates)
    by_index: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_index.setdefault(int(candidate["row_index"]), []).append(candidate)
    input_rows = list(read_rows([args.input]))
    if args.limit is not None:
        input_rows = input_rows[: args.limit]
    expected = len(input_rows) * args.repeats
    samples: list[Sample] = []
    identities: list[tuple[int, int]] = []
    candidate_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(input_rows):
        candidates_for_row = sorted(by_index.get(row_index, []), key=lambda item: int(item["repeat"]))
        if len(candidates_for_row) != args.repeats:
            raise RuntimeError(
                f"row {row_index}: expected {args.repeats} candidates, found {len(candidates_for_row)}"
            )
        for candidate in candidates_for_row:
            samples.append(
                Sample(
                    response=candidate["response"],
                    label=row.get("label"),
                    metadata=row.get("metadata") or {},
                )
            )
            identities.append((row_index, int(candidate["repeat"])))
            candidate_records.append(candidate)
    rewards = [0.0] * len(samples)
    scorable_indices = [
        index
        for index, sample in enumerate(samples)
        if str(sample.response or "").strip()
    ]
    for start in range(0, len(scorable_indices), args.score_batch_size):
        batch_indices = scorable_indices[start : start + args.score_batch_size]
        batch_rewards = await _score_batch(
            args.task,
            [samples[index] for index in batch_indices],
        )
        for index, reward in zip(batch_indices, batch_rewards, strict=True):
            rewards[index] = reward
    if len(rewards) != expected:
        raise RuntimeError(f"expected {expected} rewards, found {len(rewards)}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as handle:
        for (row_index, repeat), reward in zip(identities, rewards, strict=True):
            handle.write(
                json.dumps(
                    {"row_index": row_index, "repeat": repeat, "reward": reward},
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(partial, output)
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    stderr = math.sqrt(mean * (1 - mean) / len(rewards)) if rewards else 0.0
    per_prompt = []
    for row_index in range(len(input_rows)):
        row_rewards = [
            reward
            for (identity_index, _), reward in zip(identities, rewards, strict=True)
            if identity_index == row_index
        ]
        per_prompt.append(sum(row_rewards) / len(row_rewards))
    prompt_mean = sum(per_prompt) / len(per_prompt) if per_prompt else 0.0
    empty_final_response_count = sum(
        not str(candidate.get("response") or "").strip()
        for candidate in candidate_records
    )
    length_finished_count = sum(
        candidate.get("finish_reason") == "length"
        for candidate in candidate_records
    )
    summary = {
        "task": args.task,
        "input": str(args.input),
        "candidates": str(args.candidates),
        "scores": str(output),
        "prompts": len(input_rows),
        "repeats": args.repeats,
        "samples": len(rewards),
        "sample_accuracy": mean,
        "sample_accuracy_stderr": stderr,
        "prompt_avg_accuracy": prompt_mean,
        "empty_final_response_count": empty_final_response_count,
        "length_finished_count": length_finished_count,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    summary_partial = summary_path.with_name(summary_path.name + ".partial")
    summary_partial.write_text(rendered + "\n", encoding="utf-8")
    os.replace(summary_partial, summary_path)
    print(rendered)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--input", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--endpoint", required=True)
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--repeats", type=int, required=True)
    generate_parser.add_argument("--max-tokens", type=int, required=True)
    generate_parser.add_argument("--temperature", type=float, default=0.6)
    generate_parser.add_argument("--top-p", type=float, default=0.95)
    generate_parser.add_argument("--top-k", type=int, default=20)
    generate_parser.add_argument("--concurrency", type=int, default=256)
    generate_parser.add_argument("--request-timeout", type=int, default=3600)
    generate_parser.add_argument("--max-retries", type=int, default=5)
    generate_parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    generate_parser.add_argument("--limit", type=int)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--task", required=True)
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument("--candidates", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--summary", type=Path, required=True)
    score_parser.add_argument("--repeats", type=int, required=True)
    score_parser.add_argument("--score-batch-size", type=int, default=8)
    score_parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.command == "generate":
        asyncio.run(generate(args))
    else:
        asyncio.run(score(args))


if __name__ == "__main__":
    main()
