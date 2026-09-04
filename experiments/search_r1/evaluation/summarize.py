#!/usr/bin/env python3
"""Summarize held-out Search-R1 episode records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


WEIGHTED_FIELDS = (
    "response_len_mean",
    "observation_len_mean",
    "total_response_len_mean",
    "truncated_frac",
    "search_calls_mean",
    "turns_mean",
    "searched_frac",
    "answered_frac",
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluation records: {path}")
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate prompt and trajectory metrics from one benchmark."""
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    trajectory_count = sum(int(record["n_samples"]) for record in records)
    if trajectory_count <= 0:
        raise ValueError("evaluation records contain no trajectories")
    correct_count = sum(int(record["n_correct"]) for record in records)
    summary: dict[str, float | int] = {
        "prompts": len(records),
        "trajectories": trajectory_count,
        "correct": correct_count,
        "exact_match": correct_count / trajectory_count,
        "prompt_mean_exact_match": sum(float(record["pass_rate"]) for record in records) / len(records),
    }
    for field in WEIGHTED_FIELDS:
        weighted_sum = sum(float(record[field]) * int(record["n_samples"]) for record in records)
        summary[field] = weighted_sum / trajectory_count
    return summary


def build_summary(result_root: Path, benchmarks: list[str]) -> dict[str, Any]:
    """Build per-benchmark and macro summaries from a result directory."""
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    results = {
        benchmark: summarize_records(_load_records(result_root / benchmark / "records.jsonl"))
        for benchmark in benchmarks
    }
    macro_exact_match = sum(float(result["exact_match"]) for result in results.values()) / len(results)
    return {
        "benchmarks": results,
        "macro_exact_match": macro_exact_match,
        "benchmark_count": len(results),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_summary(args.result_root, args.benchmarks)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.partial")
    temporary.write_text(f"{rendered}\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(rendered)


if __name__ == "__main__":
    main()
