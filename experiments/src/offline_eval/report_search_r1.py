#!/usr/bin/env python3
"""Report Search-R1 exact match and interaction cost from offline-eval JSONL."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


IN_DOMAIN = frozenset({"nq", "hotpotqa"})


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _weighted_mean(rows: list[dict], key: str) -> float:
    weighted = [(float(row[key]), int(row.get("n_samples", 1))) for row in rows if row.get(key) is not None]
    denominator = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / denominator if denominator else math.nan


def summarise(rows: list[dict]) -> dict:
    rates = [float(row["pass_rate"]) for row in rows if "pass_rate" in row]
    if not rates:
        return {}
    n = len(rates)
    return {
        "prompts": n,
        "episodes": sum(int(row.get("n_samples", 1)) for row in rows),
        "accuracy": statistics.mean(rates),
        "se": statistics.stdev(rates) / math.sqrt(n) if n > 1 else math.nan,
        "pass_at_n": statistics.mean(float(row.get("n_correct", 0) > 0) for row in rows),
        "search_calls": _weighted_mean(rows, "search_calls_mean"),
        "turns": _weighted_mean(rows, "turns_mean"),
        "searched": _weighted_mean(rows, "searched_frac"),
        "answered": _weighted_mean(rows, "answered_frac"),
        "truncated": _weighted_mean(rows, "truncated_frac"),
        "generated_tokens": _weighted_mean(rows, "response_len_mean"),
        "observation_tokens": _weighted_mean(rows, "observation_len_mean"),
    }


def _fmt_percent(value: float, width: int = 7) -> str:
    return f"{100 * value:>{width - 1}.1f}%" if math.isfinite(value) else f"{'-':>{width}}"


def _print_row(name: str, summary: dict) -> None:
    print(
        f"  {name:<18} {summary['prompts']:>6} {summary['episodes']:>7}"
        f"{_fmt_percent(summary['accuracy'], 8)}{_fmt_percent(summary['se'], 8)}"
        f"{_fmt_percent(summary['pass_at_n'], 8)}"
        f"{summary['search_calls']:>8.2f}{summary['turns']:>8.2f}"
        f"{_fmt_percent(summary['searched'], 8)}{_fmt_percent(summary['answered'], 8)}"
        f"{_fmt_percent(summary['truncated'], 8)}"
        f"{summary['generated_tokens']:>9.0f}{summary['observation_tokens']:>9.0f}"
    )


def _macro(summaries: list[dict]) -> dict:
    keys = (
        "accuracy",
        "pass_at_n",
        "search_calls",
        "turns",
        "searched",
        "answered",
        "truncated",
        "generated_tokens",
        "observation_tokens",
    )
    return {
        "prompts": sum(summary["prompts"] for summary in summaries),
        "episodes": sum(summary["episodes"] for summary in summaries),
        "se": statistics.stdev(summary["accuracy"] for summary in summaries) / math.sqrt(len(summaries))
        if len(summaries) > 1
        else math.nan,
        **{key: statistics.mean(summary[key] for summary in summaries) for key in keys},
    }


def report(directory: Path) -> None:
    files = sorted(directory.glob("*.jsonl"))
    benchmark_rows = {path.stem: load(path) for path in files}
    benchmark_rows = {name: rows for name, rows in benchmark_rows.items() if rows and "pass_rate" in rows[0]}
    if not benchmark_rows:
        print(f"{directory}: no Search-R1 benchmark files")
        return

    print(f"\n{directory}")
    print(
        f"  {'benchmark':<18} {'prompt':>6} {'episode':>7}{'avg@N':>8}{'±SE':>8}{'pass@N':>8}"
        f"{'search':>8}{'turns':>8}{'use':>8}{'answer':>8}{'trunc':>8}{'gen_tok':>9}{'obs_tok':>9}"
    )
    print("  " + "-" * 114)

    summaries = {}
    for name, rows in benchmark_rows.items():
        summaries[name] = summarise(rows)
        _print_row(name, summaries[name])

    print("  " + "-" * 114)
    in_domain_rows = [row for name, rows in benchmark_rows.items() if name in IN_DOMAIN for row in rows]
    out_domain_rows = [row for name, rows in benchmark_rows.items() if name not in IN_DOMAIN for row in rows]
    if in_domain_rows:
        _print_row("in-domain pooled", summarise(in_domain_rows))
    if out_domain_rows:
        _print_row("out-domain pooled", summarise(out_domain_rows))
    _print_row("all pooled", summarise([row for rows in benchmark_rows.values() for row in rows]))
    _print_row("benchmark macro", _macro(list(summaries.values())))
    print()
    print("  search = retriever calls per trajectory; turns = LLM generation calls per trajectory.")
    print("  gen_tok is policy output; obs_tok is masked environment text (retrieval or invalid-action feedback).")


def main() -> int:
    directories = [Path(argument) for argument in sys.argv[1:]] or [Path(".")]
    for directory in directories:
        report(directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
