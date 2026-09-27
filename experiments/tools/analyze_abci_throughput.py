"""Summarize 20 successful optimizer updates from ABCI job logs.

Usage: python experiments/tools/analyze_abci_throughput.py t2r6.log t3r5.log
Rates are ratios of summed counters and elapsed time, never means of rates.
The first window includes pipeline startup; a second summary excludes it.
"""

import argparse
import ast
import json
import re
from pathlib import Path

_ROW = re.compile(r"rollout pipeline throughput (\d+): (\{.*\})")
_PREFIX = "throughput/"


def read_windows(path: Path) -> list[dict[str, float]]:
    rows = {}
    with path.open() as stream:
        for line in stream:
            match = _ROW.search(line)
            if match is None:
                continue
            step = int(match[1])
            values = ast.literal_eval(match[2])
            if step in rows and rows[step] != values:
                raise ValueError(f"{path}: conflicting duplicate step {step}")
            rows[step] = values
    if sorted(rows) != list(range(20)):
        raise ValueError(f"{path}: expected steps 0..19, got {sorted(rows)}")
    for step, values in rows.items():
        if values[_PREFIX + "window_seconds"] <= 0:
            raise ValueError(f"{path}: nonpositive window at step {step}")
        if values[_PREFIX + "optimizer_updates"] != 1 or values[_PREFIX + "completed_training_batches"] != 1:
            raise ValueError(f"{path}: expected one successful optimizer update at step {step}")
        if values[_PREFIX + "window_accepted_loss_tokens_available"] != 1:
            raise ValueError(f"{path}: accepted token counter missing at step {step}")
    return [rows[step] for step in range(20)]


def summarize(windows: list[dict[str, float]]) -> dict[str, float]:
    seconds = sum(row[_PREFIX + "window_seconds"] for row in windows)
    updates = sum(row[_PREFIX + "optimizer_updates"] for row in windows)
    generated = sum(row[_PREFIX + "generated_tokens"] for row in windows)
    accepted = sum(row[_PREFIX + "window_accepted_loss_tokens"] for row in windows)
    return {
        "optimizer_updates": updates,
        "wall_seconds": seconds,
        "seconds_per_update": seconds / updates,
        "updates_per_second": updates / seconds,
        "generated_tokens": generated,
        "generated_tokens_per_second": generated / seconds,
        "accepted_loss_tokens": accepted,
        "useful_tokens_per_second": accepted / seconds,
        "trainer_starvation_seconds": sum(row["queue/trainer_starvation_seconds"] for row in windows),
        "rollout_backpressure_seconds": sum(row["queue/rollout_backpressure_seconds"] for row in windows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.logs:
        windows = read_windows(path)
        results.append({"log": str(path), "all_20": summarize(windows), "excluding_first": summarize(windows[1:])})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
