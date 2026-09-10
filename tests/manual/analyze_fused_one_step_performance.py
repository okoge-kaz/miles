#!/usr/bin/env python3
"""Summarize paired legacy/fused trainer performance logs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import torch


_PERF_PATTERN = re.compile(r"perf (\d+): (\{.*\})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-log", type=Path, required=True)
    parser.add_argument("--legacy-gpu-csv", type=Path, required=True)
    parser.add_argument("--fused-log", type=Path, required=True)
    parser.add_argument("--fused-gpu-csv", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    return parser.parse_args()


def _read_perf(path: Path, warmup_steps: int) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = _PERF_PATTERN.search(line)
        if match and int(match.group(1)) >= warmup_steps:
            rows.append(ast.literal_eval(match.group(2)))
    return rows


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "median": torch.quantile(tensor, 0.50).item(),
        "p90": torch.quantile(tensor, 0.90).item(),
    }


def _peak_gpu_memory_mib(path: Path) -> int:
    with path.open(newline="") as handle:
        return max(int(row[2].strip()) for row in csv.reader(handle))


def _mode_summary(rows: list[dict[str, float]], gpu_csv: Path) -> dict:
    metrics = {}
    for key in (
        "perf/log_probs_time",
        "perf/actor_train_time",
        "perf/train_time",
        "perf/step_time",
    ):
        values = [row[key] for row in rows if key in row]
        if values:
            metrics[key] = _summary(values)
    metrics["measured_steps"] = len(rows)
    metrics["peak_gpu_memory_mib"] = _peak_gpu_memory_mib(gpu_csv)
    return metrics


def main() -> None:
    args = _parse_args()
    legacy_rows = _read_perf(args.legacy_log, args.warmup_steps)
    fused_rows = _read_perf(args.fused_log, args.warmup_steps)
    if len(legacy_rows) != len(fused_rows):
        raise RuntimeError(
            f"Measured step counts differ: legacy={len(legacy_rows)}, fused={len(fused_rows)}"
        )
    legacy = _mode_summary(legacy_rows, args.legacy_gpu_csv)
    fused = _mode_summary(fused_rows, args.fused_gpu_csv)

    legacy_actor = legacy["perf/actor_train_time"]["median"]
    fused_actor = fused["perf/actor_train_time"]["median"]
    legacy_train = legacy["perf/train_time"]["median"]
    fused_train = fused["perf/train_time"]["median"]
    legacy_logprobs = legacy["perf/log_probs_time"]["median"]
    result = {
        "legacy": legacy,
        "fused": fused,
        "comparison": {
            "fused_has_standalone_logprobs_timer": "perf/log_probs_time" in fused,
            "actor_train_median_change_fraction": (fused_actor - legacy_actor) / legacy_actor,
            "trainer_service_median_reduction_seconds": legacy_train - fused_train,
            "recovered_legacy_logprobs_fraction": (legacy_train - fused_train) / legacy_logprobs,
            "trainer_service_median_speedup": legacy_train / fused_train,
            "step_median_speedup": legacy["perf/step_time"]["median"]
            / fused["perf/step_time"]["median"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
