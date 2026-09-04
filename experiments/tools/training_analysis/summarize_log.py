#!/usr/bin/env python3
"""Summarize reward, truncation, staleness, and optimizer metrics from a training log."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any


ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ROLLOUT_PATTERN = re.compile(r"\brollout\s+(\d+):\s+(\{.*\})\s*$")
STEP_PATTERN = re.compile(r"\bstep\s+(\d+):\s+(\{.*\})\s*$")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x_mean = (len(values) - 1) / 2
    y_mean = sum(values) / len(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator


def _parse_metric_dict(blob: str) -> dict[str, Any]:
    parsed = ast.literal_eval(blob)
    if not isinstance(parsed, dict):
        raise TypeError("metric payload is not a dictionary")
    return parsed


def summarize(path: Path) -> dict[str, Any]:
    rollout_metrics: dict[int, dict[str, Any]] = {}
    step_metrics: dict[int, dict[str, Any]] = {}
    segments_enabled = False
    update_successes = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_PATTERN.sub("", raw_line.rstrip())
            segments_enabled = segments_enabled or "sglang_enable_response_weight_version_segments .. True" in line
            update_successes += int("fn=update_weights phase=end ok=true" in line and "repeated" not in line)
            if match := ROLLOUT_PATTERN.search(line):
                rollout_metrics[int(match.group(1))] = _parse_metric_dict(match.group(2))
            elif match := STEP_PATTERN.search(line):
                step_metrics[int(match.group(1))] = _parse_metric_dict(match.group(2))
    ordered_rollouts = [rollout_metrics[index] for index in sorted(rollout_metrics)]
    normalized_rewards = [
        float(metric["rollout/rewards"]) for metric in ordered_rollouts if "rollout/rewards" in metric
    ]
    raw_rewards = [
        float(metric["rollout/raw_reward"]) for metric in ordered_rollouts if "rollout/raw_reward" in metric
    ]
    # GRPO's rollout/rewards is centered within each prompt group and therefore
    # has a batch mean near zero by construction.  Learning progress must be
    # measured with the unnormalized verifier reward whenever it is available.
    rewards = raw_rewards if raw_rewards else normalized_rewards
    truncation = [float(metric["rollout/truncated"]) for metric in ordered_rollouts if "rollout/truncated" in metric]
    staleness = [
        float(metric["rollout/sample_staleness"])
        for metric in ordered_rollouts
        if "rollout/sample_staleness" in metric
    ]
    optimizer_applied = sum(
        int(float(metric.get("train/optimizer_step_applied", 0)) == 1.0) for metric in step_metrics.values()
    )
    window = max(1, len(rewards) // 4) if rewards else 0
    first = rewards[:window]
    last = rewards[-window:] if window else []
    first_mean = _mean(first)
    last_mean = _mean(last)
    return {
        "log": str(path),
        "rollout_metric_count": len(rollout_metrics),
        "train_step_metric_count": len(step_metrics),
        "optimizer_steps_applied": optimizer_applied,
        "weight_update_success_lines": update_successes,
        "response_weight_version_segments": segments_enabled,
        "reward": {
            "first_quarter_mean": first_mean,
            "last_quarter_mean": last_mean,
            "last_minus_first": None if first_mean is None or last_mean is None else last_mean - first_mean,
            "mean": _mean(rewards),
            "min": min(rewards) if rewards else None,
            "max": max(rewards) if rewards else None,
            "linear_slope_per_rollout": _slope(rewards),
            "normalized_mean": _mean(normalized_rewards),
            "source": "raw_reward" if raw_rewards else "rewards",
        },
        "truncated_mean": _mean(truncation),
        "sample_staleness": {
            "mean": _mean(staleness),
            "max": max(staleness) if staleness else None,
            "within_requested_bound_4": bool(staleness) and max(staleness) <= 4,
        },
        "finite": all(
            math.isfinite(value)
            for value in raw_rewards + normalized_rewards + truncation + staleness
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.log)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        partial = args.output.with_name(args.output.name + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(rendered + "\n", encoding="utf-8")
        os.replace(partial, args.output)


if __name__ == "__main__":
    main()
