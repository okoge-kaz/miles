#!/usr/bin/env python3
"""Check learning parity and summarize telemetry OFF/ON GPU overhead."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path

import torch

_TRAIN_PATTERN = re.compile(r"(?:^|\s)step (\d+): (\{.*\})")
_PERF_PATTERN = re.compile(r"perf (\d+): (\{.*\})")
_PERF_KEYS = (
    "perf/actor_train_time",
    "perf/train_time",
    "perf/step_time",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic-baseline", type=Path, required=True)
    parser.add_argument("--deterministic-off", type=Path, required=True)
    parser.add_argument("--deterministic-on", type=Path, required=True)
    parser.add_argument("--deterministic-baseline-dir", type=Path, required=True)
    parser.add_argument("--deterministic-off-dir", type=Path, required=True)
    parser.add_argument("--deterministic-on-dir", type=Path, required=True)
    parser.add_argument("--performance-baseline", type=Path, required=True)
    parser.add_argument("--performance-off", type=Path, required=True)
    parser.add_argument("--performance-on", type=Path, required=True)
    parser.add_argument("--performance-histogram", type=Path, required=True)
    parser.add_argument("--performance-baseline-gpu", type=Path, required=True)
    parser.add_argument("--performance-off-gpu", type=Path, required=True)
    parser.add_argument("--performance-on-gpu", type=Path, required=True)
    parser.add_argument("--performance-histogram-gpu", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _literal_dict(text: str) -> dict[str, float]:
    value = ast.literal_eval(text)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a metric dictionary, got {type(value).__name__}")
    return value


def _read_indexed_metrics(path: Path, pattern: re.Pattern) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows[int(match.group(1))] = _literal_dict(match.group(2))
    if not rows:
        raise RuntimeError(f"No matching metrics found in {path}")
    return rows


def _compare_training_metrics(
    off_rows: dict[int, dict[str, float]],
    on_rows: dict[int, dict[str, float]],
) -> dict:
    if off_rows.keys() != on_rows.keys():
        raise AssertionError(f"Deterministic train steps differ: off={sorted(off_rows)}, on={sorted(on_rows)}")
    compared = 0
    maximum_abs_difference = 0.0
    mismatches = []
    for step in sorted(off_rows):
        off = off_rows[step]
        on = on_rows[step]
        common = sorted(set(off) & set(on))
        for key in common:
            if not key.startswith("train/") or key.startswith("train/staleness_gradient/"):
                continue
            off_value = float(off[key])
            on_value = float(on[key])
            difference = abs(off_value - on_value)
            maximum_abs_difference = max(maximum_abs_difference, difference)
            compared += 1
            if not math.isclose(off_value, on_value, rel_tol=0.0, abs_tol=0.0):
                mismatches.append({"step": step, "metric": key, "off": off_value, "on": on_value})
    if mismatches:
        raise AssertionError(f"Deterministic training metrics changed: {mismatches[:10]}")
    return {
        "steps": len(off_rows),
        "common_train_scalars_compared": compared,
        "max_abs_difference": maximum_abs_difference,
        "bitwise_equal_as_logged": True,
    }


def _compare_grad_norms(off_dir: Path, on_dir: Path) -> dict:
    off_paths = sorted(off_dir.glob("grad_norm-*.pt"))
    on_paths = sorted(on_dir.glob("grad_norm-*.pt"))
    if [path.name for path in off_paths] != [path.name for path in on_paths] or not off_paths:
        raise AssertionError(
            f"Gradient norm files differ: off={[p.name for p in off_paths]}, on={[p.name for p in on_paths]}"
        )
    maximum_abs_difference = 0.0
    for off_path, on_path in zip(off_paths, on_paths, strict=True):
        off = float(torch.load(off_path, map_location="cpu", weights_only=False))
        on = float(torch.load(on_path, map_location="cpu", weights_only=False))
        maximum_abs_difference = max(maximum_abs_difference, abs(off - on))
        if not math.isclose(off, on, rel_tol=0.0, abs_tol=0.0):
            raise AssertionError(f"Gradient norm changed in {off_path.name}: off={off}, on={on}")
    return {
        "files_compared": len(off_paths),
        "max_abs_difference": maximum_abs_difference,
        "bitwise_equal_as_logged": True,
    }


def _check_staleness_metrics(
    off_rows: dict[int, dict[str, float]],
    on_rows: dict[int, dict[str, float]],
) -> dict:
    off_keys = {key for row in off_rows.values() for key in row}
    if any(key.startswith("train/staleness_gradient/") for key in off_keys):
        raise AssertionError("OFF run unexpectedly logged staleness-gradient metrics")

    checked_distributions = 0
    bins_seen = set()
    for step, row in on_rows.items():
        if row.get("train/staleness_gradient/effective_contribution_available") != 1.0:
            raise AssertionError(f"Step {step} has no effective contribution")
        for suffix in ("consumed_sequence_mass", "effective_contribution_mass"):
            values = [
                float(value)
                for key, value in row.items()
                if key.startswith("train/staleness_gradient/s_") and key.endswith(f"/{suffix}")
            ]
            if not values or not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(f"Step {step} {suffix} is not a normalized distribution: {sum(values):.9g}")
            checked_distributions += 1
        bins_seen.update(key.split("/")[2] for key in row if key.startswith("train/staleness_gradient/s_"))
    return {
        "steps": len(on_rows),
        "normalized_distributions_checked": checked_distributions,
        "bins_seen": sorted(bins_seen),
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "median": torch.quantile(tensor, 0.50).item(),
        "p90": torch.quantile(tensor, 0.90).item(),
    }


def _peak_gpu_memory_mib(path: Path) -> int:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise RuntimeError(f"GPU monitor file is empty: {path}")
    return max(int(float(row[2].strip())) for row in rows)


def _performance_summary(log_path: Path, gpu_path: Path, warmup_steps: int) -> dict:
    rows = _read_indexed_metrics(log_path, _PERF_PATTERN)
    measured = [row for step, row in sorted(rows.items()) if step >= warmup_steps]
    if not measured:
        raise RuntimeError(f"No measured performance rows remain after {warmup_steps} warmups")
    result = {key: _quantiles([float(row[key]) for row in measured if key in row]) for key in _PERF_KEYS}
    result["measured_steps"] = len(measured)
    result["peak_gpu_memory_mib"] = _peak_gpu_memory_mib(gpu_path)
    wall_path = log_path.parent / "wall_seconds.txt"
    result["end_to_end_wall_seconds"] = float(wall_path.read_text())
    result["end_to_end_wall_seconds_per_step"] = result["end_to_end_wall_seconds"] / len(rows)
    return result


def _overhead(off: dict, on: dict) -> dict:
    if off["measured_steps"] != on["measured_steps"]:
        raise AssertionError(f"Measured step counts differ: off={off['measured_steps']}, on={on['measured_steps']}")
    changes = {}
    for key in _PERF_KEYS:
        if not off[key] or not on[key]:
            continue
        off_median = off[key]["median"]
        on_median = on[key]["median"]
        changes[key] = {
            "median_delta_seconds": on_median - off_median,
            "median_change_fraction": (on_median - off_median) / off_median,
        }
    return {
        "metric_changes": changes,
        "peak_gpu_memory_delta_mib": on["peak_gpu_memory_mib"] - off["peak_gpu_memory_mib"],
        "end_to_end_wall_delta_seconds": (on["end_to_end_wall_seconds"] - off["end_to_end_wall_seconds"]),
        "end_to_end_wall_change_fraction": (on["end_to_end_wall_seconds"] - off["end_to_end_wall_seconds"])
        / off["end_to_end_wall_seconds"],
    }


def main() -> None:
    args = _parse_args()
    deterministic_baseline = _read_indexed_metrics(args.deterministic_baseline, _TRAIN_PATTERN)
    deterministic_off = _read_indexed_metrics(args.deterministic_off, _TRAIN_PATTERN)
    deterministic_on = _read_indexed_metrics(args.deterministic_on, _TRAIN_PATTERN)
    performance_baseline = _performance_summary(
        args.performance_baseline,
        args.performance_baseline_gpu,
        args.warmup_steps,
    )
    performance_off = _performance_summary(
        args.performance_off,
        args.performance_off_gpu,
        args.warmup_steps,
    )
    performance_on = _performance_summary(
        args.performance_on,
        args.performance_on_gpu,
        args.warmup_steps,
    )
    performance_histogram = _performance_summary(
        args.performance_histogram,
        args.performance_histogram_gpu,
        args.warmup_steps,
    )
    result = {
        "learning_parity": {
            "always_on_telemetry_vs_clean_head": {
                "training_metrics": _compare_training_metrics(
                    deterministic_baseline,
                    deterministic_off,
                ),
                "gradient_norms": _compare_grad_norms(
                    args.deterministic_baseline_dir,
                    args.deterministic_off_dir,
                ),
            },
            "gradient_bins_vs_telemetry_off": {
                "training_metrics": _compare_training_metrics(deterministic_off, deterministic_on),
                "gradient_norms": _compare_grad_norms(
                    args.deterministic_off_dir,
                    args.deterministic_on_dir,
                ),
            },
        },
        "metric_validation": _check_staleness_metrics(deterministic_off, deterministic_on),
        "performance": {
            "clean_head": performance_baseline,
            "off": performance_off,
            "base_bins": performance_on,
            "ratio_histogram": performance_histogram,
            "always_on_telemetry_minus_clean_head": _overhead(
                performance_baseline,
                performance_off,
            ),
            "base_bins_minus_off": _overhead(performance_off, performance_on),
            "ratio_histogram_minus_off": _overhead(
                performance_off,
                performance_histogram,
            ),
            "ratio_histogram_minus_base_bins": _overhead(
                performance_on,
                performance_histogram,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
