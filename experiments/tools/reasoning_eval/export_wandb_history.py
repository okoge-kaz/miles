#!/usr/bin/env python3
"""Export the selected W&B training lineage for reasoning-evaluation analysis."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARM_PATTERN = re.compile(r"^(s(?:0-colocated|[1248]-t[1-4]r[4-7]))-")
ROLLOUT_STEP = "rollout/step"
TRAIN_STEP = "train/step"
ROLLOUT_METRICS = (
    "perf/step_time",
    "throughput/generated_tokens_per_second",
    "throughput/accepted_tokens_per_second",
    "throughput/useful_tokens_per_second",
    "throughput/optimizer_updates_per_second",
    "throughput/window_useful_efficiency",
    "throughput/cohort_useful_efficiency",
    "queue/trainer_starvation_seconds",
    "queue/rollout_backpressure_seconds",
    "staleness/total/mean",
    "staleness/total/variance",
    "staleness/total/std",
    "staleness/total/max",
    "staleness/total/p90",
    "staleness/pre_queue/mean",
    "staleness/in_queue/mean",
    "staleness/bound_exceeded_sample_frac",
    "rollout/fully_async/wasted_token_frac",
    "rollout/fully_async/stale_groups_recycled",
    "rollout/raw_reward",
    "rollout/truncated_ratio",
    "rollout/response_len/mean",
)
TRAIN_METRICS = (
    "train/policy_rollout_kl",
    "train/policy_rollout_abs_diff",
    "train/policy_rollout_token_ess",
    "train/policy_rollout_sequence_ess",
    "train/train_rollout_kl",
    "train/tis",
    "train/tis_abs",
    "train/tis_clipfrac",
    "train/ppo_kl",
    "train/final_loss_tokens",
    "train/optimizer_step_applied",
    "train/grad_norm_pre_clip",
    "train/grad_clip_coefficient",
    "train/num_zeros_in_grad",
    "train/update_norm",
    "train/parameter_norm",
    "train/relative_update_norm",
    "train/cumulative_update_path_norm",
    "train/advantage_std",
    "train/advantage_rms",
    "train/advantage_abs_mean",
    "train/loss",
)
OUTPUT_FIELDS = (
    "arm",
    "update_index",
    "training_step",
    "rollout_timestamp",
    "train_timestamp",
    "calendar_elapsed_seconds",
    "active_wallclock_seconds",
    "active_wallclock_coverage",
    *ROLLOUT_METRICS,
    *TRAIN_METRICS,
)


@dataclass(frozen=True)
class RunHistory:
    """One W&B run segment before checkpoint-lineage selection."""

    arm: str
    run_id: str
    created_at: str
    rollout: dict[int, dict[str, float]]
    train: dict[int, dict[str, float]]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _arm_from_run(run: Any) -> str:
    candidates = (
        run.config.get("wandb_group", ""),
        getattr(run, "group", ""),
        getattr(run, "name", ""),
    )
    for candidate in candidates:
        match = ARM_PATTERN.match(str(candidate))
        if match is not None:
            return match.group(1)
    raise ValueError(f"cannot identify sweep arm for W&B run {run.id}")


def _merge_row_metrics(
    target: dict[int, dict[str, float]],
    *,
    row: dict[str, Any],
    step_key: str,
    metric_keys: Iterable[str],
) -> None:
    raw_step = _finite_number(row.get(step_key))
    if raw_step is None:
        return
    step = int(raw_step)
    record = target.setdefault(step, {})
    timestamp = _finite_number(row.get("_timestamp"))
    if timestamp is not None:
        record["_timestamp"] = timestamp
    for metric in metric_keys:
        value = _finite_number(row.get(metric))
        if value is not None:
            record[metric] = value


def _fetch_run_history(run: Any) -> RunHistory:
    rollout: dict[int, dict[str, float]] = {}
    train: dict[int, dict[str, float]] = {}
    # W&B treats a long ``keys`` list as a sparse-row intersection. Rollout and
    # train metrics are logged on different rows, so fetch the complete stream
    # and project the required metrics locally.
    for row in run.scan_history(page_size=1000):
        _merge_row_metrics(
            rollout,
            row=row,
            step_key=ROLLOUT_STEP,
            metric_keys=ROLLOUT_METRICS,
        )
        _merge_row_metrics(
            train,
            row=row,
            step_key=TRAIN_STEP,
            metric_keys=TRAIN_METRICS,
        )
    return RunHistory(
        arm=_arm_from_run(run),
        run_id=run.id,
        created_at=run.created_at or "",
        rollout=rollout,
        train=train,
    )


def _merge_lineage(
    histories: Iterable[RunHistory],
) -> tuple[dict[str, dict[str, dict[int, dict[str, float]]]], int]:
    lineage: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    replacements = 0
    for history in sorted(histories, key=lambda item: (item.created_at, item.run_id)):
        arm_lineage = lineage.setdefault(history.arm, {"rollout": {}, "train": {}})
        for axis in ("rollout", "train"):
            source = getattr(history, axis)
            for step, values in source.items():
                target = arm_lineage[axis].setdefault(step, {})
                for metric, value in values.items():
                    if metric in target and target[metric] != value:
                        replacements += 1
                    target[metric] = value
    return lineage, replacements


def _selected_timestamps(arm_lineage: dict[str, dict[int, dict[str, float]]]) -> list[float]:
    timestamps = []
    for axis in ("rollout", "train"):
        for record in arm_lineage[axis].values():
            if "_timestamp" in record:
                timestamps.append(record["_timestamp"])
    return timestamps


def _history_rows(
    lineage: dict[str, dict[str, dict[int, dict[str, float]]]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for arm, arm_lineage in sorted(lineage.items()):
        rollout = arm_lineage["rollout"]
        train = arm_lineage["train"]
        steps = sorted(set(rollout) | set(train))
        if not steps:
            continue
        timestamps = _selected_timestamps(arm_lineage)
        first_timestamp = min(timestamps) if timestamps else None
        cumulative_seconds = 0.0
        covered_steps = 0
        for update_index in range(max(steps) + 1):
            step_time = rollout.get(update_index, {}).get("perf/step_time")
            if step_time is not None and step_time >= 0.0:
                cumulative_seconds += step_time
                covered_steps += 1
            if update_index not in rollout and update_index not in train:
                continue
            rollout_timestamp = rollout.get(update_index, {}).get("_timestamp")
            train_timestamp = train.get(update_index, {}).get("_timestamp")
            selected_timestamp = max(
                (value for value in (rollout_timestamp, train_timestamp) if value is not None),
                default=None,
            )
            row: dict[str, float | int | str] = {
                "arm": arm,
                "update_index": update_index,
                "training_step": update_index + 1,
                "rollout_timestamp": rollout_timestamp if rollout_timestamp is not None else "",
                "train_timestamp": train_timestamp if train_timestamp is not None else "",
                "calendar_elapsed_seconds": (
                    selected_timestamp - first_timestamp
                    if selected_timestamp is not None and first_timestamp is not None
                    else ""
                ),
                "active_wallclock_seconds": cumulative_seconds,
                "active_wallclock_coverage": covered_steps / (update_index + 1),
            }
            row.update({metric: rollout.get(update_index, {}).get(metric, "") for metric in ROLLOUT_METRICS})
            row.update({metric: train.get(update_index, {}).get(metric, "") for metric in TRAIN_METRICS})
            rows.append(row)
    return rows


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="async-rl-dapo-math")
    parser.add_argument("--entity")
    parser.add_argument("--namespace", default="sr-20260819-212906")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def _wandb_runs(*, entity: str | None, project: str, namespace: str) -> tuple[Any, list[Any]]:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required to export training history") from error
    api = wandb.Api(timeout=120)
    resolved_entity = entity or api.default_entity
    runs = list(
        api.runs(
            f"{resolved_entity}/{project}",
            filters={
                "$or": [
                    {"group": {"$regex": namespace}},
                    {"display_name": {"$regex": namespace}},
                ]
            },
        )
    )
    return resolved_entity, runs


def main() -> None:
    args = _parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    entity, runs = _wandb_runs(entity=args.entity, project=args.project, namespace=args.namespace)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        histories = list(executor.map(_fetch_run_history, runs))
    lineage, replacements = _merge_lineage(histories)
    rows = _history_rows(lineage)
    output_csv = args.output_csv.resolve()
    metadata_json = args.metadata_json or output_csv.with_suffix(".metadata.json")
    _atomic_write_csv(output_csv, rows)
    _atomic_write_json(
        metadata_json.resolve(),
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "entity": entity,
            "project": args.project,
            "namespace": args.namespace,
            "run_count": len(runs),
            "arms": sorted(lineage),
            "lineage_replacements": replacements,
            "history_rows": len(rows),
        },
    )
    print(output_csv)


if __name__ == "__main__":
    main()
