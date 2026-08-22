#!/usr/bin/env python3
"""Collect AIME24/25/26 scores from the staleness-ratio sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


TASKS = ("aime24", "aime25", "aime26")
DEFAULT_PROTOCOL = (
    "eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-"
    "thinking-t0.6-p0.95-k20-aime64-v1"
)
ASYNC_ARM_PATTERN = re.compile(r"^s(?P<staleness>\d+)-t(?P<train>\d+)r(?P<rollout>\d+)$")


@dataclass(frozen=True)
class ScoreRecord:
    """One completed benchmark score."""

    arm: str
    placement: str
    max_weight_staleness: int
    trainer_nodes: int
    rollout_nodes: int
    training_step: int
    checkpoint_directory: int
    task: str
    metric: str
    score_percent: float
    num_repeats: int | None
    verified_output_records: int | None
    checkpoint: str
    result_directory: str


@dataclass(frozen=True)
class AggregateRecord:
    """Three-task macro score for one arm and training step."""

    arm: str
    placement: str
    max_weight_staleness: int
    trainer_nodes: int
    rollout_nodes: int
    training_step: int
    checkpoint_directory: int
    completed_tasks: int
    aime24_percent: float | None
    aime25_percent: float | None
    aime26_percent: float | None
    aime_macro_mean_percent: float | None


def _expected_arms() -> tuple[str, ...]:
    async_arms = tuple(
        f"s{staleness}-t{train_nodes}r{8 - train_nodes}"
        for staleness in (1, 2, 4, 8)
        for train_nodes in (1, 2, 3, 4)
    )
    return (*async_arms, "s0-colocated")


def _arm_metadata(arm: str) -> tuple[str, int, int, int]:
    if arm == "s0-colocated":
        return "colocated", 0, 8, 0
    match = ASYNC_ARM_PATTERN.fullmatch(arm)
    if match is None:
        raise ValueError(f"unsupported arm name: {arm}")
    return (
        "disaggregated",
        int(match["staleness"]),
        int(match["train"]),
        int(match["rollout"]),
    )


def _read_provenance(task_directory: Path) -> dict[str, str]:
    candidates = sorted(task_directory.glob("provenance-*.env"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return {}
    values: dict[str, str] = {}
    for line in candidates[-1].read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _read_score(task_directory: Path, task: str) -> float:
    metrics_path = task_directory / "evaluator" / "eval-results" / task / "metrics.json"
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    score = float(metrics[task]["pass@1"]["symbolic_correct"])
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"score outside [0, 100] in {metrics_path}: {score}")
    return score


def _collect_score(
    *,
    root: Path,
    arm: str,
    step: int,
    task: str,
    protocol: str,
    mode: str,
) -> ScoreRecord | None:
    task_directory = root / arm / f"step_{step:04d}" / protocol / mode / task
    if not (task_directory / "_SUCCESS").is_file():
        return None
    placement, staleness, train_nodes, rollout_nodes = _arm_metadata(arm)
    provenance = _read_provenance(task_directory)
    repeats_text = provenance.get("num_repeats", "")
    repeats = int(repeats_text) if repeats_text.isdigit() else None
    output_records_text = provenance.get("expected_output_records", "")
    output_records = int(output_records_text) if output_records_text.isdigit() else None
    return ScoreRecord(
        arm=arm,
        placement=placement,
        max_weight_staleness=staleness,
        trainer_nodes=train_nodes,
        rollout_nodes=rollout_nodes,
        training_step=step,
        checkpoint_directory=step - 1,
        task=task,
        metric="pass@1.symbolic_correct",
        score_percent=_read_score(task_directory, task),
        num_repeats=repeats,
        verified_output_records=output_records,
        checkpoint=provenance.get("checkpoint", ""),
        result_directory=str(task_directory),
    )


def _collect_records(
    *, root: Path, protocol: str, mode: str, steps: Iterable[int]
) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    for arm in _expected_arms():
        for step in steps:
            for task in TASKS:
                record = _collect_score(
                    root=root,
                    arm=arm,
                    step=step,
                    task=task,
                    protocol=protocol,
                    mode=mode,
                )
                if record is not None:
                    records.append(record)
    return records


def _aggregate_records(records: Iterable[ScoreRecord], steps: Iterable[int]) -> list[AggregateRecord]:
    score_lookup = {(record.arm, record.training_step, record.task): record.score_percent for record in records}
    aggregates: list[AggregateRecord] = []
    for arm in _expected_arms():
        placement, staleness, train_nodes, rollout_nodes = _arm_metadata(arm)
        for step in steps:
            scores = {task: score_lookup.get((arm, step, task)) for task in TASKS}
            completed_scores = [score for score in scores.values() if score is not None]
            macro_mean = sum(completed_scores) / 3.0 if len(completed_scores) == 3 else None
            aggregates.append(
                AggregateRecord(
                    arm=arm,
                    placement=placement,
                    max_weight_staleness=staleness,
                    trainer_nodes=train_nodes,
                    rollout_nodes=rollout_nodes,
                    training_step=step,
                    checkpoint_directory=step - 1,
                    completed_tasks=len(completed_scores),
                    aime24_percent=scores["aime24"],
                    aime25_percent=scores["aime25"],
                    aime26_percent=scores["aime26"],
                    aime_macro_mean_percent=macro_mean,
                )
            )
    return aggregates


def _write_csv(
    path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None
) -> None:
    columns = fieldnames or (list(rows[0]) if rows else None)
    if columns is None:
        raise ValueError(f"CSV columns are required for empty output: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _latest_complete(aggregates: Iterable[AggregateRecord], arm: str) -> AggregateRecord | None:
    candidates = [
        record
        for record in aggregates
        if record.arm == arm and record.aime_macro_mean_percent is not None
    ]
    return max(candidates, key=lambda record: record.training_step) if candidates else None


def _render_markdown(
    *, root: Path, protocol: str, mode: str, records: list[ScoreRecord], aggregates: list[AggregateRecord]
) -> str:
    complete_suites = sum(record.aime_macro_mean_percent is not None for record in aggregates)
    lines = [
        "# Staleness-ratio sweep reasoning evaluation",
        "",
        f"Result root: `{root}`",
        f"Protocol: `{protocol}`; mode: `{mode}`",
        "",
        f"Completed task evaluations: **{len(records)} / {len(aggregates) * 3}**",
        f"Complete AIME24/25/26 checkpoint suites: **{complete_suites} / {len(aggregates)}**",
        "",
        "## Latest complete checkpoint per arm",
        "",
        "| Arm | Step | AIME24 | AIME25 | AIME26 | Macro mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in _expected_arms():
        latest = _latest_complete(aggregates, arm)
        if latest is None:
            lines.append(f"| {arm} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {arm} | {latest.training_step} | {latest.aime24_percent:.4f} | "
            f"{latest.aime25_percent:.4f} | {latest.aime26_percent:.4f} | "
            f"{latest.aime_macro_mean_percent:.4f} |"
        )
    lines.extend(
        [
            "",
            "The macro mean is emitted only when all three AIME tasks completed for that checkpoint.",
            "Detailed and aggregate machine-readable data are in `task-results.csv`, "
            "`aggregate-results.csv`, and `summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-study-root", type=Path, required=True)
    parser.add_argument("--protocol-name", default=DEFAULT_PROTOCOL)
    parser.add_argument("--eval-mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--start-step", type=int, default=10)
    parser.add_argument("--end-step", type=int, default=300)
    parser.add_argument("--step-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.start_step <= 0 or args.end_step < args.start_step or args.step_interval <= 0:
        raise ValueError("invalid step range")
    root = args.result_study_root.resolve()
    output_dir = args.output_dir or root / "analysis" / args.protocol_name / args.eval_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = tuple(range(args.start_step, args.end_step + 1, args.step_interval))
    records = _collect_records(root=root, protocol=args.protocol_name, mode=args.eval_mode, steps=steps)
    aggregates = _aggregate_records(records, steps)
    record_rows = [asdict(record) for record in records]
    score_columns = [field.name for field in ScoreRecord.__dataclass_fields__.values()]
    _write_csv(output_dir / "task-results.csv", record_rows, fieldnames=score_columns)
    aggregate_rows = [asdict(record) for record in aggregates]
    _write_csv(output_dir / "aggregate-results.csv", aggregate_rows)
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "result_study_root": str(root),
        "protocol_name": args.protocol_name,
        "eval_mode": args.eval_mode,
        "expected_arms": list(_expected_arms()),
        "steps": list(steps),
        "tasks": list(TASKS),
        "completed_task_evaluations": len(records),
        "expected_task_evaluations": len(aggregates) * len(TASKS),
        "complete_checkpoint_suites": sum(
            record.aime_macro_mean_percent is not None for record in aggregates
        ),
        "records": record_rows,
        "aggregates": aggregate_rows,
    }
    _write_json(output_dir / "summary.json", summary)
    markdown = _render_markdown(
        root=root,
        protocol=args.protocol_name,
        mode=args.eval_mode,
        records=records,
        aggregates=aggregates,
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
