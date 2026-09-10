"""Parse the metric records printed to a Slurm training log."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(
    r"\[(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) [^\]]*\]\s*"
    r"\S+:\d+\s*-\s*(?P<kind>rollout batch consumption|rollout pipeline throughput|"
    r"eval|perf|rollout|step) (?P<id>\d+): (?P<payload>\{.*\})\s*$"
)
NUMBER = re.compile(
    r"'(?P<key>[^']+)':\s*(?P<value>-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf|-inf))"
)
KIND_STEP_KEY = {
    "eval": "eval/step",
    "perf": "rollout/step",
    "rollout": "rollout/step",
    "rollout batch consumption": "rollout/step",
    "rollout pipeline throughput": "rollout/step",
    "step": "train/step",
}


def parse_payload(payload: str) -> dict[str, Any]:
    """Parse a logged metric dictionary, including non-finite float values."""
    try:
        parsed = ast.literal_eval(payload)
        if isinstance(parsed, dict):
            return {
                key: value for key, value in parsed.items() if isinstance(value, (int, float))
            }
    except (ValueError, SyntaxError):
        pass
    return {match["key"]: float(match["value"]) for match in NUMBER.finditer(payload)}


def parse_log(path: Path) -> list[dict[str, Any]]:
    """Return timestamped metric records from one Slurm log."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for line in _iter_lines(path):
        match = LINE.search(ANSI.sub("", line))
        if not match:
            continue
        identity = (match["kind"], int(match["id"]), match["payload"])
        if identity in seen:
            continue
        seen.add(identity)
        metrics = parse_payload(match["payload"])
        if not metrics:
            continue
        step_key = KIND_STEP_KEY[match["kind"]]
        metrics.setdefault(step_key, int(match["id"]))
        records.append(
            {
                "ts": datetime.strptime(match["stamp"], "%Y-%m-%d %H:%M:%S.%f").timestamp(),
                "step_key": step_key,
                "step": int(match["id"]),
                "metrics": metrics,
            }
        )
    return sorted(records, key=lambda record: record["ts"])


def merge_step_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge metric streams which belong to the same rollout or train step."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        identity = (record["step_key"], record["step"])
        if identity not in merged:
            merged[identity] = {**record, "metrics": dict(record["metrics"])}
            continue
        merged[identity]["metrics"].update(record["metrics"])
        merged[identity]["ts"] = max(merged[identity]["ts"], record["ts"])
    return sorted(merged.values(), key=lambda record: record["ts"])


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open(errors="replace") as handle:
        yield from handle
