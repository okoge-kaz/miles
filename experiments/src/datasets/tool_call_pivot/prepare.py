#!/usr/bin/env python3
"""Build leakage-free, balanced static tool-call Pivot train and evaluation splits.

Only expert actions with an exactly verifiable function call are eligible.
Free-form message actions require a semantic judge and are deliberately excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import read_rows
from experiments.src.protocols.openai_responses import expected_action_signature
from experiments.src.protocols.tool_call_pivot import (
    TOOL_CALL_PIVOT_INTERACTION_MODE,
    TOOL_CALL_PIVOT_POLICY,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_fingerprint(row: dict[str, Any]) -> str:
    """Return a stable identity independent of source-file ordering."""
    metadata = row.get("metadata") or {}
    identity = {
        "source": metadata.get("source"),
        "trajectory_id": metadata.get("trajectory_id"),
        "prompt": row.get("prompt"),
        "tools": row.get("tools"),
        "expected_action": metadata.get("expected_action"),
        "interaction_mode": metadata.get("interaction_mode"),
        "stateful_environment": metadata.get("stateful_environment"),
    }
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()


def _eligible(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    signature = expected_action_signature(metadata.get("expected_action"))
    return (
        metadata.get("verifier") == "expert_action"
        and metadata.get("interaction_mode") == TOOL_CALL_PIVOT_INTERACTION_MODE
        and metadata.get("stateful_environment") is False
        and signature is not None
        and signature.get("kind") == "function_call"
        and bool(signature.get("name"))
        and bool(row.get("tools"))
    )


def _ranked_fingerprints(path: Path, source: str) -> tuple[list[str], Counter[str]]:
    fingerprints: set[str] = set()
    counts: Counter[str] = Counter()
    for row in read_rows([path]):
        counts["input"] += 1
        metadata = row.get("metadata") or {}
        if metadata.get("source") != source:
            counts["wrong_source"] += 1
            continue
        signature = expected_action_signature(metadata.get("expected_action"))
        if signature is not None:
            counts[str(signature.get("kind") or "unknown")] += 1
        if not _eligible(row):
            counts["ineligible"] += 1
            continue
        fingerprint = row_fingerprint(row)
        if fingerprint in fingerprints:
            counts["duplicate"] += 1
            continue
        fingerprints.add(fingerprint)
        counts["eligible"] += 1
    return sorted(fingerprints), counts


def _atomic_handles(paths: list[Path]):
    partials = [path.with_name(path.name + ".partial") for path in paths]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return partials, [partial.open("w", encoding="utf-8") for partial in partials]


def _write_selected(
    sources: dict[str, Path],
    assignments: dict[str, tuple[set[str], set[str]]],
    train_output: Path,
    eval_output: Path,
) -> tuple[Counter[str], Counter[str]]:
    partials, handles = _atomic_handles([train_output, eval_output])
    train_handle, eval_handle = handles
    train_counts: Counter[str] = Counter()
    eval_counts: Counter[str] = Counter()
    try:
        with train_handle, eval_handle:
            for source, path in sources.items():
                train_ids, eval_ids = assignments[source]
                emitted_train: set[str] = set()
                emitted_eval: set[str] = set()
                for row in read_rows([path]):
                    fingerprint = row_fingerprint(row)
                    if fingerprint in train_ids and fingerprint not in emitted_train:
                        row.setdefault("metadata", {})["split_fingerprint"] = fingerprint
                        train_handle.write(_canonical(row) + "\n")
                        emitted_train.add(fingerprint)
                        train_counts[source] += 1
                    elif fingerprint in eval_ids and fingerprint not in emitted_eval:
                        row.setdefault("metadata", {})["split_fingerprint"] = fingerprint
                        row["metadata"]["eval_only"] = True
                        eval_handle.write(_canonical(row) + "\n")
                        emitted_eval.add(fingerprint)
                        eval_counts[source] += 1
        os.replace(partials[0], train_output)
        os.replace(partials[1], eval_output)
    except Exception:
        for partial in partials:
            partial.unlink(missing_ok=True)
        raise
    return train_counts, eval_counts


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    sources = dict(args.source)
    audits: dict[str, dict[str, int]] = {}
    assignments: dict[str, tuple[set[str], set[str]]] = {}
    expected_train_counts: dict[str, int] = {}
    for source, path in sources.items():
        ranked, counts = _ranked_fingerprints(path, source)
        required = args.eval_per_source
        if args.train_per_source is not None:
            required += args.train_per_source
        if len(ranked) < required:
            raise ValueError(
                f"{source} has {len(ranked)} eligible unique rows; {required} required"
            )
        eval_ids = set(ranked[: args.eval_per_source])
        train_stop = required if args.train_per_source is not None else len(ranked)
        train_ids = set(ranked[args.eval_per_source : train_stop])
        assignments[source] = (train_ids, eval_ids)
        expected_train_counts[source] = len(train_ids)
        audits[source] = dict(sorted(counts.items()))

    train_counts, eval_counts = _write_selected(
        sources, assignments, args.train_output, args.eval_output
    )
    if any(
        train_counts[source] != expected_train_counts[source] for source in sources
    ):
        raise RuntimeError(f"incomplete train split: {dict(train_counts)}")
    if any(eval_counts[source] != args.eval_per_source for source in sources):
        raise RuntimeError(f"incomplete eval split: {dict(eval_counts)}")
    train_fingerprints = set().union(*(assignments[source][0] for source in sources))
    eval_fingerprints = set().union(*(assignments[source][1] for source in sources))
    if train_fingerprints & eval_fingerprints:
        raise RuntimeError("train/eval fingerprint overlap")
    return {
        "policy": TOOL_CALL_PIVOT_POLICY,
        "interaction_mode": TOOL_CALL_PIVOT_INTERACTION_MODE,
        "stateful_environment": False,
        "sources": {source: str(path) for source, path in sources.items()},
        "audit": audits,
        "train_output": str(args.train_output),
        "train_sha256": _file_sha256(args.train_output),
        "eval_output": str(args.eval_output),
        "eval_sha256": _file_sha256(args.eval_output),
        "train_counts": dict(sorted(train_counts.items())),
        "eval_counts": dict(sorted(eval_counts.items())),
        "train_total": sum(train_counts.values()),
        "eval_total": sum(eval_counts.values()),
        "overlap": 0,
    }


def _source(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("source must be NAME=PATH")
    return name, Path(path)


def _train_count(value: str) -> int | None:
    if value == "all":
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "train count must be a positive integer or 'all'"
        ) from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("train count must be positive")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--train-per-source", type=_train_count, default=4700)
    parser.add_argument("--eval-per-source", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(dict(args.source)) != len(args.source):
        raise ValueError("source names must be unique")
    summary = prepare(args)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    partial = args.summary.with_name(args.summary.name + ".partial")
    try:
        partial.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, args.summary)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
