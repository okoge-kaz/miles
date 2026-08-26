"""Merge converted Miles JSONL files into one contamination-safe training set.

The current Miles ``--prompt-data`` interface accepts one path. This utility
streams several already converted files into one atomic output while rejecting
benchmark rows marked ``metadata.eval_only`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import expand_paths, read_rows


def _validate_training_row(row: dict[str, Any], source: Path, index: int) -> str:
    prompt = row.get("prompt")
    metadata = row.get("metadata")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"{source}:{index}: prompt must be a non-empty message list")
    if row.get("label") is None:
        raise ValueError(f"{source}:{index}: label is missing")
    if not isinstance(metadata, dict) or not metadata.get("verifier"):
        raise ValueError(f"{source}:{index}: metadata.verifier is required")
    if metadata.get("eval_only") is True:
        raise ValueError(f"{source}:{index}: refusing to mix an eval_only benchmark row into training")
    return str(metadata["verifier"])


def _apply_metadata_defaults(
    row: dict[str, Any],
    *,
    default_verifier: str | None,
    default_source: str | None,
) -> None:
    if default_verifier is None and default_source is None:
        return
    metadata = row.get("metadata")
    if metadata is None:
        metadata = {}
        row["metadata"] = metadata
    if not isinstance(metadata, dict):
        return
    if default_verifier is not None:
        metadata.setdefault("verifier", default_verifier)
    if default_source is not None:
        metadata.setdefault("source", default_source)


def merge_training_files(
    inputs: list[str],
    output: Path,
    *,
    expected_rows: int | None = None,
    require_verifiers: list[str] | None = None,
    default_verifier: str | None = None,
    default_source: str | None = None,
) -> dict[str, Any]:
    paths = expand_paths(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    rows = 0
    per_file: Counter[str] = Counter()
    verifiers: Counter[str] = Counter()
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for path in paths:
                for index, row in enumerate(read_rows([path]), start=1):
                    _apply_metadata_defaults(
                        row,
                        default_verifier=default_verifier,
                        default_source=default_source,
                    )
                    verifier = _validate_training_row(row, path, index)
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    rows += 1
                    per_file[str(path)] += 1
                    verifiers[verifier] += 1
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(f"expected {expected_rows} rows, found {rows}")
        missing = sorted(set(require_verifiers or []) - verifiers.keys())
        if missing:
            raise ValueError(f"required verifiers absent from merged data: {', '.join(missing)}")
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "inputs": [str(path) for path in paths],
        "output": str(output),
        "rows": rows,
        "rows_by_input": dict(sorted(per_file.items())),
        "verifiers": dict(sorted(verifiers.items())),
        "eval_only_rejected": True,
    }


def _selection_priority(*, seed: int, row: dict[str, Any]) -> int:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{seed}:".encode() + payload.encode()).digest()
    return int.from_bytes(digest[:16], byteorder="big")


def balanced_merge_training_files(
    inputs: list[str],
    output: Path,
    *,
    rows_per_input: int,
    seed: int = 42,
    require_verifiers: list[str] | None = None,
    default_verifier: str | None = None,
    default_source: str | None = None,
) -> dict[str, Any]:
    """Select the same deterministic row count from every input, then merge.

    Selection is two-pass so large code rows never accumulate in memory. The
    first pass retains only row indices and hash priorities; the second writes
    selected rows in source order to an atomic output.
    """
    if rows_per_input <= 0:
        raise ValueError("rows_per_input must be positive")
    paths = expand_paths(inputs)
    selected_by_path: dict[Path, set[int]] = {}
    available: Counter[str] = Counter()
    for path in paths:
        selected: list[tuple[int, int]] = []
        for index, row in enumerate(read_rows([path]), start=1):
            _apply_metadata_defaults(
                row,
                default_verifier=default_verifier,
                default_source=default_source,
            )
            _validate_training_row(row, path, index)
            available[str(path)] += 1
            priority = _selection_priority(seed=seed, row=row)
            candidate = (-priority, index)
            if len(selected) < rows_per_input:
                heapq.heappush(selected, candidate)
            elif priority < -selected[0][0]:
                heapq.heapreplace(selected, candidate)
        if available[str(path)] < rows_per_input:
            raise ValueError(
                f"{path}: requested {rows_per_input} rows, only {available[str(path)]} are available"
            )
        selected_by_path[path] = {index for _, index in selected}

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    rows = 0
    per_file: Counter[str] = Counter()
    verifiers: Counter[str] = Counter()
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for path in paths:
                selected_indices = selected_by_path[path]
                for index, row in enumerate(read_rows([path]), start=1):
                    if index not in selected_indices:
                        continue
                    _apply_metadata_defaults(
                        row,
                        default_verifier=default_verifier,
                        default_source=default_source,
                    )
                    verifier = _validate_training_row(row, path, index)
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    rows += 1
                    per_file[str(path)] += 1
                    verifiers[verifier] += 1
        expected_rows = rows_per_input * len(paths)
        if rows != expected_rows:
            raise ValueError(f"expected {expected_rows} balanced rows, found {rows}")
        missing = sorted(set(require_verifiers or []) - verifiers.keys())
        if missing:
            raise ValueError(f"required verifiers absent from balanced data: {', '.join(missing)}")
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "inputs": [str(path) for path in paths],
        "output": str(output),
        "rows": rows,
        "rows_per_input": rows_per_input,
        "rows_by_input": dict(sorted(per_file.items())),
        "available_by_input": dict(sorted(available.items())),
        "verifiers": dict(sorted(verifiers.items())),
        "eval_only_rejected": True,
        "balanced": True,
        "seed": seed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--rows-per-input", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-verifier")
    parser.add_argument("--default-source")
    parser.add_argument("--require-verifiers", nargs="*", default=[])
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rows_per_input is not None:
        if args.expected_rows is not None:
            raise ValueError("--expected-rows and --rows-per-input are mutually exclusive")
        summary = balanced_merge_training_files(
            args.input,
            args.output,
            rows_per_input=args.rows_per_input,
            seed=args.seed,
            require_verifiers=args.require_verifiers,
            default_verifier=args.default_verifier,
            default_source=args.default_source,
        )
    else:
        summary = merge_training_files(
            args.input,
            args.output,
            expected_rows=args.expected_rows,
            require_verifiers=args.require_verifiers,
            default_verifier=args.default_verifier,
            default_source=args.default_source,
        )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
