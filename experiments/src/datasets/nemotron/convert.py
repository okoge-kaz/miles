"""Convert staged benchmark and NeMo Gym data into Miles prompt JSONL.

Every output row uses the same contract::

    {"prompt": [...], "label": "...", "metadata": {...}, "tools": [...]}

Use ``--input-key prompt --label-key label --tool-key tools
--apply-chat-template``. Conversion preserves ``metadata.verifier``; a training
recipe selects the matching fail-closed entry point under
``experiments.src.reward_sets``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import expand_paths, read_rows
from experiments.src.datasets.nemotron.adapters import ADAPTERS, adapt_nano


def _open_partial(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    return partial, partial.open("w", encoding="utf-8")


def _commit_partial(partial: Path, output: Path) -> None:
    os.replace(partial, output)


def _write_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def convert_one_dataset(args: argparse.Namespace) -> dict[str, Any]:
    adapter = ADAPTERS[args.dataset]
    paths = expand_paths(args.input)
    partial, handle = _open_partial(args.output)
    kept = skipped = 0
    verifiers: Counter[str] = Counter()
    try:
        with handle:
            for index, row in enumerate(read_rows(paths)):
                converted = adapter(row)
                if converted is None:
                    skipped += 1
                    continue
                _write_row(handle, converted)
                kept += 1
                verifiers[str(converted["metadata"].get("verifier"))] += 1
                if args.limit and kept >= args.limit:
                    break
        _commit_partial(partial, args.output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        "dataset": args.dataset,
        "inputs": [str(path) for path in paths],
        "output": str(args.output),
        "kept": kept,
        "skipped": skipped,
        "verifiers": dict(sorted(verifiers.items())),
    }


def convert_nano(args: argparse.Namespace) -> dict[str, Any]:
    if args.environment_output is None or args.unverifiable_output is None:
        raise ValueError("--environment-output and --unverifiable-output are required for nano-blend")
    paths = expand_paths(args.input)
    ready_partial, ready_handle = _open_partial(args.output)
    env_partial, env_handle = _open_partial(args.environment_output)
    unverifiable_partial, unverifiable_handle = _open_partial(args.unverifiable_output)
    ready = environment = unverifiable = skipped = placeholders = 0
    verifiers: Counter[str] = Counter()
    try:
        with ready_handle, env_handle, unverifiable_handle:
            for row in read_rows(paths):
                if row.get("_hf_placeholder"):
                    placeholders += 1
                    continue
                converted, route = adapt_nano(row)
                if converted is None:
                    skipped += 1
                    continue
                if route == "environment":
                    _write_row(env_handle, converted)
                    environment += 1
                elif route == "unverifiable":
                    _write_row(unverifiable_handle, converted)
                    unverifiable += 1
                else:
                    _write_row(ready_handle, converted)
                    ready += 1
                    verifiers[str(converted["metadata"].get("verifier"))] += 1
                if args.limit and ready + environment + unverifiable >= args.limit:
                    break
        if placeholders:
            raise RuntimeError(
                f"input still contains {placeholders} math placeholders; run restore_nano first"
            )
        if skipped:
            raise RuntimeError(f"Nano conversion skipped {skipped} unknown or invalid rows")
        _commit_partial(ready_partial, args.output)
        _commit_partial(env_partial, args.environment_output)
        _commit_partial(unverifiable_partial, args.unverifiable_output)
    except Exception:
        ready_partial.unlink(missing_ok=True)
        env_partial.unlink(missing_ok=True)
        unverifiable_partial.unlink(missing_ok=True)
        raise
    return {
        "dataset": "nano-blend",
        "inputs": [str(path) for path in paths],
        "output": str(args.output),
        "environment_output": str(args.environment_output),
        "unverifiable_output": str(args.unverifiable_output),
        "ready": ready,
        "requires_environment": environment,
        "unverifiable": unverifiable,
        "skipped": skipped,
        "verifiers": dict(sorted(verifiers.items())),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*sorted(ADAPTERS), "nano-blend"], required=True)
    parser.add_argument("--input", nargs="+", required=True, help="input paths or globs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--environment-output",
        type=Path,
        help="Nano rows needing a NeMo Gym environment are written here",
    )
    parser.add_argument(
        "--unverifiable-output",
        type=Path,
        help="Nano rows whose published source has no ground truth are written here",
    )
    parser.add_argument("--summary", type=Path, help="optional machine-readable conversion summary")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = convert_nano(args) if args.dataset == "nano-blend" else convert_one_dataset(args)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
