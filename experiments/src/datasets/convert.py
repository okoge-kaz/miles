from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import expand_paths, read_rows
from experiments.src.datasets.common.parallel import add_parallel_arguments, map_rows
from experiments.src.datasets.source_adapters import ADAPTERS


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
        with handle, map_rows(read_rows(paths), adapter, num_proc=args.num_proc, chunk_size=args.chunk_size) as rows:
            for converted in rows:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert source datasets into Miles prompt JSONL for the selected reward/runtime"
    )
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--input", nargs="+", required=True, help="input paths or globs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="optional machine-readable conversion summary")
    parser.add_argument("--limit", type=int)
    add_parallel_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = convert_one_dataset(args)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
