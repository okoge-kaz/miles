#!/usr/bin/env python3
"""Export pinned NeMo Skills AIME data to the Miles evaluation schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_RECORDS = 30
BENCHMARK_OUTPUTS = {
    "aime24": Path("aime-2024/aime-2024.jsonl"),
    "aime25": Path("aime-2025/aime-2025.jsonl"),
    "aime26": Path("aime-2026/aime-2026.jsonl"),
}
BOXED_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response "
    "must be of the form Answer: \\boxed{$Answer}, where $Answer is the answer.\n\n"
)
BOXED_REMINDER = '\n\nPut the final answer on its own line in the form `Answer: \\boxed{...}`.'


@dataclass(frozen=True)
class ExportInfo:
    """Provenance for one canonical Miles AIME export."""

    benchmark: str
    output: Path
    output_sha256: str
    records: int
    source: Path
    source_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _converted_rows(source: Path, benchmark: str) -> Iterator[dict[str, Any]]:
    record_ids: set[str] = set()
    records = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {source}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {source}:{line_number}")
            values: dict[str, str] = {}
            for field in ("id", "problem", "expected_answer"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"invalid {field!r} at {source}:{line_number}")
                values[field] = value.strip()
            if values["id"] in record_ids:
                raise ValueError(f"duplicate id {values['id']!r} in {source}")
            record_ids.add(values["id"])
            records += 1
            yield {
                "prompt": [
                    {
                        "role": "user",
                        "content": BOXED_INSTRUCTION + values["problem"] + BOXED_REMINDER,
                    }
                ],
                "label": values["expected_answer"],
                "metadata": {
                    "benchmark": benchmark,
                    "eval_only": True,
                    "record_id": values["id"],
                    "rm_type": "math",
                    "source": "nemo-skills-26.03",
                    "verifier": "math",
                },
            }
    if records != EXPECTED_RECORDS:
        raise ValueError(f"{benchmark} has {records} records; expected {EXPECTED_RECORDS}: {source}")


def _write_jsonl_atomic(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    records = 0
    try:
        with partial.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
                records += 1
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return records


def _write_provenance(info: ExportInfo) -> None:
    path = info.output.with_name(f"{info.output.name}.provenance.json")
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    payload = {
        "adapter": "experiments.tools.reasoning_eval.export_miles_aime",
        "benchmark": info.benchmark,
        "output_sha256": info.output_sha256,
        "records": info.records,
        "source": f"{info.benchmark}/test.jsonl",
        "source_sha256": info.source_sha256,
    }
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def export_benchmark(*, source_root: Path, output_root: Path, benchmark: str) -> ExportInfo:
    """Export one benchmark atomically and return its checksummed provenance."""
    try:
        relative_output = BENCHMARK_OUTPUTS[benchmark]
    except KeyError as error:
        raise ValueError(f"unsupported benchmark: {benchmark}") from error
    source = source_root / benchmark / "test.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"prepared AIME source not found: {source}")
    output = output_root / relative_output
    records = _write_jsonl_atomic(output, _converted_rows(source, benchmark))
    info = ExportInfo(
        benchmark=benchmark,
        output=output,
        output_sha256=_sha256(output),
        records=records,
        source=source,
        source_sha256=_sha256(source),
    )
    _write_provenance(info)
    return info


def export_all(
    *,
    source_root: Path,
    output_root: Path,
    benchmarks: tuple[str, ...] = tuple(BENCHMARK_OUTPUTS),
) -> tuple[ExportInfo, ...]:
    """Export each requested benchmark in deterministic order."""
    if not benchmarks or len(benchmarks) != len(set(benchmarks)):
        raise ValueError("benchmarks must be nonempty and unique")
    return tuple(
        export_benchmark(source_root=source_root, output_root=output_root, benchmark=benchmark)
        for benchmark in benchmarks
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", dest="benchmarks")
    return parser.parse_args()


def main() -> None:
    """Export requested data and print machine-readable provenance."""
    args = _parse_args()
    benchmarks = tuple(args.benchmarks) if args.benchmarks else tuple(BENCHMARK_OUTPUTS)
    try:
        exports = export_all(
            source_root=args.source_root,
            output_root=args.output_root,
            benchmarks=benchmarks,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"failed to export Miles AIME data: {error}") from error
    for info in exports:
        print(
            json.dumps(
                {
                    "benchmark": info.benchmark,
                    "output": str(info.output),
                    "output_sha256": info.output_sha256,
                    "records": info.records,
                    "source_sha256": info.source_sha256,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
