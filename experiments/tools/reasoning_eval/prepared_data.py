#!/usr/bin/env python3
"""Validate prepared AIME data and its evaluator-image provenance marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


EXPECTED_AIME_RECORDS = 30
_BENCHMARK_NAME = re.compile(r"^aime(?:24|25|26)$")
_MARKER_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class DatasetInfo:
    """Validated identity for one prepared benchmark file."""

    benchmark: str
    path: Path
    sha256: str
    records: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_marker(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"prepared-data marker not found: {path}")
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or "=" not in line:
            raise ValueError(f"malformed marker line {line_number}: {path}")
        key, value = line.split("=", 1)
        if not _MARKER_KEY.fullmatch(key) or not value:
            raise ValueError(f"invalid marker entry on line {line_number}: {path}")
        if key in values:
            raise ValueError(f"duplicate marker key {key!r}: {path}")
        values[key] = value
    return values


def _validate_dataset(path: Path, benchmark: str) -> DatasetInfo:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"prepared benchmark data not found: {path}")
    record_ids: set[str] = set()
    records = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            for field in ("id", "problem", "expected_answer"):
                field_value = value.get(field)
                if not isinstance(field_value, str) or not field_value:
                    raise ValueError(f"invalid {field!r} at {path}:{line_number}")
            record_id = value["id"]
            if record_id in record_ids:
                raise ValueError(f"duplicate id {record_id!r} in {path}")
            record_ids.add(record_id)
            records += 1
    if records != EXPECTED_AIME_RECORDS:
        raise ValueError(
            f"{benchmark} has {records} records; expected {EXPECTED_AIME_RECORDS}: {path}"
        )
    return DatasetInfo(
        benchmark=benchmark,
        path=path,
        sha256=_sha256(path),
        records=records,
    )


def validate_prepared_data(
    *,
    data_root: Path,
    image: Path,
    benchmarks: tuple[str, ...],
    marker: Path | None = None,
) -> tuple[DatasetInfo, ...]:
    """Validate required data, accepting a marker with extra prepared benchmarks."""
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    if len(set(benchmarks)) != len(benchmarks):
        raise ValueError("benchmarks must not contain duplicates")
    for benchmark in benchmarks:
        if not _BENCHMARK_NAME.fullmatch(benchmark):
            raise ValueError(f"unsupported benchmark: {benchmark}")

    marker_values: dict[str, str] = {}
    if marker is not None:
        marker_values = _read_marker(marker)
        if marker_values.get("nemo_skills_image") != str(image):
            raise ValueError("prepared data uses a different NeMo Skills image")
        prepared_benchmarks = marker_values.get("benchmarks", "").split()
        if len(set(prepared_benchmarks)) != len(prepared_benchmarks):
            raise ValueError("prepared-data marker contains duplicate benchmarks")
        missing = sorted(set(benchmarks).difference(prepared_benchmarks))
        if missing:
            raise ValueError(f"prepared-data marker is missing benchmarks: {' '.join(missing)}")

    datasets = tuple(
        _validate_dataset(data_root / benchmark / "test.jsonl", benchmark)
        for benchmark in benchmarks
    )
    for dataset in datasets:
        recorded_sha256 = marker_values.get(f"dataset_{dataset.benchmark}_sha256")
        if recorded_sha256 is not None and recorded_sha256 != dataset.sha256:
            raise ValueError(f"prepared-data checksum mismatch for {dataset.benchmark}")
        recorded_count = marker_values.get(f"dataset_{dataset.benchmark}_records")
        if recorded_count is not None and recorded_count != str(dataset.records):
            raise ValueError(f"prepared-data record-count mismatch for {dataset.benchmark}")
    return datasets


def format_dataset_environment(datasets: tuple[DatasetInfo, ...]) -> str:
    """Render deterministic environment-style provenance lines."""
    lines: list[str] = []
    for dataset in datasets:
        lines.extend(
            (
                f"dataset_{dataset.benchmark}_sha256={dataset.sha256}",
                f"dataset_{dataset.benchmark}_records={dataset.records}",
            )
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", dest="benchmarks", required=True)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--print-env", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate prepared data for shell callers."""
    args = _parse_args()
    try:
        datasets = validate_prepared_data(
            data_root=args.data_root,
            image=args.image,
            benchmarks=tuple(args.benchmarks),
            marker=args.marker,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid prepared AIME data: {error}") from error
    if args.print_env:
        print(format_dataset_environment(datasets))


if __name__ == "__main__":
    main()
