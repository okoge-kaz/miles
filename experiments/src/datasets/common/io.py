"""Streaming readers and writers shared by dataset converters."""

from __future__ import annotations

import csv
import glob
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    """Expand globs deterministically and reject an empty input set."""
    paths = sorted({Path(match) for pattern in patterns for match in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"no input files matched: {list(patterns)}")
    return paths


def read_rows(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Stream JSONL, CSV, parquet, or Hugging Face Arrow rows."""
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        elif suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                yield from csv.DictReader(handle)
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("parquet conversion requires pyarrow") from exc
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=16_384):
                yield from batch.to_pylist()
        elif suffix == ".arrow":
            try:
                import pyarrow as pa
            except ImportError as exc:
                raise RuntimeError("Arrow conversion requires pyarrow") from exc
            with pa.memory_map(str(path), "r") as source:
                try:
                    reader = pa.ipc.open_stream(source)
                    batches = reader
                except pa.ArrowInvalid:
                    source.seek(0)
                    reader = pa.ipc.open_file(source)
                    batches = (
                        reader.get_batch(batch_index)
                        for batch_index in range(reader.num_record_batches)
                    )
                for batch in batches:
                    yield from batch.to_pylist()
        else:
            raise ValueError(f"unsupported input format: {path}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as UTF-8 JSONL and return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
