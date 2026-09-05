#!/usr/bin/env python3
"""Measure net parameter displacement between adjacent evaluated checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ASYNC_ARM_PATTERN = re.compile(r"^s(?P<staleness>\d+)-t\d+r\d+$")
OUTPUT_FIELDS = (
    "arm",
    "study_identity",
    "namespace",
    "start_step",
    "end_step",
    "optimizer_steps",
    "parameter_count",
    "start_parameter_norm",
    "net_parameter_displacement_norm",
    "net_parameter_displacement_per_update",
    "relative_net_parameter_displacement",
    "elapsed_seconds",
    "start_checkpoint",
    "end_checkpoint",
)


@dataclass(frozen=True)
class CheckpointInterval:
    """One adjacent pair of evaluated checkpoints."""

    arm: str
    study_identity: str
    namespace: str
    start_step: int
    end_step: int
    start_checkpoint: Path
    end_checkpoint: Path


@dataclass(frozen=True)
class Displacement:
    """Norms calculated from one checkpoint interval."""

    arm: str
    study_identity: str
    namespace: str
    start_step: int
    end_step: int
    optimizer_steps: int
    parameter_count: int
    start_parameter_norm: float
    net_parameter_displacement_norm: float
    net_parameter_displacement_per_update: float
    relative_net_parameter_displacement: float
    elapsed_seconds: float
    start_checkpoint: str
    end_checkpoint: str


def _checkpoint_root(
    study_root: Path,
    *,
    arm: str,
    namespace: str,
    async_max_concurrent_samples: int | None = None,
    training_buffer_queue_size: int = 1000,
) -> Path:
    if arm == "s0-colocated":
        return study_root / "colocated/on-policy/max-weight-staleness-0" / f"{arm}-{namespace}-zero-trunc/hf"
    match = ASYNC_ARM_PATTERN.fullmatch(arm)
    if match is None:
        raise ValueError(f"unsupported sweep arm: {arm}")
    staleness = match.group("staleness")
    identity_suffix = ""
    if async_max_concurrent_samples is not None:
        identity_suffix += f"-concurrency-{async_max_concurrent_samples}"
    if training_buffer_queue_size != 1000:
        identity_suffix += f"-tbq{training_buffer_queue_size}"
    return (
        study_root
        / "async/off-policy"
        / f"max-weight-staleness-{staleness}-from-prefill"
        / f"{arm}-{namespace}-zero-trunc-rb-inflight{identity_suffix}/hf"
    )


def _read_intervals(
    aggregates: Path,
    *,
    study_root: Path,
    study_identity: str,
    namespace: str,
    step_interval: int,
    selected_arm: str | None,
    async_max_concurrent_samples: int | None = None,
    training_buffer_queue_size: int = 1000,
) -> list[CheckpointInterval]:
    rows_by_arm: dict[str, list[dict[str, str]]] = defaultdict(list)
    with aggregates.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("aime_macro_mean_percent", "").strip():
                continue
            arm = row["arm"]
            if selected_arm is None or arm == selected_arm:
                rows_by_arm[arm].append(row)

    intervals: list[CheckpointInterval] = []
    for arm, rows in sorted(rows_by_arm.items()):
        checkpoint_root = _checkpoint_root(
            study_root,
            arm=arm,
            namespace=namespace,
            async_max_concurrent_samples=async_max_concurrent_samples,
            training_buffer_queue_size=training_buffer_queue_size,
        )
        ordered = sorted(rows, key=lambda row: int(row["training_step"]))
        for start, end in zip(ordered, ordered[1:], strict=False):
            start_step = int(start["training_step"])
            end_step = int(end["training_step"])
            if end_step - start_step != step_interval:
                continue
            intervals.append(
                CheckpointInterval(
                    arm=arm,
                    study_identity=study_identity,
                    namespace=namespace,
                    start_step=start_step,
                    end_step=end_step,
                    start_checkpoint=checkpoint_root / start["checkpoint_directory"],
                    end_checkpoint=checkpoint_root / end["checkpoint_directory"],
                )
            )
    return intervals


def _checkpoint_shards(checkpoint: Path) -> tuple[Path, ...]:
    shards = tuple(sorted(checkpoint.glob("model-*.safetensors")))
    if shards:
        return shards
    unsharded = checkpoint / "model.safetensors"
    if unsharded.is_file():
        return (unsharded,)
    raise FileNotFoundError(f"no safetensors model found in {checkpoint}")


def _tensor_squares(
    start_tensor: torch.Tensor,
    end_tensor: torch.Tensor,
    *,
    chunk_elements: int,
) -> tuple[float, float]:
    if start_tensor.shape != end_tensor.shape:
        raise ValueError(f"tensor shape changed: {start_tensor.shape} != {end_tensor.shape}")
    start_flat = start_tensor.reshape(-1)
    end_flat = end_tensor.reshape(-1)
    parameter_square_sum = 0.0
    displacement_square_sum = 0.0
    for offset in range(0, start_flat.numel(), chunk_elements):
        stop = min(offset + chunk_elements, start_flat.numel())
        start_chunk = start_flat[offset:stop].float()
        end_chunk = end_flat[offset:stop].float()
        delta = end_chunk - start_chunk
        parameter_square_sum += float(torch.sum(start_chunk.square(), dtype=torch.float64))
        displacement_square_sum += float(torch.sum(delta.square(), dtype=torch.float64))
    return parameter_square_sum, displacement_square_sum


def _measure_interval(interval: CheckpointInterval, *, chunk_elements: int) -> Displacement:
    started = time.monotonic()
    start_shards = _checkpoint_shards(interval.start_checkpoint)
    end_shards = _checkpoint_shards(interval.end_checkpoint)
    if tuple(path.name for path in start_shards) != tuple(path.name for path in end_shards):
        raise ValueError(f"checkpoint shard layout changed for {interval.arm} step {interval.end_step}")

    parameter_count = 0
    parameter_square_sum = 0.0
    displacement_square_sum = 0.0
    for start_shard, end_shard in zip(start_shards, end_shards, strict=True):
        with safe_open(start_shard, framework="pt", device="cpu") as start, safe_open(
            end_shard, framework="pt", device="cpu"
        ) as end:
            if set(start.keys()) != set(end.keys()):
                raise ValueError(f"tensor keys changed in {start_shard.name}")
            for name in start.keys():
                start_tensor = start.get_tensor(name)
                end_tensor = end.get_tensor(name)
                start_squares, displacement_squares = _tensor_squares(
                    start_tensor,
                    end_tensor,
                    chunk_elements=chunk_elements,
                )
                parameter_count += start_tensor.numel()
                parameter_square_sum += start_squares
                displacement_square_sum += displacement_squares

    optimizer_steps = interval.end_step - interval.start_step
    start_parameter_norm = math.sqrt(parameter_square_sum)
    displacement_norm = math.sqrt(displacement_square_sum)
    return Displacement(
        arm=interval.arm,
        study_identity=interval.study_identity,
        namespace=interval.namespace,
        start_step=interval.start_step,
        end_step=interval.end_step,
        optimizer_steps=optimizer_steps,
        parameter_count=parameter_count,
        start_parameter_norm=start_parameter_norm,
        net_parameter_displacement_norm=displacement_norm,
        net_parameter_displacement_per_update=displacement_norm / optimizer_steps,
        relative_net_parameter_displacement=displacement_norm / start_parameter_norm,
        elapsed_seconds=time.monotonic() - started,
        start_checkpoint=str(interval.start_checkpoint),
        end_checkpoint=str(interval.end_checkpoint),
    )


def _read_completed(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _matching_completed(
    rows: Iterable[dict[str, str]],
    intervals: Iterable[CheckpointInterval],
) -> list[dict[str, str]]:
    valid_by_key = {_interval_key(interval): interval for interval in intervals}
    by_key: dict[tuple[str, str, str, int, int, str, str], dict[str, str]] = {}
    for row in rows:
        key = _completed_row_key(row)
        if key is not None and key in valid_by_key and _completed_row_is_valid(row, valid_by_key[key]):
            by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _interval_key(interval: CheckpointInterval) -> tuple[str, str, str, int, int, str, str]:
    return (
        interval.arm,
        interval.study_identity,
        interval.namespace,
        interval.start_step,
        interval.end_step,
        str(interval.start_checkpoint),
        str(interval.end_checkpoint),
    )


def _completed_row_key(
    row: dict[str, str],
) -> tuple[str, str, str, int, int, str, str] | None:
    try:
        return (
            row["arm"],
            row["study_identity"],
            row["namespace"],
            int(row["start_step"]),
            int(row["end_step"]),
            row["start_checkpoint"],
            row["end_checkpoint"],
        )
    except (KeyError, TypeError, ValueError):
        return None


def _completed_row_is_valid(row: dict[str, str], interval: CheckpointInterval) -> bool:
    try:
        optimizer_steps = int(row["optimizer_steps"])
        parameter_count = int(row["parameter_count"])
        finite_values = (
            float(row["start_parameter_norm"]),
            float(row["net_parameter_displacement_norm"]),
            float(row["net_parameter_displacement_per_update"]),
            float(row["relative_net_parameter_displacement"]),
            float(row["elapsed_seconds"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    (
        start_norm,
        displacement,
        displacement_per_update,
        relative_displacement,
        elapsed,
    ) = finite_values
    return (
        optimizer_steps == interval.end_step - interval.start_step
        and optimizer_steps > 0
        and parameter_count > 0
        and all(math.isfinite(value) and value >= 0.0 for value in finite_values)
        and start_norm > 0.0
        and elapsed > 0.0
        and math.isclose(
            displacement_per_update,
            displacement / optimizer_steps,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        and math.isclose(
            relative_displacement,
            displacement / start_norm,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    )


def _merge_completed_parts(
    parts_root: Path,
    intervals: Iterable[CheckpointInterval],
) -> list[dict[str, str]]:
    interval_list = list(intervals)
    source_rows = [row for path in sorted(parts_root.glob("*.csv")) for row in _read_completed(path)]
    completed = _matching_completed(source_rows, interval_list)
    expected_keys = {(interval.arm, interval.start_step, interval.end_step) for interval in interval_list}
    completed_keys = {(row["arm"], int(row["start_step"]), int(row["end_step"])) for row in completed}
    missing = sorted(expected_keys - completed_keys)
    if missing:
        preview = ", ".join(f"{arm}:{start}->{end}" for arm, start, end in missing[:5])
        raise ValueError(f"checkpoint displacement merge is missing {len(missing)} interval(s): {preview}")
    return completed


def _atomic_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-results-csv", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--study-identity", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--step-interval", type=int, default=10)
    parser.add_argument("--chunk-elements", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--async-max-concurrent-samples", type=int)
    parser.add_argument("--training-buffer-queue-size", type=int, default=1000)
    parser.add_argument("--arm")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--merge-parts-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.step_interval <= 0 or args.chunk_elements <= 0 or args.torch_threads <= 0:
        raise ValueError("step interval, chunk size, and torch threads must be positive")
    if args.async_max_concurrent_samples is not None and args.async_max_concurrent_samples <= 0:
        raise ValueError("async max concurrent samples must be positive")
    if args.training_buffer_queue_size <= 0:
        raise ValueError("training buffer queue size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    torch.set_num_threads(args.torch_threads)

    intervals = _read_intervals(
        args.aggregate_results_csv,
        study_root=args.study_root,
        study_identity=args.study_identity,
        namespace=args.namespace,
        step_interval=args.step_interval,
        selected_arm=args.arm,
        async_max_concurrent_samples=args.async_max_concurrent_samples,
        training_buffer_queue_size=args.training_buffer_queue_size,
    )
    if args.merge_parts_root is not None:
        if args.arm is not None or args.limit is not None:
            raise ValueError("merge mode cannot be combined with --arm or --limit")
        rows = _merge_completed_parts(args.merge_parts_root.resolve(), intervals)
        _atomic_write(args.output_csv, rows)
        print(f"merged {len(rows)} interval(s) to {args.output_csv.resolve()}")
        return

    completed = _matching_completed(_read_completed(args.output_csv), intervals)
    completed_keys = {(row["arm"], int(row["start_step"]), int(row["end_step"])) for row in completed}
    pending = [
        interval
        for interval in intervals
        if (interval.arm, interval.start_step, interval.end_step) not in completed_keys
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    rows: list[dict[str, Any]] = list(completed)
    for index, interval in enumerate(pending, start=1):
        result = _measure_interval(interval, chunk_elements=args.chunk_elements)
        rows.append(asdict(result))
        rows.sort(key=lambda row: (str(row["arm"]), int(row["start_step"])))
        _atomic_write(args.output_csv, rows)
        print(
            f"[{index}/{len(pending)}] {result.arm} {result.start_step}->{result.end_step} "
            f"delta={result.net_parameter_displacement_norm:.8g} "
            f"relative={result.relative_net_parameter_displacement:.8g} "
            f"seconds={result.elapsed_seconds:.1f}",
            flush=True,
        )

    _atomic_write(args.output_csv, rows)
    print(f"wrote {len(rows)} interval(s) to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
