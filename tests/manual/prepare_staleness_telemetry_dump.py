#!/usr/bin/env python3
"""Add deterministic sample-staleness provenance to a fixed rollout dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from miles.rollout.recycle_compute_metrics import (
    BOUND_REFERENCE_VERSION_KEY,
    DRAIN_VERSION_KEY,
    SAMPLE_REFERENCE_VERSION_KEY,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--lags", type=int, nargs="+", default=(0, 1, 2, 4, 8, 17))
    parser.add_argument("--drain-version", type=int, default=100)
    return parser.parse_args()


def _group_key(sample: dict, row_index: int, group_size: int) -> tuple[str, int]:
    group_index = sample.get("group_index")
    if isinstance(group_index, int):
        return "group_index", group_index
    return "row_chunk", row_index // group_size


def main() -> None:
    args = _parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if not args.lags or min(args.lags) < 0:
        raise ValueError("--lags must contain non-negative integers")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{args.input} does not contain a non-empty samples list")

    lag_by_group: dict[tuple[str, int], int] = {}
    lag_counts = {lag: 0 for lag in args.lags}
    for row_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(f"sample row {row_index} is not a dictionary")
        group_key = _group_key(sample, row_index, args.group_size)
        if group_key not in lag_by_group:
            lag_by_group[group_key] = args.lags[len(lag_by_group) % len(args.lags)]
        lag = lag_by_group[group_key]
        reference = args.drain_version - lag
        metadata = sample.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError(f"sample row {row_index} metadata is not a dictionary")
        metadata[SAMPLE_REFERENCE_VERSION_KEY] = reference
        metadata[BOUND_REFERENCE_VERSION_KEY] = reference
        metadata[DRAIN_VERSION_KEY] = args.drain_version
        lag_counts[lag] += 1

    payload_metadata = payload.setdefault("metadata", {})
    payload_metadata["staleness_telemetry_validation"] = {
        "source": str(args.input),
        "drain_version": args.drain_version,
        "lag_counts": lag_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "samples": len(samples),
                "groups": len(lag_by_group),
                "lag_counts": lag_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
