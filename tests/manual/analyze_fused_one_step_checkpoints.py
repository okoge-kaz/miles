#!/usr/bin/env python3
"""Compare one-step legacy and fused optimizer states without loading the model.

The parameter delta reconstruction assumes zero-initialized Adam moments,
zero weight decay, and exactly one update. The omitted learning-rate factor
does not affect cosine similarity or relative L2 error.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


_EXP_AVG_SUFFIX = ".exp_avg"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--chunk-elements", type=int, default=8_000_000)
    return parser.parse_args()


def _tensor_metadata(reader: dcp.FileSystemReader) -> dict:
    return reader.read_metadata().state_dict_metadata


def _load_tensor(reader: dcp.FileSystemReader, metadata: dict, key: str) -> torch.Tensor:
    tensor_metadata = metadata[key]
    tensor = torch.empty(tuple(tensor_metadata.size), dtype=tensor_metadata.properties.dtype)
    dcp.load({key: tensor}, storage_reader=reader, no_dist=True)
    return tensor


def _comparison_sums(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_elements: int,
    scale: float = 1.0,
) -> dict[str, float]:
    sums = {
        "dot": 0.0,
        "left_sq": 0.0,
        "right_sq": 0.0,
        "diff_sq": 0.0,
        "max_abs": 0.0,
    }
    for start in range(0, left.numel(), chunk_elements):
        end = min(start + chunk_elements, left.numel())
        left_chunk = left[start:end].double().mul_(scale)
        right_chunk = right[start:end].double().mul_(scale)
        difference = left_chunk - right_chunk
        sums["dot"] += torch.dot(left_chunk, right_chunk).item()
        sums["left_sq"] += torch.dot(left_chunk, left_chunk).item()
        sums["right_sq"] += torch.dot(right_chunk, right_chunk).item()
        sums["diff_sq"] += torch.dot(difference, difference).item()
        sums["max_abs"] = max(sums["max_abs"], difference.abs().max().item())
    return sums


def _adam_first_step_delta(
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    beta1: float,
    beta2: float,
    epsilon: float,
    chunk_elements: int,
) -> torch.Tensor:
    delta = torch.empty_like(exp_avg)
    for start in range(0, exp_avg.numel(), chunk_elements):
        end = min(start + chunk_elements, exp_avg.numel())
        first_moment = exp_avg[start:end] / (1.0 - beta1)
        second_moment = exp_avg_sq[start:end] / (1.0 - beta2)
        delta[start:end] = first_moment / (second_moment.sqrt() + epsilon)
    return delta


def _merge_sums(total: dict[str, float], value: dict[str, float]) -> None:
    for key in ("dot", "left_sq", "right_sq", "diff_sq"):
        total[key] += value[key]
    total["max_abs"] = max(total["max_abs"], value["max_abs"])


def _finalize(sums: dict[str, float]) -> dict[str, float]:
    return {
        "left_l2": math.sqrt(sums["left_sq"]),
        "right_l2": math.sqrt(sums["right_sq"]),
        "cosine_similarity": sums["dot"]
        / math.sqrt(sums["left_sq"] * sums["right_sq"]),
        "relative_l2_error": math.sqrt(sums["diff_sq"] / sums["left_sq"]),
        "max_abs_difference": sums["max_abs"],
    }


def main() -> None:
    args = _parse_args()
    legacy_reader = dcp.FileSystemReader(args.legacy)
    fused_reader = dcp.FileSystemReader(args.fused)
    legacy_metadata = _tensor_metadata(legacy_reader)
    fused_metadata = _tensor_metadata(fused_reader)
    exp_avg_keys = sorted(key for key in legacy_metadata if key.endswith(_EXP_AVG_SUFFIX))
    if not exp_avg_keys or any(key not in fused_metadata for key in exp_avg_keys):
        raise RuntimeError("The checkpoints do not contain matching Adam first-moment tensors")

    empty_sums = {
        "dot": 0.0,
        "left_sq": 0.0,
        "right_sq": 0.0,
        "diff_sq": 0.0,
        "max_abs": 0.0,
    }
    gradient_sums = dict(empty_sums)
    delta_sums = dict(empty_sums)
    for exp_avg_key in exp_avg_keys:
        exp_avg_sq_key = exp_avg_key.removesuffix(_EXP_AVG_SUFFIX) + ".exp_avg_sq"
        legacy_exp_avg = _load_tensor(legacy_reader, legacy_metadata, exp_avg_key)
        fused_exp_avg = _load_tensor(fused_reader, fused_metadata, exp_avg_key)
        _merge_sums(
            gradient_sums,
            _comparison_sums(
                legacy_exp_avg,
                fused_exp_avg,
                chunk_elements=args.chunk_elements,
                scale=1.0 / (1.0 - args.beta1),
            ),
        )

        legacy_exp_avg_sq = _load_tensor(legacy_reader, legacy_metadata, exp_avg_sq_key)
        legacy_delta = _adam_first_step_delta(
            legacy_exp_avg,
            legacy_exp_avg_sq,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            chunk_elements=args.chunk_elements,
        )
        del legacy_exp_avg, legacy_exp_avg_sq
        gc.collect()

        fused_exp_avg_sq = _load_tensor(fused_reader, fused_metadata, exp_avg_sq_key)
        fused_delta = _adam_first_step_delta(
            fused_exp_avg,
            fused_exp_avg_sq,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            chunk_elements=args.chunk_elements,
        )
        del fused_exp_avg, fused_exp_avg_sq
        gc.collect()
        _merge_sums(
            delta_sums,
            _comparison_sums(
                legacy_delta,
                fused_delta,
                chunk_elements=args.chunk_elements,
            ),
        )
        del legacy_delta, fused_delta
        gc.collect()

    print(
        json.dumps(
            {
                "gradient": _finalize(gradient_sums),
                "adam_first_step_parameter_delta_without_lr": _finalize(delta_sums),
                "optimizer_tensor_count": len(exp_avg_keys),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
