#!/usr/bin/env python3
"""Compare train-data dumps from fixed-batch legacy and fused one-step runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--fused-dir", type=Path, required=True)
    parser.add_argument("--tis-clip-low", type=float, default=0.0)
    parser.add_argument("--tis-clip", type=float, default=2.0)
    return parser.parse_args()


def _load_rollout_data(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["rollout_data"]


def _flatten_valid(
    values: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    return torch.cat(
        [
            value.detach().float().reshape(-1)[loss_mask.detach().bool().reshape(-1)]
            for value, loss_mask in zip(values, loss_masks, strict=True)
        ]
    )


def _absolute_summary(values: list[torch.Tensor]) -> dict[str, float]:
    absolute = torch.cat(values).abs()
    return {
        "mean": absolute.mean().item(),
        "p50": torch.quantile(absolute, 0.50).item(),
        "p95": torch.quantile(absolute, 0.95).item(),
        "p99": torch.quantile(absolute, 0.99).item(),
        "max": absolute.max().item(),
    }


def main() -> None:
    args = _parse_args()
    legacy_paths = sorted(args.legacy_dir.glob("*.pt"))
    fused_paths = sorted(args.fused_dir.glob("*.pt"))
    if not legacy_paths or [path.name for path in legacy_paths] != [path.name for path in fused_paths]:
        raise RuntimeError("Legacy and fused train-data dump files do not match")

    differences: dict[str, list[torch.Tensor]] = {
        "advantages": [],
        "returns": [],
        "actor_anchor_logprobs": [],
        "rollout_logprobs": [],
        "raw_tis_weights": [],
        "clipped_tis_weights": [],
    }
    disagreement_count = 0
    valid_token_count = 0
    for legacy_path, fused_path in zip(legacy_paths, fused_paths, strict=True):
        legacy = _load_rollout_data(legacy_path)
        fused = _load_rollout_data(fused_path)
        legacy_masks = legacy["loss_masks"]
        fused_masks = fused["loss_masks"]
        legacy_actor = _flatten_valid(legacy["log_probs"], legacy_masks)
        fused_actor = _flatten_valid(fused["legacy_actor_log_probs"], fused_masks)
        legacy_rollout = _flatten_valid(legacy["rollout_log_probs"], legacy_masks)
        fused_rollout = _flatten_valid(fused["rollout_log_probs"], fused_masks)

        differences["advantages"].append(
            _flatten_valid(legacy["advantages"], legacy_masks)
            - _flatten_valid(fused["advantages"], fused_masks)
        )
        differences["returns"].append(
            _flatten_valid(legacy["returns"], legacy_masks)
            - _flatten_valid(fused["returns"], fused_masks)
        )
        differences["actor_anchor_logprobs"].append(legacy_actor - fused_actor)
        differences["rollout_logprobs"].append(legacy_rollout - fused_rollout)

        legacy_tis = torch.exp(legacy_actor - legacy_rollout)
        fused_tis = torch.exp(fused_actor - fused_rollout)
        differences["raw_tis_weights"].append(legacy_tis - fused_tis)
        legacy_clipped = torch.clamp(legacy_tis, min=args.tis_clip_low, max=args.tis_clip)
        fused_clipped = torch.clamp(fused_tis, min=args.tis_clip_low, max=args.tis_clip)
        differences["clipped_tis_weights"].append(legacy_clipped - fused_clipped)
        disagreement_count += (
            (legacy_clipped != legacy_tis) != (fused_clipped != fused_tis)
        ).sum().item()
        valid_token_count += legacy_tis.numel()

    result = {name: _absolute_summary(values) for name, values in differences.items()}
    result["tis_clip_decision_disagreement_fraction"] = disagreement_count / valid_token_count
    result["dump_file_count"] = len(legacy_paths)
    result["valid_token_count"] = valid_token_count
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
