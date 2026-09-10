"""Low-overhead diagnostics describing the data and optimizer update."""

from __future__ import annotations

import math
from argparse import Namespace

import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.utils.types import RolloutBatch

UPDATE_PART_PREFIX = "_update_diagnostic_part/"


def compute_update_diagnostic_parts(
    args: Namespace,
    batch: RolloutBatch,
    advantages: torch.Tensor,
    final_loss_masks: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return additive token statistics which piggyback on loss reduction."""
    if not getattr(args, "log_update_diagnostics", False):
        return {}

    token_sum = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        final_loss_masks,
        calculate_per_token_loss=True,
        qkv_format=args.qkv_format,
        max_seq_lens=batch.get("max_seq_lens"),
    )
    values = advantages.detach().float()
    return {
        f"{UPDATE_PART_PREFIX}final_loss_tokens": token_sum(torch.ones_like(values)),
        f"{UPDATE_PART_PREFIX}advantage_sum": token_sum(values),
        f"{UPDATE_PART_PREFIX}advantage_square_sum": token_sum(values.square()),
        f"{UPDATE_PART_PREFIX}advantage_abs_sum": token_sum(values.abs()),
    }


def finalize_update_diagnostic_parts(metric_sums: dict[str, float]) -> dict[str, float]:
    """Build public update diagnostics from globally reduced additive parts."""
    final_loss_tokens = metric_sums.get(f"{UPDATE_PART_PREFIX}final_loss_tokens")
    if final_loss_tokens is None:
        return {}
    if final_loss_tokens <= 0:
        return {
            "final_loss_tokens": 0.0,
            "advantage_std": 0.0,
            "advantage_rms": 0.0,
            "advantage_abs_mean": 0.0,
        }

    advantage_mean = metric_sums[f"{UPDATE_PART_PREFIX}advantage_sum"] / final_loss_tokens
    advantage_square_mean = metric_sums[f"{UPDATE_PART_PREFIX}advantage_square_sum"] / final_loss_tokens
    advantage_abs_mean = metric_sums[f"{UPDATE_PART_PREFIX}advantage_abs_sum"] / final_loss_tokens
    variance = max(advantage_square_mean - advantage_mean * advantage_mean, 0.0)
    return {
        "final_loss_tokens": final_loss_tokens,
        "advantage_std": math.sqrt(variance),
        "advantage_rms": math.sqrt(max(advantage_square_mean, 0.0)),
        "advantage_abs_mean": advantage_abs_mean,
    }


def optimizer_step_diagnostics(
    args: Namespace,
    *,
    optimizer_step_applied: bool,
    grad_norm: float | torch.Tensor,
    num_zeros_in_grad: float | torch.Tensor | None,
) -> dict[str, float]:
    """Format statistics already produced by the Megatron optimizer step."""
    if not getattr(args, "log_update_diagnostics", False):
        return {}

    grad_norm_value = float(grad_norm)
    clip_grad = float(args.clip_grad)
    clip_coefficient = 1.0
    if clip_grad > 0.0 and grad_norm_value > 0.0:
        clip_coefficient = min(1.0, clip_grad / (grad_norm_value + 1.0e-6))

    metrics = {
        "optimizer_step_applied": float(optimizer_step_applied),
        "grad_norm_pre_clip": grad_norm_value,
        "grad_clip_coefficient": clip_coefficient,
    }
    if num_zeros_in_grad is not None:
        metrics["num_zeros_in_grad"] = float(num_zeros_in_grad)
    return metrics
