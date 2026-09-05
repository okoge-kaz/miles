"""Staleness-aware attenuation for truncated zero-reward policy loss."""

from __future__ import annotations

from argparse import Namespace

import torch

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

STALENESS_AWARE_LOSS_PART_PREFIX = "_staleness_aware_loss_part/"


def compute_staleness_decay_weights(
    sample_staleness: list[int],
    *,
    safe_training_staleness: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``1 / (1 + excess staleness)`` for each sample."""
    staleness = torch.as_tensor(sample_staleness, device=device, dtype=torch.float32)
    excess = (staleness - safe_training_staleness).clamp_min(0.0)
    return (1.0 + excess).reciprocal()


def _token_sample_ids(local_masks: list[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    lengths = torch.as_tensor([mask.numel() for mask in local_masks], device=device, dtype=torch.long)
    return torch.repeat_interleave(
        torch.arange(len(local_masks), device=device, dtype=torch.long),
        lengths,
    )


def _validate_batch_rows(
    batch: RolloutBatch,
    local_masks: list[torch.Tensor],
) -> tuple[list[int], list[int]]:
    sample_staleness = batch.get("sample_staleness")
    truncated = batch.get("truncated")
    if sample_staleness is None:
        raise RuntimeError("--use-staleness-aware-loss requires complete per-sample training-staleness provenance")
    if truncated is None:
        raise RuntimeError("--use-staleness-aware-loss requires the per-sample truncated indicator")
    expected_samples = len(local_masks)
    for name, values in (("sample_staleness", sample_staleness), ("truncated", truncated)):
        if len(values) != expected_samples:
            raise ValueError(f"{name} has {len(values)} rows for {expected_samples} samples")
    return [int(value) for value in sample_staleness], [int(value) for value in truncated]


def _additive_logging_parts(
    *,
    local_mask: torch.Tensor,
    truncated_token_indicator: torch.Tensor,
    token_decay: torch.Tensor,
    truncated_samples: torch.Tensor,
    pg_loss_tokens: torch.Tensor,
    log_details: bool,
) -> dict[str, torch.Tensor]:
    parallel_state = get_parallel_state()
    local_sample_count = truncated_samples.float().sum()
    sample_count = (
        local_sample_count
        if parallel_state.cp.size == 1 or parallel_state.cp.rank == 0
        else local_sample_count.new_zeros(())
    )
    truncated_loss_tokens = truncated_token_indicator * local_mask
    parts = {
        "sample_count": sample_count,
        "loss_token_count": local_mask.sum(),
        "truncated_loss_token_count": truncated_loss_tokens.sum(),
        "truncated_weighted_loss_token_count": (truncated_loss_tokens * token_decay).sum(),
    }
    if log_details:
        active_tokens = local_mask.bool()
        pre_scaling_objective = torch.where(
            active_tokens,
            pg_loss_tokens.detach().float().abs(),
            pg_loss_tokens.new_zeros((), dtype=torch.float32),
        )
        post_scaling_objective = pre_scaling_objective * token_decay
        parts |= {
            "post_tis_pre_scaling_abs_pg_objective": pre_scaling_objective.sum(),
            "truncated_zero_post_tis_pre_scaling_abs_pg_objective": (
                pre_scaling_objective * truncated_token_indicator
            ).sum(),
            "post_tis_post_scaling_abs_pg_objective": post_scaling_objective.sum(),
            "truncated_zero_post_tis_post_scaling_abs_pg_objective": (
                post_scaling_objective * truncated_token_indicator
            ).sum(),
        }
    return {f"{STALENESS_AWARE_LOSS_PART_PREFIX}{name}": value.detach() for name, value in parts.items()}


def apply_staleness_aware_loss(
    *,
    args: Namespace,
    batch: RolloutBatch,
    pg_loss_tokens: torch.Tensor,
    final_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Attenuate truncated zero-reward samples and return additive log parts.

    Non-truncated samples retain unit weight regardless of their staleness. Log
    parts use the final post-correction loss mask and piggyback on the existing
    training-metric reduction.
    """
    if not getattr(args, "use_staleness_aware_loss", False):
        return pg_loss_tokens, {}
    if not getattr(args, "zero_reward_on_truncated", False):
        raise RuntimeError("--use-staleness-aware-loss requires --zero-reward-on-truncated")
    if getattr(args, "log_staleness_aware_loss_details", False) and not getattr(args, "use_tis", False):
        raise RuntimeError("--log-staleness-aware-loss-details requires --use-tis")

    local_masks = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        final_masks,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    sample_staleness, truncated = _validate_batch_rows(batch, local_masks)
    device = pg_loss_tokens.device
    sample_ids = _token_sample_ids(local_masks, device=device)
    local_mask = torch.cat(local_masks).to(device=device, dtype=torch.float32)
    if pg_loss_tokens.numel() != sample_ids.numel():
        raise ValueError(
            f"pg_loss_tokens has {pg_loss_tokens.numel()} tokens, expected {sample_ids.numel()} from response masks"
        )

    truncated_samples = torch.as_tensor(truncated, device=device, dtype=torch.bool)
    sample_decay = compute_staleness_decay_weights(
        sample_staleness,
        safe_training_staleness=args.safe_training_staleness,
        device=device,
    )
    sample_decay = torch.where(truncated_samples, sample_decay, torch.ones_like(sample_decay))
    token_decay = sample_decay[sample_ids]
    truncated_token_indicator = truncated_samples[sample_ids].float()
    weighted_pg_loss_tokens = pg_loss_tokens * token_decay.to(dtype=pg_loss_tokens.dtype)
    parts = _additive_logging_parts(
        local_mask=local_mask,
        truncated_token_indicator=truncated_token_indicator,
        token_decay=token_decay,
        truncated_samples=truncated_samples,
        pg_loss_tokens=pg_loss_tokens,
        log_details=getattr(args, "log_staleness_aware_loss_details", False),
    )
    return weighted_pg_loss_tokens, parts


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def finalize_staleness_aware_loss_parts(metric_sums: dict[str, float]) -> dict[str, float]:
    """Convert globally summed implementation parts into public metrics."""
    prefix = STALENESS_AWARE_LOSS_PART_PREFIX
    sample_count = metric_sums.get(f"{prefix}sample_count")
    if sample_count is None:
        return {}
    loss_tokens = metric_sums[f"{prefix}loss_token_count"]
    truncated_loss_tokens = metric_sums[f"{prefix}truncated_loss_token_count"]
    metrics = {
        "staleness_aware_loss/truncated_zero_reward_sample_count": sample_count,
        "staleness_aware_loss/truncated_zero_reward_loss_token_fraction": _safe_ratio(
            truncated_loss_tokens, loss_tokens
        ),
        "staleness_aware_loss/truncated_zero_reward_mean_gradient_scale": _safe_ratio(
            metric_sums[f"{prefix}truncated_weighted_loss_token_count"], truncated_loss_tokens
        ),
    }
    pre_scaling_objective = metric_sums.get(f"{prefix}post_tis_pre_scaling_abs_pg_objective")
    if pre_scaling_objective is None:
        return metrics

    truncated_pre_scaling_objective = metric_sums[f"{prefix}truncated_zero_post_tis_pre_scaling_abs_pg_objective"]
    post_scaling_objective = metric_sums[f"{prefix}post_tis_post_scaling_abs_pg_objective"]
    truncated_post_scaling_objective = metric_sums[f"{prefix}truncated_zero_post_tis_post_scaling_abs_pg_objective"]
    metrics |= {
        "staleness_aware_loss/post_tis_pre_scaling_abs_pg_objective_per_loss_token": _safe_ratio(
            pre_scaling_objective, loss_tokens
        ),
        "staleness_aware_loss/truncated_zero_reward_post_tis_pre_scaling_abs_pg_objective_per_all_loss_token": _safe_ratio(
            truncated_pre_scaling_objective, loss_tokens
        ),
        "staleness_aware_loss/truncated_zero_reward_post_tis_pre_scaling_abs_pg_objective_fraction": _safe_ratio(
            truncated_pre_scaling_objective, pre_scaling_objective
        ),
        "staleness_aware_loss/post_tis_post_scaling_abs_pg_objective_per_loss_token": _safe_ratio(
            post_scaling_objective, loss_tokens
        ),
        "staleness_aware_loss/truncated_zero_reward_post_tis_post_scaling_abs_pg_objective_per_all_loss_token": _safe_ratio(
            truncated_post_scaling_objective, loss_tokens
        ),
        "staleness_aware_loss/truncated_zero_reward_post_tis_post_scaling_abs_pg_objective_fraction": _safe_ratio(
            truncated_post_scaling_objective, post_scaling_objective
        ),
    }
    return metrics
