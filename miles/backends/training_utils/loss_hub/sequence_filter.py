"""Detached, response-wide train/rollout probability-error masking."""

from argparse import Namespace

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks, get_sum_of_sample_mean
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

SEQUENCE_FILTER_PART_PREFIX = "_train_rollout_sequence_filter/"
SEQUENCE_FILTER_TOKEN_COUNT = "_train_rollout_sequence_filter_loss_tokens"


@torch.no_grad()
def filter_train_rollout_logprob_sequences(
    args: Namespace,
    batch: RolloutBatch,
    train_log_probs: list[torch.Tensor],
) -> tuple[RolloutBatch, dict[str, torch.Tensor]]:
    """Keep responses with mean(exp(abs(log pi_train - log mu_rollout))) <= threshold.

    The mask uses the actual training forward. CP ranks sum numerator/count pairs
    so the decision covers the full response. Inputs and rewards are unchanged;
    rejected responses contribute neither loss nor normalization tokens.
    """
    rollout_log_probs = batch.get("rollout_log_probs")
    if rollout_log_probs is None:
        raise ValueError("--use-train-rollout-logprob-sequence-filter requires rollout_log_probs")
    cp = get_parallel_state().cp
    local_masks = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    parts = []
    for train, rollout, mask in zip(train_log_probs, rollout_log_probs, local_masks, strict=True):
        mask = mask.to(device=train.device, dtype=torch.float32)
        delta = (train.detach().float() - rollout.detach().float()).abs()
        # Ignore padding before exp: inf * 0 and nan * 0 would pollute the sum.
        error = torch.where(mask.bool(), delta, torch.zeros_like(delta)).exp()
        parts.append(torch.stack(((error * mask).sum(), mask.sum())))
    totals = torch.stack(parts)
    if cp.size > 1:
        dist.all_reduce(totals, group=cp.group)
    errors = totals[:, 0] / totals[:, 1].clamp_min(1)
    valid = totals[:, 1] > 0
    keep = valid & torch.isfinite(errors) & (errors <= args.train_rollout_logprob_sequence_filter_threshold)
    masks = [mask * kept.to(mask) for mask, kept in zip(batch["loss_masks"], keep, strict=True)]
    kept_tokens = torch.stack([mask.sum() for mask in masks]).sum()
    # Full-response counters are replicated over CP, unlike token-loss sums.
    counters = {
        "sequences": valid.sum(),
        "rejected_sequences": (valid & ~keep).sum(),
        "tokens": totals[:, 1].sum(),
        "rejected_tokens": (totals[:, 1] * ~keep).sum(),
    }
    metrics = {SEQUENCE_FILTER_PART_PREFIX + key: value.float() / cp.size for key, value in counters.items()}
    metrics[SEQUENCE_FILTER_TOKEN_COUNT] = kept_tokens
    return {**batch, "loss_masks": masks}, metrics


def sequence_filter_reducer(args: Namespace, batch: RolloutBatch):
    """Rebuild the ordinary reducer with the response masks from the filter."""
    return get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
        denominators=batch.get("rollout_mask_sums"),
    )


def finalize_sequence_filter_metrics(parts: dict[str, float]) -> dict[str, float]:
    """Form rejection fractions after summing across microbatches and DP/CP."""
    if SEQUENCE_FILTER_PART_PREFIX + "sequences" not in parts:
        return {}
    counts = {
        key.removeprefix(SEQUENCE_FILTER_PART_PREFIX): value
        for key, value in parts.items()
        if key.startswith(SEQUENCE_FILTER_PART_PREFIX)
    }
    prefix = "train_rollout_logprob_sequence_filter/"
    return {
        prefix + "rejected_sequence_fraction": counts["rejected_sequences"] / max(counts["sequences"], 1),
        prefix + "rejected_token_fraction": counts["rejected_tokens"] / max(counts["tokens"], 1),
        prefix + "rejected_sequences": counts["rejected_sequences"],
        prefix + "kept_tokens": counts["tokens"] - counts["rejected_tokens"],
    }
