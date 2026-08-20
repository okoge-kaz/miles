"""Feature-gated trainer diagnostics grouped by sample staleness."""

from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

_PART_PREFIX = "_sample_staleness_part/"
_RATIO_HISTOGRAM_EDGES = (
    -1.0,
    -0.3,
    -0.1,
    -0.03,
    -0.01,
    -0.003,
    -0.001,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
)


@contextmanager
def _allow_nondeterministic_telemetry_reductions():
    """Permit detached CUDA bincounts without weakening training kernels."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    if deterministic:
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)


def _bin_label(index: int, max_staleness: int) -> str:
    return f"s_{index}" if index <= max_staleness else f"s_ge_{max_staleness + 1}"


def _part_key(statistic: str, index: int) -> str:
    return f"{_PART_PREFIX}{statistic}/{index}"


def _bincount(
    indices: torch.Tensor,
    *,
    weights: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    return torch.bincount(indices, weights=weights, minlength=num_bins)[:num_bins]


def _importance_clip_indicator(
    tis_metrics: dict[str, torch.Tensor] | None,
    template: torch.Tensor,
) -> torch.Tensor:
    clipped = torch.zeros_like(template, dtype=torch.bool)
    for name, value in (tis_metrics or {}).items():
        if not any(token in name for token in ("clipfrac", "clip_fraction", "truncate_fraction")):
            continue
        if isinstance(value, torch.Tensor) and value.shape == template.shape:
            clipped |= value.detach() != 0
    return clipped.float()


def _token_layout(
    sample_staleness: list[int],
    local_masks: list[torch.Tensor],
    *,
    max_staleness: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([mask.numel() for mask in local_masks], device=device, dtype=torch.long)
    sample_bins = torch.as_tensor(sample_staleness, device=device, dtype=torch.long).clamp(
        min=0,
        max=max_staleness + 1,
    )
    sample_ids = torch.repeat_interleave(
        torch.arange(len(local_masks), device=device, dtype=torch.long),
        lengths,
    )
    return sample_bins, sample_ids, sample_bins[sample_ids]


def _objective_normalization(
    args: Namespace,
    final_masks: list[torch.Tensor],
    final_local_masks: torch.Tensor,
    sample_ids: torch.Tensor,
) -> torch.Tensor:
    if args.calculate_per_token_loss:
        return final_local_masks
    denominators = torch.stack(
        [mask.to(device=final_local_masks.device, dtype=torch.float32).sum() for mask in final_masks]
    ).clamp_min(1.0)
    return final_local_masks / denominators[sample_ids]


def _sequence_ess_parts(
    *,
    policy_log_ratio: torch.Tensor,
    final_local_masks: torch.Tensor,
    sample_ids: torch.Tensor,
    sample_bins: torch.Tensor,
    num_samples: int,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parallel_state = get_parallel_state()
    log_weights = _bincount(
        sample_ids,
        weights=policy_log_ratio * final_local_masks,
        num_bins=num_samples,
    )
    active_token_counts = _bincount(
        sample_ids,
        weights=final_local_masks,
        num_bins=num_samples,
    )
    if parallel_state.cp.size > 1:
        sequence_parts = torch.stack((log_weights, active_token_counts))
        dist.all_reduce(sequence_parts, op=dist.ReduceOp.SUM, group=parallel_state.cp.group)
        log_weights, active_token_counts = sequence_parts
    if parallel_state.cp.size > 1 and parallel_state.cp.rank != 0:
        zeros = log_weights.new_zeros(num_bins, dtype=torch.float64)
        return zeros, zeros, zeros

    active_samples = active_token_counts > 0
    active_bins = sample_bins[active_samples]
    active_log_weights = log_weights[active_samples]
    if active_log_weights.numel() == 0:
        zeros = log_weights.new_zeros(num_bins, dtype=torch.float64)
        return zeros, zeros, zeros
    # These are additive parts across microbatches and DP ranks. A per-call max
    # shift would make otherwise identical populations produce different ESS
    # after aggregation. Float64 keeps the unshifted sequence weights usable;
    # the cap is only a guard beyond the representable policy-ratio regime.
    sequence_weights = active_log_weights.double().clamp(min=-300.0, max=300.0).exp()
    sum_w = _bincount(active_bins, weights=sequence_weights, num_bins=num_bins)
    sum_w2 = _bincount(active_bins, weights=sequence_weights * sequence_weights, num_bins=num_bins)
    counts = _bincount(
        active_bins,
        weights=torch.ones_like(sequence_weights),
        num_bins=num_bins,
    )
    return sum_w, sum_w2, counts


def _token_parts(
    *,
    token_bins: torch.Tensor,
    pre_mask: torch.Tensor,
    final_mask: torch.Tensor,
    objective_contribution: torch.Tensor,
    policy_log_ratio: torch.Tensor,
    objective_log_ratio: torch.Tensor,
    ppo_clipfrac: torch.Tensor,
    importance_clip: torch.Tensor,
    num_bins: int,
) -> dict[str, torch.Tensor]:
    safe_policy_log_ratio = torch.nan_to_num(
        policy_log_ratio.detach().float(), nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    safe_objective_log_ratio = torch.nan_to_num(
        objective_log_ratio.detach().float(), nan=0.0, posinf=20.0, neginf=-20.0
    ).clamp(-20.0, 20.0)
    policy_ratio = safe_policy_log_ratio.exp()
    correction_masked = ((pre_mask > 0) & (final_mask <= 0)).float()
    return {
        "response_token_count": _bincount(
            token_bins,
            weights=torch.ones_like(pre_mask),
            num_bins=num_bins,
        ),
        "token_count": _bincount(token_bins, weights=pre_mask, num_bins=num_bins),
        "final_token_count": _bincount(token_bins, weights=final_mask, num_bins=num_bins),
        "effective_contribution": _bincount(token_bins, weights=objective_contribution, num_bins=num_bins),
        "nonzero_contribution_count": _bincount(
            token_bins,
            weights=((objective_contribution > 0) & (pre_mask > 0)).float(),
            num_bins=num_bins,
        ),
        "abs_policy_log_ratio_sum": _bincount(
            token_bins,
            weights=safe_policy_log_ratio.abs() * pre_mask,
            num_bins=num_bins,
        ),
        "abs_ppo_objective_log_ratio_sum": _bincount(
            token_bins,
            weights=safe_objective_log_ratio.abs() * pre_mask,
            num_bins=num_bins,
        ),
        "ppo_clip_count": _bincount(token_bins, weights=ppo_clipfrac.detach() * pre_mask, num_bins=num_bins),
        "importance_clip_count": _bincount(token_bins, weights=importance_clip * pre_mask, num_bins=num_bins),
        "correction_mask_count": _bincount(token_bins, weights=correction_masked, num_bins=num_bins),
        "token_ratio_sum_w": _bincount(
            token_bins,
            weights=policy_ratio * final_mask,
            num_bins=num_bins,
        ),
        "token_ratio_sum_w2": _bincount(
            token_bins,
            weights=policy_ratio.square() * final_mask,
            num_bins=num_bins,
        ),
    }


def _histogram_parts(
    *,
    token_bins: torch.Tensor,
    policy_log_ratio: torch.Tensor,
    pre_mask: torch.Tensor,
    num_bins: int,
) -> dict[str, torch.Tensor]:
    edges = torch.tensor(_RATIO_HISTOGRAM_EDGES, device=policy_log_ratio.device, dtype=torch.float32)
    safe_log_ratio = torch.nan_to_num(
        policy_log_ratio.detach().float(),
        nan=0.0,
        posinf=float(_RATIO_HISTOGRAM_EDGES[-1]),
        neginf=float(_RATIO_HISTOGRAM_EDGES[0]),
    )
    ratio_bins = torch.bucketize(safe_log_ratio, edges)
    num_ratio_bins = len(_RATIO_HISTOGRAM_EDGES) + 1
    joint_bins = token_bins * num_ratio_bins + ratio_bins
    counts = _bincount(
        joint_bins,
        weights=pre_mask,
        num_bins=num_bins * num_ratio_bins,
    ).reshape(num_bins, num_ratio_bins)
    return {f"ratio_hist_{index}": counts[:, index] for index in range(num_ratio_bins)}


def compute_sample_staleness_parts(
    *,
    args: Namespace,
    batch: RolloutBatch,
    original_local_masks: list[torch.Tensor],
    final_masks: list[torch.Tensor],
    pg_loss_tokens: torch.Tensor,
    ppo_clipfrac: torch.Tensor,
    policy_log_ratio: torch.Tensor,
    objective_log_ratio: torch.Tensor,
    tis_metrics: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Return detached additive parts; the global reducer forms all ratios."""
    sample_staleness = batch.get("sample_staleness")
    if not getattr(args, "log_sample_staleness_metrics", False) or sample_staleness is None:
        return {}
    if len(sample_staleness) != len(original_local_masks):
        raise ValueError(f"sample_staleness has {len(sample_staleness)} rows for {len(original_local_masks)} samples")

    max_staleness = int(getattr(args, "sample_staleness_max_bin", 16))
    num_bins = max_staleness + 2
    device = pg_loss_tokens.device
    final_local_mask_list = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        final_masks,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )
    pre_mask = torch.cat(original_local_masks, dim=0).to(device=device, dtype=torch.float32)
    final_mask = torch.cat(final_local_mask_list, dim=0).to(device=device, dtype=torch.float32)
    sample_bins, sample_ids, token_bins = _token_layout(
        sample_staleness,
        original_local_masks,
        max_staleness=max_staleness,
        device=device,
    )
    expected_tokens = token_bins.numel()
    for name, tensor in (
        ("pg_loss_tokens", pg_loss_tokens),
        ("ppo_clipfrac", ppo_clipfrac),
        ("policy_log_ratio", policy_log_ratio),
        ("objective_log_ratio", objective_log_ratio),
        ("pre_mask", pre_mask),
        ("final_mask", final_mask),
    ):
        if tensor.numel() != expected_tokens:
            raise ValueError(f"{name} has {tensor.numel()} tokens, expected {expected_tokens} from response masks")
    objective_norm = _objective_normalization(args, final_masks, final_mask, sample_ids)
    contribution = pg_loss_tokens.detach().float().abs() * objective_norm
    parallel_state = get_parallel_state()
    sample_weights = torch.ones_like(sample_bins, dtype=torch.float32)
    if parallel_state.cp.size > 1 and parallel_state.cp.rank != 0:
        sample_weights.zero_()
    # CUDA bincount uses atomic additions and is not admitted by PyTorch's
    # deterministic-algorithm guard. These tensors are detached diagnostics;
    # keep the exception scoped to their dispatch and restore the training
    # setting before returning to the loss.
    with _allow_nondeterministic_telemetry_reductions():
        parts = _token_parts(
            token_bins=token_bins,
            pre_mask=pre_mask,
            final_mask=final_mask,
            objective_contribution=contribution,
            policy_log_ratio=policy_log_ratio,
            objective_log_ratio=objective_log_ratio,
            ppo_clipfrac=ppo_clipfrac,
            importance_clip=_importance_clip_indicator(tis_metrics, pg_loss_tokens),
            num_bins=num_bins,
        )
        seq_sum_w, seq_sum_w2, seq_count = _sequence_ess_parts(
            policy_log_ratio=policy_log_ratio.detach().float(),
            final_local_masks=final_mask,
            sample_ids=sample_ids,
            sample_bins=sample_bins,
            num_samples=len(original_local_masks),
            num_bins=num_bins,
        )
        parts |= {
            "sequence_ratio_sum_w": seq_sum_w,
            "sequence_ratio_sum_w2": seq_sum_w2,
            "sequence_count": seq_count,
        }
        if getattr(args, "log_sample_staleness_ratio_histogram", False):
            parts |= _histogram_parts(
                token_bins=token_bins,
                policy_log_ratio=policy_log_ratio,
                pre_mask=pre_mask,
                num_bins=num_bins,
            )
        parts["sample_count"] = _bincount(sample_bins, weights=sample_weights, num_bins=num_bins)
    return {
        _part_key(statistic, index): values[index].detach()
        for statistic, values in parts.items()
        for index in range(num_bins)
    }


def _pop_vector(metrics: dict[str, float], statistic: str, num_bins: int) -> list[float]:
    return [float(metrics.pop(_part_key(statistic, index), 0.0)) for index in range(num_bins)]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _ratio_histogram_labels() -> tuple[str, ...]:
    edges = _RATIO_HISTOGRAM_EDGES
    labels = [f"lt_{edges[0]:g}"]
    labels.extend(f"{left:g}_to_{right:g}" for left, right in zip(edges[:-1], edges[1:], strict=True))
    labels.append(f"ge_{edges[-1]:g}")
    return tuple(label.replace("-", "neg_").replace(".", "p") for label in labels)


def _approx_abs_p95(histogram: list[float]) -> float:
    total = sum(histogram)
    if total <= 0.0:
        return 0.0
    thresholds = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    edges = _RATIO_HISTOGRAM_EDGES
    target = 0.95 * total
    for threshold in thresholds:
        inside = sum(
            count
            for index, count in enumerate(histogram)
            if 0 < index < len(edges) and edges[index - 1] >= -threshold and edges[index] <= threshold
        )
        if inside >= target:
            return threshold
    return 1.0


def finalize_sample_staleness_metrics(
    metrics: dict[str, float],
    *,
    max_staleness: int | None = None,
) -> None:
    """Replace additive implementation parts with global, interpretable ratios."""
    if max_staleness is None:
        token_prefix = f"{_PART_PREFIX}token_count/"
        indices = [int(key.removeprefix(token_prefix)) for key in metrics if key.startswith(token_prefix)]
        if not indices:
            return
        max_staleness = max(indices) - 1
    num_bins = max_staleness + 2
    if _part_key("token_count", 0) not in metrics:
        return
    statistics = (
        "sample_count",
        "response_token_count",
        "token_count",
        "final_token_count",
        "effective_contribution",
        "nonzero_contribution_count",
        "abs_policy_log_ratio_sum",
        "abs_ppo_objective_log_ratio_sum",
        "ppo_clip_count",
        "importance_clip_count",
        "correction_mask_count",
        "token_ratio_sum_w",
        "token_ratio_sum_w2",
        "sequence_ratio_sum_w",
        "sequence_ratio_sum_w2",
        "sequence_count",
    )
    parts = {statistic: _pop_vector(metrics, statistic, num_bins) for statistic in statistics}
    total_samples = sum(parts["sample_count"])
    total_response_tokens = sum(parts["response_token_count"])
    total_pre_loss_tokens = sum(parts["token_count"])
    total_contribution = sum(parts["effective_contribution"])
    metrics["sample_staleness/mean_abs_pg_objective"] = _safe_ratio(
        total_contribution,
        total_samples,
    )
    metrics["sample_staleness/effective_contribution_available"] = float(total_contribution > 0.0)
    for index in range(num_bins):
        label = _bin_label(index, max_staleness)
        response_count = parts["response_token_count"][index]
        token_count = parts["token_count"][index]
        final_count = parts["final_token_count"][index]
        contribution = parts["effective_contribution"][index]
        prefix = f"sample_staleness/{label}"
        metrics[f"{prefix}/consumed_sequence_mass"] = _safe_ratio(parts["sample_count"][index], total_samples)
        metrics[f"{prefix}/consumed_response_token_mass"] = _safe_ratio(
            response_count,
            total_response_tokens,
        )
        metrics[f"{prefix}/consumed_pre_loss_token_mass"] = _safe_ratio(
            token_count,
            total_pre_loss_tokens,
        )
        metrics[f"{prefix}/effective_contribution_mass"] = _safe_ratio(contribution, total_contribution)
        metrics[f"{prefix}/mean_abs_policy_rollout_log_ratio"] = _safe_ratio(
            parts["abs_policy_log_ratio_sum"][index], token_count
        )
        metrics[f"{prefix}/mean_abs_ppo_objective_log_ratio"] = _safe_ratio(
            parts["abs_ppo_objective_log_ratio_sum"][index], token_count
        )
        metrics[f"{prefix}/ppo_clip_fraction"] = _safe_ratio(parts["ppo_clip_count"][index], token_count)
        metrics[f"{prefix}/importance_clip_fraction"] = _safe_ratio(parts["importance_clip_count"][index], token_count)
        metrics[f"{prefix}/correction_mask_fraction"] = _safe_ratio(parts["correction_mask_count"][index], token_count)
        metrics[f"{prefix}/initial_mask_fraction"] = (
            1.0 - token_count / response_count if response_count > 0.0 else 0.0
        )
        metrics[f"{prefix}/final_mask_fraction"] = 1.0 - final_count / response_count if response_count > 0.0 else 0.0
        metrics[f"{prefix}/nonzero_contribution_fraction"] = _safe_ratio(
            parts["nonzero_contribution_count"][index], token_count
        )
        metrics[f"{prefix}/mean_abs_pg_contribution_per_pre_loss_token"] = _safe_ratio(
            contribution,
            token_count,
        )
        metrics[f"{prefix}/policy_rollout_ratio_token_ess"] = _safe_ratio(
            parts["token_ratio_sum_w"][index] ** 2,
            final_count * parts["token_ratio_sum_w2"][index],
        )
        metrics[f"{prefix}/policy_rollout_ratio_sequence_ess"] = _safe_ratio(
            parts["sequence_ratio_sum_w"][index] ** 2,
            parts["sequence_count"][index] * parts["sequence_ratio_sum_w2"][index],
        )

    histogram_labels = _ratio_histogram_labels()
    if _part_key("ratio_hist_0", 0) not in metrics:
        return
    histograms = [
        _pop_vector(metrics, f"ratio_hist_{ratio_index}", num_bins) for ratio_index in range(len(histogram_labels))
    ]
    for staleness_index in range(num_bins):
        label = _bin_label(staleness_index, max_staleness)
        histogram = [values[staleness_index] for values in histograms]
        total = sum(histogram)
        metrics[f"sample_staleness/{label}/approx_p95_abs_policy_rollout_log_ratio_capped_1"] = _approx_abs_p95(
            histogram
        )
        for ratio_label, count in zip(histogram_labels, histogram, strict=True):
            metrics[f"sample_staleness/{label}/policy_rollout_log_ratio_hist/{ratio_label}"] = _safe_ratio(
                count,
                total,
            )
