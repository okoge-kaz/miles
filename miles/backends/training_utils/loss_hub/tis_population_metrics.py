"""TIS absolute-deviation diagnostics over explicit response populations."""

from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager

import torch

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

TIS_POPULATION_PART_PREFIX = "_tis_population_part/"

_POPULATIONS = ("active", "all_response", "truncated", "non_truncated")
_QUANTILE_UPPER_BOUNDS = (
    0.0,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    2.0e-2,
    3.0e-2,
    5.0e-2,
    1.0e-1,
    2.0e-1,
    3.0e-1,
    5.0e-1,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
)
_QUANTILE_CAP_LABEL = "1e3"


@contextmanager
def _allow_nondeterministic_telemetry_reductions():
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


def _validate_truncated(batch: RolloutBatch, *, expected_samples: int) -> list[bool] | None:
    truncated = batch.get("truncated")
    if truncated is None:
        return None
    if len(truncated) != expected_samples:
        raise ValueError(f"truncated has {len(truncated)} rows for {expected_samples} samples")

    indicators = [int(value) for value in truncated]
    if any(value not in (0, 1) for value in indicators):
        raise ValueError("truncated indicators must be 0 or 1")
    return [bool(value) for value in indicators]


def _population_masks(
    loss_masks: list[torch.Tensor],
    truncated: list[bool],
) -> dict[str, tuple[list[torch.Tensor], list[bool]]]:
    all_response_masks = [torch.ones_like(mask) for mask in loss_masks]
    truncated_masks = [
        torch.ones_like(mask) if is_truncated else torch.zeros_like(mask)
        for mask, is_truncated in zip(loss_masks, truncated, strict=True)
    ]
    non_truncated_masks = [
        torch.zeros_like(mask) if is_truncated else torch.ones_like(mask)
        for mask, is_truncated in zip(loss_masks, truncated, strict=True)
    ]
    return {
        "active": (loss_masks, [True] * len(loss_masks)),
        "all_response": (all_response_masks, [True] * len(loss_masks)),
        "truncated": (truncated_masks, truncated),
        "non_truncated": (non_truncated_masks, [not value for value in truncated]),
    }


def _local_metric_weights(
    args: Namespace,
    batch: RolloutBatch,
    full_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    local_masks = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        full_masks,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    local_weights = [mask.detach().float() for mask in local_masks]
    if not args.calculate_per_token_loss:
        denominators = [mask.detach().float().sum().clamp_min(1.0) for mask in full_masks]
        local_weights = [weight / denominator for weight, denominator in zip(local_weights, denominators, strict=True)]
    return torch.cat(local_weights), torch.stack([mask.detach().float().sum() for mask in full_masks])


def _population_normalizer_and_missing_weight(
    args: Namespace,
    full_mask_sums: torch.Tensor,
    selected_samples: list[bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.as_tensor(selected_samples, device=full_mask_sums.device, dtype=torch.bool)
    if args.calculate_per_token_loss:
        contributions = full_mask_sums.clamp_min(1.0)
        observed = full_mask_sums
    else:
        contributions = torch.ones_like(full_mask_sums)
        observed = (full_mask_sums > 0).to(dtype=full_mask_sums.dtype)
    normalizer = contributions[selected].sum()
    missing_weight = (contributions[selected] - observed[selected]).sum()
    parallel_state = get_parallel_state()
    if parallel_state.cp.size > 1 and parallel_state.cp.rank != 0:
        normalizer = normalizer.new_zeros(())
        missing_weight = missing_weight.new_zeros(())
    return normalizer, missing_weight


def _part_key(population: str, statistic: str) -> str:
    return f"{TIS_POPULATION_PART_PREFIX}{population}/{statistic}"


def compute_tis_abs_population_parts(
    *,
    args: Namespace,
    batch: RolloutBatch,
    tis_abs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return additive TIS statistics before zero-loss population censoring."""
    loss_masks = batch["loss_masks"]
    truncated = _validate_truncated(batch, expected_samples=len(loss_masks))
    if truncated is None:
        return {}

    values = tis_abs.detach().float()
    safe_values = torch.nan_to_num(
        values,
        nan=0.0,
        posinf=_QUANTILE_UPPER_BOUNDS[-1],
        neginf=0.0,
    ).clamp(min=0.0, max=_QUANTILE_UPPER_BOUNDS[-1])
    boundaries = torch.tensor(_QUANTILE_UPPER_BOUNDS, device=values.device, dtype=values.dtype)
    histogram_bins = torch.bucketize(safe_values, boundaries)
    num_histogram_bins = len(_QUANTILE_UPPER_BOUNDS) + 1
    parts: dict[str, torch.Tensor] = {}

    with _allow_nondeterministic_telemetry_reductions():
        for population, (full_masks, selected_samples) in _population_masks(loss_masks, truncated).items():
            local_weights, full_mask_sums = _local_metric_weights(args, batch, full_masks)
            local_weights = local_weights.to(device=values.device)
            if local_weights.numel() != values.numel():
                raise ValueError(
                    f"TIS population weight has {local_weights.numel()} tokens, expected {values.numel()}"
                )

            normalizer, missing_weight = _population_normalizer_and_missing_weight(
                args,
                full_mask_sums.to(device=values.device),
                selected_samples,
            )
            histogram = torch.bincount(
                histogram_bins,
                weights=local_weights,
                minlength=num_histogram_bins,
            )[:num_histogram_bins]
            parallel_state = get_parallel_state()
            if parallel_state.cp.size == 1 or parallel_state.cp.rank == 0:
                histogram[0] += missing_weight

            weighted_values = torch.where(
                local_weights > 0,
                values * local_weights,
                torch.zeros_like(values),
            )
            parts[_part_key(population, "sum")] = weighted_values.sum().detach()
            parts[_part_key(population, "weight")] = normalizer.detach()
            for index, count in enumerate(histogram):
                parts[_part_key(population, f"hist_{index}")] = count.detach()
    return parts


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _approx_quantile(histogram: list[float], quantile: float) -> float:
    total = sum(histogram)
    if total <= 0.0:
        return 0.0
    target = quantile * total
    cumulative = 0.0
    for index, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            if index < len(_QUANTILE_UPPER_BOUNDS):
                return _QUANTILE_UPPER_BOUNDS[index]
            return _QUANTILE_UPPER_BOUNDS[-1]
    return _QUANTILE_UPPER_BOUNDS[-1]


def finalize_tis_abs_population_parts(metric_sums: dict[str, float]) -> dict[str, float]:
    """Convert globally summed TIS population parts into public metrics."""
    if _part_key("active", "weight") not in metric_sums:
        return {}

    metrics: dict[str, float] = {}
    for population in _POPULATIONS:
        weight = metric_sums[_part_key(population, "weight")]
        metrics[f"tis_abs/{population}"] = _safe_ratio(metric_sums[_part_key(population, "sum")], weight)
        histogram = [
            metric_sums[_part_key(population, f"hist_{index}")] for index in range(len(_QUANTILE_UPPER_BOUNDS) + 1)
        ]
        metrics[f"tis_abs_approx_p95_capped_{_QUANTILE_CAP_LABEL}/{population}"] = _approx_quantile(histogram, 0.95)
        metrics[f"tis_abs_approx_p99_capped_{_QUANTILE_CAP_LABEL}/{population}"] = _approx_quantile(histogram, 0.99)
    return metrics
