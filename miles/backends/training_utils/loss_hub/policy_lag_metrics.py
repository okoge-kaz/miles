"""Additive diagnostics for selected-token policy lag.

For a response token ``j``, ``delta_j`` is the current-policy selected-token
log-probability minus the rollout-policy value. ``loss_sensitivity_j`` is the
absolute scalar derivative of the policy-surrogate reference loss with respect
to that current selected-token log-probability. It is not a parameter-gradient
norm and excludes entropy, KL, value, and auxiliary losses.

The collector keeps the reference reducer fixed between ``pre`` and ``post``.
The only difference is the explicit target weight (zero-loss filtering,
staleness-aware attenuation, or their composition). Statistics are emitted as
additive float64 parts, summed across microbatches and effective DP/CP ranks,
and finalized only after the global sums are available. Consequently, a
token's coefficient is identical in ``all_response`` and either truncation
subpopulation; no population-specific renormalization is performed.

Only genuine response tokens enter the collector. Padding is never multiplied
by a possibly nonfinite value. A nonfinite response-token input is counted and
invalidates dependent public metrics instead of being replaced by zero.
"""

from __future__ import annotations

import math
from argparse import Namespace

import torch

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

POLICY_LAG_PART_PREFIX = "_policy_lag_part/"

_LOG_RATIO_EXP_CLAMP = 20.0
_POPULATIONS = ("all_response", "truncated", "non_truncated")
_STAGES = ("pre", "post")


def _branch_derivative(
    first: torch.Tensor,
    second: torch.Tensor,
    first_derivative: torch.Tensor,
    second_derivative: torch.Tensor,
    *,
    select_maximum: bool,
) -> torch.Tensor:
    """Match PyTorch maximum/minimum's equal-input subgradient."""
    first_selected = first > second if select_maximum else first < second
    second_selected = second > first if select_maximum else second < first
    tie_derivative = (first_derivative + second_derivative) * 0.5
    return torch.where(
        first_selected,
        first_derivative,
        torch.where(second_selected, second_derivative, tie_derivative),
    )


def compute_ppo_loss_sensitivity(
    *,
    ppo_log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None,
) -> torch.Tensor:
    """Return ``abs(d surrogate / d selected_logprob)`` for built-in PPO.

    This mirrors :func:`compute_policy_loss`, including its log-ratio clamp,
    asymmetric PPO clipping, dual clipping, and PyTorch's boundary
    subgradients. Inputs must be the same post-preprocessing values used by the
    loss. The returned tensor is detached and does not retain the training
    graph.
    """
    if ppo_log_ratio.shape != advantages.shape:
        raise ValueError(
            f"ppo_log_ratio shape {tuple(ppo_log_ratio.shape)} does not match advantages {tuple(advantages.shape)}"
        )

    raw_log_ratio = ppo_log_ratio.detach().float()
    advantage = advantages.detach().float()
    safe_log_ratio = torch.nan_to_num(
        raw_log_ratio,
        nan=0.0,
        posinf=_LOG_RATIO_EXP_CLAMP,
        neginf=-_LOG_RATIO_EXP_CLAMP,
    ).clamp(min=-_LOG_RATIO_EXP_CLAMP, max=_LOG_RATIO_EXP_CLAMP)
    ratio = safe_log_ratio.exp()

    exponent_derivative = (
        torch.isfinite(raw_log_ratio)
        & (raw_log_ratio >= -_LOG_RATIO_EXP_CLAMP)
        & (raw_log_ratio <= _LOG_RATIO_EXP_CLAMP)
    ).to(dtype=ratio.dtype)
    ratio_derivative = ratio * exponent_derivative

    low, high = 1.0 - eps_clip, 1.0 + eps_clip_high
    clipped_ratio = ratio.clamp(low, high)
    clipped_derivative = ratio_derivative * ((ratio >= low) & (ratio <= high)).to(dtype=ratio.dtype)
    loss_unclipped = -ratio * advantage
    loss_clipped = -clipped_ratio * advantage
    derivative_unclipped = -ratio_derivative * advantage
    derivative_clipped = -clipped_derivative * advantage
    selected_loss = torch.maximum(loss_unclipped, loss_clipped)
    signed_coefficient = _branch_derivative(
        loss_unclipped,
        loss_clipped,
        derivative_unclipped,
        derivative_clipped,
        select_maximum=True,
    )

    if eps_clip_c is not None:
        if eps_clip_c <= 1.0:
            raise ValueError(f"eps_clip_c must be greater than 1.0, got {eps_clip_c}")
        dual_clip_loss = -eps_clip_c * advantage
        signed_coefficient = torch.where(
            advantage < 0,
            _branch_derivative(
                dual_clip_loss,
                selected_loss,
                torch.zeros_like(signed_coefficient),
                signed_coefficient,
                select_maximum=False,
            ),
            signed_coefficient,
        )

    return signed_coefficient.abs().detach()


def _local_masks(
    args: Namespace,
    batch: RolloutBatch,
    masks: list[torch.Tensor],
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    local = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        masks,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    return [mask.detach().to(device=device, dtype=torch.float32) for mask in local]


def _cat_or_empty(values: list[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    return torch.cat(values) if values else torch.empty(0, device=device, dtype=torch.float32)


def _sample_ids(local_masks: list[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    lengths = torch.as_tensor([mask.numel() for mask in local_masks], device=device, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(len(local_masks), device=device), lengths)


def _truncated_indicator(
    batch: RolloutBatch,
    local_masks: list[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    truncated = batch.get("truncated")
    if truncated is None:
        raise RuntimeError("Policy-lag populations require the per-sample truncated indicator")
    if len(truncated) != len(local_masks):
        raise ValueError(f"truncated has {len(truncated)} rows for {len(local_masks)} samples")
    indicators = [int(value) for value in truncated]
    if any(value not in (0, 1) for value in indicators):
        raise ValueError("truncated values must be 0 or 1")
    values = torch.as_tensor(indicators, device=device, dtype=torch.long)
    return values[_sample_ids(local_masks, device=device)].bool()


def _reference_reducer_weights(
    args: Namespace,
    full_masks: list[torch.Tensor],
    local_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    if args.calculate_per_token_loss:
        return local_masks
    return [
        local / full.detach().to(device=local.device, dtype=torch.float32).sum().clamp_min(1.0)
        for local, full in zip(local_masks, full_masks, strict=True)
    ]


def _normalizer_part(args: Namespace, full_masks: list[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    parallel_state = get_parallel_state()
    if args.calculate_per_token_loss:
        value = sum(
            (mask.detach().to(device=device, dtype=torch.float64).sum().clamp_min(1.0) for mask in full_masks),
            start=torch.zeros((), device=device, dtype=torch.float64),
        )
    else:
        value = torch.tensor(float(len(full_masks)), device=device, dtype=torch.float64)
    if parallel_state.cp.size > 1 and parallel_state.cp.rank != 0:
        return value.new_zeros(())
    return value


def _masked_finite_sum(values: torch.Tensor, population: torch.Tensor) -> torch.Tensor:
    include = population & torch.isfinite(values)
    return torch.where(include, values, values.new_zeros(())).sum(dtype=torch.float64)


def _population_parts(
    *,
    population: torch.Tensor,
    delta: torch.Tensor,
    delta_sq: torch.Tensor,
    initial_mask: torch.Tensor,
    target_weight: torch.Tensor,
    sensitivity_base: torch.Tensor | None,
    reducer_weight: torch.Tensor | None,
    actual_reducer_weight: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    finite_delta = torch.isfinite(delta)
    parts = {
        "response_token_count": population.sum(dtype=torch.float64),
        "delta_abs_sum": _masked_finite_sum(delta.abs(), population),
        "delta_sq_sum": _masked_finite_sum(delta_sq, population),
        "initial_loss_mask_sum": _masked_finite_sum(initial_mask, population),
        "final_loss_mask_nonzero_sum": _masked_finite_sum(
            initial_mask * (target_weight > 0).to(dtype=initial_mask.dtype), population
        ),
        "final_loss_mask_weight_sum": _masked_finite_sum(initial_mask * target_weight, population),
        "nonfinite_delta_count": (population & ~finite_delta).sum(dtype=torch.float64),
    }
    if sensitivity_base is None or reducer_weight is None:
        return parts

    stage_bases = {"pre": sensitivity_base, "post": sensitivity_base * target_weight}
    for stage, stage_base in stage_bases.items():
        nonfinite = ~torch.isfinite(sensitivity_base)
        if stage == "post":
            nonfinite = nonfinite | ~torch.isfinite(target_weight)
        coefficient = torch.where(
            torch.isfinite(stage_base) & torch.isfinite(reducer_weight),
            stage_base * reducer_weight,
            stage_base.new_zeros(()),
        )
        joint_finite = torch.isfinite(delta) & torch.isfinite(stage_base) & torch.isfinite(reducer_weight)
        parts[f"{stage}/loss_sensitivity_sum_raw"] = _masked_finite_sum(coefficient, population)
        parts[f"{stage}/loss_sensitivity_delta_sq_sum_raw"] = _masked_finite_sum(
            torch.where(joint_finite, coefficient * delta_sq, coefficient.new_zeros(())), population
        )
        parts[f"{stage}/nonfinite_loss_sensitivity_count"] = (population & nonfinite).sum(dtype=torch.float64)

    if actual_reducer_weight is not None:
        actual_coefficient = torch.where(
            torch.isfinite(sensitivity_base) & torch.isfinite(actual_reducer_weight),
            sensitivity_base * actual_reducer_weight,
            sensitivity_base.new_zeros(()),
        )
        joint_finite = torch.isfinite(delta) & torch.isfinite(sensitivity_base) & torch.isfinite(actual_reducer_weight)
        parts["actual/loss_sensitivity_sum_raw"] = _masked_finite_sum(actual_coefficient, population)
        parts["actual/loss_sensitivity_delta_sq_sum_raw"] = _masked_finite_sum(
            torch.where(joint_finite, actual_coefficient * delta_sq, actual_coefficient.new_zeros(())),
            population,
        )
    return parts


def compute_policy_lag_parts(
    *,
    args: Namespace,
    batch: RolloutBatch,
    delta: torch.Tensor,
    initial_loss_masks: list[torch.Tensor],
    target_weights: torch.Tensor,
    loss_sensitivity_base: torch.Tensor | None,
    actual_post_loss_masks: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build additive Phase-1 statistics for the three response populations.

    ``loss_sensitivity_base`` contains the PPO sensitivity after common applied
    weights such as OPSM/TIS, but before the initial mask, reducer coefficient,
    and target weight. Pass ``None`` for an unsupported surrogate; delta and
    mask diagnostics are still emitted with
    ``policy_lag/loss_sensitivity_supported=0``.
    """
    device = delta.device
    values = delta.detach().to(dtype=torch.float32)
    delta_sq = values.to(dtype=torch.float64).square()
    targets = target_weights.detach().to(device=device, dtype=torch.float32)
    if values.shape != targets.shape:
        raise ValueError(f"delta shape {tuple(values.shape)} does not match target weights {tuple(targets.shape)}")
    target_weights_are_valid = torch.isfinite(targets) & (targets >= 0) & (targets <= 1)
    if targets.is_cuda:
        torch._assert_async(target_weights_are_valid.all())
    elif not bool(target_weights_are_valid.all()):
        raise ValueError("Policy-lag target weights must be finite and in [0, 1]")

    local_initial = _local_masks(args, batch, initial_loss_masks, device=device)
    initial_flat = _cat_or_empty(local_initial, device=device)
    if initial_flat.shape != values.shape:
        raise ValueError(f"initial mask has {initial_flat.numel()} tokens, expected {values.numel()}")
    truncated = _truncated_indicator(batch, local_initial, device=device)

    reducer_flat = None
    sensitivity = None
    if loss_sensitivity_base is not None:
        sensitivity = loss_sensitivity_base.detach().to(device=device, dtype=torch.float32)
        if sensitivity.shape != values.shape:
            raise ValueError(f"loss sensitivity has shape {tuple(sensitivity.shape)}, expected {tuple(values.shape)}")
        reducer_flat = _cat_or_empty(
            _reference_reducer_weights(args, initial_loss_masks, local_initial), device=device
        )

    actual_reducer_flat = None
    if actual_post_loss_masks is not None:
        if not args.calculate_per_token_loss:
            raise ValueError("Actual post-normalization parts are only needed for a changing token normalizer")
        actual_local = _local_masks(args, batch, actual_post_loss_masks, device=device)
        actual_reducer_flat = _cat_or_empty(actual_local, device=device)

    parts = {
        "reference_normalizer": _normalizer_part(args, initial_loss_masks, device=device),
        "support_count": values.new_tensor(float(loss_sensitivity_base is not None)),
        "support_total": values.new_ones(()),
    }
    if actual_post_loss_masks is not None:
        parts["actual_normalizer"] = _normalizer_part(args, actual_post_loss_masks, device=device)

    population_masks = {
        "all_response": torch.ones_like(truncated),
        "truncated": truncated,
        "non_truncated": ~truncated,
    }
    for population, population_mask in population_masks.items():
        population_values = _population_parts(
            population=population_mask,
            delta=values,
            delta_sq=delta_sq,
            initial_mask=initial_flat,
            target_weight=targets,
            sensitivity_base=sensitivity,
            reducer_weight=reducer_flat,
            actual_reducer_weight=actual_reducer_flat,
        )
        parts.update({f"{population}/{key}": value for key, value in population_values.items()})
    return {f"{POLICY_LAG_PART_PREFIX}{key}": value.detach() for key, value in parts.items()}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.nan


def _normalized_sum(raw_value: float, normalizer: float) -> float:
    if normalizer > 0.0:
        return raw_value / normalizer
    return 0.0 if raw_value == 0.0 else math.nan


def _finalize_stage(
    sums: dict[str, float],
    *,
    population_prefix: str,
    stage: str,
    normalizer: float,
    delta_is_finite: bool,
) -> tuple[dict[str, float], float, float]:
    nonfinite = sums[f"{population_prefix}{stage}/nonfinite_loss_sensitivity_count"]
    raw_mass = sums[f"{population_prefix}{stage}/loss_sensitivity_sum_raw"]
    raw_delta_sq = sums[f"{population_prefix}{stage}/loss_sensitivity_delta_sq_sum_raw"]
    mass = _normalized_sum(raw_mass, normalizer) if nonfinite == 0 else math.nan
    delta_sq = _normalized_sum(raw_delta_sq, normalizer) if nonfinite == 0 and delta_is_finite else math.nan
    valid = nonfinite == 0 and delta_is_finite and math.isfinite(mass) and mass > 0.0
    weighted_rms = math.sqrt(max(raw_delta_sq / raw_mass, 0.0)) if valid else math.nan
    metrics = {
        f"loss_sensitivity_sum_{stage}": mass,
        f"loss_sensitivity_weighted_delta_rms_{stage}": weighted_rms,
        f"loss_sensitivity_delta_sq_sum_{stage}": delta_sq,
        f"loss_sensitivity_weighted_metrics_valid_{stage}": float(valid),
        f"nonfinite_loss_sensitivity_count_{stage}": nonfinite,
    }
    return metrics, mass, delta_sq


def _finalize_actual(
    sums: dict[str, float],
    *,
    population_prefix: str,
    normalizer: float,
    sensitivity_is_finite: bool,
    delta_is_finite: bool,
) -> dict[str, float]:
    raw_mass = sums[f"{population_prefix}actual/loss_sensitivity_sum_raw"]
    raw_delta_sq = sums[f"{population_prefix}actual/loss_sensitivity_delta_sq_sum_raw"]
    mass = _normalized_sum(raw_mass, normalizer) if sensitivity_is_finite else math.nan
    joint_inputs_are_finite = sensitivity_is_finite and delta_is_finite
    delta_sq = _normalized_sum(raw_delta_sq, normalizer) if joint_inputs_are_finite else math.nan
    weighted_rms = (
        math.sqrt(max(raw_delta_sq / raw_mass, 0.0)) if joint_inputs_are_finite and raw_mass > 0 else math.nan
    )
    return {
        "loss_sensitivity_sum_actual": mass,
        "loss_sensitivity_weighted_delta_rms_actual": weighted_rms,
        "loss_sensitivity_delta_sq_sum_actual": delta_sq,
    }


def finalize_policy_lag_parts(metric_sums: dict[str, float]) -> dict[str, float]:
    """Finalize globally summed policy-lag parts without scalar re-averaging."""
    prefix = POLICY_LAG_PART_PREFIX
    support_total = metric_sums.get(f"{prefix}support_total")
    if support_total is None:
        return {}
    supported = support_total > 0 and metric_sums[f"{prefix}support_count"] == support_total
    metrics = {"policy_lag/loss_sensitivity_supported": float(supported)}
    reference_normalizer = metric_sums[f"{prefix}reference_normalizer"]
    actual_normalizer = metric_sums.get(f"{prefix}actual_normalizer")

    for population in _POPULATIONS:
        part_prefix = f"{prefix}{population}/"
        public_prefix = f"policy_lag/{population}/"
        count = metric_sums[f"{part_prefix}response_token_count"]
        nonfinite_delta = metric_sums[f"{part_prefix}nonfinite_delta_count"]
        delta_is_finite = nonfinite_delta == 0
        population_metrics = {
            "response_token_count": count,
            "delta_abs_mean": (
                metric_sums[f"{part_prefix}delta_abs_sum"] / count if count > 0 and delta_is_finite else math.nan
            ),
            "delta_rms": (
                math.sqrt(max(metric_sums[f"{part_prefix}delta_sq_sum"] / count, 0.0))
                if count > 0 and delta_is_finite
                else math.nan
            ),
            "initial_loss_mask_fraction": (
                metric_sums[f"{part_prefix}initial_loss_mask_sum"] / count if count > 0 else math.nan
            ),
            "final_loss_mask_nonzero_fraction": (
                metric_sums[f"{part_prefix}final_loss_mask_nonzero_sum"] / count if count > 0 else math.nan
            ),
            "final_loss_mask_weight_mean": (
                metric_sums[f"{part_prefix}final_loss_mask_weight_sum"] / count if count > 0 else math.nan
            ),
            "nonfinite_delta_count": nonfinite_delta,
        }
        if supported:
            stage_values = {}
            stages_are_finite = True
            for stage in _STAGES:
                stage_metrics, mass, delta_sq = _finalize_stage(
                    metric_sums,
                    population_prefix=part_prefix,
                    stage=stage,
                    normalizer=reference_normalizer,
                    delta_is_finite=delta_is_finite,
                )
                population_metrics.update(stage_metrics)
                stage_values[stage] = (mass, delta_sq)
                stages_are_finite &= metric_sums[f"{part_prefix}{stage}/nonfinite_loss_sensitivity_count"] == 0
            pre_mass, pre_delta_sq = stage_values["pre"]
            post_mass, post_delta_sq = stage_values["post"]
            population_metrics["loss_sensitivity_retained_fraction"] = _ratio(post_mass, pre_mass)
            population_metrics["loss_sensitivity_delta_sq_retained_fraction"] = _ratio(post_delta_sq, pre_delta_sq)
            if actual_normalizer is not None:
                population_metrics.update(
                    _finalize_actual(
                        metric_sums,
                        population_prefix=part_prefix,
                        normalizer=actual_normalizer,
                        sensitivity_is_finite=stages_are_finite,
                        delta_is_finite=delta_is_finite,
                    )
                )
        metrics.update({f"{public_prefix}{key}": value for key, value in population_metrics.items()})
    return metrics
