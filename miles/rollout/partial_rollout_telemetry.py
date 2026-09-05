"""Train-time staleness telemetry for synchronous partial rollouts."""

from collections import defaultdict

from miles.rollout.queue_telemetry import Group, _iter_samples
from miles.rollout.recycle_compute_metrics import (
    SAMPLE_REFERENCE_VERSION_KEY,
    TRAIN_VERSION_KEY,
    weighted_lag_distribution_metrics,
)
from miles.rollout.staleness_distribution import staleness_distribution_metrics
from miles.utils.types import Sample

START_ROLLOUT_ID_KEY = "start_rollout_id"
TOKEN_BOUNDARY_LEDGER_KEY = "partial_rollout_token_boundaries"


def stamp_partial_rollout_start(group: Group, rollout_id: int) -> None:
    """Remember the first rollout boundary that retained each non-empty prefix."""
    for sample in _iter_samples(group):
        if sample.response_length <= 0:
            continue
        start = sample.metadata.get(START_ROLLOUT_ID_KEY)
        if start is None:
            sample.metadata[START_ROLLOUT_ID_KEY] = rollout_id
            sample.metadata[TOKEN_BOUNDARY_LEDGER_KEY] = [[sample.response_length, rollout_id]]
        else:
            _validate_start_rollout_id(sample, start, rollout_id)
            _append_token_boundary(sample, rollout_id)


def collect_partial_rollout_staleness_metrics(groups: list[Group], rollout_id: int) -> dict[str, float]:
    """Measure accepted-batch lag in synchronous weight-update boundaries.

    Synchronous training pushes weights once after every rollout. A retained
    prefix stamped in rollout ``S`` and consumed in rollout ``T`` therefore has
    train-time staleness ``T - S``. Fresh samples use ``S = T``. This relative
    coordinate remains valid when a chained job restarts SGLang and its absolute
    engine weight-version counter starts over.
    """
    if not groups:
        return {}

    samples = [sample for group in groups for sample in _iter_samples(group)]
    sample_ages = [_sample_rollout_age(sample, rollout_id) for sample in samples]
    group_ages = [
        max((_sample_rollout_age(sample, rollout_id) for sample in _iter_samples(group)), default=0)
        for group in groups
    ]

    for sample, age in zip(samples, sample_ages, strict=True):
        sample.metadata[SAMPLE_REFERENCE_VERSION_KEY] = rollout_id - age
        sample.metadata[TRAIN_VERSION_KEY] = rollout_id

    metrics: dict[str, float] = {
        "staleness/partial_rollout/reference_is_rollout_boundary": 1.0,
        "staleness/partial_rollout/train_rollout_id": float(rollout_id),
        "staleness/partial_rollout/resumed_groups": float(sum(age > 0 for age in group_ages)),
        "staleness/partial_rollout/resumed_group_frac": sum(age > 0 for age in group_ages) / len(group_ages),
        "staleness/partial_rollout/resumed_samples": float(sum(age > 0 for age in sample_ages)),
        "staleness/partial_rollout/resumed_sample_frac": sum(age > 0 for age in sample_ages) / len(sample_ages),
    }
    for population, values in (
        ("total", group_ages),
        # There is no output queue in synchronous rollout. All crossed updates
        # belong to generation/carry-over; the ready-to-train interval is zero.
        ("pre_queue", group_ages),
        ("in_queue", [0] * len(group_ages)),
    ):
        metrics.update(
            {f"staleness/{population}/{name}": value for name, value in staleness_distribution_metrics(values).items()}
        )
    sample_metrics = staleness_distribution_metrics(sample_ages)
    sample_metrics["num_samples"] = sample_metrics.pop("num_groups")
    metrics.update({f"staleness/partial_rollout/sample_total/{name}": value for name, value in sample_metrics.items()})
    metrics.update(_token_staleness_metrics(samples, rollout_id))
    return metrics


def collect_partial_rollout_work_metrics(
    *,
    launched_groups: int,
    launched_trajectories: int,
    launched_existing_response_tokens: int,
    accepted: list[Group],
    carried: list[Group],
    dynamic_filter_discarded: list[Group],
    completed_surplus_discarded: list[Group],
    generation_failed_groups: int,
    rollout_id: int,
) -> dict[str, float]:
    """Account for synchronous partial-rollout work at one abort boundary."""
    metrics = {
        "rollout/partial_rollout/launched_groups": float(launched_groups),
        "rollout/partial_rollout/launched_trajectories": float(launched_trajectories),
        "rollout/partial_rollout/launched_existing_response_tokens": float(launched_existing_response_tokens),
        "rollout/partial_rollout/generation_failed_groups": float(generation_failed_groups),
    }
    for population, groups in (
        ("accepted", accepted),
        ("carried", carried),
        ("dynamic_filter_discarded", dynamic_filter_discarded),
        ("completed_surplus_discarded", completed_surplus_discarded),
    ):
        prefix = f"rollout/partial_rollout/{population}"
        metrics.update(
            {f"{prefix}/{name}": value for name, value in _work_population_metrics(groups, rollout_id).items()}
        )
    accounted_groups = (
        len(accepted)
        + len(carried)
        + len(dynamic_filter_discarded)
        + len(completed_surplus_discarded)
        + generation_failed_groups
    )
    metrics["rollout/partial_rollout/accounting_error_groups"] = float(launched_groups - accounted_groups)
    return metrics


def _sample_rollout_age(sample: Sample, rollout_id: int) -> int:
    start = sample.metadata.get(START_ROLLOUT_ID_KEY)
    if start is None:
        return 0
    _validate_start_rollout_id(sample, start, rollout_id)
    return rollout_id - start


def _validate_start_rollout_id(sample: Sample, start: object, rollout_id: int) -> None:
    if type(start) is not int or start < 0 or start > rollout_id:
        raise RuntimeError(
            f"Invalid partial-rollout origin for sample {sample.index}: "
            f"start_rollout_id={start!r}, current_rollout_id={rollout_id}"
        )


def _append_token_boundary(sample: Sample, rollout_id: int) -> None:
    ledger = sample.metadata.get(TOKEN_BOUNDARY_LEDGER_KEY)
    # A checkpoint made before the ledger existed cannot recover the old/new
    # split. Keep group/sample lag available, but report incomplete token
    # coverage instead of inventing a boundary.
    if ledger is None:
        return
    parsed = _parse_token_boundaries(sample, rollout_id)
    last_end = parsed[-1][0] if parsed else 0
    if sample.response_length < last_end:
        raise RuntimeError(
            f"Partial-rollout response shrank for sample {sample.index}: "
            f"ledger_end={last_end}, response_length={sample.response_length}"
        )
    if sample.response_length > last_end:
        ledger.append([sample.response_length, rollout_id])


def _parse_token_boundaries(sample: Sample, rollout_id: int) -> list[tuple[int, int]] | None:
    raw = sample.metadata.get(TOKEN_BOUNDARY_LEDGER_KEY)
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise RuntimeError(f"Malformed partial-rollout token ledger for sample {sample.index}: expected list")

    parsed = []
    previous_end = 0
    previous_rollout = -1
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise RuntimeError(f"Malformed partial-rollout token boundary for sample {sample.index}: {entry!r}")
        end, origin = entry
        if (
            type(end) is not int
            or type(origin) is not int
            or end <= previous_end
            or end > sample.response_length
            or origin <= previous_rollout
            or origin > rollout_id
        ):
            raise RuntimeError(f"Invalid partial-rollout token boundary for sample {sample.index}: {entry!r}")
        parsed.append((end, origin))
        previous_end = end
        previous_rollout = origin
    return parsed


def _response_token_origin_segments(sample: Sample, rollout_id: int) -> list[tuple[int, int, int]] | None:
    parsed = _parse_token_boundaries(sample, rollout_id)
    if parsed is None and _sample_rollout_age(sample, rollout_id) > 0:
        return None

    segments = []
    start = 0
    for end, origin in parsed or []:
        segments.append((start, end, origin))
        start = end
    if start < sample.response_length:
        segments.append((start, sample.response_length, rollout_id))
    return segments


def _work_population_metrics(groups: list[Group], rollout_id: int) -> dict[str, float]:
    samples = [sample for group in groups for sample in _iter_samples(group)]
    response_tokens = sum(sample.response_length for sample in samples)
    covered_response_tokens = 0
    current_rollout_response_tokens = 0
    for sample in samples:
        segments = _response_token_origin_segments(sample, rollout_id)
        if segments is None:
            continue
        covered_response_tokens += sample.response_length
        current_rollout_response_tokens += sum(end - start for start, end, origin in segments if origin == rollout_id)
    return {
        "groups": float(len(groups)),
        "trajectories": float(len(samples)),
        "response_tokens": float(response_tokens),
        "current_rollout_response_tokens": float(current_rollout_response_tokens),
        "current_rollout_response_token_coverage_frac": (
            covered_response_tokens / response_tokens if response_tokens else 0.0
        ),
    }


def _token_staleness_metrics(samples: list[Sample], rollout_id: int) -> dict[str, float]:
    response_lag_weights: dict[int, int | float] = defaultdict(int)
    loss_lag_weights: dict[int, int | float] = defaultdict(int)
    total_response_tokens = sum(sample.response_length for sample in samples)
    total_loss_tokens = 0
    covered_response_tokens = 0
    covered_loss_tokens = 0
    covered_loss_samples = 0
    invalid_loss_masks = 0

    for sample in samples:
        loss_mask = sample.loss_mask
        if loss_mask is None:
            sample_loss_tokens = sample.response_length
        elif len(loss_mask) == sample.response_length:
            sample_loss_tokens = sum(loss_mask)
        else:
            sample_loss_tokens = 0
            invalid_loss_masks += 1
        total_loss_tokens += sample_loss_tokens

        origin_segments = _response_token_origin_segments(sample, rollout_id)
        if origin_segments is None:
            continue
        segments = [(start, end, rollout_id - origin) for start, end, origin in origin_segments]

        covered_response_tokens += sample.response_length
        for start, end, lag in segments:
            response_lag_weights[lag] += end - start

        if loss_mask is not None and len(loss_mask) != sample.response_length:
            continue
        covered_loss_samples += 1
        for start, end, lag in segments:
            segment_loss_tokens = end - start if loss_mask is None else sum(loss_mask[start:end])
            loss_lag_weights[lag] += segment_loss_tokens
            covered_loss_tokens += segment_loss_tokens

    metrics = {
        "staleness/token_lag/exact/covered_response_token_frac": (
            covered_response_tokens / total_response_tokens if total_response_tokens else 0.0
        ),
        "staleness/token_lag/exact/invalid_segments": 0.0,
        "staleness/token_lag/exact/invalid_turns": 0.0,
        "staleness/token_lag/exact/invalid_samples": 0.0,
        "staleness/token_lag/exact/loss_token/covered_loss_token_frac": (
            covered_loss_tokens / total_loss_tokens if total_loss_tokens else 0.0
        ),
        "staleness/token_lag/exact/loss_token/covered_sample_frac": (
            covered_loss_samples / len(samples) if samples else 0.0
        ),
        "staleness/token_lag/exact/loss_token/invalid_loss_masks": float(invalid_loss_masks),
        "staleness/partial_rollout/carried_prefix_tokens": float(
            sum(tokens for lag, tokens in response_lag_weights.items() if lag > 0)
        ),
        "staleness/partial_rollout/current_suffix_tokens": float(response_lag_weights.get(0, 0)),
    }
    metrics["staleness/partial_rollout/carried_prefix_token_frac"] = (
        metrics["staleness/partial_rollout/carried_prefix_tokens"] / covered_response_tokens
        if covered_response_tokens
        else 0.0
    )
    for prefix, weights in (
        ("staleness/token_lag/exact", response_lag_weights),
        ("staleness/token_lag/exact/loss_token", loss_lag_weights),
    ):
        metrics.update(
            {f"{prefix}/{name}": value for name, value in weighted_lag_distribution_metrics(weights).items()}
        )
    return metrics
