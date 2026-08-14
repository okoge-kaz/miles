"""Additive telemetry for fully-async rollout lag and discarded compute.

The existing ``staleness/*`` metrics intentionally remain group-weighted and
selection-time based.  This module owns new sample/token-weighted views and the
reason-coded accounting for work that never reaches the training loss.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from miles.utils.types import Sample

SUBMISSION_VERSION_KEY = "submission_weight_version"
GROUP_READY_VERSION_KEY = "group_ready_weight_version"
QUEUE_PUT_VERSION_KEY = "queue_put_weight_version"
DRAIN_VERSION_KEY = "drain_weight_version"
BOUND_REFERENCE_VERSION_KEY = "bound_reference_weight_version"
SAMPLE_REFERENCE_VERSION_KEY = "sample_staleness_reference_weight_version"

# Additive trajectory lifecycle boundaries. These are deliberately separate
# from the historical staleness keys above: changing GROUP_READY_VERSION_KEY or
# the selected bound reference would silently change existing plots.
TRAJECTORY_START_VERSION_KEY = "trajectory_start_weight_version"
SAMPLE_GENERATION_COMPLETE_VERSION_KEY = "sample_generation_complete_weight_version"
GROUP_GENERATION_COMPLETE_VERSION_KEY = "group_generation_complete_weight_version"
TRAJECTORY_START_TIME_KEY = "trajectory_start_wall_time"
SAMPLE_GENERATION_COMPLETE_TIME_KEY = "sample_generation_complete_wall_time"
GROUP_GENERATION_COMPLETE_TIME_KEY = "group_generation_complete_wall_time"
GROUP_READY_TIME_KEY = "group_ready_wall_time"
QUEUE_PUT_TIME_KEY = "queue_put_wall_time"
DRAIN_TIME_KEY = "drain_wall_time"
LIFECYCLE_EXACT_KEY = "fully_async_lifecycle_exact"

ATTEMPT_WALL_SECONDS_KEY = "fully_async_attempt_wall_seconds"
REWARD_SECONDS_KEY = "fully_async_reward_seconds"

GENERATED_TOKENS_KEY = "rollout/fully_async/useful_rollout/generated_tokens"
ADMITTED_TOKENS_KEY = "rollout/fully_async/useful_rollout/admitted_tokens"

RECYCLE_DEBUG_SCHEMA_VERSION = 2
SELECTION_POPULATIONS = ("generated", "admitted", "consumed", "recycled", "dropped")

STALE_AT_GENERATION_COMPLETION = "stale_at_generation_completion"
STALE_DURING_REWARD_FINALIZE = "stale_during_reward_finalize"
STALE_DURING_QUEUE_BACKPRESSURE = "stale_during_queue_backpressure"
STALE_IN_OUTPUT_QUEUE = "stale_in_output_queue"
STALE_STAGE_UNKNOWN = "stale_stage_unknown"
GENERATION_ABORTED = "generation_aborted"
ACTOR_WEIGHT_SYNC_OVERLAP = "actor_weight_sync_overlap"

RECYCLE_REASONS = (
    STALE_AT_GENERATION_COMPLETION,
    STALE_DURING_REWARD_FINALIZE,
    STALE_DURING_QUEUE_BACKPRESSURE,
    STALE_IN_OUTPUT_QUEUE,
    STALE_STAGE_UNKNOWN,
    GENERATION_ABORTED,
    ACTOR_WEIGHT_SYNC_OVERLAP,
)
RECYCLE_AUX_REASONS = ("group_straggler_collateral",)
DISCARD_REASONS = (*RECYCLE_REASONS, "dynamic_filter_dropped")

WASTE_COMPONENTS = (
    "decode_tokens",
    "prefill_uncached_tokens",
    "tool_env_seconds",
    "reward_seconds",
)

_ATTEMPT_METADATA_KEYS = (
    ATTEMPT_WALL_SECONDS_KEY,
    REWARD_SECONDS_KEY,
    SUBMISSION_VERSION_KEY,
    BOUND_REFERENCE_VERSION_KEY,
    SAMPLE_REFERENCE_VERSION_KEY,
    TRAJECTORY_START_VERSION_KEY,
    SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
    GROUP_GENERATION_COMPLETE_VERSION_KEY,
    GROUP_READY_VERSION_KEY,
    QUEUE_PUT_VERSION_KEY,
    DRAIN_VERSION_KEY,
    TRAJECTORY_START_TIME_KEY,
    SAMPLE_GENERATION_COMPLETE_TIME_KEY,
    GROUP_GENERATION_COMPLETE_TIME_KEY,
    GROUP_READY_TIME_KEY,
    QUEUE_PUT_TIME_KEY,
    DRAIN_TIME_KEY,
    LIFECYCLE_EXACT_KEY,
)


def reset_attempt_telemetry(samples: Iterable[Sample]) -> None:
    """Clear retry-local stamps while preserving prompt identity metadata."""
    for sample in samples:
        for key in _ATTEMPT_METADATA_KEYS:
            sample.metadata.pop(key, None)


def add_apportioned_reward_seconds(samples: Iterable[Sample], elapsed_seconds: float) -> None:
    """Charge one reward invocation once, split across the samples it scored."""
    samples = list(samples)
    if not samples:
        return
    share = max(0.0, elapsed_seconds) / len(samples)
    for sample in samples:
        sample.metadata[REWARD_SECONDS_KEY] = float(sample.metadata.get(REWARD_SECONDS_KEY, 0.0)) + share


def stamp_attempt_wall_seconds(samples: Iterable[Sample], elapsed_seconds: float) -> None:
    """Attach the group attempt wall time without pretending it is additive GPU time."""
    for sample in samples:
        sample.metadata[ATTEMPT_WALL_SECONDS_KEY] = max(0.0, elapsed_seconds)


def stamp_sample_lifecycle_boundary(
    samples: Iterable[Sample],
    *,
    version_key: str,
    version: int,
    time_key: str,
    wall_time: float,
) -> None:
    """Stamp one sample-level lifecycle boundary on generated outputs."""
    for sample in samples:
        sample.metadata[version_key] = int(version)
        sample.metadata[time_key] = float(wall_time)


def stamp_sample_reference_versions(samples: Iterable[Sample], reference_mode: str) -> None:
    """Persist each row's own reference for train-side staleness joins."""
    for sample in samples:
        reference = sample_reference_version(sample, reference_mode)
        if reference is not None:
            sample.metadata[SAMPLE_REFERENCE_VERSION_KEY] = reference


def waste_vector(samples: Iterable[Sample]) -> dict[str, float]:
    """Return heterogeneous waste components with their units kept explicit."""
    samples = list(samples)
    return {
        "decode_tokens": float(sum(sample.response_length for sample in samples)),
        "prefill_uncached_tokens": float(
            sum(
                max(0, sample.prefix_cache_info.total_prompt_tokens - sample.prefix_cache_info.cached_tokens)
                for sample in samples
            )
        ),
        "tool_env_seconds": float(sum(max(0.0, sample.non_generation_time) for sample in samples)),
        "reward_seconds": float(
            sum(max(0.0, float(sample.metadata.get(REWARD_SECONDS_KEY, 0.0))) for sample in samples)
        ),
    }


def add_discard_accounting(
    waste_by_reason: dict[str, dict[str, float]],
    *,
    reason: str,
    samples: Iterable[Sample],
) -> dict[str, float]:
    """Accumulate one disposition and return its per-attempt waste vector."""
    waste = waste_vector(samples)
    totals = waste_by_reason.setdefault(reason, {component: 0.0 for component in WASTE_COMPONENTS})
    for component, value in waste.items():
        totals[component] += value
    return waste


def discard_waste_metrics(waste_by_reason: dict[str, dict[str, float]]) -> dict[str, float]:
    """Render fixed-unit waste vectors by reason and for all discarded work."""
    metrics: dict[str, float] = {}
    totals = {component: 0.0 for component in WASTE_COMPONENTS}
    unknown_reasons = set(waste_by_reason) - set(DISCARD_REASONS)
    if unknown_reasons:
        raise ValueError(f"Unknown discard reasons: {sorted(unknown_reasons)}")
    for reason in DISCARD_REASONS:
        waste = waste_by_reason.get(reason, {})
        for component in WASTE_COMPONENTS:
            value = float(waste.get(component, 0.0))
            metrics[f"rollout/fully_async/waste/{reason}/{component}"] = value
            totals[component] += value
    for component, value in totals.items():
        metrics[f"rollout/fully_async/waste/all_discarded/{component}"] = value
    return metrics


def group_generation_completion_version(samples: Iterable[Sample]) -> int | None:
    """Return the latest scheduler-authoritative final forward in the group."""
    samples = list(samples)
    lifecycle_versions = [sample.metadata.get(GROUP_GENERATION_COMPLETE_VERSION_KEY) for sample in samples]
    numeric_lifecycle = [version for version in lifecycle_versions if isinstance(version, int)]
    lifecycle_exact = samples and all(bool(sample.metadata.get(LIFECYCLE_EXACT_KEY, False)) for sample in samples)
    if numeric_lifecycle and lifecycle_exact:
        if len(numeric_lifecycle) != len(samples) or len(set(numeric_lifecycle)) != 1:
            raise RuntimeError("Inconsistent group generation-completion lifecycle versions: " f"{lifecycle_versions}")
        return numeric_lifecycle[0]

    ends = [max(sample.last_forward_weight_versions) for sample in samples if sample.last_forward_weight_versions]
    if ends:
        return max(ends)
    completion_versions = [sample.newest_weight_version for sample in samples]
    numeric = [version for version in completion_versions if version is not None]
    return max(numeric) if numeric else None


def classify_stale_recycle_stage(
    *,
    reference_version: int | None,
    generation_completion_version: int | None,
    group_ready_version: int | None,
    queue_put_version: int | None,
    drain_version: int,
    bound: int,
) -> str:
    """Locate the first lifecycle boundary at which the configured bound failed."""
    if reference_version is None:
        return STALE_STAGE_UNKNOWN
    stages = (
        (STALE_AT_GENERATION_COMPLETION, generation_completion_version),
        (STALE_DURING_REWARD_FINALIZE, group_ready_version),
        (STALE_DURING_QUEUE_BACKPRESSURE, queue_put_version),
        (STALE_IN_OUTPUT_QUEUE, drain_version),
    )
    for reason, version in stages:
        if version is not None and version - reference_version > bound:
            return reason
    return STALE_STAGE_UNKNOWN


def aborted_recycle_reason(
    *,
    submission_version: int | None,
    group_ready_version: int | None,
) -> str:
    """Identify a proven weight-sync overlap without calling every abort one."""
    if submission_version is not None and group_ready_version is not None and group_ready_version > submission_version:
        return ACTOR_WEIGHT_SYNC_OVERLAP
    return GENERATION_ABORTED


def sample_reference_version(sample: Sample, reference_mode: str) -> int | None:
    if reference_mode == "completion":
        return sample.oldest_weight_version
    if reference_mode == "submission":
        version = sample.metadata.get(SUBMISSION_VERSION_KEY)
        return version if isinstance(version, int) else None
    if reference_mode == "prefill":
        return min(sample.first_prefill_weight_versions) if sample.first_prefill_weight_versions else None
    raise ValueError(f"Unknown staleness reference mode: {reference_mode}")


def straggler_collateral_indices(
    samples: Iterable[Sample],
    *,
    reference_mode: str,
    drain_version: int,
    bound: int,
) -> list[int | None]:
    """Samples that pass independently or cross only while awaiting a straggler."""
    samples = list(samples)
    group_completion = group_generation_completion_version(samples)
    collateral = []
    for sample in samples:
        reference = sample_reference_version(sample, reference_mode)
        if reference is None:
            continue
        passes_at_drain = drain_version - reference <= bound
        sample_completion = sample.metadata.get(SAMPLE_GENERATION_COMPLETE_VERSION_KEY)
        crosses_during_group_wait = (
            isinstance(sample_completion, int)
            and group_completion is not None
            and sample_completion < group_completion
            and sample_completion - reference <= bound
            and group_completion - reference > bound
        )
        if passes_at_drain or crosses_during_group_wait:
            collateral.append(sample.index)
    return collateral


def recycle_record(
    samples: Iterable[Sample],
    *,
    disposition: str,
    reason_code: str,
    reference_mode: str,
    reference_version: int | None,
    generation_completion_version: int | None,
    group_ready_version: int | None,
    queue_put_version: int | None,
    drain_version: int,
    bound: int | None,
    waste: dict[str, float],
    collateral_indices: list[int | None] | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """Build one primitive-only record suitable for the rollout debug dump."""
    samples = list(samples)
    numeric_rewards = [_numeric_reward(sample) for sample in samples]
    group_rewards = [reward for reward in numeric_rewards if reward is not None]
    group_reward_mean = float(np.mean(group_rewards)) if group_rewards else None
    group_reward_variance = float(np.var(group_rewards)) if group_rewards else None
    difficulty_features = [_difficulty_features(sample) for sample in samples]
    record: dict[str, Any] = {
        "schema_version": RECYCLE_DEBUG_SCHEMA_VERSION,
        "disposition": disposition,
        "reason_code": reason_code,
        "group_index": samples[0].group_index if samples else None,
        "prompt_id": (samples[0].metadata.get("prompt_id", samples[0].group_index) if samples else None),
        "sample_indices": [sample.index for sample in samples],
        "recycle_count_before": max((sample.retry_count for sample in samples), default=0),
        "generation_attempt_id": (
            f"{samples[0].group_index}:{max((sample.retry_count for sample in samples), default=0)}"
            if samples
            else None
        ),
        "reference_mode": reference_mode,
        "bound": bound,
        "versions": {
            "reference": reference_version,
            "generation_completion": generation_completion_version,
            "group_ready": group_ready_version,
            "queue_put": queue_put_version,
            "drain": drain_version,
        },
        "response_lengths": [sample.response_length for sample in samples],
        "generation_duration_seconds": [_sample_generation_duration(sample) for sample in samples],
        "rewards": numeric_rewards,
        "group_reward_mean": group_reward_mean,
        "group_reward_variance": group_reward_variance,
        "difficulty": [difficulty for difficulty, _ in difficulty_features],
        "prompt_pass_rates": [pass_rate for _, pass_rate in difficulty_features],
        "pre_queue_active": _sample_phase_values(
            samples,
            TRAJECTORY_START_VERSION_KEY,
            SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
        ),
        "pre_queue_group_wait": _sample_phase_values(
            samples,
            SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
            GROUP_GENERATION_COMPLETE_VERSION_KEY,
        ),
        "pre_queue_postprocess": _sample_phase_values(
            samples,
            GROUP_GENERATION_COMPLETE_VERSION_KEY,
            GROUP_READY_VERSION_KEY,
        ),
        "in_queue_staleness": _sample_phase_values(samples, GROUP_READY_VERSION_KEY, DRAIN_VERSION_KEY),
        "queue_wait_seconds": _sample_phase_values(samples, QUEUE_PUT_TIME_KEY, DRAIN_TIME_KEY),
        "attempt_wall_seconds": max(
            (float(sample.metadata.get(ATTEMPT_WALL_SECONDS_KEY, 0.0)) for sample in samples),
            default=0.0,
        ),
        "waste": dict(waste),
        "straggler_collateral_sample_indices": list(collateral_indices or []),
    }
    if detail is not None:
        record["detail"] = detail
    return record


def append_final_consumed_records(
    debug_metadata: dict[str, Any] | None,
    samples: list[Sample],
    *,
    reference_mode: str,
    bound: int | None,
    training_step: int,
) -> None:
    """Add postprocess-surviving rows to the optional joinable debug table."""
    if debug_metadata is None:
        return
    section = debug_metadata.setdefault(
        "recycle_compute",
        {"schema_version": RECYCLE_DEBUG_SCHEMA_VERSION, "records": []},
    )
    existing_consumed = {
        (
            record.get("training_step"),
            record.get("group_index"),
            tuple(record.get("sample_indices", [])),
        )
        for record in section["records"]
        if record.get("disposition") == "consumed"
    }
    grouped: dict[int | None, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_index].append(sample)
    for group in grouped.values():
        record_key = (training_step, group[0].group_index, tuple(sample.index for sample in group))
        if record_key in existing_consumed:
            continue
        references = [sample.metadata.get(BOUND_REFERENCE_VERSION_KEY) for sample in group]
        numeric_references = [version for version in references if isinstance(version, int)]
        ready_versions = [sample.metadata.get(GROUP_READY_VERSION_KEY) for sample in group]
        queue_versions = [sample.metadata.get(QUEUE_PUT_VERSION_KEY) for sample in group]
        drain_versions = [sample.metadata.get(DRAIN_VERSION_KEY) for sample in group]
        record = recycle_record(
            group,
            disposition="consumed",
            reason_code="entered_training_loss_input",
            reference_mode=reference_mode,
            reference_version=min(numeric_references) if numeric_references else None,
            generation_completion_version=group_generation_completion_version(group),
            group_ready_version=_single_numeric_value(ready_versions),
            queue_put_version=_single_numeric_value(queue_versions),
            drain_version=_single_numeric_value(drain_versions, default=-1),
            bound=bound,
            waste={},
        )
        record["training_step"] = training_step
        record["loss_input_tokens"] = [_loss_input_tokens(sample) for sample in group]
        section["records"].append(record)


def _single_numeric_value(values: list[Any], default: int | None = None) -> int | None:
    numeric = [value for value in values if isinstance(value, int)]
    if not numeric:
        return default
    if len(numeric) != len(values) or len(set(numeric)) != 1:
        raise RuntimeError(f"Inconsistent lifecycle values in one group: {values}")
    return numeric[0]


def _sample_phase_values(samples: list[Sample], start_key: str, end_key: str) -> list[float | int | None]:
    values = []
    for sample in samples:
        start = sample.metadata.get(start_key)
        end = sample.metadata.get(end_key)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            values.append(end - start)
        else:
            values.append(None)
    return values


def _sample_generation_duration(sample: Sample) -> float | None:
    [duration] = _sample_phase_values(
        [sample],
        TRAJECTORY_START_TIME_KEY,
        SAMPLE_GENERATION_COMPLETE_TIME_KEY,
    )
    if isinstance(duration, (int, float)):
        return float(duration)
    attempt_duration = sample.metadata.get(ATTEMPT_WALL_SECONDS_KEY)
    return float(attempt_duration) if isinstance(attempt_duration, (int, float)) else None


def _append_population_value(population: dict[str, list[float]], key: str, value: float | int | None) -> None:
    if isinstance(value, (int, float)):
        population.setdefault(key, []).append(float(value))


def add_selection_population(
    populations: dict[str, dict[str, list[float]]],
    *,
    population_name: str,
    samples: Iterable[Sample],
) -> None:
    """Accumulate small numeric marginals; joint analysis stays in debug records."""
    if population_name not in SELECTION_POPULATIONS:
        raise ValueError(f"Unknown selection population: {population_name}")
    samples = list(samples)
    population = populations.setdefault(population_name, {})
    numeric_rewards = [_numeric_reward(sample) for sample in samples]
    group_rewards = [reward for reward in numeric_rewards if reward is not None]
    group_mean = float(np.mean(group_rewards)) if group_rewards else None
    group_variance = float(np.var(group_rewards)) if group_rewards else None
    for sample, reward in zip(samples, numeric_rewards, strict=True):
        difficulty, prompt_pass_rate = _difficulty_features(sample)
        _append_population_value(population, "_sample_count", 1)
        # Queue telemetry owns response-length distributions at every admission
        # disposition. Only ``consumed`` is a later, postprocess-surviving
        # boundary and therefore has no equivalent queue/selection population.
        if population_name == "consumed":
            _append_population_value(population, "response_length", sample.response_length)
        _append_population_value(
            population,
            "generation_duration_seconds",
            _sample_generation_duration(sample),
        )
        _append_population_value(population, "reward", reward)
        _append_population_value(population, "group_reward_mean", group_mean)
        _append_population_value(population, "group_reward_variance", group_variance)
        _append_population_value(population, "tool_env_seconds", sample.non_generation_time)
        _append_population_value(population, "reward_seconds", sample.metadata.get(REWARD_SECONDS_KEY))
        _append_population_value(population, "difficulty", difficulty)
        _append_population_value(population, "prompt_pass_rate", prompt_pass_rate)
        for name, start_key, end_key in (
            ("pre_queue_active", TRAJECTORY_START_VERSION_KEY, SAMPLE_GENERATION_COMPLETE_VERSION_KEY),
            ("pre_queue_group_wait", SAMPLE_GENERATION_COMPLETE_VERSION_KEY, GROUP_GENERATION_COMPLETE_VERSION_KEY),
            ("pre_queue_postprocess", GROUP_GENERATION_COMPLETE_VERSION_KEY, GROUP_READY_VERSION_KEY),
            ("in_queue_staleness", GROUP_READY_VERSION_KEY, DRAIN_VERSION_KEY),
            ("queue_wait_seconds", QUEUE_PUT_TIME_KEY, DRAIN_TIME_KEY),
        ):
            [value] = _sample_phase_values([sample], start_key, end_key)
            _append_population_value(population, name, value)


def selection_population_metrics(
    populations: dict[str, dict[str, list[float]]],
    *,
    population_names: Iterable[str] = SELECTION_POPULATIONS,
) -> dict[str, float]:
    """Render generated/consumed/recycled marginals under a new namespace."""
    metrics: dict[str, float] = {}
    for population_name in population_names:
        if population_name not in SELECTION_POPULATIONS:
            raise ValueError(f"Unknown selection population: {population_name}")
        population = populations.get(population_name, {})
        sample_count = population.get("_sample_count", [])
        metrics[f"selection_bias/{population_name}/samples"] = float(len(sample_count))
        for field, values in population.items():
            if field == "_sample_count":
                continue
            for name, value in _distribution_metrics(values).items():
                metrics[f"selection_bias/{population_name}/{field}/{name}"] = value
    return metrics


def _loss_input_tokens(sample: Sample) -> int:
    if sample.remove_sample:
        return 0
    return sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length


def finalize_useful_rollout_metrics(
    samples: list[Sample],
    metrics: dict[str, Any] | None,
    *,
    has_custom_converter: bool,
) -> None:
    """Finalize efficiency after flattening/trimming, immediately before logging."""
    if metrics is None or GENERATED_TOKENS_KEY not in metrics:
        return

    metrics["rollout/fully_async/useful_rollout/available"] = float(not has_custom_converter)
    if has_custom_converter:
        return

    generated_tokens = int(metrics[GENERATED_TOKENS_KEY])
    admitted_tokens = int(metrics[ADMITTED_TOKENS_KEY])
    selected_tokens = sum(sample.response_length for sample in samples)
    loss_input_tokens = sum(_loss_input_tokens(sample) for sample in samples)
    postprocess_trimmed_tokens = admitted_tokens - selected_tokens
    loss_masked_tokens = selected_tokens - loss_input_tokens
    useful_efficiency = loss_input_tokens / generated_tokens if generated_tokens else 0.0

    known_discarded_tokens = int(metrics.get("rollout/fully_async/aborted_tokens", 0))
    known_discarded_tokens += int(metrics.get("rollout/fully_async/stale_tokens", 0))
    known_discarded_tokens += int(metrics.get("rollout/fully_async/age_cutoff_tokens", 0))
    known_discarded_tokens += int(metrics.get("rollout/fully_async/queue_evicted_tokens", 0))
    known_discarded_tokens += int(metrics.get("rollout/fully_async/dynamic_filter_tokens", 0))
    accounted_waste = known_discarded_tokens + postprocess_trimmed_tokens + loss_masked_tokens

    metrics.update(
        {
            "rollout/fully_async/useful_rollout/loss_input_tokens": loss_input_tokens,
            "rollout/fully_async/useful_rollout/efficiency": useful_efficiency,
            "rollout/fully_async/useful_rollout/wasted_tokens": generated_tokens - loss_input_tokens,
            "rollout/fully_async/useful_rollout/postprocess_trimmed_tokens": postprocess_trimmed_tokens,
            "rollout/fully_async/useful_rollout/loss_masked_tokens": loss_masked_tokens,
            "rollout/fully_async/useful_rollout/accounting_error_tokens": (
                generated_tokens - loss_input_tokens - accounted_waste
            ),
        }
    )

    consumed_populations: dict[str, dict[str, list[float]]] = {}
    add_selection_population(
        consumed_populations,
        population_name="consumed",
        samples=samples,
    )
    metrics.update(
        selection_population_metrics(
            consumed_populations,
            population_names=("consumed",),
        )
    )

    train_version = metrics.get("rollout/fully_async/current_weight_version")
    if isinstance(train_version, int):
        metrics.update(sample_lag_metrics(samples, train_version=train_version))
    metrics.update(prequeue_phase_metrics(samples))


def _weighted_mean(values: list[int], weights: list[int]) -> float | None:
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight


def _distribution_metrics(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
    }


def _weighted_distribution_metrics(weight_by_value: dict[int, int]) -> dict[str, float]:
    positive = sorted((value, weight) for value, weight in weight_by_value.items() if weight > 0)
    if not positive:
        return {}
    total = sum(weight for _, weight in positive)

    def percentile(q: float) -> float:
        target = q * total
        cumulative = 0
        for value, weight in positive:
            cumulative += weight
            if cumulative >= target:
                return float(value)
        return float(positive[-1][0])

    return {
        "mean": sum(value * weight for value, weight in positive) / total,
        "max": float(positive[-1][0]),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "num_tokens": float(total),
    }


def _numeric_reward(sample: Sample) -> float | None:
    return float(sample.reward) if isinstance(sample.reward, (int, float)) else None


def _difficulty_features(sample: Sample) -> tuple[float | None, float | None]:
    """Return a numeric difficulty proxy and an explicit measured pass rate."""
    raw = sample.metadata.get("difficulty")
    if isinstance(raw, (int, float)):
        return float(raw), None
    if not isinstance(raw, dict):
        return None, None

    selected_policy = sample.metadata.get("difficulty_policy")
    selected = raw.get(selected_policy) if isinstance(selected_policy, str) else None
    candidates = [selected] if isinstance(selected, dict) else list(raw.values())
    pass_rates = [
        float(candidate["pass_rate"])
        for candidate in candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("pass_rate"), (int, float))
    ]
    if len(pass_rates) != 1 or not 0.0 <= pass_rates[0] <= 1.0:
        return None, None
    pass_rate = pass_rates[0]
    return 1.0 - pass_rate, pass_rate


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _metadata_number(sample: Sample, key: str, expected_type: type) -> int | float | None:
    value = sample.metadata.get(key)
    return value if isinstance(value, expected_type) else None


def _phase_rows(samples: list[Sample], keys: tuple[str, str, str, str], expected_type: type):
    rows = []
    for sample in samples:
        values = tuple(_metadata_number(sample, key, expected_type) for key in keys)
        if any(value is None for value in values):
            continue
        start, sample_complete, group_complete, ready = values
        if not start <= sample_complete <= group_complete <= ready:
            raise RuntimeError(
                "Non-monotonic fully-async lifecycle boundaries for sample "
                f"{sample.index}: start={start}, sample_complete={sample_complete}, "
                f"group_complete={group_complete}, ready={ready}"
            )
        rows.append((sample, start, sample_complete, group_complete, ready))
    return rows


def _phase_component_metrics(
    rows,
    *,
    prefix: str,
) -> dict[str, float]:
    components = {
        "active": [sample_complete - start for _, start, sample_complete, _, _ in rows],
        "group_wait": [group_complete - sample_complete for _, _, sample_complete, group_complete, _ in rows],
        "postprocess": [ready - group_complete for _, _, _, group_complete, ready in rows],
        "total": [ready - start for _, start, _, _, ready in rows],
    }
    metrics: dict[str, float] = {}
    generated_weights = [sample.response_length for sample, *_ in rows]
    loss_weights = [_loss_input_tokens(sample) for sample, *_ in rows]
    for component, values in components.items():
        for name, value in _distribution_metrics(values).items():
            metrics[f"{prefix}/{component}/sequence_{name}"] = value
        for weighting, weights in (("generated_token", generated_weights), ("loss_token", loss_weights)):
            mean = _weighted_mean(values, weights)
            if mean is not None:
                metrics[f"{prefix}/{component}/{weighting}_mean"] = mean

    identity_errors = [
        total - active - group_wait - postprocess
        for total, active, group_wait, postprocess in zip(
            components["total"],
            components["active"],
            components["group_wait"],
            components["postprocess"],
            strict=True,
        )
    ]
    metrics[f"{prefix}/identity_max_abs_error"] = max((abs(value) for value in identity_errors), default=0.0)
    return metrics


def prequeue_phase_metrics(samples: list[Sample]) -> dict[str, float]:
    """Split trajectory-start-to-ready lag without redefining legacy pre-queue."""
    version_rows = _phase_rows(
        samples,
        (
            TRAJECTORY_START_VERSION_KEY,
            SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
            GROUP_GENERATION_COMPLETE_VERSION_KEY,
            GROUP_READY_VERSION_KEY,
        ),
        int,
    )
    wall_rows = _phase_rows(
        samples,
        (
            TRAJECTORY_START_TIME_KEY,
            SAMPLE_GENERATION_COMPLETE_TIME_KEY,
            GROUP_GENERATION_COMPLETE_TIME_KEY,
            GROUP_READY_TIME_KEY,
        ),
        float,
    )
    exact = sum(bool(sample.metadata.get(LIFECYCLE_EXACT_KEY, False)) for sample in samples)
    metrics = {
        "staleness/pre_queue_phase/provenance_sample_frac": len(version_rows) / len(samples) if samples else 0.0,
        "staleness/pre_queue_phase/exact_sample_frac": exact / len(samples) if samples else 0.0,
        # Reward work for an early sample may overlap generation of a straggler.
        # The additive partition follows the critical-path boundaries and assigns
        # that overlap to group_wait rather than double-counting it.
        "staleness/pre_queue_phase/critical_path_overlap_semantics": 1.0,
    }
    metrics.update(_phase_component_metrics(version_rows, prefix="staleness/pre_queue_phase/version"))
    metrics.update(_phase_component_metrics(wall_rows, prefix="staleness/pre_queue_phase/wall_seconds"))
    reference_deltas = []
    for sample, active_start, _, _, _ in version_rows:
        selected_reference = sample.metadata.get(SAMPLE_REFERENCE_VERSION_KEY)
        if isinstance(selected_reference, int):
            reference_deltas.append(active_start - selected_reference)
    metrics["staleness/pre_queue_phase/selected_reference_alignment_sample_frac"] = (
        len(reference_deltas) / len(samples) if samples else 0.0
    )
    for name, value in _distribution_metrics(reference_deltas).items():
        metrics[f"staleness/pre_queue_phase/active_start_minus_selected_reference/{name}"] = value
    return metrics


def _exact_token_lag_metrics(samples: list[Sample], train_version: int) -> dict[str, float]:
    lag_weights: dict[int, int] = defaultdict(int)
    covered_tokens = 0
    total_response_tokens = sum(sample.response_length for sample in samples)
    invalid_segments = 0
    invalid_turns = 0
    invalid_samples = 0
    for sample in samples:
        sample_lag_weights: dict[int, int] = defaultdict(int)
        sample_covered_tokens = 0
        for turn_segments in sample.response_weight_version_segments:
            parsed_segments = []
            expected_start = 0
            turn_is_valid = True
            for segment in turn_segments:
                if not isinstance(segment, (list, tuple)) or len(segment) != 3:
                    invalid_segments += 1
                    turn_is_valid = False
                    break
                start, end, version = segment
                if not all(isinstance(value, int) for value in (start, end, version)):
                    invalid_segments += 1
                    turn_is_valid = False
                    break
                if start != expected_start or end <= start or version < 0 or version > train_version:
                    invalid_segments += 1
                    turn_is_valid = False
                    break
                parsed_segments.append((start, end, version))
                expected_start = end

            if not turn_is_valid:
                invalid_turns += 1
                continue
            for start, end, version in parsed_segments:
                tokens = end - start
                sample_lag_weights[train_version - version] += tokens
                sample_covered_tokens += tokens

        if sample_covered_tokens > sample.response_length:
            invalid_samples += 1
            continue
        for lag, tokens in sample_lag_weights.items():
            lag_weights[lag] += tokens
        covered_tokens += sample_covered_tokens

    metrics = {
        "staleness/token_lag/exact/covered_response_token_frac": (
            covered_tokens / total_response_tokens if total_response_tokens else 0.0
        ),
        "staleness/token_lag/exact/invalid_segments": float(invalid_segments),
        "staleness/token_lag/exact/invalid_turns": float(invalid_turns),
        "staleness/token_lag/exact/invalid_samples": float(invalid_samples),
    }
    metrics.update(
        {
            f"staleness/token_lag/exact/{name}": value
            for name, value in _weighted_distribution_metrics(lag_weights).items()
        }
    )
    return metrics


def sample_lag_metrics(samples: list[Sample], *, train_version: int) -> dict[str, float]:
    """Compute sequence/token-weighted D2 lag components over the trained rows."""
    grouped: dict[int | None, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_index].append(sample)

    rows: list[tuple[Sample, int, int, int]] = []
    group_spreads: list[int] = []
    for group in grouped.values():
        ends = [max(sample.last_forward_weight_versions) for sample in group if sample.last_forward_weight_versions]
        if len(ends) != len(group):
            continue
        group_completion = max(ends)
        group_spreads.append(max(ends) - min(ends))
        for sample, end in zip(group, ends, strict=True):
            if not sample.first_prefill_weight_versions:
                continue
            start = min(sample.first_prefill_weight_versions)
            if not (start <= end <= group_completion <= train_version):
                continue
            rows.append((sample, start, end, group_completion))

    metrics: dict[str, float] = {
        "staleness/sample_lag/provenance_sample_frac": len(rows) / len(samples) if samples else 0.0,
    }
    if not rows:
        metrics.update(_exact_token_lag_metrics(samples, train_version))
        return metrics

    components = {
        "generation": [end - start for _, start, end, _ in rows],
        "group_sync": [group_completion - end for _, _, end, group_completion in rows],
        "train_handoff": [train_version - group_completion for _, _, _, group_completion in rows],
        "total": [train_version - start for _, start, _, _ in rows],
    }
    generated_weights = [sample.response_length for sample, *_ in rows]
    loss_weights = [_loss_input_tokens(sample) for sample, *_ in rows]
    for component, values in components.items():
        for name, value in _distribution_metrics(values).items():
            metrics[f"staleness/sample_lag/{component}/sequence_{name}"] = value
        for weighting, weights in (("generated_token", generated_weights), ("loss_token", loss_weights)):
            mean = _weighted_mean(values, weights)
            if mean is not None:
                metrics[f"staleness/sample_lag/{component}/{weighting}_mean"] = mean

    metrics.update(
        {
            f"staleness/sample_lag/within_group_end_spread/{name}": value
            for name, value in _distribution_metrics(group_spreads).items()
        }
    )
    total_lags = components["total"]
    length_corr = _correlation([float(weight) for weight in generated_weights], [float(lag) for lag in total_lags])
    if length_corr is not None:
        metrics["staleness/sample_lag/corr_response_length_total"] = length_corr
    reward_pairs = [(_numeric_reward(sample), lag) for (sample, *_), lag in zip(rows, total_lags, strict=True)]
    numeric_pairs = [(reward, lag) for reward, lag in reward_pairs if reward is not None]
    reward_corr = _correlation(
        [reward for reward, _ in numeric_pairs],
        [float(lag) for _, lag in numeric_pairs],
    )
    if reward_corr is not None:
        metrics["staleness/sample_lag/corr_reward_total"] = reward_corr

    metrics.update(_exact_token_lag_metrics(samples, train_version))
    return metrics


def build_batch_consumption_snapshot(
    samples: list[Sample],
    *,
    selection_version: int | None,
    bound: int | None,
    optimizer_updates: int = 1,
    cohort_generated_tokens: int | None = None,
    has_custom_converter: bool = False,
) -> dict[str, Any]:
    """Retain only compact group primitives until the trainer consumes the batch."""
    groups: dict[int | None, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_index].append(sample)
    group_rows = []
    for group in groups.values():
        references = [sample.metadata.get(BOUND_REFERENCE_VERSION_KEY) for sample in group]
        numeric = [version for version in references if isinstance(version, int)]
        group_rows.append(
            {
                "reference_version": min(numeric) if numeric else None,
                "response_tokens": sum(sample.response_length for sample in group),
            }
        )
    queue_wait_seconds = []
    for sample in samples:
        queue_put = sample.metadata.get(QUEUE_PUT_TIME_KEY)
        drain = sample.metadata.get(DRAIN_TIME_KEY)
        if isinstance(queue_put, (int, float)) and isinstance(drain, (int, float)) and drain >= queue_put:
            queue_wait_seconds.append(float(drain - queue_put))
    return {
        "selection_weight_version": selection_version,
        "bound": bound,
        "groups": group_rows,
        "loss_input_tokens": (None if has_custom_converter else sum(_loss_input_tokens(sample) for sample in samples)),
        "queue_wait_seconds": queue_wait_seconds,
        "optimizer_updates": optimizer_updates,
        "cohort_generated_tokens": cohort_generated_tokens,
    }


def batch_consumption_metrics(
    snapshot: dict[str, Any],
    *,
    train_start_version: int,
    pipeline_snapshot: dict[str, float] | None = None,
) -> dict[str, float | int]:
    """Attribute bound crossings after queue selection without calling them recycled."""
    selection_version = snapshot.get("selection_weight_version")
    metrics = {"queue/consumption/train_start_weight_version": train_start_version}
    if isinstance(selection_version, int):
        gap = train_start_version - selection_version
        if gap < 0:
            raise RuntimeError(
                "Applied weight version moved backwards between queue selection and training: "
                f"selection={selection_version}, train_start={train_start_version}"
            )
        metrics["queue/consumption/selection_weight_version"] = selection_version
        metrics["queue/consumption/selection_to_train_gap"] = gap

    bound = snapshot.get("bound")
    late_groups = 0
    late_tokens = 0
    if isinstance(bound, int):
        for group in snapshot.get("groups", []):
            reference = group.get("reference_version")
            if isinstance(reference, int) and train_start_version - reference > bound:
                late_groups += 1
                late_tokens += int(group.get("response_tokens", 0))
    metrics["staleness/late_stale_trained/forward_handoff_groups"] = late_groups
    metrics["staleness/late_stale_trained/forward_handoff_tokens"] = late_tokens
    for name, value in _distribution_metrics(snapshot.get("queue_wait_seconds", [])).items():
        metrics[f"queue/consumption/wall_wait_seconds/{name}"] = value

    raw_accepted_tokens = snapshot.get("loss_input_tokens")
    accepted_tokens = int(raw_accepted_tokens) if isinstance(raw_accepted_tokens, int) else None
    optimizer_updates = int(snapshot.get("optimizer_updates", 1))
    cohort_generated = snapshot.get("cohort_generated_tokens")
    metrics["throughput/accepted_loss_tokens_available"] = float(accepted_tokens is not None)
    if accepted_tokens is not None:
        metrics["throughput/accepted_loss_tokens"] = accepted_tokens
        metrics["throughput/cohort_accepted_loss_tokens"] = accepted_tokens
    metrics["throughput/planned_optimizer_updates"] = optimizer_updates
    if accepted_tokens is not None and isinstance(cohort_generated, int) and cohort_generated >= 0:
        metrics["throughput/cohort_useful_efficiency"] = (
            accepted_tokens / cohort_generated if cohort_generated > 0 else 0.0
        )
    if pipeline_snapshot is not None:
        metrics.update(
            pipeline_throughput_metrics(
                pipeline_snapshot,
                cohort_accepted_tokens=accepted_tokens,
                cohort_generated_tokens=(cohort_generated if isinstance(cohort_generated, int) else None),
            )
        )
    return metrics


def pipeline_throughput_metrics(
    pipeline_snapshot: dict[str, float],
    *,
    cohort_accepted_tokens: int | None,
    cohort_generated_tokens: int | None,
) -> dict[str, float]:
    """Render one completion-to-completion producer/trainer wall window."""
    window_seconds = float(pipeline_snapshot.get("window_seconds", 0.0))
    generated_tokens = float(pipeline_snapshot.get("generated_tokens", 0.0))
    completed_updates_available = "optimizer_updates" in pipeline_snapshot
    completed_updates = float(pipeline_snapshot.get("optimizer_updates", 0.0))
    window_accepted_available = bool(pipeline_snapshot.get("accepted_tokens_available", False))
    window_accepted_tokens = float(pipeline_snapshot.get("accepted_tokens", 0.0))
    metrics = {
        "throughput/window_seconds": window_seconds,
        "throughput/generated_tokens": generated_tokens,
        "throughput/generated_tokens_per_second": (generated_tokens / window_seconds if window_seconds > 0.0 else 0.0),
        "throughput/completed_training_batches": float(pipeline_snapshot.get("completed_training_batches", 0.0)),
        "throughput/optimizer_updates_available": float(completed_updates_available),
        "throughput/window_accepted_loss_tokens_available": float(window_accepted_available),
        "queue/depth_time_mean": float(pipeline_snapshot.get("queue_depth_time_mean", 0.0)),
        "queue/depth_current": float(pipeline_snapshot.get("queue_depth_current", 0.0)),
        "queue/trainer_starvation_seconds": float(pipeline_snapshot.get("trainer_starvation_seconds", 0.0)),
        "queue/rollout_backpressure_seconds": float(pipeline_snapshot.get("rollout_backpressure_seconds", 0.0)),
        "queue/rollout_idle_capacity_seconds": float(pipeline_snapshot.get("rollout_idle_capacity_seconds", 0.0)),
        "queue/active_group_capacity_fraction": float(pipeline_snapshot.get("active_group_capacity_fraction", 0.0)),
        "queue/active_group_capacity_time_mean": float(pipeline_snapshot.get("active_group_capacity_time_mean", 0.0)),
    }
    if completed_updates_available:
        metrics["throughput/optimizer_updates"] = completed_updates
        metrics["throughput/optimizer_updates_per_second"] = (
            completed_updates / window_seconds if window_seconds > 0.0 else 0.0
        )
    if window_accepted_available:
        metrics["throughput/window_accepted_loss_tokens"] = window_accepted_tokens
        metrics["throughput/accepted_tokens_per_second"] = (
            window_accepted_tokens / window_seconds if window_seconds > 0.0 else 0.0
        )
        # v_useful = eta_useful * v_generated = accepted tokens / second.
        metrics["throughput/useful_tokens_per_second"] = (
            window_accepted_tokens / window_seconds if window_seconds > 0.0 else 0.0
        )
        metrics["throughput/window_useful_efficiency"] = (
            window_accepted_tokens / generated_tokens if generated_tokens > 0.0 else 0.0
        )
    if cohort_accepted_tokens is not None and cohort_generated_tokens is not None and cohort_generated_tokens >= 0:
        cohort_efficiency = cohort_accepted_tokens / cohort_generated_tokens if cohort_generated_tokens > 0 else 0.0
        metrics["throughput/cohort_projected_useful_tokens_per_second"] = (
            cohort_efficiency * generated_tokens / window_seconds if window_seconds > 0.0 else 0.0
        )
    return metrics
