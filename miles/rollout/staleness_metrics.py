import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from urllib.parse import quote

from miles.rollout.filter_hub.base_types import iter_samples
from miles.rollout.filter_hub.common_filters import GroupWeightVersionStats
from miles.utils.types import Sample


def sample_category(sample: Sample) -> str:
    metadata = sample.metadata or {}
    extra = metadata.get("extra_env_info") or {}
    if not isinstance(extra, Mapping):
        raise ValueError("Sample extra_env_info must be a mapping")
    agent = extra.get("agent_ref") or metadata.get("agent_ref") or {}
    candidates = (
        metadata.get("category"),
        metadata.get("task_source"),
        extra.get("task_source"),
        agent.get("name") if isinstance(agent, Mapping) else None,
        metadata.get("task_name"),
    )
    for value in candidates:
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Sample category must be a string, got {value!r}")
        if value.strip():
            return value.strip()
    return "unknown"


def category_key(category: str) -> str:
    # Escape '/' too: literal category hierarchy cannot collide with metric names.
    return quote(category, safe="")


def _numeric_version(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(f"Policy version must be a non-negative integer, got {value!r}")
    return int(value)


def lifecycle_version(group, key: str) -> int | None:
    values = [
        _numeric_version((sample.metadata or {}).get("policy_provenance", {}).get(key))
        for sample in iter_samples(group)
    ]
    return min(values) if values and all(v is not None for v in values) else None


def staleness_value(group, stats: GroupWeightVersionStats, current_version: int | None, *, definition: str):
    if definition == "oldest":
        return stats.oldest_lag(current_version)
    if definition == "newest":
        return stats.newest_lag(current_version)
    if definition == "token-mean":
        return stats.token_weighted_lag(current_version)
    keys = {
        "submission": "submitted_weight_version",
        "first-prefill": "first_prefill_weight_version",
        "ready": "ready_weight_version",
    }
    if definition not in keys:
        raise ValueError(f"Unknown staleness definition: {definition}")
    version = lifecycle_version(group, keys[definition])
    return current_version - version if current_version is not None and version is not None else None


def summarize_staleness(values: Sequence[float], *, prefix: str, histogram: bool = False) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(fraction):
        pos = (len(ordered) - 1) * fraction
        low, high = math.floor(pos), math.ceil(pos)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

    variance = statistics.pvariance(values)
    result = {
        "mean": statistics.fmean(values),
        "variance": variance,
        "std": math.sqrt(variance),
        "max": max(values),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "num_groups": len(values),
        "frac_zero": values.count(0) / len(values),
    }
    if histogram:
        if any(v < 0 or int(v) != v for v in values):
            raise ValueError("Version-lag histogram requires non-negative integer lags")
        counts = Counter(values)
        result.update({f"count_{i}": counts[i] for i in range(41)})
        result["count_ge_41"] = sum(n for v, n in counts.items() if v >= 41)
    return {f"{prefix}/{k}": float(v) for k, v in result.items()}


def _response_token_stats(group, *, loss_enabled: bool) -> GroupWeightVersionStats:
    samples, known_samples, count, version_sum = 0, 0, 0, 0
    oldest = newest = None
    for sample in group:
        samples += 1
        mask = sample.loss_mask if loss_enabled else None
        if mask is not None and len(mask) != sample.response_length:
            raise ValueError("Staleness telemetry requires response-aligned loss masks")
        prompt_len = len(sample.tokens) - sample.response_length
        if prompt_len < 0:
            raise ValueError("Response length exceeds sample token length")
        # Multimodal expansion does not yet remap spans; publish missing coverage rather than guessed alignment.
        if (loss_enabled and sample.remove_sample) or sample.multimodal_train_inputs:
            continue
        known = False
        for span in sample.all_weight_version_spans:
            start, end = max(span.abs_start, prompt_len), min(span.abs_end, len(sample.tokens))
            version = _numeric_version(span.version)
            if version is None:
                continue
            tokens = (
                (end - start if mask is None else sum(bool(x) for x in mask[start - prompt_len : end - prompt_len]))
                if end > start
                else 0
            )
            if tokens:
                count += tokens
                version_sum += tokens * version
                oldest = version if oldest is None else min(oldest, version)
                newest = version if newest is None else max(newest, version)
                known = True
        known_samples += known
    return GroupWeightVersionStats(samples, known_samples, count, version_sum, oldest, newest)


def _category_balance_metrics(categories, category_groups, *, targets):
    num_samples = sum(map(len, categories.values()))
    num_groups = sum(map(len, category_groups.values()))
    total_tokens = sum(s.response_length for rows in categories.values() for s in rows)
    result = {"rollout/category/num_samples": float(num_samples), "rollout/category/num_groups": float(num_groups)}
    if targets is not None and (
        any(not math.isfinite(value) or value < 0 for value in targets.values())
        or not math.isclose(sum(targets.values()), 1.0)
    ):
        raise ValueError("Category target fractions must be finite, non-negative, and sum to one")
    deltas = []
    for category in sorted(set(categories) | set(targets or {})):
        prefix = f"rollout/category/{category_key(category)}"
        rows = categories.get(category, [])
        lengths = [s.response_length for s in rows]
        fraction = len(category_groups.get(category, [])) / num_groups
        result.update(
            {
                f"{prefix}/sample_count": float(len(rows)),
                f"{prefix}/sample_fraction": len(rows) / num_samples,
                f"{prefix}/group_count": float(len(category_groups.get(category, []))),
                f"{prefix}/group_fraction": fraction,
                f"{prefix}/response_token_count": float(sum(lengths)),
                f"{prefix}/response_length/count": float(len(rows)),
            }
        )
        if total_tokens:
            result[f"{prefix}/response_token_fraction"] = sum(lengths) / total_tokens
        if lengths:
            statistics_by_key = summarize_staleness(lengths, prefix=f"{prefix}/response_length")
            result.update({key: value for key, value in statistics_by_key.items() if not key.endswith("/num_groups")})
            result[f"{prefix}/truncated_fraction"] = sum(s.status == Sample.Status.TRUNCATED for s in rows) / len(rows)
        if targets is not None:
            target = targets.get(category, 0.0)
            delta = fraction - target
            result[f"{prefix}/target_group_fraction"] = target
            result[f"{prefix}/group_fraction_delta"] = delta
            deltas.append(abs(delta))
    if targets is not None:
        result["rollout/category/balance/total_variation"] = sum(deltas) / 2
        result["rollout/category/balance/max_abs_group_fraction_delta"] = max(deltas)
    return result


def batch_category_staleness_metrics(
    samples: list[Sample], *, current_version: int | None, target_group_fractions: Mapping[str, float] | None = None
) -> dict[str, float]:
    if not samples:
        return {}
    groups = defaultdict(list)
    categories = defaultdict(list)
    for i, sample in enumerate(samples):
        category = sample_category(sample)
        categories[category].append(sample)
        group_id = ("group", sample.group_index) if sample.group_index is not None else ("row", i)
        groups[group_id].append(sample)
    category_groups = defaultdict(list)
    for group in groups.values():
        labels = {sample_category(sample) for sample in group}
        if len(labels) != 1:
            raise ValueError("Siblings in one prompt group must share a category")
        category_groups[next(iter(labels))].append(group)
    result = _category_balance_metrics(categories, category_groups, targets=target_group_fractions)
    populations = [("rollout/staleness/total", list(groups.values()))]
    for category in sorted(categories):
        populations.append((f"rollout/staleness/category/{category_key(category)}", category_groups[category]))
    if current_version is None:
        result["rollout/staleness/current_version_known"] = 0.0
        return result
    _numeric_version(current_version)
    result["rollout/staleness/current_version_known"] = 1.0
    result["rollout/staleness/current_version"] = float(current_version)
    for prefix, selected in populations:
        result.update(_population_metrics(selected, current_version=current_version, prefix=prefix))
    return result


def _population_metrics(groups, *, current_version: int, prefix: str) -> dict[str, float]:
    for group in groups:
        for sample in group:
            previous = 0
            for span in sample.all_weight_version_spans:
                version = _numeric_version(span.version)
                if version is not None and version > current_version:
                    raise ValueError("Generated policy version is newer than the published training reference")
                if not previous <= span.abs_start < span.abs_end <= len(sample.tokens):
                    raise ValueError("Staleness telemetry has overlapping or out-of-range token spans")
                previous = span.abs_end
    # Historical prompt/prefill spans must not enter the selected response population.
    stats = [_response_token_stats(group, loss_enabled=False) for group in groups]
    for value in stats:
        if value.newest_version is not None and value.newest_version > current_version:
            raise ValueError("Generated policy version is newer than the published training reference")
    known = sum(s.oldest_version is not None for s in stats)
    result = {
        f"{prefix}/version_coverage": known / len(groups),
        f"{prefix}/missing_groups": float(len(groups) - known),
    }
    for definition in ("oldest", "newest", "token-mean", "submission", "first-prefill", "ready"):
        values = [
            staleness_value(g, s, current_version, definition=definition) for g, s in zip(groups, stats, strict=True)
        ]
        known_values = [v for v in values if v is not None]
        if any(v < 0 for v in known_values):
            raise ValueError("Lifecycle policy version is newer than the training reference")
        result[f"{prefix}/{definition}/coverage"] = len(known_values) / len(groups)
        result.update(
            summarize_staleness(known_values, prefix=f"{prefix}/{definition}", histogram=definition != "token-mean")
        )
    token_count = sum(s.versioned_token_count for s in stats)
    response_tokens = sum(s.response_length for group in groups for s in group)
    if response_tokens:
        result[f"{prefix}/generated_token_version_coverage"] = token_count / response_tokens
    result[f"{prefix}/fully_sample_versioned_group_fraction"] = sum(
        s.versioned_sample_count == s.sample_count for s in stats
    ) / len(groups)
    if token_count:
        result[f"{prefix}/generated_token_weighted_lag"] = (
            current_version - sum(s.token_version_sum for s in stats) / token_count
        )
    loss_stats = [_response_token_stats(group, loss_enabled=True) for group in groups]
    enabled_tokens = sum(
        s.response_length if s.loss_mask is None else sum(bool(x) for x in s.loss_mask)
        for group in groups
        for s in group
        if not s.remove_sample
    )
    loss_count = sum(s.versioned_token_count for s in loss_stats)
    result[f"{prefix}/loss_enabled_versioned_tokens"] = float(loss_count)
    if enabled_tokens:
        result[f"{prefix}/loss_enabled_token_version_coverage"] = loss_count / enabled_tokens
    if loss_count:
        result[f"{prefix}/loss_enabled_token_weighted_lag"] = (
            current_version - sum(s.token_version_sum for s in loss_stats) / loss_count
        )
    return result
