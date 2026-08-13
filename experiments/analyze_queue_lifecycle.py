#!/usr/bin/env python3
"""Summarize fully-async queue lifecycle records from rollout dumps.

Run this on trusted ``--save-debug-rollout-data`` files from one training run:

    python experiments/analyze_queue_lifecycle.py /path/to/rollout_data/*.pt

For queue-drop, also pass the measured rollout/trainer throughput ratio and the
actual sample concurrency to compare observed prefill staleness with the
closed-form approximation:

    python experiments/analyze_queue_lifecycle.py /path/to/*.pt \
        --rho 1.14 --concurrency-samples 128

The lifecycle metadata is primitive-only. For the usual tensor-free rollout
dumps, this script can therefore run without torch or numpy. As with
``torch.load``, only open trusted pickle files.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEBUG_METADATA_KEY = "rollout_fn_debug"
COMPLETED_DISPOSITIONS = {
    "trained",
    "stale_recycled",
    "age_cutoff_dropped",
    "dynamic_filter_dropped",
    "queue_evicted",
}


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def distribution(values: Iterable[int | float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "sum": 0.0}
    return {
        "count": len(ordered),
        "sum": sum(ordered),
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _load_primitive_torch_dump(path: Path) -> dict[str, Any]:
    """Load a tensor-free torch.save archive with only the standard library."""
    with zipfile.ZipFile(path) as archive:
        pickle_members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(pickle_members) != 1:
            raise ValueError(f"{path}: expected one data.pkl member, found {pickle_members}")
        return pickle.loads(archive.read(pickle_members[0]))  # noqa: S301 - trusted experiment dump


def load_lifecycle(path: Path) -> dict[str, Any]:
    payload = _load_primitive_torch_dump(path)
    metadata = payload.get("metadata") or {}
    lifecycle = metadata.get(DEBUG_METADATA_KEY)
    if lifecycle is None:
        raise ValueError(f"{path}: no metadata[{DEBUG_METADATA_KEY!r}]; recording was not enabled")
    if lifecycle.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported queue lifecycle schema {lifecycle.get('schema_version')!r}")
    return lifecycle


def _lengths(records: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    sample_lengths = [length for record in records for length in record.get("response_lengths", [])]
    group_max_lengths = [max(record["response_lengths"]) for record in records if record.get("response_lengths")]
    return sample_lengths, group_max_lengths


def _length_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples, group_maxima = _lengths(records)
    return {
        "groups": len(group_maxima),
        "sample_length": distribution(samples),
        "group_max_length": distribution(group_maxima),
    }


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        return None
    covariance = sum((x - left_mean) * (y - right_mean) for x, y in pairs)
    return covariance / math.sqrt(left_ss * right_ss)


def _reward_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    sample_rewards = []
    group_mean_rewards = []
    sample_length_reward_pairs = []
    group_length_reward_pairs = []
    records_with_values = 0
    samples_missing_reward = 0
    group_categories: Counter[str] = Counter()

    for record in records:
        lengths = record.get("response_lengths", [])
        raw_rewards = record.get("reward_values")
        if raw_rewards is None:
            samples_missing_reward += len(lengths)
            continue
        if len(raw_rewards) != len(lengths):
            raise ValueError(
                f"attempt {record.get('attempt_id')}: {len(raw_rewards)} rewards "
                f"do not align with {len(lengths)} response lengths"
            )
        records_with_values += 1
        rewards = []
        for length, raw_reward in zip(lengths, raw_rewards, strict=True):
            if raw_reward is None:
                samples_missing_reward += 1
                continue
            reward = float(raw_reward)
            rewards.append(reward)
            sample_rewards.append(reward)
            sample_length_reward_pairs.append((float(length), reward))

        if len(rewards) != len(lengths) or not rewards:
            continue
        group_mean = statistics.fmean(rewards)
        group_mean_rewards.append(group_mean)
        group_length_reward_pairs.append((float(max(lengths)), group_mean))
        if all(reward == 0 for reward in rewards):
            group_categories["all_zero"] += 1
        elif all(reward == 1 for reward in rewards):
            group_categories["all_one"] += 1
        elif len(set(rewards)) > 1:
            group_categories["mixed"] += 1
        else:
            group_categories["uniform_other"] += 1

    complete_groups = len(group_mean_rewards)
    return {
        "records_with_reward_values": records_with_values,
        "groups_with_complete_rewards": complete_groups,
        "samples_missing_reward": samples_missing_reward,
        "sample_reward": distribution(sample_rewards),
        "group_mean_reward": distribution(group_mean_rewards),
        "all_zero_group_frac": group_categories["all_zero"] / complete_groups if complete_groups else None,
        "all_one_group_frac": group_categories["all_one"] / complete_groups if complete_groups else None,
        "mixed_reward_group_frac": group_categories["mixed"] / complete_groups if complete_groups else None,
        "uniform_other_group_frac": group_categories["uniform_other"] / complete_groups if complete_groups else None,
        "sample_length_reward_pearson": _pearson(sample_length_reward_pairs),
        "group_max_length_mean_reward_pearson": _pearson(group_length_reward_pairs),
    }


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {**_length_summary(records), "reward": _reward_summary(records)}


def _version_gap(records: list[dict[str, Any]], end: str, start: str) -> dict[str, float | int]:
    gaps = []
    for record in records:
        end_version = record.get(end)
        start_version = record.get(start)
        if isinstance(end_version, int) and isinstance(start_version, int):
            gap = end_version - start_version
            if gap < 0:
                raise ValueError(
                    f"attempt {record.get('attempt_id')}: negative version gap "
                    f"{end}={end_version} - {start}={start_version}"
                )
            gaps.append(gap)
    return distribution(gaps)


def _infer_batch_shape(trained: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    groups_by_rollout = Counter(record["rollout_id"] for record in trained if record.get("rollout_id") is not None)
    group_sizes = Counter(len(record.get("response_lengths", [])) for record in trained)
    if not groups_by_rollout or not group_sizes:
        return None, None
    groups_per_batch = groups_by_rollout.most_common(1)[0][1]
    samples_per_group = group_sizes.most_common(1)[0][0]
    return groups_per_batch, samples_per_group


def queue_drop_prediction(
    *,
    rho: float,
    concurrency_samples: int,
    batch_samples: int,
    queue_factor: float,
    tailness: float,
) -> dict[str, float]:
    if rho <= 0:
        raise ValueError("--rho must be positive")
    if concurrency_samples < 1:
        raise ValueError("--concurrency-samples must be positive")
    pre_queue = concurrency_samples * tailness / (batch_samples * max(1.0, rho))
    if rho < 1:
        in_queue = rho
    elif rho > 1:
        in_queue = (2 * queue_factor + rho - 1) / (2 * rho)
    else:
        raise ValueError(
            "--rho=1 is the boundary between the rollout- and train-bound "
            "approximations; use the measured non-rounded throughput ratio"
        )
    return {
        "pre_queue": pre_queue,
        "in_queue": in_queue,
        "total": pre_queue + in_queue,
    }


def summarize(
    lifecycles: list[dict[str, Any]],
    *,
    rho: float | None,
    concurrency_samples: int | None,
) -> dict[str, Any]:
    policies = {lifecycle["policy"] for lifecycle in lifecycles}
    capacities = {lifecycle["capacity_groups"] for lifecycle in lifecycles}
    if len(policies) != 1 or len(capacities) != 1:
        raise ValueError(f"dumps must come from one run; found policies={policies}, capacities={capacities}")
    policy = next(iter(policies))

    records = [record for lifecycle in lifecycles for record in lifecycle["records"]]
    by_disposition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_disposition[record["disposition"]].append(record)

    completed = [record for record in records if record["disposition"] in COMPLETED_DISPOSITIONS]
    trained = by_disposition["trained"]
    completed_summary = _record_summary(completed)
    trained_summary = _record_summary(trained)
    sampled_mean = completed_summary["sample_length"].get("mean")
    trained_mean = trained_summary["sample_length"].get("mean")
    group_max_mean = completed_summary["group_max_length"].get("mean")
    tailness = group_max_mean / sampled_mean if sampled_mean else None
    completed_reward_mean = completed_summary["reward"]["sample_reward"].get("mean")
    trained_reward_mean = trained_summary["reward"]["sample_reward"].get("mean")

    groups_per_batch, samples_per_group = _infer_batch_shape(trained)
    batch_samples = groups_per_batch * samples_per_group if groups_per_batch and samples_per_group else None
    capacity_groups = capacities.pop()
    queue_factor = capacity_groups / groups_per_batch if policy == "queue-drop" and groups_per_batch else None

    summary: dict[str, Any] = {
        "schema_version": 1,
        "policy": policy,
        "dump_count": len(lifecycles),
        "terminal_records": len(records),
        "capacity_groups": capacity_groups,
        "inferred_shape": {
            "groups_per_batch": groups_per_batch,
            "samples_per_group": samples_per_group,
            "batch_samples": batch_samples,
            "queue_factor": queue_factor,
        },
        "dispositions": {
            disposition: _record_summary(disposition_records)
            for disposition, disposition_records in sorted(by_disposition.items())
        },
        "terminal_completed": completed_summary,
        "trained": trained_summary,
        "length_selection": {
            "trained_minus_terminal_completed_mean": (
                trained_mean - sampled_mean if trained_mean is not None and sampled_mean is not None else None
            ),
            "trained_to_terminal_completed_mean_ratio": (
                trained_mean / sampled_mean if trained_mean is not None and sampled_mean else None
            ),
            "group_tailness_multiplier": tailness,
        },
        "reward_selection": {
            "trained_minus_terminal_completed_sample_mean": (
                trained_reward_mean - completed_reward_mean
                if trained_reward_mean is not None and completed_reward_mean is not None
                else None
            ),
            "trained_to_terminal_completed_sample_mean_ratio": (
                trained_reward_mean / completed_reward_mean
                if trained_reward_mean is not None and completed_reward_mean
                else None
            ),
        },
        "trained_selection_staleness": {
            "pre_queue_ready_minus_first_prefill": _version_gap(trained, "ready_version", "first_prefill_version"),
            "in_queue_selection_minus_ready": _version_gap(trained, "decision_version", "ready_version"),
            "total_selection_minus_first_prefill": _version_gap(trained, "decision_version", "first_prefill_version"),
        },
        "notes": [
            "terminal_completed excludes aborted attempts but includes trained, age/filter drops, recycles, and evictions",
            "finite runs censor groups still in flight or queued at the final dump; discard warmup/final windows for inference",
            "trained means queue-admitted; exclude a debug-exit prefetch dump whose rollout_id was never consumed",
            "decision_version is queue-selection time; join queue/consumption/selection_to_train_gap for legacy prefetch slack",
            "reward summaries are absent for lifecycle dumps written before reward_values was added",
        ],
    }

    if rho is not None or concurrency_samples is not None:
        if rho is None or concurrency_samples is None:
            raise ValueError("--rho and --concurrency-samples must be provided together")
        if summary["policy"] != "queue-drop":
            raise ValueError("the closed-form prediction implemented here applies to queue-drop")
        if batch_samples is None or queue_factor is None or tailness is None:
            raise ValueError("could not infer batch shape or tailness from the supplied dumps")
        summary["queue_drop_formula"] = {
            "inputs": {
                "rho": rho,
                "concurrency_samples": concurrency_samples,
                "batch_samples": batch_samples,
                "queue_factor": queue_factor,
                "group_tailness_multiplier": tailness,
            },
            "predicted_staleness": queue_drop_prediction(
                rho=rho,
                concurrency_samples=concurrency_samples,
                batch_samples=batch_samples,
                queue_factor=queue_factor,
                tailness=tailness,
            ),
            "comparison_target": "trained_selection_staleness.total_selection_minus_first_prefill.mean",
        }

    return summary


def _expand_paths(paths: list[Path]) -> list[Path]:
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(path.glob("*.pt"))
        else:
            expanded.append(path)
    resolved = sorted({path.resolve() for path in expanded})
    if not resolved:
        raise ValueError("no rollout dump files found")
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise ValueError(f"rollout dump files do not exist: {missing}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", nargs="+", type=Path, help="rollout .pt files or directories containing them")
    parser.add_argument("--rho", type=float, help="measured rollout/train token-throughput ratio")
    parser.add_argument("--concurrency-samples", type=int, help="actual simultaneous rollout sample slots C")
    args = parser.parse_args()

    paths = _expand_paths(args.dumps)
    lifecycles = [load_lifecycle(path) for path in paths]
    result = summarize(lifecycles, rho=args.rho, concurrency_samples=args.concurrency_samples)
    result["files"] = [str(path) for path in paths]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
