#!/usr/bin/env python3
"""Join AIME checkpoints with training history and analyze staleness effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCORE_COLUMNS = {
    "aime24": "aime24_percent",
    "aime25": "aime25_percent",
    "aime26": "aime26_percent",
    "macro": "aime_macro_mean_percent",
}
STALENESS_FEATURES = (
    "staleness/total/mean",
    "staleness/total/variance",
    "staleness/total/std",
    "staleness/total/p90",
    "staleness/pre_queue/mean",
    "staleness/in_queue/mean",
)
MEDIATOR_METRICS = (
    "train/policy_rollout_kl",
    "train/policy_rollout_abs_diff",
    "train/policy_rollout_token_ess",
    "train/policy_rollout_sequence_ess",
    "train/tis_abs",
    "train/tis_clipfrac",
    "train/grad_norm_pre_clip",
    "train/grad_clip_coefficient",
    "train/update_norm",
    "train/relative_update_norm",
    "train/advantage_std",
    "rollout/raw_reward",
    "rollout/truncated_ratio",
    "throughput/useful_tokens_per_second",
    "throughput/cohort_useful_efficiency",
    "staleness/bound_exceeded_sample_frac",
    "rollout/fully_async/wasted_token_frac",
    "perf/step_time",
)
INTERVAL_METRICS = (*STALENESS_FEATURES, *MEDIATOR_METRICS)


@dataclass(frozen=True)
class Correlation:
    """One fixed-effect correlation estimate."""

    predictor: str
    outcome: str
    observations: int
    correlation: float
    slope: float
    ci_low: float | None
    ci_high: float | None


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def _correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return math.nan
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    centered_x = [value - x_mean for value in xs]
    centered_y = [value - y_mean for value in ys]
    denominator = math.sqrt(sum(value * value for value in centered_x) * sum(value * value for value in centered_y))
    if denominator == 0.0:
        return math.nan
    return sum(x * centered_y[index] for index, x in enumerate(centered_x)) / denominator


def _slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return math.nan
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0.0:
        return math.nan
    return sum((x - x_mean) * (ys[index] - y_mean) for index, x in enumerate(xs)) / denominator


def _ratio_from_arm(arm: str) -> str:
    return "colocated" if arm == "s0-colocated" else arm.split("-", 1)[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _history_index(rows: Iterable[dict[str, str]]) -> dict[str, dict[int, dict[str, str]]]:
    indexed: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        indexed[row["arm"]][int(row["training_step"])] = row
    return dict(indexed)


def _window_metric(
    history: dict[int, dict[str, str]],
    *,
    start_step: int,
    end_step: int,
    metric: str,
) -> float | None:
    values = [_optional_float(history.get(step, {}).get(metric)) for step in range(start_step + 1, end_step + 1)]
    if any(value is None for value in values):
        return None
    return _mean([value for value in values if value is not None])


def _score_value(row: dict[str, str], label: str) -> float | None:
    return _optional_float(row[SCORE_COLUMNS[label]])


def _checkpoint_series(
    aggregates: list[dict[str, str]],
    histories: dict[str, dict[int, dict[str, str]]],
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for aggregate in aggregates:
        if int(aggregate["completed_tasks"]) == 0:
            continue
        arm = aggregate["arm"]
        step = int(aggregate["training_step"])
        history = histories.get(arm, {}).get(step, {})
        row: dict[str, Any] = dict(aggregate)
        active_seconds = _optional_float(history.get("active_wallclock_seconds"))
        calendar_seconds = _optional_float(history.get("calendar_elapsed_seconds"))
        row["active_wallclock_hours"] = active_seconds / 3600.0 if active_seconds is not None else ""
        row["calendar_elapsed_hours"] = calendar_seconds / 3600.0 if calendar_seconds is not None else ""
        row["active_wallclock_coverage"] = history.get("active_wallclock_coverage", "")
        window_start = max(0, step - 10)
        for metric in INTERVAL_METRICS:
            value = _window_metric(
                histories.get(arm, {}),
                start_step=window_start,
                end_step=step,
                metric=metric,
            )
            row[f"window/{metric}"] = value if value is not None else ""
        series.append(row)
    return series


def _complete_scores_by_arm(
    aggregates: Iterable[dict[str, str]],
) -> dict[str, dict[int, dict[str, str]]]:
    by_arm: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in aggregates:
        if int(row["completed_tasks"]) == len(SCORE_COLUMNS) - 1:
            by_arm[row["arm"]][int(row["training_step"])] = row
    return dict(by_arm)


def _interval_records(
    aggregates: list[dict[str, str]],
    histories: dict[str, dict[int, dict[str, str]]],
) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for arm, score_rows in _complete_scores_by_arm(aggregates).items():
        for end_step in sorted(score_rows):
            start_step = end_step - 10
            if start_step not in score_rows:
                continue
            start = score_rows[start_step]
            end = score_rows[end_step]
            record: dict[str, Any] = {
                "arm": arm,
                "ratio": _ratio_from_arm(arm),
                "max_weight_staleness": end["max_weight_staleness"],
                "start_step": start_step,
                "end_step": end_step,
            }
            for label in SCORE_COLUMNS:
                start_score = _score_value(start, label)
                end_score = _score_value(end, label)
                record[f"delta_{label}"] = (
                    end_score - start_score if start_score is not None and end_score is not None else ""
                )
            start_hours = _optional_float(histories.get(arm, {}).get(start_step, {}).get("active_wallclock_seconds"))
            end_hours = _optional_float(histories.get(arm, {}).get(end_step, {}).get("active_wallclock_seconds"))
            record["active_interval_hours"] = (
                (end_hours - start_hours) / 3600.0 if start_hours is not None and end_hours is not None else ""
            )
            interval_hours = _optional_float(record["active_interval_hours"])
            record["updates_per_active_hour"] = (
                (end_step - start_step) / interval_hours if interval_hours is not None and interval_hours > 0.0 else ""
            )
            for label in SCORE_COLUMNS:
                delta = _optional_float(record[f"delta_{label}"])
                record[f"{label}_points_per_active_hour"] = (
                    delta / interval_hours
                    if delta is not None and interval_hours is not None and interval_hours > 0.0
                    else ""
                )
            for metric in INTERVAL_METRICS:
                value = _window_metric(
                    histories.get(arm, {}),
                    start_step=start_step,
                    end_step=end_step,
                    metric=metric,
                )
                record[metric] = value if value is not None else ""
            intervals.append(record)
    return intervals


def _wallclock_decomposition(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for interval in intervals:
        hours = _optional_float(interval.get("active_interval_hours"))
        delta_macro = _optional_float(interval.get("delta_macro"))
        if hours is not None and hours > 0.0 and delta_macro is not None:
            by_arm[interval["arm"]].append(interval)

    rows: list[dict[str, Any]] = []
    for arm, arm_intervals in sorted(by_arm.items()):
        active_hours = sum(float(interval["active_interval_hours"]) for interval in arm_intervals)
        covered_updates = sum(int(interval["end_step"]) - int(interval["start_step"]) for interval in arm_intervals)
        first = arm_intervals[0]
        row: dict[str, Any] = {
            "arm": arm,
            "ratio": first["ratio"],
            "max_weight_staleness": first["max_weight_staleness"],
            "interval_count": len(arm_intervals),
            "covered_updates": covered_updates,
            "active_interval_hours": active_hours,
            "updates_per_active_hour": covered_updates / active_hours,
        }
        for label in SCORE_COLUMNS:
            score_change = sum(float(interval[f"delta_{label}"]) for interval in arm_intervals)
            row[f"{label}_score_change"] = score_change
            row[f"{label}_points_per_update"] = score_change / covered_updates
            row[f"{label}_points_per_active_hour"] = score_change / active_hours
        rows.append(row)
    return rows


def _center_records(
    records: list[dict[str, Any]],
    *,
    predictor: str,
    outcome: str,
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        x = _optional_float(record.get(predictor))
        y = _optional_float(record.get(outcome))
        if x is None or y is None or record["arm"] == "s0-colocated":
            continue
        groups[tuple(record[key] for key in group_keys)].append(record)
    centered: list[dict[str, Any]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        x_mean = _mean([float(record[predictor]) for record in group])
        y_mean = _mean([float(record[outcome]) for record in group])
        centered.extend(
            {
                "arm": record["arm"],
                "x": float(record[predictor]) - x_mean,
                "y": float(record[outcome]) - y_mean,
            }
            for record in group
        )
    return centered


def _bootstrap_ci(
    records: list[dict[str, Any]],
    *,
    predictor: str,
    outcome: str,
    group_keys: tuple[str, ...],
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if samples == 0:
        return None, None
    arms = sorted({record["arm"] for record in records if record["arm"] != "s0-colocated"})
    if len(arms) < 2:
        return None, None
    records_by_arm = {arm: [record for record in records if record["arm"] == arm] for arm in arms}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[dict[str, Any]] = []
        for copy_index, arm in enumerate(rng.choices(arms, k=len(arms))):
            sampled.extend({**record, "arm": f"{arm}#{copy_index}"} for record in records_by_arm[arm])
        centered = _center_records(
            sampled,
            predictor=predictor,
            outcome=outcome,
            group_keys=group_keys,
        )
        estimate = _correlation(
            [float(record["x"]) for record in centered],
            [float(record["y"]) for record in centered],
        )
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None, None
    estimates.sort()
    return (
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    )


def _fixed_effect_correlation(
    records: list[dict[str, Any]],
    *,
    predictor: str,
    outcome: str,
    group_keys: tuple[str, ...],
    bootstrap_samples: int,
    seed: int,
) -> Correlation:
    centered = _center_records(
        records,
        predictor=predictor,
        outcome=outcome,
        group_keys=group_keys,
    )
    xs = [float(record["x"]) for record in centered]
    ys = [float(record["y"]) for record in centered]
    ci_low, ci_high = _bootstrap_ci(
        records,
        predictor=predictor,
        outcome=outcome,
        group_keys=group_keys,
        samples=bootstrap_samples,
        seed=seed,
    )
    return Correlation(
        predictor=predictor,
        outcome=outcome,
        observations=len(centered),
        correlation=_correlation(xs, ys),
        slope=_slope(xs, ys),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _downstream_correlations(intervals: list[dict[str, Any]], bootstrap_samples: int) -> list[Correlation]:
    correlations = []
    for metric_index, predictor in enumerate(INTERVAL_METRICS):
        for outcome_index, label in enumerate(SCORE_COLUMNS):
            correlations.append(
                _fixed_effect_correlation(
                    intervals,
                    predictor=predictor,
                    outcome=f"delta_{label}",
                    group_keys=("end_step", "ratio"),
                    bootstrap_samples=bootstrap_samples,
                    seed=20260823 + metric_index * 10 + outcome_index,
                )
            )
    return correlations


def _per_step_records(histories: dict[str, dict[int, dict[str, str]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm, history in histories.items():
        for step, source in history.items():
            record: dict[str, Any] = {"arm": arm, "ratio": _ratio_from_arm(arm), "training_step": step}
            record.update(source)
            records.append(record)
    return records


def _staleness_correlations(
    histories: dict[str, dict[int, dict[str, str]]],
) -> list[Correlation]:
    records = _per_step_records(histories)
    correlations = []
    for predictor in ("staleness/total/mean", "staleness/total/variance"):
        for outcome in MEDIATOR_METRICS:
            correlations.append(
                _fixed_effect_correlation(
                    records,
                    predictor=predictor,
                    outcome=outcome,
                    group_keys=("training_step", "ratio"),
                    bootstrap_samples=0,
                    seed=0,
                )
            )
    return correlations


def _correlation_rows(correlations: Iterable[Correlation]) -> list[dict[str, Any]]:
    return [
        {
            "predictor": record.predictor,
            "outcome": record.outcome,
            "observations": record.observations,
            "correlation": record.correlation if math.isfinite(record.correlation) else "",
            "slope": record.slope if math.isfinite(record.slope) else "",
            "ci_low": record.ci_low if record.ci_low is not None else "",
            "ci_high": record.ci_high if record.ci_high is not None else "",
        }
        for record in correlations
    ]


def _selected_metrics(correlations: Iterable[Correlation]) -> list[Correlation]:
    selected = [
        record
        for record in correlations
        if record.outcome == "delta_macro"
        and record.ci_low is not None
        and record.ci_high is not None
        and (record.ci_low > 0.0 or record.ci_high < 0.0)
        and abs(record.correlation) >= 0.2
    ]
    return sorted(selected, key=lambda record: abs(record.correlation), reverse=True)[:3]


def _selected_arm_groups(intervals: list[dict[str, Any]], selected: Iterable[Correlation]) -> list[dict[str, Any]]:
    groups = []
    for correlation in selected:
        values_by_arm: dict[str, list[float]] = defaultdict(list)
        for record in intervals:
            value = _optional_float(record.get(correlation.predictor))
            if value is not None and record["arm"] != "s0-colocated":
                values_by_arm[record["arm"]].append(value)
        ranked = sorted(
            ((arm, statistics.median(values)) for arm, values in values_by_arm.items()),
            key=lambda item: item[1],
        )
        groups.append(
            {
                "metric": correlation.predictor,
                "correlation": correlation.correlation,
                "low_arms": [arm for arm, _ in ranked[:2]],
                "high_arms": [arm for arm, _ in ranked[-2:]],
            }
        )
    return groups


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _summary_markdown(
    *,
    series: list[dict[str, Any]],
    downstream: list[Correlation],
    selected_groups: list[dict[str, Any]],
) -> str:
    complete = [row for row in series if _optional_float(row.get("aime_macro_mean_percent")) is not None]
    macro = [record for record in downstream if record.outcome == "delta_macro"]
    macro.sort(key=lambda record: abs(record.correlation) if math.isfinite(record.correlation) else -1, reverse=True)
    lines = [
        "# Staleness and downstream analysis",
        "",
        f"Complete checkpoint suites represented: **{len(complete)}**",
        "",
        "Correlations use ten-update AIME score changes. Predictor values are averaged over the same interval,",
        "then centered within the same ending step and trainer:rollout ratio. Active wall-clock is the cumulative",
        "selected-lineage `perf/step_time`; scheduler requeue gaps are excluded.",
        "",
        "## Largest absolute macro correlations",
        "",
        "| Predictor | n | r | 95% arm-bootstrap interval |",
        "|---|---:|---:|---:|",
    ]
    for record in macro[:10]:
        interval = (
            f"[{record.ci_low:.3f}, {record.ci_high:.3f}]"
            if record.ci_low is not None and record.ci_high is not None
            else "-"
        )
        lines.append(f"| `{record.predictor}` | {record.observations} | {record.correlation:.3f} | {interval} |")
    lines.extend(["", "## Robustly selected downstream relationships", ""])
    if selected_groups:
        for group in selected_groups:
            lines.append(
                f"- `{group['metric']}`: r={group['correlation']:.3f}; low={group['low_arms']}; "
                f"high={group['high_arms']}"
            )
    else:
        lines.append(
            "No metric met both |r| >= 0.2 and an arm-cluster bootstrap interval excluding zero. "
            "No selective downstream trajectory should be interpreted as established."
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--training-history-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be nonnegative")
    aggregates = _read_csv(args.aggregate_csv.resolve())
    histories = _history_index(_read_csv(args.training_history_csv.resolve()))
    series = _checkpoint_series(aggregates, histories)
    intervals = _interval_records(aggregates, histories)
    decomposition = _wallclock_decomposition(intervals)
    downstream = _downstream_correlations(intervals, args.bootstrap_samples)
    staleness = _staleness_correlations(histories)
    selected_groups = _selected_arm_groups(intervals, _selected_metrics(downstream))
    output_dir = args.output_dir.resolve()
    series_fields = list(series[0]) if series else []
    interval_fields = list(intervals[0]) if intervals else []
    if not series_fields or not interval_fields or not decomposition:
        raise ValueError("analysis requires completed checkpoint series and adjacent intervals")
    _atomic_write_csv(output_dir / "checkpoint-series.csv", series, series_fields)
    _atomic_write_csv(output_dir / "score-intervals.csv", intervals, interval_fields)
    _atomic_write_csv(
        output_dir / "wallclock-decomposition.csv",
        decomposition,
        list(decomposition[0]),
    )
    correlation_fields = ["predictor", "outcome", "observations", "correlation", "slope", "ci_low", "ci_high"]
    _atomic_write_csv(
        output_dir / "downstream-correlations.csv",
        _correlation_rows(downstream),
        correlation_fields,
    )
    _atomic_write_csv(
        output_dir / "staleness-metric-correlations.csv",
        _correlation_rows(staleness),
        correlation_fields,
    )
    _atomic_write_json(output_dir / "selected-relationships.json", selected_groups)
    _atomic_write_text(
        output_dir / "staleness-summary.md",
        _summary_markdown(series=series, downstream=downstream, selected_groups=selected_groups),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
