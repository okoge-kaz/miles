#!/usr/bin/env python3
"""Plot TIS diagnostics and summarize their relation to training collapse."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from export_reward_conditioned_response_lengths import rollout_values
from plot_cross_cohort import color, polyline, svg_canvas


OUTPUT_ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = OUTPUT_ROOT / "figures"
TRAINING_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/"
    "async-rl/checkpoints/training/math/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/"
    "Qwen3-4B-Base-LR2e-5-Step4000/grpo-clip0.2-0.28-tis2.0/async/off-policy"
)
OBJECTIVE_PATH = OUTPUT_ROOT / "truncated-zero-reward-objective.csv"
SUMMARY_PATH = OUTPUT_ROOT / "tis-diagnostics-summary.csv"
AGE_SUMMARY_PATH = OUTPUT_ROOT / "tis-staleness-bin-summary.csv"
NO_TREATMENT_REWARD_PATH = OUTPUT_ROOT / "no-treatment-truncated-reward-summary.csv"
REPORT_PATH = OUTPUT_ROOT / "tis-analysis.md"
MAX_STEP = 300
ROLLING_WINDOW = 5
REWARD_COLLAPSE_THRESHOLD = 0.30
TIS_ABS_WARNING_THRESHOLD = 0.05
TIS_CLIP_WARNING_THRESHOLD = 0.001
SHORT_RESPONSE_THRESHOLD = 2_048.0
OVERLONG_THRESHOLD = 0.50
ARM_PATTERN = re.compile(r"^(?P<arm>s(?P<staleness>\d+)-t(?P<train>\d+)r(?P<rollout>\d+))")
AGE_METRIC_PATTERN = re.compile(r"^sample_staleness/s_(?P<age>\d+)/(?P<metric>.+)$")


@dataclass(frozen=True)
class Source:
    namespace: str
    treatment: str
    treatment_label: str


@dataclass(frozen=True)
class Arm:
    namespace: str
    treatment: str
    treatment_label: str
    arm: str
    staleness: int
    dashboard_path: Path


@dataclass(frozen=True)
class MetricSpec:
    name: str
    title: str
    ticks: tuple[float, ...]
    tick_labels: tuple[str, ...]
    lower: float
    upper: float
    logarithmic: bool = False


@dataclass(frozen=True)
class AgeCase:
    label: str
    treatment: str
    staleness: int
    first_step: int
    last_step: int


SOURCES = (
    Source(
        "zero-reward-s12-control-t1r7-cb1c041f",
        "zero-reward",
        "zero reward on truncated",
    ),
    Source(
        "sr-20260826-141753-p1497131",
        "zero-reward",
        "zero reward on truncated",
    ),
    Source(
        "hiso-no-trunc-treatment-s8-16-r12-tbq6000-20260831-v1",
        "none",
        "no truncation treatment",
    ),
    Source(
        "hiso-zero-loss-trunc-s8-16-20-r12-tbq6000-20260831-v1",
        "zero-loss",
        "zero loss on truncated",
    ),
    Source(
        "staleness-aware-safe4-t1r7-cb1c041f",
        "staleness-aware",
        "staleness-aware truncated loss",
    ),
)
TREATMENT_ORDER = ("zero-reward", "none", "zero-loss", "staleness-aware")
TREATMENT_LABELS = {source.treatment: source.treatment_label for source in SOURCES}
COLLAPSE_KEYS = (
    ("none", 16),
    ("zero-reward", 16),
    ("zero-reward", 20),
    ("zero-reward", 24),
    ("zero-reward", 28),
)
AGE_CASES = (
    AgeCase("zero-reward healthy", "zero-reward", 20, 91, 110),
    AgeCase("zero-reward transition", "zero-reward", 20, 128, 147),
    AgeCase("zero-loss late", "zero-loss", 20, 281, 300),
    AgeCase("staleness-aware late", "staleness-aware", 20, 241, 260),
)
NO_TREATMENT_REWARD_WINDOWS = (
    ("pre-collapse", 1, 149),
    ("healthy", 80, 120),
    ("transition", 125, 150),
    ("overlong", 180, 210),
    ("all-observed", 1, 249),
)
EXACT_METRICS = {
    "rollout/raw_reward",
    "rollout/truncated_ratio",
    "rollout/response_len/mean",
    "staleness/total/mean",
    "train/tis",
    "train/tis_abs",
    "train/tis_clipfrac",
    "train/policy_rollout_abs_diff",
    "train/policy_rollout_kl",
    "train/policy_rollout_token_ess",
    "train/policy_rollout_sequence_ess",
    "train/grad_norm_pre_clip",
    "train/final_loss_tokens",
}
STALE_AWARE_PREFIX = "train/staleness_aware_loss/"
TRUNCATED_TOKEN_METRIC = (
    f"{STALE_AWARE_PREFIX}truncated_zero_reward_loss_token_fraction"
)
TRUNCATED_SCALE_METRIC = (
    f"{STALE_AWARE_PREFIX}truncated_zero_reward_mean_gradient_scale"
)
TRUNCATED_PRE_OBJECTIVE_METRIC = (
    f"{STALE_AWARE_PREFIX}"
    "truncated_zero_reward_post_tis_pre_scaling_abs_pg_objective_fraction"
)
TRUNCATED_POST_OBJECTIVE_METRIC = (
    f"{STALE_AWARE_PREFIX}"
    "truncated_zero_reward_post_tis_post_scaling_abs_pg_objective_fraction"
)
PRE_OBJECTIVE_PER_TOKEN_METRIC = (
    f"{STALE_AWARE_PREFIX}post_tis_pre_scaling_abs_pg_objective_per_loss_token"
)
POST_OBJECTIVE_PER_TOKEN_METRIC = (
    f"{STALE_AWARE_PREFIX}post_tis_post_scaling_abs_pg_objective_per_loss_token"
)
MASKED_RESPONSE_TOKEN_METRIC = "derived/initial_mask_response_token_fraction"
OBJECTIVE_RETENTION_METRIC = "derived/post_tis_objective_retention"
BASELINE_TRUNCATED_OBJECTIVE_METRIC = (
    "diagnostic/truncated_zero_pre_tis_abs_pg_objective_fraction"
)
BASELINE_TRUNCATED_TOKEN_METRIC = "diagnostic/truncated_zero_loss_token_fraction"


def discover_arms() -> list[Arm]:
    arms: list[Arm] = []
    seen: set[tuple[str, str]] = set()
    for source in SOURCES:
        paths = TRAINING_ROOT.glob(
            f"max-weight-staleness-*-from-prefill/*-{source.namespace}-*/"
            "dump/dashboard/metrics.jsonl"
        )
        for path in paths:
            match = ARM_PATTERN.match(path.parents[2].name)
            if match is None:
                continue
            if (int(match["train"]), int(match["rollout"])) != (1, 7):
                continue
            key = (source.namespace, match["arm"])
            if key in seen:
                raise ValueError(f"duplicate dashboard for {key}: {path}")
            seen.add(key)
            arms.append(
                Arm(
                    namespace=source.namespace,
                    treatment=source.treatment,
                    treatment_label=source.treatment_label,
                    arm=match["arm"],
                    staleness=int(match["staleness"]),
                    dashboard_path=path,
                )
            )
    return sorted(arms, key=lambda arm: (TREATMENT_ORDER.index(arm.treatment), arm.staleness))


def keep_metric(name: str) -> bool:
    return (
        name in EXACT_METRICS
        or name.startswith(STALE_AWARE_PREFIX)
        or name.startswith("sample_staleness/s_")
    )


def read_dashboard(path: Path) -> dict[str, dict[int, float]]:
    histories: dict[str, dict[int, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            step = int(record["step"]) + 1
            if not 1 <= step <= MAX_STEP:
                continue
            for metric, raw_value in record.get("metrics", {}).items():
                if not keep_metric(metric) or not isinstance(raw_value, (int, float)):
                    continue
                value = float(raw_value)
                if math.isfinite(value):
                    histories.setdefault(metric, {})[step] = value
    derive_metrics(histories)
    return histories


def age_metrics_at_step(
    histories: dict[str, dict[int, float]],
    step: int,
) -> dict[int, dict[str, float]]:
    by_age: dict[int, dict[str, float]] = {}
    for metric, values in histories.items():
        match = AGE_METRIC_PATTERN.fullmatch(metric)
        if match is None or step not in values:
            continue
        by_age.setdefault(int(match["age"]), {})[match["metric"]] = values[step]
    return by_age


def derive_metrics(histories: dict[str, dict[int, float]]) -> None:
    observed_steps = sorted({step for values in histories.values() for step in values})
    masked_fractions: dict[int, float] = {}
    for step in observed_steps:
        age_metrics = age_metrics_at_step(histories, step)
        if not age_metrics:
            continue
        masked_fractions[step] = sum(
            metrics.get("consumed_response_token_mass", 0.0)
            * metrics.get("initial_mask_fraction", 0.0)
            for metrics in age_metrics.values()
        )
    if masked_fractions:
        histories[MASKED_RESPONSE_TOKEN_METRIC] = masked_fractions

    pre_values = histories.get(PRE_OBJECTIVE_PER_TOKEN_METRIC, {})
    post_values = histories.get(POST_OBJECTIVE_PER_TOKEN_METRIC, {})
    retention = {
        step: post_values[step] / pre_value
        for step, pre_value in pre_values.items()
        if step in post_values and pre_value > 0.0
    }
    if retention:
        histories[OBJECTIVE_RETENTION_METRIC] = retention


def merge_objective_diagnostics(
    arms: list[Arm],
    histories: dict[tuple[str, int], dict[str, dict[int, float]]],
) -> None:
    by_identity = {(arm.namespace, arm.arm): (arm.treatment, arm.staleness) for arm in arms}
    with OBJECTIVE_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = by_identity.get((row["namespace"], row["arm"]))
            if identity is None:
                continue
            step = int(row["training_step"])
            if not 1 <= step <= MAX_STEP:
                continue
            target = histories[identity]
            objective_value = row["truncated_zero_pre_tis_abs_pg_objective_fraction"]
            if objective_value:
                target.setdefault(BASELINE_TRUNCATED_OBJECTIVE_METRIC, {})[step] = float(
                    objective_value
                )
            target.setdefault(BASELINE_TRUNCATED_TOKEN_METRIC, {})[step] = float(
                row["truncated_zero_loss_token_fraction"]
            )


def rolling(values: dict[int, float], window: int = ROLLING_WINDOW) -> dict[int, float]:
    ordered = sorted(values.items())
    return {
        step: statistics.fmean(
            value for _, value in ordered[max(0, index - window + 1) : index + 1]
        )
        for index, (step, _) in enumerate(ordered)
    }


def first_threshold(
    values: dict[int, float],
    threshold: float,
    *,
    above: bool,
    minimum_step: int = 20,
) -> int | None:
    for step, value in sorted(rolling(values).items()):
        if step < minimum_step:
            continue
        if (above and value >= threshold) or (not above and value <= threshold):
            return step
    return None


def collapse_onset(values: dict[int, float]) -> int | None:
    smoothed = rolling(values)
    if not smoothed:
        return None
    peak_step = max(smoothed, key=smoothed.get)
    for step, value in sorted(smoothed.items()):
        if step <= peak_step or value >= REWARD_COLLAPSE_THRESHOLD:
            continue
        tail = [tail_value for tail_step, tail_value in smoothed.items() if tail_step >= step]
        if len(tail) >= ROLLING_WINDOW and max(tail) < REWARD_COLLAPSE_THRESHOLD:
            return step
    return None


def values_in_window(values: dict[int, float], first_step: int, last_step: int) -> list[float]:
    return [value for step, value in sorted(values.items()) if first_step <= step <= last_step]


def window_mean(
    values: dict[int, float],
    first_step: int,
    last_step: int,
) -> float | None:
    selected = values_in_window(values, first_step, last_step)
    return statistics.fmean(selected) if selected else None


def late_mean(values: dict[int, float], window: int = 20) -> float | None:
    ordered = [value for _, value in sorted(values.items())]
    return statistics.fmean(ordered[-window:]) if ordered else None


def last_step(histories: dict[str, dict[int, float]]) -> int:
    return max(step for values in histories.values() for step in values)


def optional_number(value: float | int | None) -> float | int | str:
    return "" if value is None else value


def summary_rows(
    arms: list[Arm],
    histories: dict[tuple[str, int], dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    late_metrics = {
        "raw_reward": "rollout/raw_reward",
        "truncated_ratio": "rollout/truncated_ratio",
        "response_len": "rollout/response_len/mean",
        "realized_staleness": "staleness/total/mean",
        "tis": "train/tis",
        "tis_abs": "train/tis_abs",
        "tis_clipfrac": "train/tis_clipfrac",
        "policy_rollout_kl": "train/policy_rollout_kl",
        "policy_rollout_token_ess": "train/policy_rollout_token_ess",
        "policy_rollout_sequence_ess": "train/policy_rollout_sequence_ess",
        "grad_norm_pre_clip": "train/grad_norm_pre_clip",
        "masked_response_token_fraction": MASKED_RESPONSE_TOKEN_METRIC,
        "aware_truncated_loss_token_fraction": TRUNCATED_TOKEN_METRIC,
        "aware_truncated_mean_gradient_scale": TRUNCATED_SCALE_METRIC,
        "aware_truncated_pre_objective_fraction": TRUNCATED_PRE_OBJECTIVE_METRIC,
        "aware_truncated_post_objective_fraction": TRUNCATED_POST_OBJECTIVE_METRIC,
        "aware_total_objective_retention": OBJECTIVE_RETENTION_METRIC,
        "baseline_truncated_pre_tis_objective_fraction": BASELINE_TRUNCATED_OBJECTIVE_METRIC,
        "baseline_truncated_loss_token_fraction": BASELINE_TRUNCATED_TOKEN_METRIC,
    }
    for arm in arms:
        key = (arm.treatment, arm.staleness)
        metrics = histories[key]
        reward_values = metrics.get("rollout/raw_reward", {})
        collapse_step = collapse_onset(reward_values)
        tis_abs_step = first_threshold(
            metrics.get("train/tis_abs", {}),
            TIS_ABS_WARNING_THRESHOLD,
            above=True,
        )
        tis_clip_step = first_threshold(
            metrics.get("train/tis_clipfrac", {}),
            TIS_CLIP_WARNING_THRESHOLD,
            above=True,
        )
        short_step = first_threshold(
            metrics.get("rollout/response_len/mean", {}),
            SHORT_RESPONSE_THRESHOLD,
            above=False,
        )
        overlong_step = first_threshold(
            metrics.get("rollout/truncated_ratio", {}),
            OVERLONG_THRESHOLD,
            above=True,
        )
        row: dict[str, Any] = {
            "namespace": arm.namespace,
            "treatment": arm.treatment,
            "arm": arm.arm,
            "max_weight_staleness": arm.staleness,
            "last_observed_step": last_step(metrics),
            "collapse_onset_reward_rolling5_lt_0p30": optional_number(collapse_step),
            "first_tis_abs_rolling5_ge_0p05": optional_number(tis_abs_step),
            "tis_abs_lead_to_collapse_steps": optional_number(
                collapse_step - tis_abs_step
                if collapse_step is not None and tis_abs_step is not None
                else None
            ),
            "first_tis_clipfrac_rolling5_ge_0p001": optional_number(tis_clip_step),
            "first_response_len_rolling5_le_2048": optional_number(short_step),
            "first_truncated_ratio_rolling5_ge_0p50": optional_number(overlong_step),
        }
        for column, metric in late_metrics.items():
            row[f"late20_{column}"] = optional_number(late_mean(metrics.get(metric, {})))
            if collapse_step is None:
                row[f"collapse20_{column}"] = ""
            else:
                row[f"collapse20_{column}"] = optional_number(
                    window_mean(
                        metrics.get(metric, {}),
                        max(1, collapse_step - 19),
                        collapse_step,
                    )
                )
        rows.append(row)
    return rows


def age_summary_rows(
    histories: dict[tuple[str, int], dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suffixes = (
        "consumed_sequence_mass",
        "consumed_response_token_mass",
        "consumed_pre_loss_token_mass",
        "effective_contribution_mass",
        "mean_abs_policy_rollout_log_ratio",
        "importance_clip_fraction",
        "initial_mask_fraction",
        "final_mask_fraction",
        "nonzero_contribution_fraction",
        "mean_abs_pg_contribution_per_pre_loss_token",
        "policy_rollout_ratio_token_ess",
        "policy_rollout_ratio_sequence_ess",
    )
    for case in AGE_CASES:
        metrics = histories[(case.treatment, case.staleness)]
        ages = sorted(
            {
                int(match["age"])
                for metric in metrics
                if (match := AGE_METRIC_PATTERN.fullmatch(metric)) is not None
            }
        )
        for age in ages:
            row: dict[str, Any] = {
                "case": case.label,
                "treatment": case.treatment,
                "max_weight_staleness": case.staleness,
                "first_step": case.first_step,
                "last_step": case.last_step,
                "sample_staleness": age,
            }
            for suffix in suffixes:
                metric = f"sample_staleness/s_{age}/{suffix}"
                row[suffix] = optional_number(
                    window_mean(metrics.get(metric, {}), case.first_step, case.last_step)
                )
            if any(row[suffix] not in ("", 0.0) for suffix in suffixes):
                rows.append(row)
    return rows


def no_treatment_truncated_reward_rows(arms: list[Arm]) -> list[dict[str, Any]]:
    arm = next(
        arm
        for arm in arms
        if arm.treatment == "none" and arm.staleness == 16
    )
    rollout_root = arm.dashboard_path.parents[1] / "rollout_data"
    accumulators = {
        label: {
            "sample_count": 0,
            "reward_sum": 0.0,
            "truncated_count": 0,
            "truncated_reward_sum": 0.0,
            "truncated_positive_count": 0,
        }
        for label, _, _ in NO_TREATMENT_REWARD_WINDOWS
    }
    paths = sorted(
        (path for path in rollout_root.glob("*.pt") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    for path in paths:
        step = int(path.stem) + 1
        matching_windows = [
            label
            for label, first_step, last_step in NO_TREATMENT_REWARD_WINDOWS
            if first_step <= step <= last_step
        ]
        if not matching_windows:
            continue
        lengths, rewards, statuses, _ = rollout_values(path)
        if not (len(lengths) == len(rewards) == len(statuses)):
            raise ValueError(f"unaligned rollout metadata in {path}")
        for reward, status in zip(rewards, statuses, strict=True):
            truncated = status == "truncated"
            for label in matching_windows:
                accumulator = accumulators[label]
                accumulator["sample_count"] += 1
                accumulator["reward_sum"] += reward
                if truncated:
                    accumulator["truncated_count"] += 1
                    accumulator["truncated_reward_sum"] += reward
                    accumulator["truncated_positive_count"] += int(reward > 0.5)
    rows: list[dict[str, Any]] = []
    for label, first_step, last_step in NO_TREATMENT_REWARD_WINDOWS:
        accumulator = accumulators[label]
        sample_count = int(accumulator["sample_count"])
        truncated_count = int(accumulator["truncated_count"])
        rows.append(
            {
                "namespace": arm.namespace,
                "arm": arm.arm,
                "window": label,
                "first_step": first_step,
                "last_step": last_step,
                "sample_count": sample_count,
                "raw_reward_mean": accumulator["reward_sum"] / sample_count,
                "truncated_count": truncated_count,
                "truncated_sample_fraction": truncated_count / sample_count,
                "truncated_raw_reward_mean": (
                    accumulator["truncated_reward_sum"] / truncated_count
                    if truncated_count
                    else ""
                ),
                "truncated_positive_count": accumulator["truncated_positive_count"],
                "truncated_positive_fraction": (
                    accumulator["truncated_positive_count"] / truncated_count
                    if truncated_count
                    else ""
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def scale_value(spec: MetricSpec, value: float) -> float:
    bounded = min(max(value, spec.lower), spec.upper)
    if spec.logarithmic:
        numerator = math.log10(bounded) - math.log10(spec.lower)
        denominator = math.log10(spec.upper) - math.log10(spec.lower)
        return numerator / denominator
    return (bounded - spec.lower) / (spec.upper - spec.lower)


def metric_specs() -> tuple[MetricSpec, ...]:
    return (
        MetricSpec(
            "rollout/raw_reward",
            "Raw reward",
            (0.0, 0.25, 0.5, 0.75, 1.0),
            ("0", ".25", ".50", ".75", "1"),
            0.0,
            1.0,
        ),
        MetricSpec(
            "rollout/response_len/mean",
            "Mean response length",
            (0.0, 4_096.0, 8_192.0, 12_288.0, 16_384.0),
            ("0", "4K", "8K", "12K", "16K"),
            0.0,
            16_384.0,
        ),
        MetricSpec(
            "rollout/truncated_ratio",
            "Truncated sample fraction",
            (0.0, 0.25, 0.5, 0.75, 1.0),
            ("0", "25%", "50%", "75%", "100%"),
            0.0,
            1.0,
        ),
        MetricSpec(
            "staleness/total/mean",
            "Realized training staleness",
            (0.0, 7.0, 14.0, 21.0, 28.0),
            ("0", "7", "14", "21", "28"),
            0.0,
            28.0,
        ),
        MetricSpec(
            "train/tis_abs",
            "Mean |TIS ratio - 1|",
            (0.001, 0.01, 0.1, 1.0, 10.0),
            (".001", ".01", ".1", "1", "10"),
            0.001,
            10.0,
            True,
        ),
        MetricSpec(
            "train/tis_clipfrac",
            "TIS upper-clip fraction",
            (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            (".0001%", ".001%", ".01%", ".1%", "1%", "10%"),
            1e-6,
            0.1,
            True,
        ),
        MetricSpec(
            "train/policy_rollout_kl",
            "Sampled-token policy/rollout KL",
            (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
            ("1e-5", "1e-4", ".001", ".01", ".1", "1", "10"),
            1e-5,
            10.0,
            True,
        ),
        MetricSpec(
            "train/policy_rollout_token_ess",
            "Token importance-weight ESS",
            (0.0, 0.25, 0.5, 0.75, 1.0),
            ("0", ".25", ".50", ".75", "1"),
            0.0,
            1.0,
        ),
    )


def append_panel_axes(
    elements: list[str],
    *,
    spec: MetricSpec,
    panel_x: float,
    panel_y: float,
    panel_width: float,
    panel_height: float,
    show_y_labels: bool,
    show_x_labels: bool,
    title: str,
) -> tuple[Callable[[float], float], Callable[[float], float], str]:
    plot_x = panel_x + 64.0
    plot_y = panel_y + 25.0
    plot_width = panel_width - 82.0
    plot_height = panel_height - 58.0
    x_map = lambda value: plot_x + plot_width * (value - 1.0) / (MAX_STEP - 1.0)
    y_map = lambda value: plot_y + plot_height * (1.0 - scale_value(spec, value))
    clip_id = f"clip-{len(elements)}"
    elements.append(
        f'<defs><clipPath id="{clip_id}"><rect x="{plot_x:.1f}" y="{plot_y:.1f}" '
        f'width="{plot_width:.1f}" height="{plot_height:.1f}"/></clipPath></defs>'
    )
    for tick, label in zip(spec.ticks, spec.tick_labels, strict=True):
        tick_y = y_map(tick)
        elements.append(
            f'<line class="grid" x1="{plot_x:.1f}" y1="{tick_y:.1f}" '
            f'x2="{plot_x + plot_width:.1f}" y2="{tick_y:.1f}"/>'
        )
        if show_y_labels:
            elements.append(
                f'<text class="tick" x="{plot_x - 6:.1f}" y="{tick_y + 4:.1f}" '
                f'text-anchor="end">{html.escape(label)}</text>'
            )
    for tick in (1, 100, 200, 300):
        tick_x = x_map(float(tick))
        if show_x_labels:
            elements.append(
                f'<text class="tick" x="{tick_x:.1f}" y="{plot_y + plot_height + 16:.1f}" '
                f'text-anchor="middle">{tick}</text>'
            )
    elements.extend(
        [
            f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
            f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
            f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 14:.1f}" text-anchor="middle">{html.escape(title)}</text>',
        ]
    )
    return x_map, y_map, clip_id


def append_series(
    elements: list[str],
    *,
    values: dict[int, float],
    series_color: str,
    x_map: Callable[[float], float],
    y_map: Callable[[float], float],
    clip_id: str,
    raw: bool = True,
) -> None:
    if not values:
        return
    if raw:
        raw_points = [(float(step), value) for step, value in sorted(values.items())]
        elements.append(
            f'<polyline class="raw" clip-path="url(#{clip_id})" stroke="{series_color}" '
            f'points="{polyline(raw_points, x_map, y_map)}"/>'
        )
    smooth_points = [(float(step), value) for step, value in rolling(values).items()]
    elements.append(
        f'<polyline class="smooth" clip-path="url(#{clip_id})" stroke="{series_color}" '
        f'points="{polyline(smooth_points, x_map, y_map)}"/>'
    )


def render_cross_treatment(
    histories: dict[tuple[str, int], dict[str, dict[int, float]]],
) -> str:
    specs = metric_specs()
    panel_width, panel_height = 390.0, 170.0
    column_gap, row_gap = 10.0, 14.0
    left, top = 18.0, 100.0
    width = int(left + len(TREATMENT_ORDER) * (panel_width + column_gap) + 18)
    height = int(top + len(specs) * (panel_height + row_gap) + 20)
    elements = svg_canvas(width, height)
    elements.append(
        "<style>.collapse-onset{stroke-width:1.1;stroke-dasharray:3 3;opacity:.75}"
        ".warning{stroke:#7a0000;stroke-width:1;stroke-dasharray:5 3;opacity:.55}"
        ".note{font-size:10px;fill:#555}</style>"
    )
    elements.extend(
        [
            f'<text class="column-label" x="{left + 64:.1f}" y="20">TIS and policy/rollout mismatch · 16K Math RL · Train:Rollout = 1:7</text>',
            f'<text class="note" x="{left + 64:.1f}" y="42">thin = per-step, bold = trailing-5 · vertical colored dashes = sustained reward &lt; 0.30</text>',
            f'<text class="note" x="{left + 64:.1f}" y="58">TIS = clamp(pi_train / pi_rollout, 0, 2); clip fraction counts only ratios above 2.</text>',
            f'<text class="note" x="{left + 64:.1f}" y="74">Zero-loss TIS metrics exclude initially masked truncated tokens; sequence ESS is omitted because full-response length confounds it.</text>',
        ]
    )
    staleness_levels = sorted({staleness for _, staleness in histories})
    legend_x = width - 88.0 * len(staleness_levels) - 22.0
    for index, staleness in enumerate(staleness_levels):
        item_x = legend_x + 88.0 * index
        elements.extend(
            [
                f'<line x1="{item_x:.1f}" y1="48" x2="{item_x + 24:.1f}" y2="48" stroke="{color(staleness)}" stroke-width="2.5"/>',
                f'<text class="legend" x="{item_x + 29:.1f}" y="52">S={staleness}</text>',
            ]
        )
    for column, treatment in enumerate(TREATMENT_ORDER):
        panel_x = left + column * (panel_width + column_gap)
        elements.append(
            f'<text class="column-label" x="{panel_x + panel_width / 2:.1f}" y="93" '
            f'text-anchor="middle">{html.escape(TREATMENT_LABELS[treatment])}</text>'
        )
        treatment_keys = sorted(
            (key for key in histories if key[0] == treatment), key=lambda key: key[1]
        )
        for row_index, spec in enumerate(specs):
            panel_y = top + row_index * (panel_height + row_gap)
            x_map, y_map, clip_id = append_panel_axes(
                elements,
                spec=spec,
                panel_x=panel_x,
                panel_y=panel_y,
                panel_width=panel_width,
                panel_height=panel_height,
                show_y_labels=True,
                show_x_labels=row_index == len(specs) - 1,
                title=spec.title,
            )
            if spec.name == "train/tis_abs":
                warning_y = y_map(TIS_ABS_WARNING_THRESHOLD)
                elements.append(
                    f'<line class="warning" x1="{x_map(1):.1f}" y1="{warning_y:.1f}" '
                    f'x2="{x_map(MAX_STEP):.1f}" y2="{warning_y:.1f}"/>'
                )
            if spec.name == "train/tis_clipfrac":
                warning_y = y_map(TIS_CLIP_WARNING_THRESHOLD)
                elements.append(
                    f'<line class="warning" x1="{x_map(1):.1f}" y1="{warning_y:.1f}" '
                    f'x2="{x_map(MAX_STEP):.1f}" y2="{warning_y:.1f}"/>'
                )
            for key in treatment_keys:
                metrics = histories[key]
                append_series(
                    elements,
                    values=metrics.get(spec.name, {}),
                    series_color=color(key[1]),
                    x_map=x_map,
                    y_map=y_map,
                    clip_id=clip_id,
                )
                onset = collapse_onset(metrics.get("rollout/raw_reward", {}))
                if onset is not None:
                    onset_x = x_map(float(onset))
                    elements.append(
                        f'<line class="collapse-onset" clip-path="url(#{clip_id})" '
                        f'x1="{onset_x:.1f}" y1="{y_map(spec.upper):.1f}" '
                        f'x2="{onset_x:.1f}" y2="{y_map(spec.lower):.1f}" '
                        f'stroke="{color(key[1])}"/>'
                    )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def render_collapse_alignment(
    histories: dict[tuple[str, int], dict[str, dict[int, float]]],
) -> str:
    selected_specs = tuple(
        spec
        for spec in metric_specs()
        if spec.name
        in {
            "rollout/raw_reward",
            "rollout/response_len/mean",
            "rollout/truncated_ratio",
            "train/tis_abs",
            "train/tis_clipfrac",
            "train/policy_rollout_kl",
            "train/policy_rollout_token_ess",
        }
    )
    panel_width, panel_height = 310.0, 174.0
    column_gap, row_gap = 4.0, 12.0
    left, top = 12.0, 96.0
    width = int(left + len(COLLAPSE_KEYS) * (panel_width + column_gap) + 12)
    height = int(top + len(selected_specs) * (panel_height + row_gap) + 18)
    elements = svg_canvas(width, height)
    elements.append(
        "<style>.mismatch-onset{stroke:#D55E00;stroke-width:1.3;stroke-dasharray:6 3}"
        ".reward-onset{stroke:#8B0000;stroke-width:1.3;stroke-dasharray:2 3}"
        ".note{font-size:10px;fill:#555}</style>"
    )
    elements.extend(
        [
            f'<text class="column-label" x="{left + 64:.1f}" y="20">Collapse time alignment: mismatch rise precedes reward collapse; overlong rebound follows</text>',
            f'<line class="mismatch-onset" x1="{left + 64:.1f}" y1="45" x2="{left + 91:.1f}" y2="45"/>',
            f'<text class="legend" x="{left + 98:.1f}" y="49">first trailing-5 mean |TIS-1| >= 0.05</text>',
            f'<line class="reward-onset" x1="{left + 340:.1f}" y1="45" x2="{left + 367:.1f}" y2="45"/>',
            f'<text class="legend" x="{left + 374:.1f}" y="49">sustained trailing-5 reward &lt; 0.30</text>',
            f'<text class="note" x="{left + 64:.1f}" y="68">All curves use the exact trained-batch step and a trailing-5 emphasis; TIS upper clipping does not suppress ratios below 1.</text>',
        ]
    )
    for column, key in enumerate(COLLAPSE_KEYS):
        treatment, staleness = key
        panel_x = left + column * (panel_width + column_gap)
        label = "none" if treatment == "none" else "zero reward"
        elements.append(
            f'<text class="column-label" x="{panel_x + panel_width / 2:.1f}" y="89" '
            f'text-anchor="middle">{label} · S={staleness}</text>'
        )
        metrics = histories[key]
        mismatch_step = first_threshold(
            metrics.get("train/tis_abs", {}),
            TIS_ABS_WARNING_THRESHOLD,
            above=True,
        )
        reward_step = collapse_onset(metrics.get("rollout/raw_reward", {}))
        for row_index, spec in enumerate(selected_specs):
            panel_y = top + row_index * (panel_height + row_gap)
            x_map, y_map, clip_id = append_panel_axes(
                elements,
                spec=spec,
                panel_x=panel_x,
                panel_y=panel_y,
                panel_width=panel_width,
                panel_height=panel_height,
                show_y_labels=True,
                show_x_labels=row_index == len(selected_specs) - 1,
                title=spec.title,
            )
            append_series(
                elements,
                values=metrics.get(spec.name, {}),
                series_color=color(staleness),
                x_map=x_map,
                y_map=y_map,
                clip_id=clip_id,
            )
            for onset, css_class in (
                (mismatch_step, "mismatch-onset"),
                (reward_step, "reward-onset"),
            ):
                if onset is None:
                    continue
                onset_x = x_map(float(onset))
                elements.append(
                    f'<line class="{css_class}" clip-path="url(#{clip_id})" '
                    f'x1="{onset_x:.1f}" y1="{y_map(spec.upper):.1f}" '
                    f'x2="{onset_x:.1f}" y2="{y_map(spec.lower):.1f}"/>'
                )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def render_staleness_aware(
    histories: dict[tuple[str, int], dict[str, dict[int, float]]],
) -> str:
    keys = sorted(
        (key for key in histories if key[0] == "staleness-aware"),
        key=lambda key: key[1],
    )
    columns = 3
    rows = math.ceil(len(keys) / columns)
    panel_width, panel_height = 500.0, 285.0
    column_gap, row_gap = 14.0, 22.0
    left, top = 36.0, 105.0
    width = int(left + columns * panel_width + (columns - 1) * column_gap + 28)
    height = int(top + rows * panel_height + (rows - 1) * row_gap + 20)
    elements = svg_canvas(width, height)
    elements.append(
        "<style>.aware-token{fill:none;stroke:#0072B2;stroke-width:2.1}"
        ".aware-scale{fill:none;stroke:#6A3D9A;stroke-width:2.1;stroke-dasharray:7 3}"
        ".aware-pre{fill:none;stroke:#D55E00;stroke-width:2.2}"
        ".aware-post{fill:none;stroke:#009E73;stroke-width:2.2}"
        ".aware-retain{fill:none;stroke:#222;stroke-width:1.9;stroke-dasharray:2 3}"
        ".panel-note{font-size:10px;fill:#444}.note{font-size:10px;fill:#555}</style>"
    )
    legend = (
        ("aware-token", "truncated loss-token share"),
        ("aware-scale", "mean truncated gradient scale"),
        ("aware-pre", "truncated objective share: post-TIS, pre-scaling"),
        ("aware-post", "truncated objective share: post-scaling"),
        ("aware-retain", "total absolute objective retained"),
    )
    elements.append(
        f'<text class="column-label" x="{left:.1f}" y="20">Staleness-aware truncated-loss attenuation (safe staleness = 4)</text>'
    )
    legend_x = left
    for css_class, label in legend:
        elements.extend(
            [
                f'<line class="{css_class}" x1="{legend_x:.1f}" y1="48" x2="{legend_x + 26:.1f}" y2="48"/>',
                f'<text class="legend" x="{legend_x + 32:.1f}" y="52">{html.escape(label)}</text>',
            ]
        )
        legend_x += 240.0 if "objective" not in label else 330.0
    elements.append(
        f'<text class="note" x="{left:.1f}" y="76">Objective shares are exact after token-level TIS; retention = total post-scaling / pre-scaling absolute objective.</text>'
    )
    metric_styles = (
        (TRUNCATED_TOKEN_METRIC, "aware-token"),
        (TRUNCATED_SCALE_METRIC, "aware-scale"),
        (TRUNCATED_PRE_OBJECTIVE_METRIC, "aware-pre"),
        (TRUNCATED_POST_OBJECTIVE_METRIC, "aware-post"),
        (OBJECTIVE_RETENTION_METRIC, "aware-retain"),
    )
    for index, key in enumerate(keys):
        row_index, column = divmod(index, columns)
        panel_x = left + column * (panel_width + column_gap)
        panel_y = top + row_index * (panel_height + row_gap)
        plot_x, plot_y = panel_x + 54.0, panel_y + 58.0
        plot_width, plot_height = panel_width - 76.0, panel_height - 96.0
        x_map = lambda value, base=plot_x: base + plot_width * (value - 1.0) / (MAX_STEP - 1.0)
        y_map = lambda value, base=plot_y: base + plot_height * (1.0 - min(max(value, 0.0), 1.0))
        clip_id = f"aware-{key[1]}"
        elements.append(
            f'<defs><clipPath id="{clip_id}"><rect x="{plot_x:.1f}" y="{plot_y:.1f}" '
            f'width="{plot_width:.1f}" height="{plot_height:.1f}"/></clipPath></defs>'
        )
        metrics = histories[key]
        late_token = late_mean(metrics.get(TRUNCATED_TOKEN_METRIC, {}))
        late_scale = late_mean(metrics.get(TRUNCATED_SCALE_METRIC, {}))
        late_pre = late_mean(metrics.get(TRUNCATED_PRE_OBJECTIVE_METRIC, {}))
        late_post = late_mean(metrics.get(TRUNCATED_POST_OBJECTIVE_METRIC, {}))
        late_retain = late_mean(metrics.get(OBJECTIVE_RETENTION_METRIC, {}))
        subtitle = (
            f"late-20: tokens {100 * (late_token or 0):.1f}% · scale {late_scale or 0:.3f} · "
            f"obj {100 * (late_pre or 0):.1f}% -> {100 * (late_post or 0):.1f}% · "
            f"retain {100 * (late_retain or 0):.1f}%"
        )
        elements.extend(
            [
                f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 18:.1f}" text-anchor="middle">S={key[1]}</text>',
                f'<text class="panel-note" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 36:.1f}" text-anchor="middle">{html.escape(subtitle)}</text>',
            ]
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            tick_y = y_map(tick)
            elements.extend(
                [
                    f'<line class="grid" x1="{plot_x:.1f}" y1="{tick_y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{tick_y:.1f}"/>',
                    f'<text class="tick" x="{plot_x - 6:.1f}" y="{tick_y + 4:.1f}" text-anchor="end">{tick * 100:.0f}%</text>',
                ]
            )
        for tick in (1, 100, 200, 300):
            tick_x = x_map(float(tick))
            elements.append(
                f'<text class="tick" x="{tick_x:.1f}" y="{plot_y + plot_height + 17:.1f}" text-anchor="middle">{tick}</text>'
            )
        elements.extend(
            [
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height - 4:.1f}" text-anchor="middle">training step</text>',
            ]
        )
        for metric, css_class in metric_styles:
            points = [
                (float(step), value)
                for step, value in rolling(metrics.get(metric, {})).items()
            ]
            if points:
                elements.append(
                    f'<polyline class="{css_class}" clip-path="url(#{clip_id})" '
                    f'points="{polyline(points, x_map, y_map)}"/>'
                )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def render_age_profiles(rows: list[dict[str, Any]]) -> str:
    specs = (
        (
            "mass",
            "Age mass / objective share",
            False,
            (0.0, 0.25, 0.5, 0.75, 1.0),
            ("0", "25%", "50%", "75%", "100%"),
        ),
        (
            "mean_abs_policy_rollout_log_ratio",
            "Mean |log pi_train - log pi_rollout|",
            True,
            (0.001, 0.01, 0.1, 1.0, 10.0),
            (".001", ".01", ".1", "1", "10"),
        ),
        (
            "importance_clip_fraction",
            "TIS upper-clip fraction",
            True,
            (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
            (".0001%", ".001%", ".01%", ".1%", "1%", "10%"),
        ),
        (
            "initial_mask_fraction",
            "Initially masked response tokens in age bin",
            False,
            (0.0, 0.25, 0.5, 0.75, 1.0),
            ("0", "25%", "50%", "75%", "100%"),
        ),
    )
    panel_width, panel_height = 380.0, 215.0
    column_gap, row_gap = 12.0, 18.0
    left, top = 24.0, 95.0
    width = int(left + len(AGE_CASES) * (panel_width + column_gap) + 20)
    height = int(top + len(specs) * (panel_height + row_gap) + 18)
    elements = svg_canvas(width, height)
    elements.append(
        "<style>.token-mass{fill:none;stroke:#0072B2;stroke-width:2.1}"
        ".response-mass{fill:none;stroke:#56B4E9;stroke-width:1.8;stroke-dasharray:6 3}"
        ".contribution-mass{fill:none;stroke:#D55E00;stroke-width:2.2}"
        ".profile{fill:none;stroke:#222;stroke-width:2.2}"
        ".point{stroke:white;stroke-width:.7}.note{font-size:10px;fill:#555}</style>"
    )
    elements.extend(
        [
            f'<text class="column-label" x="{left + 55:.1f}" y="20">S=20 staleness-bin profiles: healthy, collapse transition, and stabilized treatments</text>',
            f'<line class="token-mass" x1="{left + 55:.1f}" y1="47" x2="{left + 82:.1f}" y2="47"/>',
            f'<text class="legend" x="{left + 89:.1f}" y="51">pre-loss token mass</text>',
            f'<line class="response-mass" x1="{left + 220:.1f}" y1="47" x2="{left + 247:.1f}" y2="47"/>',
            f'<text class="legend" x="{left + 254:.1f}" y="51">response-token mass</text>',
            f'<line class="contribution-mass" x1="{left + 405:.1f}" y1="47" x2="{left + 432:.1f}" y2="47"/>',
            f'<text class="legend" x="{left + 439:.1f}" y="51">post-mask/TIS objective mass</text>',
            f'<text class="note" x="{left + 55:.1f}" y="70">Zero-loss mismatch and clip statistics condition on unmasked tokens; its mask panel shows the censored token share.</text>',
        ]
    )
    rows_by_case: dict[str, list[dict[str, Any]]] = {
        case.label: [row for row in rows if row["case"] == case.label]
        for case in AGE_CASES
    }
    for column, case in enumerate(AGE_CASES):
        panel_x = left + column * (panel_width + column_gap)
        elements.append(
            f'<text class="column-label" x="{panel_x + panel_width / 2:.1f}" y="89" text-anchor="middle">{html.escape(case.label)} · steps {case.first_step}-{case.last_step}</text>'
        )
        case_rows = rows_by_case[case.label]
        ages = [int(row["sample_staleness"]) for row in case_rows]
        age_lower = min(ages, default=0)
        age_upper = max(ages, default=1)
        if age_lower == age_upper:
            age_lower = max(0, age_lower - 1)
        for row_index, (metric, title, logarithmic, ticks, tick_labels) in enumerate(specs):
            panel_y = top + row_index * (panel_height + row_gap)
            plot_x, plot_y = panel_x + 65.0, panel_y + 29.0
            plot_width, plot_height = panel_width - 84.0, panel_height - 66.0
            x_map = lambda value, base=plot_x: base + plot_width * (value - age_lower) / max(
                age_upper - age_lower, 1
            )
            lower, upper = (1e-6, 0.1) if metric == "importance_clip_fraction" else (
                (0.001, 10.0) if logarithmic else (0.0, 1.0)
            )
            if logarithmic:
                y_map = lambda value, base=plot_y: base + plot_height * (
                    1.0
                    - (
                        math.log10(min(max(value, lower), upper)) - math.log10(lower)
                    )
                    / (math.log10(upper) - math.log10(lower))
                )
            else:
                y_map = lambda value, base=plot_y: base + plot_height * (
                    1.0 - min(max(value, lower), upper)
                )
            clip_id = f"age-{column}-{row_index}"
            elements.append(
                f'<defs><clipPath id="{clip_id}"><rect x="{plot_x:.1f}" y="{plot_y:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}"/></clipPath></defs>'
            )
            for tick, label in zip(ticks, tick_labels, strict=True):
                tick_y = y_map(tick)
                elements.extend(
                    [
                        f'<line class="grid" x1="{plot_x:.1f}" y1="{tick_y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{tick_y:.1f}"/>',
                        f'<text class="tick" x="{plot_x - 6:.1f}" y="{tick_y + 4:.1f}" text-anchor="end">{html.escape(label)}</text>',
                    ]
                )
            age_ticks = sorted({age_lower, age_upper, *range(age_lower, age_upper + 1, 2)})
            for age in age_ticks:
                tick_x = x_map(float(age))
                if row_index == len(specs) - 1:
                    elements.append(
                        f'<text class="tick" x="{tick_x:.1f}" y="{plot_y + plot_height + 17:.1f}" text-anchor="middle">{age}</text>'
                    )
            elements.extend(
                [
                    f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                    f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                    f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 15:.1f}" text-anchor="middle">{html.escape(title)}</text>',
                ]
            )
            if row_index == len(specs) - 1:
                elements.append(
                    f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height - 2:.1f}" text-anchor="middle">sample training staleness</text>'
                )
            if metric == "mass":
                plotted = (
                    ("consumed_pre_loss_token_mass", "token-mass", "#0072B2"),
                    ("consumed_response_token_mass", "response-mass", "#56B4E9"),
                    ("effective_contribution_mass", "contribution-mass", "#D55E00"),
                )
            else:
                plotted = ((metric, "profile", "#222222"),)
            for value_metric, css_class, point_color in plotted:
                points = [
                    (float(row["sample_staleness"]), float(row[value_metric]))
                    for row in case_rows
                    if row[value_metric] != ""
                ]
                if len(points) >= 2:
                    elements.append(
                        f'<polyline class="{css_class}" clip-path="url(#{clip_id})" points="{polyline(points, x_map, y_map)}"/>'
                    )
                for age, value in points:
                    elements.append(
                        f'<circle class="point" cx="{x_map(age):.2f}" cy="{y_map(value):.2f}" r="2.7" fill="{point_color}"/>'
                    )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def format_optional(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def find_row(rows: list[dict[str, Any]], treatment: str, staleness: int) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["treatment"] == treatment and int(row["max_weight_staleness"]) == staleness
    )


def render_report(
    summary: list[dict[str, Any]],
    age_rows: list[dict[str, Any]],
    no_treatment_rows: list[dict[str, Any]],
) -> str:
    collapsed_rows = [find_row(summary, treatment, staleness) for treatment, staleness in COLLAPSE_KEYS]
    stable_keys = (
        ("zero-reward", 12),
        ("none", 8),
        ("zero-loss", 8),
        ("zero-loss", 16),
        ("zero-loss", 20),
        ("staleness-aware", 12),
        ("staleness-aware", 16),
        ("staleness-aware", 20),
        ("staleness-aware", 24),
        ("staleness-aware", 28),
    )
    stable_rows = [find_row(summary, treatment, staleness) for treatment, staleness in stable_keys]
    lines = [
        "# TIS・truncation・staleness 解析",
        "",
        "Snapshot: 2026-09-06。ローカル `metrics.jsonl` を run 内の記録順で latest-write-wins にし、",
        "Train:Rollout=1:7、max response length=16,384、step<=300 に限定した。太線は trailing-5 mean。",
        "",
        "## 結論",
        "",
        "- `zero-loss-on-truncated` は、この設定で S=16/20 の catastrophic collapse を止めるには十分だった。",
        "  ただし、これだけで『staleness の問題を一般に解決した』または『truncated sample が唯一の原因』とは言えない。",
        "- collapse arm では `train/tis_abs` の trailing-5 mean が 0.05 を越えるのが reward collapse より",
        "  22--41 update 早い。したがって policy/rollout mismatch は単なる末期指標ではなく、collapse loop の一部である。",
        "- 一方、truncation 50% 超は reward collapse 後に現れる。末期の truncated-objective dominance は",
        "  collapse の増幅・固定化を示すが、発火点を単独では同定しない。観測される順序は",
        "  `TIS mismatch 上昇 -> 短文相 -> reward collapse -> overlong/truncated 相` である。",
        "- staleness-aware loss は独立な反証ではない。高 staleness では truncated gradient を 4--12% 程度まで",
        "  落とす soft zero-loss として働き、同じ経路を遮断している。",
        "- 最も妥当な解釈は、staleness 自体が一定値を越えて即座に壊すのではなく、",
        "  長文・truncation・group-normalized negative advantage が作る遅延フィードバックを staleness が増幅する、である。",
        "",
        "## Collapse の時間順序",
        "",
        "| treatment | S | |TIS-1|>=0.05 | reward<0.30 | lead | mean len<=2K | trunc>=50% | collapse-window TIS abs | clip | token ESS |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in collapsed_rows:
        label = "none" if row["treatment"] == "none" else "zero reward"
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    str(row["max_weight_staleness"]),
                    str(row["first_tis_abs_rolling5_ge_0p05"]),
                    str(row["collapse_onset_reward_rolling5_lt_0p30"]),
                    str(row["tis_abs_lead_to_collapse_steps"]),
                    str(row["first_response_len_rolling5_le_2048"]),
                    str(row["first_truncated_ratio_rolling5_ge_0p50"]),
                    format_optional(row["collapse20_tis_abs"]),
                    f"{100 * float(row['collapse20_tis_clipfrac']):.2f}%",
                    format_optional(row["collapse20_policy_rollout_token_ess"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`tis_clipfrac` は `[0, 2]` の上限 2 を越えた token だけを数える。ratio<1 は clip されないため、",
            "S=24 のように mean TIS が下側へ崩れる局面では clip fraction だけを見ると mismatch を過小評価する。",
            "",
            "## Collapse しなかった観測窓",
            "",
            "| treatment | S | last step | realized stale | reward | trunc | TIS abs | TIS clip | policy KL | token ESS |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for row in stable_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["treatment"]),
                    str(row["max_weight_staleness"]),
                    str(row["last_observed_step"]),
                    format_optional(row["late20_realized_staleness"], 2),
                    format_optional(row["late20_raw_reward"]),
                    f"{100 * float(row['late20_truncated_ratio']):.1f}%",
                    format_optional(row["late20_tis_abs"]),
                    f"{100 * float(row['late20_tis_clipfrac']):.4f}%",
                    format_optional(row["late20_policy_rollout_kl"], 4),
                    format_optional(row["late20_policy_rollout_token_ess"], 4),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "同じ realized staleness 16--24 でも、安定 arm の TIS abs は約0.017--0.022、clip率は約0.001--0.006%で、",
            "collapse-window の TIS abs 0.15--0.87、clip率0.5--1.7%より桁違いに小さい。",
            "よって lag は mismatch の十分統計ではない。実際の mismatch は lag 中に積み上がった update の大きさと方向にも依存する。",
            "",
            "## Staleness-aware loss は何をしているか",
            "",
            "| S | truncated token share | mean scale | truncated objective pre -> post | total objective retained | late reward |",
            "|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for staleness in (12, 16, 20, 24, 28):
        row = find_row(summary, "staleness-aware", staleness)
        lines.append(
            f"| {staleness} | {100 * float(row['late20_aware_truncated_loss_token_fraction']):.1f}% "
            f"| {float(row['late20_aware_truncated_mean_gradient_scale']):.3f} "
            f"| {100 * float(row['late20_aware_truncated_pre_objective_fraction']):.1f}% -> "
            f"{100 * float(row['late20_aware_truncated_post_objective_fraction']):.1f}% "
            f"| {100 * float(row['late20_aware_total_objective_retention']):.1f}% "
            f"| {float(row['late20_raw_reward']):.3f} |"
        )
    zero_loss_s20 = find_row(summary, "zero-loss", 20)
    transition_age20 = next(
        row
        for row in age_rows
        if row["case"] == "zero-reward transition" and int(row["sample_staleness"]) == 20
    )
    zero_loss_age20 = next(
        row
        for row in age_rows
        if row["case"] == "zero-loss late" and int(row["sample_staleness"]) == 20
    )
    no_treatment_pre_collapse = next(
        row for row in no_treatment_rows if row["window"] == "pre-collapse"
    )
    lines.extend(
        [
            "",
            f"S=20 aware は truncated objective share を late-20 で "
            f"{100 * float(find_row(summary, 'staleness-aware', 20)['late20_aware_truncated_pre_objective_fraction']):.1f}% から "
            f"{100 * float(find_row(summary, 'staleness-aware', 20)['late20_aware_truncated_post_objective_fraction']):.1f}% へ落としている。",
            "これは『高 staleness のまま通常の truncated loss を保持しても安全』という結果ではなく、",
            "その loss をほぼ消すと安全、という結果である。",
            "",
            "## Age-bin で見えるもの",
            "",
            f"- zero-reward S=20 の transition window (step 128--147) では、staleness=20 が pre-loss token mass の "
            f"{100 * float(transition_age20['consumed_pre_loss_token_mass']):.1f}% を占め、mean absolute log-ratio は "
            f"{float(transition_age20['mean_abs_policy_rollout_log_ratio']):.3f}、upper-clip率は "
            f"{100 * float(transition_age20['importance_clip_fraction']):.2f}% だった。",
            f"- zero-loss S=20 late では、staleness=20 bin の response token の "
            f"{100 * float(zero_loss_age20['initial_mask_fraction']):.1f}% が initial loss mask で消えている。全 age 合計でも "
            f"{100 * float(zero_loss_s20['late20_masked_response_token_fraction']):.1f}% の response-token mass が masked である。",
            "- したがって zero-loss の低い global TIS/clip は、安定化の結果に加えて、問題になり得る truncated token を",
            "  TIS 集計の母集団から外した censoring を含む。zero-loss の TIS 図だけでは masked token の counterfactual mismatch は分からない。",
            "",
            "## No-treatment が zero-reward と似る理由",
            "",
            f"no-treatment S=16 の collapse 前 step 1--149 では、truncated sample "
            f"{int(no_treatment_pre_collapse['truncated_count']):,} 件の平均 raw reward は "
            f"{float(no_treatment_pre_collapse['truncated_raw_reward_mean']):.6f}、reward=1 は "
            f"{int(no_treatment_pre_collapse['truncated_positive_count']):,} 件 "
            f"({100 * float(no_treatment_pre_collapse['truncated_positive_fraction']):.3f}%) だけだった。",
            "したがって verifier に通常どおり truncated text を採点させても、ほぼ全件が自然に0点になる。",
            "このデータでは no-treatment と explicit zero-reward は実効的に同じ負 feedback を持つため、",
            "両者の類似挙動は truncation-gradient 仮説への反証にはならない。",
            "",
            "## なぜ『staleness 耐性』より別の現象が見えるのか",
            "",
            "1. **staleness は delay、truncation は高 gain の非線形 feedback**: truncated response は長いため token mass が大きく、",
            "   binary reward と group normalization により強い負 advantage を持ちやすい。その更新が lag 分遅れて到着する。",
            "2. **lag x policy velocity が mismatch**: policy が緩やかなら realized lag 20 でも ratio は狭い。長さ方策が動き始めると、",
            "   同じ lag でも queued trajectory が急速に off-policy になり、TIS/KL が跳ねる。",
            "3. **TIS は stale advantage を直さない**: token ratio を `[0,2]` にするだけで、古い policy が生成した trajectory の",
            "   reward/advantage、state distribution、length censoring の意味は current policy 向けに再計算されない。",
            "4. **one-sided cap**: 下限0は実質 no-op なので、ratio<<1 の token は小さくなるだけで除外されない。",
            "   upper-clip率が小さくても大きな directional mismatch や stale negative feedback は残り得る。",
            "5. **二つの attractor**: 実測は最初に短文・低 reward 相へ移り、その後 max-length 相へ反跳する。",
            "   これは単純な『overlong率が徐々に上がって collapse』より、遅延系の振動・overshoot に近い。",
            "",
            "## VCPO など既報との関係",
            "",
            "- [VCPO paper](https://arxiv.org/abs/2602.17616) が論じる heavy-tailed importance ratio / high-variance 問題と矛盾しない。",
            "  今回も collapse arm では reward 崩壊より22--41 update 先に TIS mismatch が急増している。",
            "- 一方、[official MATH recipe](https://github.com/mit-han-lab/vcpo/blob/main/recipe/fully_async_policy/shell/vcpo/math/synchronous.sh) は",
            "  max response 2K、sequence-level IS、threshold 8 であり、今回の16K、token-level TIS `[0,2]`、truncation ablation とは estimator と censoring regime が異なる。",
            "- したがって VCPO の結果を無効とはいえないが、そこでの configured lag を今回の realized lag と同一視して、",
            "  Long-CoT の length/truncation feedback まで解決済みとみなす外的妥当性もない。二つは異なる failure channel を観測し得る。",
            "",
            "## 何がまだ未証明か",
            "",
            "- 各 treatment は同じ生成 batch に対する paired counterfactual ではなく、seed repeat もない。介入後は prompt/sample selection 自体が変わる。",
            "- zero-loss は direct gradient を消すが、truncated sample の reward は group baseline/advantage に残るため、完全な sample filtering ではない。",
            "- staleness-aware は total absolute objective を約27--49%落とす。truncation-specific 効果と generic effective-step-size reduction が混ざる。",
            "- zero-loss は S=20 まで300 stepと late AIMEで確認できるが、aware S=20/24/28 の late downstream eval はまだ十分でない。",
            "- 1 model、1 math dataset、16K、LR=1e-6、300 update の結果であり、math RL 全般や長期 horizon へは外挿できない。",
            "",
            "## 決着をつける control",
            "",
            "1. S=16/20 で zero-reward、zero-loss、aware を複数 seed 反復する。",
            "2. aware と同じ total objective norm / update norm になる global attenuation または random-sample attenuation を置く。",
            "3. zero-loss と、truncated sample を group baseline からも除く full filtering を分ける。",
            "4. 同じ saved batch に対して通常 loss / zero-loss / aware の gradient cosine・norm を offline replay で比較する。",
            "5. truncated/non-truncated x staleness x advantage sign ごとの post-TIS objective と gradient proxy を保存する。",
            "",
            "## 出力",
            "",
            "- [Cross-treatment TIS diagnostics](figures/tis-cross-treatment-t1r7-maxlen16384.svg)",
            "- [Collapse time alignment](figures/tis-collapse-time-alignment-t1r7-maxlen16384.svg)",
            "- [Staleness-aware exact objective attenuation](figures/staleness-aware-post-tis-objective-t1r7-maxlen16384.svg)",
            "- [S=20 staleness-bin profiles](figures/tis-staleness-bin-profiles-s20-t1r7-maxlen16384.svg)",
            "- [Per-arm summary](tis-diagnostics-summary.csv)",
            "- [Staleness-bin summary](tis-staleness-bin-summary.csv)",
            "- [No-treatment truncated-reward summary](no-treatment-truncated-reward-summary.csv)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    arms = discover_arms()
    histories = {
        (arm.treatment, arm.staleness): read_dashboard(arm.dashboard_path)
        for arm in arms
    }
    expected_keys = {
        ("zero-reward", 12),
        ("zero-reward", 16),
        ("zero-reward", 20),
        ("zero-reward", 24),
        ("zero-reward", 28),
        ("none", 8),
        ("none", 16),
        ("zero-loss", 8),
        ("zero-loss", 16),
        ("zero-loss", 20),
        ("staleness-aware", 12),
        ("staleness-aware", 16),
        ("staleness-aware", 20),
        ("staleness-aware", 24),
        ("staleness-aware", 28),
    }
    missing = expected_keys - set(histories)
    if missing:
        raise ValueError(f"missing expected arms: {sorted(missing)}")
    merge_objective_diagnostics(arms, histories)
    summary = summary_rows(arms, histories)
    age_summary = age_summary_rows(histories)
    no_treatment_summary = no_treatment_truncated_reward_rows(arms)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(SUMMARY_PATH, summary)
    write_csv(AGE_SUMMARY_PATH, age_summary)
    write_csv(NO_TREATMENT_REWARD_PATH, no_treatment_summary)
    (FIGURE_ROOT / "tis-cross-treatment-t1r7-maxlen16384.svg").write_text(
        render_cross_treatment(histories), encoding="utf-8"
    )
    (FIGURE_ROOT / "tis-collapse-time-alignment-t1r7-maxlen16384.svg").write_text(
        render_collapse_alignment(histories), encoding="utf-8"
    )
    (FIGURE_ROOT / "staleness-aware-post-tis-objective-t1r7-maxlen16384.svg").write_text(
        render_staleness_aware(histories), encoding="utf-8"
    )
    (FIGURE_ROOT / "tis-staleness-bin-profiles-s20-t1r7-maxlen16384.svg").write_text(
        render_age_profiles(age_summary), encoding="utf-8"
    )
    REPORT_PATH.write_text(
        render_report(summary, age_summary, no_treatment_summary), encoding="utf-8"
    )
    print(
        f"arms={len(arms)} summary_rows={len(summary)} age_rows={len(age_summary)} "
        f"no_treatment_rows={len(no_treatment_summary)} "
        f"figures=4 report={REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
