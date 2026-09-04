#!/usr/bin/env python3
"""Summarize and plot realized training staleness from W&B history."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.tools.reasoning_eval.grid import reasoning_eval_grid_from_environment

ARM_PATTERN = re.compile(r"^s(?P<staleness>\d+)-t(?P<train>\d+)r(?P<rollout>\d+)$")
EVALUATION_GRID = reasoning_eval_grid_from_environment()
STALENESS_LEVELS = EVALUATION_GRID.staleness_levels
NODE_RATIOS = EVALUATION_GRID.node_ratios
STALENESS_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
STALENESS_COLORS = {
    staleness: STALENESS_PALETTE[index % len(STALENESS_PALETTE)] for index, staleness in enumerate(STALENESS_LEVELS)
}
METRIC = "staleness/total/mean"
SENSITIVE_METRIC_THRESHOLD = 0.25
SENSITIVE_METRIC_CANDIDATES = (
    "train/tis_abs",
    "train/tis_clipfrac",
    "train/policy_rollout_abs_diff",
    "train/policy_rollout_kl",
    "train/policy_rollout_token_ess",
    "train/final_loss_tokens",
    "train/advantage_abs_mean",
    "train/advantage_rms",
    "train/advantage_std",
    "train/loss",
    "train/grad_norm_pre_clip",
)
SENSITIVE_METRIC_LABELS = {
    "train/tis": "TIS signed mean (reference)",
    "train/tis_abs": "TIS absolute deviation",
    "train/tis_clipfrac": "TIS clipped-token fraction",
    "train/policy_rollout_kl": "policy–rollout KL",
    "train/policy_rollout_abs_diff": "policy–rollout |logprob diff|",
    "train/policy_rollout_token_ess": "policy–rollout token ESS",
    "train/final_loss_tokens": "final loss tokens",
    "train/advantage_abs_mean": "advantage absolute mean",
    "train/advantage_rms": "advantage RMS",
    "train/advantage_std": "advantage standard deviation",
    "train/loss": "training loss",
    "train/grad_norm_pre_clip": "gradient norm before clipping",
}
STALENESS_PREDICTOR_LABELS = {
    "staleness/total/mean": "total staleness mean",
    "staleness/total/variance": "total staleness variance",
    "staleness/pre_queue/mean": "pre-queue staleness mean",
    "staleness/pre_queue/variance": "pre-queue staleness variance",
    "staleness/in_queue/mean": "in-queue staleness mean",
    "staleness/in_queue/variance": "in-queue staleness variance",
    "staleness/token_lag/exact/mean": "exact token lag mean",
    "staleness/version_mix/train/forward_version_span/sequence_mean": "forward-version span",
}


@dataclass(frozen=True)
class StalenessPoint:
    """One realized training-staleness observation."""

    arm: str
    max_weight_staleness: int
    trainer_nodes: int
    rollout_nodes: int
    training_step: int
    staleness_total_mean: float
    rolling_mean: float
    rolling_observations: int


@dataclass(frozen=True)
class SteadyState:
    """Trailing contiguous-window summary for one async setting."""

    arm: str
    max_weight_staleness: int
    trainer_nodes: int
    rollout_nodes: int
    requested_window_updates: int
    window_start_step: int
    window_end_step: int
    observations: int
    staleness_total_mean: float
    staleness_total_std: float
    slope_per_100_updates: float
    estimated_window_change: float
    settling_tolerance: float
    settled: int


@dataclass(frozen=True)
class SensitiveMetric:
    """Training metric selected by its strongest realized-staleness correlation."""

    metric: str
    label: str
    strongest_predictor: str
    correlation: float | None
    absolute_correlation: float | None
    observations: int | None
    selection_reason: str


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _arm_metadata(arm: str) -> tuple[int, int, int] | None:
    match = ARM_PATTERN.fullmatch(arm)
    if match is None:
        return None
    return int(match["staleness"]), int(match["train"]), int(match["rollout"])


def _read_history(path: Path) -> dict[str, dict[int, float]]:
    histories: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arm = row["arm"]
            if _arm_metadata(arm) is None:
                continue
            value = _optional_float(row.get(METRIC))
            if value is None:
                continue
            histories[arm][int(row["training_step"])] = value
    return dict(histories)


def _select_sensitive_metrics(path: Path) -> list[SensitiveMetric]:
    strongest: dict[str, tuple[float, str, float, int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            metric = row["outcome"]
            if metric not in SENSITIVE_METRIC_CANDIDATES:
                continue
            correlation = _optional_float(row.get("correlation"))
            if correlation is None:
                continue
            candidate = (abs(correlation), row["predictor"], correlation, int(row["observations"]))
            if metric not in strongest or candidate[0] > strongest[metric][0]:
                strongest[metric] = candidate
    selected = [
        SensitiveMetric(
            metric="train/tis",
            label=SENSITIVE_METRIC_LABELS["train/tis"],
            strongest_predictor="",
            correlation=None,
            absolute_correlation=None,
            observations=None,
            selection_reason="signed reference for interpreting TIS absolute deviation",
        )
    ]
    for metric in SENSITIVE_METRIC_CANDIDATES:
        result = strongest.get(metric)
        if result is None or result[0] < SENSITIVE_METRIC_THRESHOLD:
            continue
        absolute_correlation, predictor, correlation, observations = result
        selected.append(
            SensitiveMetric(
                metric=metric,
                label=SENSITIVE_METRIC_LABELS[metric],
                strongest_predictor=predictor,
                correlation=correlation,
                absolute_correlation=absolute_correlation,
                observations=observations,
                selection_reason=f"max |r| >= {SENSITIVE_METRIC_THRESHOLD:.2f}",
            )
        )
    return selected


def _read_metric_histories(path: Path, metrics: list[SensitiveMetric]) -> dict[str, dict[str, dict[int, float]]]:
    metric_names = {metric.metric for metric in metrics}
    histories: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arm = row["arm"]
            if _arm_metadata(arm) is None:
                continue
            step = int(row["training_step"])
            for metric in metric_names:
                value = _optional_float(row.get(metric))
                if value is not None:
                    histories[metric][arm][step] = value
    return {
        metric: {arm: dict(values) for arm, values in arm_histories.items()}
        for metric, arm_histories in histories.items()
    }


def _rolling_metric_values(values_by_step: dict[int, float], window: int) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for step in sorted(values_by_step):
        values = [
            values_by_step[candidate]
            for candidate in range(max(1, step - window + 1), step + 1)
            if candidate in values_by_step
        ]
        rows.append((step, statistics.fmean(values)))
    return rows


def _slope(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0.0:
        return 0.0
    return sum((x - x_mean) * (ys[index] - y_mean) for index, x in enumerate(xs)) / denominator


def _trajectory_rows(histories: dict[str, dict[int, float]], *, rolling_window: int) -> list[StalenessPoint]:
    rows: list[StalenessPoint] = []
    for arm, values_by_step in sorted(histories.items()):
        metadata = _arm_metadata(arm)
        if metadata is None:
            continue
        staleness, trainer_nodes, rollout_nodes = metadata
        for step in sorted(values_by_step):
            window_values = [
                values_by_step[candidate]
                for candidate in range(max(1, step - rolling_window + 1), step + 1)
                if candidate in values_by_step
            ]
            rows.append(
                StalenessPoint(
                    arm=arm,
                    max_weight_staleness=staleness,
                    trainer_nodes=trainer_nodes,
                    rollout_nodes=rollout_nodes,
                    training_step=step,
                    staleness_total_mean=values_by_step[step],
                    rolling_mean=statistics.fmean(window_values),
                    rolling_observations=len(window_values),
                )
            )
    return rows


def _trailing_contiguous_steps(values_by_step: dict[int, float], window: int) -> list[int]:
    end_step = max(values_by_step)
    steps = [end_step]
    while len(steps) < window and steps[-1] - 1 in values_by_step:
        steps.append(steps[-1] - 1)
    return sorted(steps)


def _steady_state_rows(histories: dict[str, dict[int, float]], *, steady_window: int) -> list[SteadyState]:
    rows: list[SteadyState] = []
    for arm, values_by_step in sorted(histories.items()):
        metadata = _arm_metadata(arm)
        if metadata is None or not values_by_step:
            continue
        staleness, trainer_nodes, rollout_nodes = metadata
        steps = _trailing_contiguous_steps(values_by_step, steady_window)
        values = [values_by_step[step] for step in steps]
        mean = statistics.fmean(values)
        standard_deviation = statistics.pstdev(values)
        slope_per_100_updates = 100.0 * _slope([float(step) for step in steps], values)
        estimated_window_change = abs(slope_per_100_updates) * (steps[-1] - steps[0]) / 100.0
        settling_tolerance = max(standard_deviation, 0.1 * abs(mean), 0.05)
        rows.append(
            SteadyState(
                arm=arm,
                max_weight_staleness=staleness,
                trainer_nodes=trainer_nodes,
                rollout_nodes=rollout_nodes,
                requested_window_updates=steady_window,
                window_start_step=steps[0],
                window_end_step=steps[-1],
                observations=len(steps),
                staleness_total_mean=mean,
                staleness_total_std=standard_deviation,
                slope_per_100_updates=slope_per_100_updates,
                estimated_window_change=estimated_window_change,
                settling_tolerance=settling_tolerance,
                settled=int(estimated_window_change <= settling_tolerance),
            )
        )
    return sorted(rows, key=lambda row: (row.max_weight_staleness, row.trainer_nodes))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty steady-state staleness table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:24px;font-weight:700}"
        ".subtitle{font-size:14px;fill:#555}.axis-label{font-size:15px;font-weight:600}"
        ".tick{font-size:12px}.cell-value{font-size:25px;font-weight:700}.cell-detail{font-size:13px}"
        ".panel-title{font-size:17px;font-weight:700}.grid{stroke:#d9d9d9;stroke-width:1}"
        ".axis{stroke:#555;stroke-width:1.2}.raw{fill:none;stroke-width:1;opacity:.20}"
        ".smooth{fill:none;stroke-width:2.8}.frame{fill:#fbfcfd;stroke:#777;stroke-width:1.2}"
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{html.escape(title)}</text>',
        f'<text class="subtitle" x="{width / 2:.1f}" y="58" text-anchor="middle">' f"{html.escape(subtitle)}</text>",
    ]


def _interpolate_color(value: float, maximum: float) -> str:
    fraction = min(max(value / max(maximum, 1e-12), 0.0), 1.0)
    low = (247, 251, 255)
    middle = (107, 174, 214)
    high = (8, 48, 107)
    start, end, local = (low, middle, fraction * 2.0) if fraction <= 0.5 else (middle, high, fraction * 2.0 - 1.0)
    channels = tuple(round(start[index] + local * (end[index] - start[index])) for index in range(3))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _render_steady_grid(rows: list[SteadyState]) -> str:
    width, height = 1120, 825
    plot_x, plot_y = 285.0, 155.0
    cell_width, cell_height = 190.0, 128.0
    maximum = max(row.staleness_total_mean for row in rows)
    by_setting = {(row.max_weight_staleness, row.trainer_nodes, row.rollout_nodes): row for row in rows}
    elements = _svg_header(
        width,
        height,
        "Late-window training staleness mean and settling status",
        f"W&B {METRIC}; trailing 50 contiguous updates; changing when fitted window drift exceeds "
        "max(std, 10% of mean, 0.05)",
    )
    for column, (trainer_nodes, rollout_nodes) in enumerate(NODE_RATIOS):
        center_x = plot_x + (column + 0.5) * cell_width
        elements.append(
            f'<text class="axis-label" x="{center_x:.2f}" y="112" text-anchor="middle">'
            f"train:rollout {trainer_nodes}:{rollout_nodes}</text>"
        )
    for row_index, staleness in enumerate(STALENESS_LEVELS):
        center_y = plot_y + (row_index + 0.5) * cell_height
        elements.append(
            f'<text class="axis-label" x="{plot_x - 22:.2f}" y="{center_y + 5:.2f}" text-anchor="end">'
            f"max weight staleness {staleness}</text>"
        )
        for column, (trainer_nodes, rollout_nodes) in enumerate(NODE_RATIOS):
            summary = by_setting.get((staleness, trainer_nodes, rollout_nodes))
            cell_x = plot_x + column * cell_width
            cell_y = plot_y + row_index * cell_height
            if summary is None:
                fill, value_text, detail, status, steps = "#eeeeee", "not logged", "", "", ""
                text_fill = "#555"
            else:
                fill = _interpolate_color(summary.staleness_total_mean, maximum)
                value_text = f"{summary.staleness_total_mean:.3f}"
                detail = f"± {summary.staleness_total_std:.3f}"
                status = (
                    "settled"
                    if summary.settled
                    else f"still changing (trend {summary.slope_per_100_updates:+.2f}/100)"
                )
                steps = f"steps {summary.window_start_step}–{summary.window_end_step} (n={summary.observations})"
                text_fill = "white" if summary.staleness_total_mean / maximum > 0.55 else "#17212b"
            elements.extend(
                [
                    f'<rect x="{cell_x:.2f}" y="{cell_y:.2f}" width="{cell_width:.2f}" '
                    f'height="{cell_height:.2f}" fill="{fill}" stroke="white" stroke-width="4"/>',
                    f'<text class="cell-value" x="{cell_x + cell_width / 2:.2f}" y="{cell_y + 43:.2f}" '
                    f'text-anchor="middle" style="fill:{text_fill}">{value_text}</text>',
                    f'<text class="cell-detail" x="{cell_x + cell_width / 2:.2f}" y="{cell_y + 67:.2f}" '
                    f'text-anchor="middle" style="fill:{text_fill}">{detail}</text>',
                    f'<text class="tick" x="{cell_x + cell_width / 2:.2f}" y="{cell_y + 91:.2f}" '
                    f'text-anchor="middle" style="fill:{text_fill}">{status}</text>',
                    f'<text class="tick" x="{cell_x + cell_width / 2:.2f}" y="{cell_y + 112:.2f}" '
                    f'text-anchor="middle" style="fill:{text_fill}">{steps}</text>',
                ]
            )
    legend_y = plot_y + len(STALENESS_LEVELS) * cell_height + 62.0
    legend_x, legend_width, segments = plot_x + 135.0, 490.0, 7
    for index in range(segments):
        value = maximum * index / (segments - 1)
        x = legend_x + index * legend_width / segments
        elements.append(
            f'<rect x="{x:.2f}" y="{legend_y:.2f}" width="{legend_width / segments + 1:.2f}" '
            f'height="20" fill="{_interpolate_color(value, maximum)}"/>'
        )
    elements.extend(
        [
            f'<text class="tick" x="{legend_x:.2f}" y="{legend_y + 39:.2f}" text-anchor="middle">0</text>',
            f'<text class="tick" x="{legend_x + legend_width:.2f}" y="{legend_y + 39:.2f}" '
            f'text-anchor="middle">{maximum:.2f}</text>',
            f'<text class="axis-label" x="{legend_x + legend_width / 2:.2f}" y="{legend_y + 58:.2f}" '
            f'text-anchor="middle">steady-state training staleness mean</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def _polyline(points: list[tuple[float, float]], x_map: Any, y_map: Any) -> str:
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in points)


def _render_trajectories(points: list[StalenessPoint], *, rolling_window: int) -> str:
    width, height = 1660, 1020
    margin_x, panel_y = 105.0, 145.0
    panel_width, panel_height = 720.0, 350.0
    column_gap, row_gap = 75.0, 90.0
    max_step = max(point.training_step for point in points)
    max_value = max(max(point.staleness_total_mean, point.rolling_mean) for point in points)
    y_max = max(1.0, math.ceil(max_value * 1.08))
    elements = _svg_header(
        width,
        height,
        "Training staleness mean trajectories by train:rollout ratio",
        f"W&B {METRIC}; faint = per-update value, bold = trailing {rolling_window}-update mean",
    )
    legend_y = 96.0
    for index, staleness in enumerate(STALENESS_LEVELS):
        legend_x = 500.0 + index * 180.0
        color = STALENESS_COLORS[staleness]
        elements.extend(
            [
                f'<line x1="{legend_x:.2f}" y1="{legend_y:.2f}" x2="{legend_x + 34:.2f}" '
                f'y2="{legend_y:.2f}" stroke="{color}" stroke-width="3"/>',
                f'<text class="axis-label" x="{legend_x + 43:.2f}" y="{legend_y + 5:.2f}">s={staleness}</text>',
            ]
        )
    for panel_index, (trainer_nodes, rollout_nodes) in enumerate(NODE_RATIOS):
        column, row_index = panel_index % 2, panel_index // 2
        panel_x = margin_x + column * (panel_width + column_gap)
        current_y = panel_y + row_index * (panel_height + row_gap)
        plot_x, plot_y = panel_x + 70.0, current_y + 42.0
        plot_width, plot_height = panel_width - 92.0, panel_height - 95.0

        def x_map(value: float, base: float = plot_x, width: float = plot_width) -> float:
            return base + width * (value - 1) / max(max_step - 1, 1)

        def y_map(value: float, base: float = plot_y, height: float = plot_height) -> float:
            return base + height * (y_max - value) / y_max

        elements.extend(
            [
                f'<rect class="frame" x="{panel_x:.2f}" y="{current_y:.2f}" width="{panel_width:.2f}" '
                f'height="{panel_height:.2f}" rx="5"/>',
                f'<text class="panel-title" x="{panel_x + panel_width / 2:.2f}" y="{current_y + 25:.2f}" '
                f'text-anchor="middle">train:rollout nodes = {trainer_nodes}:{rollout_nodes}</text>',
            ]
        )
        for tick_index in range(6):
            value = y_max * tick_index / 5.0
            y = y_map(value)
            elements.extend(
                [
                    f'<line class="grid" x1="{plot_x:.2f}" y1="{y:.2f}" x2="{plot_x + plot_width:.2f}" y2="{y:.2f}"/>',
                    f'<text class="tick" x="{plot_x - 10:.2f}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>',
                ]
            )
        x_ticks = sorted({1, max_step, *range(50, max_step + 1, 50)})
        for value in x_ticks:
            x = x_map(float(value))
            elements.extend(
                [
                    f'<line class="grid" x1="{x:.2f}" y1="{plot_y:.2f}" x2="{x:.2f}" y2="{plot_y + plot_height:.2f}"/>',
                    f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_height + 20:.2f}" text-anchor="middle">{value}</text>',
                ]
            )
        elements.extend(
            [
                f'<line class="axis" x1="{plot_x:.2f}" y1="{plot_y:.2f}" x2="{plot_x:.2f}" y2="{plot_y + plot_height:.2f}"/>',
                f'<line class="axis" x1="{plot_x:.2f}" y1="{plot_y + plot_height:.2f}" '
                f'x2="{plot_x + plot_width:.2f}" y2="{plot_y + plot_height:.2f}"/>',
                f'<text class="axis-label" x="{plot_x + plot_width / 2:.2f}" y="{current_y + panel_height - 11:.2f}" '
                f'text-anchor="middle">optimizer update</text>',
                f'<text class="axis-label" transform="translate({panel_x + 18:.2f},{plot_y + plot_height / 2:.2f}) rotate(-90)" '
                f'text-anchor="middle">training staleness mean</text>',
            ]
        )
        for staleness in STALENESS_LEVELS:
            selected = sorted(
                (
                    point
                    for point in points
                    if point.trainer_nodes == trainer_nodes and point.max_weight_staleness == staleness
                ),
                key=lambda point: point.training_step,
            )
            if not selected:
                continue
            raw = [(float(point.training_step), point.staleness_total_mean) for point in selected]
            smooth = [(float(point.training_step), point.rolling_mean) for point in selected]
            color = STALENESS_COLORS[staleness]
            elements.extend(
                [
                    f'<polyline class="raw" stroke="{color}" points="{_polyline(raw, x_map, y_map)}"/>',
                    f'<polyline class="smooth" stroke="{color}" points="{_polyline(smooth, x_map, y_map)}"/>',
                ]
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _metric_bounds(metric_histories: dict[str, dict[int, float]], *, rolling_window: int) -> tuple[float, float]:
    values = [
        value
        for arm_history in metric_histories.values()
        for _, value in _rolling_metric_values(arm_history, rolling_window)
    ]
    if not values:
        return 0.0, 1.0
    lower, upper = min(values), max(values)
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1e-6)
    else:
        padding = (upper - lower) * 0.08
    return lower - padding, upper + padding


def _format_tick(value: float, span: float) -> str:
    if span < 0.01:
        return f"{value:.4f}"
    if span < 0.2:
        return f"{value:.3f}"
    if span < 5.0:
        return f"{value:.2f}"
    if span < 100.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _render_sensitive_metrics(
    metrics: list[SensitiveMetric],
    histories: dict[str, dict[str, dict[int, float]]],
    *,
    rolling_window: int,
) -> str:
    panel_width, panel_height = 475.0, 225.0
    panel_gap, row_gap = 25.0, 24.0
    plot_left, plot_top = 435.0, 168.0
    width = int(plot_left + len(NODE_RATIOS) * (panel_width + panel_gap) + 35)
    height = int(plot_top + len(metrics) * (panel_height + row_gap) + 50)
    max_step = max(
        step
        for metric_histories in histories.values()
        for arm_history in metric_histories.values()
        for step in arm_history
    )
    elements = _svg_header(
        width,
        height,
        "Staleness-associated training metrics over optimizer updates",
        f"Each curve is a trailing {rolling_window}-update W&B mean; selected when max |r| >= "
        f"{SENSITIVE_METRIC_THRESHOLD:.2f}; signed TIS is included as a cancellation reference",
    )
    for index, staleness in enumerate(STALENESS_LEVELS):
        legend_x = width / 2 - 360.0 + index * 190.0
        color = STALENESS_COLORS[staleness]
        elements.extend(
            [
                f'<line x1="{legend_x:.2f}" y1="94" x2="{legend_x + 36:.2f}" y2="94" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text class="axis-label" x="{legend_x + 45:.2f}" y="99">s={staleness}</text>',
            ]
        )
    for column, (trainer_nodes, rollout_nodes) in enumerate(NODE_RATIOS):
        center_x = plot_left + column * (panel_width + panel_gap) + panel_width / 2
        elements.append(
            f'<text class="panel-title" x="{center_x:.2f}" y="140" text-anchor="middle">'
            f"train:rollout nodes = {trainer_nodes}:{rollout_nodes}</text>"
        )
    for metric_index, selected_metric in enumerate(metrics):
        metric = selected_metric.metric
        row_y = plot_top + metric_index * (panel_height + row_gap)
        metric_histories = histories.get(metric, {})
        y_min, y_max = _metric_bounds(metric_histories, rolling_window=rolling_window)
        span = y_max - y_min
        center_y = row_y + panel_height / 2
        correlation_text = (
            "signed reference"
            if selected_metric.correlation is None
            else (
                f"r={selected_metric.correlation:+.3f} vs "
                f"{STALENESS_PREDICTOR_LABELS.get(selected_metric.strongest_predictor, selected_metric.strongest_predictor)}"
            )
        )
        elements.extend(
            [
                f'<text class="axis-label" x="{plot_left - 28:.2f}" y="{center_y - 17:.2f}" '
                f'text-anchor="end">{html.escape(selected_metric.label)}</text>',
                f'<text class="tick" x="{plot_left - 28:.2f}" y="{center_y + 5:.2f}" '
                f'text-anchor="end">{html.escape(metric)}</text>',
                f'<text class="tick" x="{plot_left - 28:.2f}" y="{center_y + 27:.2f}" '
                f'text-anchor="end">{correlation_text}</text>',
            ]
        )
        for column, (trainer_nodes, rollout_nodes) in enumerate(NODE_RATIOS):
            panel_x = plot_left + column * (panel_width + panel_gap)
            plot_x, plot_y = panel_x + 68.0, row_y + 18.0
            plot_width, plot_height = panel_width - 86.0, panel_height - 60.0

            def x_map(value: float, base: float = plot_x, width: float = plot_width) -> float:
                return base + width * (value - 1) / max(max_step - 1, 1)

            def y_map(
                value: float,
                base: float = plot_y,
                height: float = plot_height,
                maximum: float = y_max,
                value_span: float = span,
            ) -> float:
                return base + height * (maximum - value) / max(value_span, 1e-12)

            elements.append(
                f'<rect class="frame" x="{panel_x:.2f}" y="{row_y:.2f}" width="{panel_width:.2f}" '
                f'height="{panel_height:.2f}" rx="4"/>'
            )
            for tick_index in range(4):
                value = y_min + span * tick_index / 3.0
                y = y_map(value)
                elements.extend(
                    [
                        f'<line class="grid" x1="{plot_x:.2f}" y1="{y:.2f}" '
                        f'x2="{plot_x + plot_width:.2f}" y2="{y:.2f}"/>',
                        f'<text class="tick" x="{plot_x - 8:.2f}" y="{y + 4:.2f}" '
                        f'text-anchor="end">{_format_tick(value, span)}</text>',
                    ]
                )
            for step in sorted({1, max_step, *range(100, max_step + 1, 100)}):
                x = x_map(float(step))
                elements.extend(
                    [
                        f'<line class="grid" x1="{x:.2f}" y1="{plot_y:.2f}" '
                        f'x2="{x:.2f}" y2="{plot_y + plot_height:.2f}"/>',
                        f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_height + 19:.2f}" '
                        f'text-anchor="middle">{step}</text>',
                    ]
                )
            elements.extend(
                [
                    f'<line class="axis" x1="{plot_x:.2f}" y1="{plot_y:.2f}" '
                    f'x2="{plot_x:.2f}" y2="{plot_y + plot_height:.2f}"/>',
                    f'<line class="axis" x1="{plot_x:.2f}" y1="{plot_y + plot_height:.2f}" '
                    f'x2="{plot_x + plot_width:.2f}" y2="{plot_y + plot_height:.2f}"/>',
                    f'<text class="axis-label" x="{plot_x + plot_width / 2:.2f}" '
                    f'y="{row_y + panel_height - 4:.2f}" text-anchor="middle">optimizer update</text>',
                ]
            )
            for staleness in STALENESS_LEVELS:
                arm = f"s{staleness}-t{trainer_nodes}r{rollout_nodes}"
                values_by_step = metric_histories.get(arm, {})
                if not values_by_step:
                    continue
                curve = [
                    (float(step), value) for step, value in _rolling_metric_values(values_by_step, rolling_window)
                ]
                elements.append(
                    f'<polyline class="smooth" stroke="{STALENESS_COLORS[staleness]}" '
                    f'points="{_polyline(curve, x_map, y_map)}"/>'
                )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-history-csv", type=Path, required=True)
    parser.add_argument("--staleness-correlations-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steady-window", type=int, default=50)
    parser.add_argument("--rolling-window", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.steady_window <= 1 or args.rolling_window <= 1:
        raise ValueError("staleness windows must be greater than one")
    histories = _read_history(args.training_history_csv.resolve())
    trajectories = _trajectory_rows(histories, rolling_window=args.rolling_window)
    steady = _steady_state_rows(histories, steady_window=args.steady_window)
    sensitive_metrics = _select_sensitive_metrics(args.staleness_correlations_csv.resolve())
    metric_histories = _read_metric_histories(args.training_history_csv.resolve(), sensitive_metrics)
    expected_settings = len(STALENESS_LEVELS) * len(NODE_RATIOS)
    if len(steady) != expected_settings:
        raise ValueError(f"expected {expected_settings} async settings, found {len(steady)}")
    output_dir = args.output_dir.resolve()
    _atomic_write_csv(output_dir / "steady-state-staleness.csv", [asdict(row) for row in steady])
    _atomic_write_csv(output_dir / "staleness-mean-trajectories.csv", [asdict(row) for row in trajectories])
    _atomic_write_csv(
        output_dir / "staleness-sensitive-metrics.csv",
        [asdict(metric) for metric in sensitive_metrics],
    )
    figures_dir = output_dir / "figures"
    _atomic_write(figures_dir / "steady-state-staleness-grid.svg", _render_steady_grid(steady))
    _atomic_write(
        figures_dir / "staleness-mean-trajectories.svg",
        _render_trajectories(trajectories, rolling_window=args.rolling_window),
    )
    _atomic_write(
        figures_dir / "staleness-sensitive-training-metrics.svg",
        _render_sensitive_metrics(
            sensitive_metrics,
            metric_histories,
            rolling_window=args.rolling_window,
        ),
    )
    print(figures_dir)


if __name__ == "__main__":
    main()
