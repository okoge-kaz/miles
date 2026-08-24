#!/usr/bin/env python3
"""Render step, wall-clock, and staleness-analysis figures as SVG."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

STALENESS_LEVELS = (1, 2, 4, 8)
ASYNC_ARMS = tuple(
    f"s{staleness}-t{train_nodes}r{8 - train_nodes}" for staleness in STALENESS_LEVELS for train_nodes in (1, 2, 3, 4)
)
ALL_ARMS = (*ASYNC_ARMS, "s0-colocated")
TASK_FIELDS = {
    "AIME24": "aime24_percent",
    "AIME25": "aime25_percent",
    "AIME26": "aime26_percent",
    "Macro": "aime_macro_mean_percent",
}
TASK_COLORS = {
    "AIME24": "#4C78A8",
    "AIME25": "#F58518",
    "AIME26": "#54A24B",
    "Macro": "#222222",
}
RATIO_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7", 0: "#222222"}
STALENESS_FEATURES = (
    "staleness/total/mean",
    "staleness/total/variance",
    "staleness/total/std",
    "staleness/total/p90",
    "staleness/pre_queue/mean",
    "staleness/in_queue/mean",
)
SELECTED_ARM_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#999999",
)


@dataclass(frozen=True)
class SeriesRow:
    """One evaluated checkpoint joined to its training clock."""

    arm: str
    training_step: int
    active_wallclock_hours: float | None
    scores: dict[str, float | None]


@dataclass(frozen=True)
class CorrelationRow:
    """One correlation estimate from the analysis CSV."""

    predictor: str
    outcome: str
    observations: int
    correlation: float | None
    ci_low: float | None
    ci_high: float | None


@dataclass(frozen=True)
class DecompositionRow:
    """Arm-level dQ/dt factorization over evaluated adjacent intervals."""

    arm: str
    trainer_nodes: int
    macro_points_per_update: float
    updates_per_active_hour: float
    macro_points_per_active_hour: float


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_series(path: Path) -> list[SeriesRow]:
    rows: list[SeriesRow] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            rows.append(
                SeriesRow(
                    arm=record["arm"],
                    training_step=int(record["training_step"]),
                    active_wallclock_hours=_optional_float(record.get("active_wallclock_hours")),
                    scores={label: _optional_float(record.get(field)) for label, field in TASK_FIELDS.items()},
                )
            )
    return rows


def _read_correlations(path: Path) -> list[CorrelationRow]:
    rows: list[CorrelationRow] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            rows.append(
                CorrelationRow(
                    predictor=record["predictor"],
                    outcome=record["outcome"],
                    observations=int(record["observations"]),
                    correlation=_optional_float(record.get("correlation")),
                    ci_low=_optional_float(record.get("ci_low")),
                    ci_high=_optional_float(record.get("ci_high")),
                )
            )
    return rows


def _trainer_nodes(arm: str) -> int:
    if arm == "s0-colocated":
        return 0
    return int(arm.split("-t", 1)[1].split("r", 1)[0])


def _read_decomposition(path: Path) -> list[DecompositionRow]:
    rows: list[DecompositionRow] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            rows.append(
                DecompositionRow(
                    arm=record["arm"],
                    trainer_nodes=_trainer_nodes(record["arm"]),
                    macro_points_per_update=float(record["macro_points_per_update"]),
                    updates_per_active_hour=float(record["updates_per_active_hour"]),
                    macro_points_per_active_hour=float(record["macro_points_per_active_hour"]),
                )
            )
    return rows


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _svg_header(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#555;stroke-width:1}"
        ".grid{stroke:#ddd;stroke-width:1}.series{fill:none;stroke-width:2.2}"
        ".tick{font-size:11px}.label{font-size:13px}.panel-title{font-size:15px;font-weight:bold}"
        ".title{font-size:22px;font-weight:bold}.subtitle{font-size:13px;fill:#555}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{width / 2:.1f}" y="30" text-anchor="middle">{html.escape(title)}</text>',
    ]
    if subtitle:
        elements.append(
            f'<text class="subtitle" x="{width / 2:.1f}" y="52" text-anchor="middle">'
            f"{html.escape(subtitle)}</text>"
        )
    return elements


def _score_bounds(values: Iterable[float]) -> tuple[float, float]:
    scores = list(values)
    if not scores:
        return 0.0, 100.0
    lower = max(0.0, math.floor((min(scores) - 3.0) / 5.0) * 5.0)
    upper = min(100.0, math.ceil((max(scores) + 3.0) / 5.0) * 5.0)
    if upper - lower < 20.0:
        midpoint = (lower + upper) / 2.0
        return max(0.0, midpoint - 10.0), min(100.0, midpoint + 10.0)
    return lower, upper


def _numeric_bounds(values: Iterable[float], *, minimum: float = 0.0) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    maximum = max(finite, default=1.0)
    magnitude = 10 ** math.floor(math.log10(max(maximum, 1e-9)))
    interval = magnitude if maximum / magnitude <= 5 else 2 * magnitude
    upper = max(interval, math.ceil(maximum / interval) * interval)
    return minimum, upper


def _panel_axes(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    x_label: str,
) -> tuple[list[str], Callable[[float], float], Callable[[float], float]]:
    left, right, top, bottom = 48.0, 12.0, 30.0, 38.0
    plot_x, plot_y = x + left, y + top
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    def x_position(value: float) -> float:
        return plot_x + plot_width * (value - x_min) / max(x_max - x_min, 1e-12)

    def y_position(value: float) -> float:
        return plot_y + plot_height * (y_max - value) / max(y_max - y_min, 1e-12)

    elements: list[str] = []
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4.0
        axis_y = y_position(value)
        elements.append(
            f'<line class="grid" x1="{plot_x}" y1="{axis_y:.2f}" ' f'x2="{plot_x + plot_width}" y2="{axis_y:.2f}"/>'
        )
        elements.append(
            f'<text class="tick" x="{plot_x - 6}" y="{axis_y + 4:.2f}" ' f'text-anchor="end">{value:.0f}</text>'
        )
    for index in range(5):
        value = x_min + (x_max - x_min) * index / 4.0
        axis_x = x_position(value)
        label = f"{value:.0f}" if x_max >= 20.0 else f"{value:.1f}"
        elements.append(
            f'<line class="grid" x1="{axis_x:.2f}" y1="{plot_y}" ' f'x2="{axis_x:.2f}" y2="{plot_y + plot_height}"/>'
        )
        elements.append(
            f'<text class="tick" x="{axis_x:.2f}" y="{plot_y + plot_height + 16}" '
            f'text-anchor="middle">{label}</text>'
        )
    elements.extend(
        [
            f'<line class="axis" x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}"/>',
            f'<line class="axis" x1="{plot_x}" y1="{plot_y + plot_height}" '
            f'x2="{plot_x + plot_width}" y2="{plot_y + plot_height}"/>',
            f'<text class="label" x="{plot_x + plot_width / 2:.2f}" y="{y + height - 2}" '
            f'text-anchor="middle">{html.escape(x_label)}</text>',
        ]
    )
    return elements, x_position, y_position


def _line_elements(
    points: Iterable[tuple[float, float]],
    *,
    color: str,
    x_position: Callable[[float], float],
    y_position: Callable[[float], float],
    dashed: bool = False,
) -> list[str]:
    valid = sorted(points)
    if not valid:
        return []
    coordinates = " ".join(f"{x_position(x):.2f},{y_position(y):.2f}" for x, y in valid)
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    elements = [f'<polyline class="series" stroke="{color}"{dash} points="{coordinates}"/>']
    elements.extend(
        f'<circle cx="{x_position(x):.2f}" cy="{y_position(y):.2f}" r="2.5" fill="{color}"/>' for x, y in valid
    )
    return elements


def _legend(entries: Iterable[tuple[str, str, bool]], *, y: float, center_x: float, spacing: float) -> list[str]:
    values = list(entries)
    start_x = center_x - spacing * (len(values) - 1) / 2.0
    elements: list[str] = []
    for index, (label, color, dashed) in enumerate(values):
        x = start_x + index * spacing
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        elements.append(f'<line x1="{x}" y1="{y}" x2="{x + 25}" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>')
        elements.append(f'<text class="label" x="{x + 32}" y="{y + 5}">{html.escape(label)}</text>')
    return elements


def _x_value(row: SeriesRow, attribute: str) -> float | None:
    if attribute == "training_step":
        return float(row.training_step)
    if attribute == "active_wallclock_hours":
        return row.active_wallclock_hours
    raise ValueError(f"unsupported x attribute: {attribute}")


def _render_arm_score_panels(
    rows: list[SeriesRow],
    *,
    arms: tuple[str, ...],
    x_attribute: str,
    x_label: str,
    title: str,
    subtitle: str,
) -> str:
    columns = min(4, max(1, len(arms)))
    panel_rows = math.ceil(len(arms) / columns)
    panel_width, panel_height = 425, 300
    width, height = columns * panel_width + 30, panel_rows * panel_height + 105
    selected_rows = [row for row in rows if row.arm in arms]
    x_values = [value for row in selected_rows if (value := _x_value(row, x_attribute)) is not None]
    x_bounds = _numeric_bounds(x_values)
    score_values = [score for row in selected_rows for score in row.scores.values() if score is not None]
    y_bounds = _score_bounds(score_values)
    elements = _svg_header(width, height, title, subtitle)
    elements.extend(
        _legend(
            ((label, TASK_COLORS[label], label == "Macro") for label in TASK_FIELDS),
            y=73,
            center_x=width / 2,
            spacing=145,
        )
    )
    for index, arm in enumerate(arms):
        panel_x = 15 + index % columns * panel_width
        panel_y = 92 + index // columns * panel_height
        elements.append(
            f'<text class="panel-title" x="{panel_x + panel_width / 2:.2f}" '
            f'y="{panel_y + 18}" text-anchor="middle">{html.escape(arm)}</text>'
        )
        axes, x_position, y_position = _panel_axes(
            x=panel_x,
            y=panel_y,
            width=panel_width,
            height=panel_height - 8,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            x_label=x_label,
        )
        elements.extend(axes)
        arm_rows = [row for row in selected_rows if row.arm == arm]
        for label in TASK_FIELDS:
            points = [
                (x_value, score)
                for row in arm_rows
                if (x_value := _x_value(row, x_attribute)) is not None and (score := row.scores[label]) is not None
            ]
            elements.extend(
                _line_elements(
                    points,
                    color=TASK_COLORS[label],
                    dashed=label == "Macro",
                    x_position=x_position,
                    y_position=y_position,
                )
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _render_wallclock_macro(rows: list[SeriesRow]) -> str:
    width, height = 1320, 850
    panel_width, panel_height = 630, 360
    valid = [row for row in rows if row.active_wallclock_hours is not None and row.scores["Macro"] is not None]
    x_bounds = _numeric_bounds(row.active_wallclock_hours for row in valid if row.active_wallclock_hours is not None)
    y_bounds = _score_bounds(row.scores["Macro"] for row in valid if row.scores["Macro"] is not None)
    elements = _svg_header(
        width,
        height,
        "AIME macro mean versus active training wall-clock",
        "Active wall-clock sums selected-lineage perf/step_time and excludes scheduler requeue gaps",
    )
    entries = [(f"T:R={train}:{8 - train}", RATIO_COLORS[train], False) for train in (1, 2, 3, 4)]
    entries.append(("colocated", RATIO_COLORS[0], True))
    elements.extend(_legend(entries, y=72, center_x=width / 2, spacing=190))
    for index, staleness in enumerate(STALENESS_LEVELS):
        panel_x = 25 + index % 2 * 645
        panel_y = 90 + index // 2 * 370
        elements.append(
            f'<text class="panel-title" x="{panel_x + panel_width / 2:.2f}" '
            f'y="{panel_y + 18}" text-anchor="middle">max weight staleness = {staleness}</text>'
        )
        axes, x_position, y_position = _panel_axes(
            x=panel_x,
            y=panel_y,
            width=panel_width,
            height=panel_height,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            x_label="Active training wall-clock (hours)",
        )
        elements.extend(axes)
        for train_nodes in (1, 2, 3, 4):
            arm = f"s{staleness}-t{train_nodes}r{8 - train_nodes}"
            points = [
                (row.active_wallclock_hours, row.scores["Macro"])
                for row in valid
                if row.arm == arm and row.active_wallclock_hours is not None and row.scores["Macro"] is not None
            ]
            elements.extend(
                _line_elements(
                    points,
                    color=RATIO_COLORS[train_nodes],
                    x_position=x_position,
                    y_position=y_position,
                )
            )
        colocated = [
            (row.active_wallclock_hours, row.scores["Macro"])
            for row in valid
            if row.arm == "s0-colocated" and row.active_wallclock_hours is not None and row.scores["Macro"] is not None
        ]
        elements.extend(
            _line_elements(
                colocated,
                color=RATIO_COLORS[0],
                dashed=True,
                x_position=x_position,
                y_position=y_position,
            )
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _short_metric(metric: str) -> str:
    replacements = {
        "train/policy_rollout_": "policy–rollout ",
        "train/": "",
        "rollout/fully_async/": "",
        "rollout/": "",
        "throughput/": "",
        "staleness/": "staleness ",
        "perf/": "",
        "_": " ",
    }
    label = metric
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label


def _correlation_x(value: float, *, plot_x: float, plot_width: float, bound: float) -> float:
    return plot_x + plot_width * (value + bound) / (2.0 * bound)


def _correlation_grid(
    *, plot_x: float, plot_y: float, plot_width: float, plot_height: float, bound: float
) -> list[str]:
    elements: list[str] = []
    for index in range(5):
        value = -bound + 2.0 * bound * index / 4.0
        x = _correlation_x(value, plot_x=plot_x, plot_width=plot_width, bound=bound)
        css_class = "axis" if index == 2 else "grid"
        elements.append(
            f'<line class="{css_class}" x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_height}"/>'
        )
        elements.append(
            f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_height + 18}" text-anchor="middle">{value:.2f}</text>'
        )
    return elements


def _correlation_bar(
    *, value: float, y: float, height: float, color: str, plot_x: float, plot_width: float, bound: float
) -> list[str]:
    zero = _correlation_x(0.0, plot_x=plot_x, plot_width=plot_width, bound=bound)
    endpoint = _correlation_x(value, plot_x=plot_x, plot_width=plot_width, bound=bound)
    bar_x = min(zero, endpoint)
    bar_width = max(abs(endpoint - zero), 0.8)
    label_x = endpoint + (5 if value >= 0.0 else -5)
    anchor = "start" if value >= 0.0 else "end"
    return [
        f'<rect x="{bar_x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height:.2f}" fill="{color}"/>',
        f'<text class="tick" x="{label_x:.2f}" y="{y + height - 1:.2f}" text-anchor="{anchor}">{value:+.2f}</text>',
    ]


def _render_staleness_metric_correlations(rows: list[CorrelationRow]) -> str:
    predictor_colors = (
        ("staleness/total/mean", "#4C78A8"),
        ("staleness/total/variance", "#F58518"),
    )
    predictors = tuple(predictor for predictor, _ in predictor_colors)
    lookup = {(row.predictor, row.outcome): row.correlation for row in rows}
    outcomes = sorted(
        {row.outcome for row in rows},
        key=lambda outcome: max(abs(lookup.get((predictor, outcome)) or 0.0) for predictor in predictors),
        reverse=True,
    )[:14]
    width, row_height = 1540, 48
    height = 150 + row_height * len(outcomes)
    plot_x, plot_y, plot_width = 455.0, 100.0, 1000.0
    plot_height = row_height * len(outcomes)
    elements = _svg_header(
        width,
        height,
        "Metrics most associated with realized training-data staleness",
        "Fixed-effect Pearson r after centering within the same update and trainer:rollout ratio",
    )
    elements.extend(
        _legend(
            (("total mean", "#4C78A8", False), ("total variance", "#F58518", False)),
            y=73,
            center_x=width / 2,
            spacing=210,
        )
    )
    elements.extend(
        _correlation_grid(
            plot_x=plot_x,
            plot_y=plot_y,
            plot_width=plot_width,
            plot_height=plot_height,
            bound=1.0,
        )
    )
    for index, outcome in enumerate(outcomes):
        center_y = plot_y + (index + 0.5) * row_height
        elements.append(
            f'<text class="label" x="{plot_x - 18}" y="{center_y + 4:.2f}" '
            f'text-anchor="end">{html.escape(_short_metric(outcome))}</text>'
        )
        for offset, (predictor, color) in enumerate(predictor_colors):
            value = lookup.get((predictor, outcome))
            if value is None:
                continue
            elements.extend(
                _correlation_bar(
                    value=value,
                    y=center_y - 13 + offset * 14,
                    height=11,
                    color=color,
                    plot_x=plot_x,
                    plot_width=plot_width,
                    bound=1.0,
                )
            )
    elements.append(
        f'<text class="label" x="{plot_x + plot_width / 2:.2f}" y="{height - 10}" '
        'text-anchor="middle">Correlation with realized staleness</text>'
    )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _render_downstream_correlations(rows: list[CorrelationRow]) -> str:
    outcome_colors = (
        ("delta_aime24", "#4C78A8"),
        ("delta_aime25", "#F58518"),
        ("delta_aime26", "#54A24B"),
        ("delta_macro", "#222222"),
    )
    lookup = {(row.predictor, row.outcome): row.correlation for row in rows}
    observed = [abs(value) for value in lookup.values() if value is not None]
    bound = max(0.1, math.ceil(max(observed, default=0.1) * 10.0) / 10.0)
    width, group_height = 1540, 100
    height = 155 + group_height * len(STALENESS_FEATURES)
    plot_x, plot_y, plot_width = 430.0, 105.0, 1020.0
    plot_height = group_height * len(STALENESS_FEATURES)
    elements = _svg_header(
        width,
        height,
        "Realized staleness versus ten-update AIME improvement",
        "Centered within the same ending step and trainer:rollout ratio; correlation is not causation",
    )
    elements.extend(
        _legend(
            ((label.replace("delta_", "").upper(), color, False) for label, color in outcome_colors),
            y=75,
            center_x=width / 2,
            spacing=170,
        )
    )
    elements.extend(
        _correlation_grid(
            plot_x=plot_x,
            plot_y=plot_y,
            plot_width=plot_width,
            plot_height=plot_height,
            bound=bound,
        )
    )
    for group_index, predictor in enumerate(STALENESS_FEATURES):
        center_y = plot_y + (group_index + 0.5) * group_height
        elements.append(
            f'<text class="label" x="{plot_x - 18}" y="{center_y + 4:.2f}" '
            f'text-anchor="end">{html.escape(predictor.replace("staleness/", ""))}</text>'
        )
        for outcome_index, (outcome, color) in enumerate(outcome_colors):
            value = lookup.get((predictor, outcome))
            if value is None:
                continue
            elements.extend(
                _correlation_bar(
                    value=value,
                    y=center_y - 34 + outcome_index * 17,
                    height=13,
                    color=color,
                    plot_x=plot_x,
                    plot_width=plot_width,
                    bound=bound,
                )
            )
    elements.append(
        f'<text class="label" x="{plot_x + plot_width / 2:.2f}" y="{height - 10}" '
        'text-anchor="middle">Correlation with AIME score change</text>'
    )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _factor_bounds(values: Iterable[float], *, signed: bool) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not signed:
        return _numeric_bounds(finite)
    _, bound = _numeric_bounds(abs(value) for value in finite)
    return -bound, bound


def _factor_x(value: float, *, plot_x: float, plot_width: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return plot_x + plot_width * (value - lower) / max(upper - lower, 1e-12)


def _factor_grid(
    *, plot_x: float, plot_y: float, plot_width: float, plot_height: float, bounds: tuple[float, float]
) -> list[str]:
    lower, upper = bounds
    elements: list[str] = []
    for index in range(5):
        value = lower + (upper - lower) * index / 4.0
        x = _factor_x(value, plot_x=plot_x, plot_width=plot_width, bounds=bounds)
        css_class = "axis" if abs(value) < 1e-12 else "grid"
        elements.append(
            f'<line class="{css_class}" x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" y2="{plot_y + plot_height}"/>'
        )
        elements.append(
            f'<text class="tick" x="{x:.2f}" y="{plot_y + plot_height + 18}" '
            f'text-anchor="middle">{value:.3g}</text>'
        )
    return elements


def _factor_bar(
    *,
    value: float,
    y: float,
    color: str,
    plot_x: float,
    plot_width: float,
    bounds: tuple[float, float],
) -> list[str]:
    zero = _factor_x(0.0, plot_x=plot_x, plot_width=plot_width, bounds=bounds)
    endpoint = _factor_x(value, plot_x=plot_x, plot_width=plot_width, bounds=bounds)
    x = min(zero, endpoint)
    width = max(abs(endpoint - zero), 0.8)
    label_x = endpoint + (5.0 if value >= 0.0 else -5.0)
    anchor = "start" if value >= 0.0 else "end"
    return [
        f'<rect x="{x:.2f}" y="{y - 9:.2f}" width="{width:.2f}" height="18" fill="{color}"/>',
        f'<text class="tick" x="{label_x:.2f}" y="{y + 4:.2f}" text-anchor="{anchor}">{value:+.3g}</text>',
    ]


def _render_wallclock_decomposition(rows: list[DecompositionRow]) -> str:
    ordered = sorted(rows, key=lambda row: ALL_ARMS.index(row.arm))
    row_height = 40
    plot_y = 125.0
    plot_height = row_height * len(ordered)
    width, height = 1900, int(plot_y + plot_height + 70)
    panel_width = 500.0
    panel_xs = (270.0, 825.0, 1380.0)
    metrics = (
        ("macro_points_per_update", "dQ/dU", "Macro points per update", True),
        ("updates_per_active_hour", "dU/dt", "Updates per active hour", False),
        ("macro_points_per_active_hour", "dQ/dt", "Macro points per active hour", True),
    )
    elements = _svg_header(
        width,
        height,
        "Learning-effect and throughput decomposition by setting",
        "dQ/dt = (dQ/dU) × (dU/dt), aggregated over adjacent evaluated ten-update intervals",
    )
    elements.extend(
        _legend(
            ((f"T:R={trainer}:{8 - trainer}", RATIO_COLORS[trainer], False) for trainer in (1, 2, 3, 4)),
            y=78,
            center_x=width / 2,
            spacing=190,
        )
    )
    for panel_x, (attribute, symbol, label, signed) in zip(panel_xs, metrics, strict=True):
        values = [float(getattr(row, attribute)) for row in ordered]
        bounds = _factor_bounds(values, signed=signed)
        elements.append(
            f'<text class="panel-title" x="{panel_x + panel_width / 2:.2f}" y="108" '
            f'text-anchor="middle">{html.escape(symbol)} — {html.escape(label)}</text>'
        )
        elements.extend(
            _factor_grid(
                plot_x=panel_x,
                plot_y=plot_y,
                plot_width=panel_width,
                plot_height=plot_height,
                bounds=bounds,
            )
        )
        for index, row in enumerate(ordered):
            center_y = plot_y + (index + 0.5) * row_height
            if panel_x == panel_xs[0]:
                elements.append(
                    f'<text class="label" x="{panel_x - 16}" y="{center_y + 4:.2f}" '
                    f'text-anchor="end">{html.escape(row.arm)}</text>'
                )
            elements.extend(
                _factor_bar(
                    value=float(getattr(row, attribute)),
                    y=center_y,
                    color=RATIO_COLORS[row.trainer_nodes],
                    plot_x=panel_x,
                    plot_width=panel_width,
                    bounds=bounds,
                )
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _selected_arms(groups: list[dict[str, object]]) -> tuple[str, ...]:
    selected: set[str] = set()
    for group in groups:
        selected.update(str(arm) for arm in group.get("low_arms", []))
        selected.update(str(arm) for arm in group.get("high_arms", []))
    return tuple(arm for arm in ALL_ARMS if arm in selected)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-series-csv", type=Path, required=True)
    parser.add_argument("--downstream-correlations-csv", type=Path, required=True)
    parser.add_argument("--staleness-correlations-csv", type=Path, required=True)
    parser.add_argument("--wallclock-decomposition-csv", type=Path, required=True)
    parser.add_argument("--selected-relationships-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    series = _read_series(args.checkpoint_series_csv.resolve())
    downstream = _read_correlations(args.downstream_correlations_csv.resolve())
    staleness = _read_correlations(args.staleness_correlations_csv.resolve())
    decomposition = _read_decomposition(args.wallclock_decomposition_csv.resolve())
    selected_groups = json.loads(args.selected_relationships_json.resolve().read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    figures = {
        "scores-vs-training-step-by-arm.svg": _render_arm_score_panels(
            series,
            arms=ALL_ARMS,
            x_attribute="training_step",
            x_label="Training step",
            title="AIME score trajectory for every sweep setting",
            subtitle="Each point is a complete or partially complete ten-step checkpoint evaluation",
        ),
        "scores-vs-active-wallclock-by-arm.svg": _render_arm_score_panels(
            series,
            arms=ALL_ARMS,
            x_attribute="active_wallclock_hours",
            x_label="Active wall-clock (hours)",
            title="AIME score trajectory versus active wall-clock for every setting",
            subtitle="Active wall-clock excludes scheduler requeue gaps",
        ),
        "macro-vs-active-wallclock-by-staleness.svg": _render_wallclock_macro(series),
        "learning-throughput-decomposition.svg": _render_wallclock_decomposition(decomposition),
        "staleness-metric-correlations.svg": _render_staleness_metric_correlations(staleness),
        "staleness-downstream-correlations.svg": _render_downstream_correlations(downstream),
    }
    for filename, content in figures.items():
        _atomic_write(output_dir / filename, content)
    selected_path = output_dir / "selected-downstream-trajectories.svg"
    arms = _selected_arms(selected_groups)
    if arms:
        _atomic_write(
            selected_path,
            _render_arm_score_panels(
                series,
                arms=arms,
                x_attribute="training_step",
                x_label="Training step",
                title="AIME trajectories selected by robust staleness relationships",
                subtitle="Only arms at the low/high extremes of relationships passing the preregistered threshold",
            ),
        )
    elif selected_path.exists():
        selected_path.unlink()
    print(output_dir)


if __name__ == "__main__":
    main()
