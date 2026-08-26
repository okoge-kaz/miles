#!/usr/bin/env python3
"""Render dependency-free SVG figures from reasoning-evaluation CSV results."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from experiments.tools.reasoning_eval.grid import reasoning_eval_grid_from_environment


EVALUATION_GRID = reasoning_eval_grid_from_environment()
STALENESS_LEVELS = EVALUATION_GRID.staleness_levels
NODE_RATIOS = EVALUATION_GRID.node_ratios
ALL_ARMS = EVALUATION_GRID.all_arms
SERIES_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
COLORS = {
    trainer_nodes: SERIES_PALETTE[index % len(SERIES_PALETTE)]
    for index, (trainer_nodes, _) in enumerate(NODE_RATIOS)
} | {0: "#222222"}
TASK_COLORS = {"aime24": "#4C78A8", "aime25": "#F58518", "aime26": "#54A24B"}


@dataclass(frozen=True)
class ResultRow:
    """One arm/step aggregate row."""

    arm: str
    max_weight_staleness: int
    trainer_nodes: int
    rollout_nodes: int
    training_step: int
    aime24: float | None
    aime25: float | None
    aime26: float | None
    macro_mean: float | None


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def _read_rows(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            rows.append(
                ResultRow(
                    arm=record["arm"],
                    max_weight_staleness=int(record["max_weight_staleness"]),
                    trainer_nodes=int(record["trainer_nodes"]),
                    rollout_nodes=int(record["rollout_nodes"]),
                    training_step=int(record["training_step"]),
                    aime24=_optional_float(record["aime24_percent"]),
                    aime25=_optional_float(record["aime25_percent"]),
                    aime26=_optional_float(record["aime26_percent"]),
                    macro_mean=_optional_float(record["aime_macro_mean_percent"]),
                )
            )
    return rows


def _score_bounds(values: Iterable[float]) -> tuple[float, float]:
    scores = list(values)
    if not scores:
        return 0.0, 100.0
    lower = max(0.0, math.floor((min(scores) - 5.0) / 10.0) * 10.0)
    upper = min(100.0, math.ceil((max(scores) + 5.0) / 10.0) * 10.0)
    if upper - lower < 20.0:
        midpoint = (upper + lower) / 2.0
        lower = max(0.0, midpoint - 10.0)
        upper = min(100.0, midpoint + 10.0)
    return lower, upper


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#555;stroke-width:1}"
        ".grid{stroke:#ddd;stroke-width:1}.series{fill:none;stroke-width:2.5}"
        ".tick{font-size:11px}.label{font-size:13px}.panel-title{font-size:16px;font-weight:bold}"
        ".title{font-size:22px;font-weight:bold}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{width / 2:.1f}" y="32" text-anchor="middle">{html.escape(title)}</text>',
    ]


def _panel_axes(
    *, x: float, y: float, width: float, height: float, min_step: int, max_step: int, y_bounds: tuple[float, float]
) -> tuple[list[str], Callable[[int], float], Callable[[float], float]]:
    left, right, top, bottom = 55.0, 15.0, 32.0, 42.0
    plot_x, plot_y = x + left, y + top
    plot_width, plot_height = width - left - right, height - top - bottom
    y_min, y_max = y_bounds

    def x_position(step: int) -> float:
        denominator = max(max_step - min_step, 1)
        return plot_x + plot_width * (step - min_step) / denominator

    def y_position(score: float) -> float:
        return plot_y + plot_height * (y_max - score) / (y_max - y_min)

    elements: list[str] = []
    for index in range(5):
        score = y_min + (y_max - y_min) * index / 4.0
        axis_y = y_position(score)
        elements.append(f'<line class="grid" x1="{plot_x}" y1="{axis_y:.2f}" x2="{plot_x + plot_width}" y2="{axis_y:.2f}"/>')
        elements.append(f'<text class="tick" x="{plot_x - 8}" y="{axis_y + 4:.2f}" text-anchor="end">{score:.0f}</text>')
    step_ticks = sorted({min_step, max_step, *range(((min_step + 49) // 50) * 50, max_step + 1, 50)})
    for step in step_ticks:
        axis_x = x_position(step)
        elements.append(f'<line class="grid" x1="{axis_x:.2f}" y1="{plot_y}" x2="{axis_x:.2f}" y2="{plot_y + plot_height}"/>')
        elements.append(f'<text class="tick" x="{axis_x:.2f}" y="{plot_y + plot_height + 18}" text-anchor="middle">{step}</text>')
    elements.extend(
        [
            f'<line class="axis" x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}"/>',
            f'<line class="axis" x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}"/>',
            f'<text class="label" x="{plot_x + plot_width / 2:.2f}" y="{y + height - 5}" text-anchor="middle">Training step</text>',
            f'<text class="label" x="{x + 14}" y="{plot_y + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 {x + 14} {plot_y + plot_height / 2:.2f})">Score (%)</text>',
        ]
    )
    return elements, x_position, y_position


def _line_elements(
    points: list[ResultRow],
    *,
    color: str,
    dashed: bool,
    x_position: Callable[[int], float],
    y_position: Callable[[float], float],
) -> list[str]:
    valid = sorted((row for row in points if row.macro_mean is not None), key=lambda row: row.training_step)
    if not valid:
        return []
    coordinates = " ".join(
        f"{x_position(row.training_step):.2f},{y_position(row.macro_mean):.2f}" for row in valid
    )
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    elements = [f'<polyline class="series" stroke="{color}"{dash} points="{coordinates}"/>']
    elements.extend(
        f'<circle cx="{x_position(row.training_step):.2f}" cy="{y_position(row.macro_mean):.2f}" r="2.8" fill="{color}"/>'
        for row in valid
    )
    return elements


def _render_learning_curves(rows: list[ResultRow]) -> str:
    width, height = 1240, 820
    panel_width, panel_height = 580, 340
    min_step = min((row.training_step for row in rows), default=10)
    max_step = max((row.training_step for row in rows), default=300)
    y_bounds = _score_bounds(row.macro_mean for row in rows if row.macro_mean is not None)
    elements = _svg_header(width, height, "AIME mean by staleness and node ratio")
    legend_y = 60
    for index, (train_nodes, rollout_nodes) in enumerate(NODE_RATIOS):
        legend_x = 210 + index * 180
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{COLORS[train_nodes]}" stroke-width="3"/>')
        elements.append(f'<text class="label" x="{legend_x + 36}" y="{legend_y + 5}">T:R={train_nodes}:{rollout_nodes}</text>')
    if EVALUATION_GRID.include_colocated:
        elements.append('<line x1="930" y1="60" x2="958" y2="60" stroke="#222" stroke-width="3" stroke-dasharray="7 5"/>')
        elements.append('<text class="label" x="966" y="65">colocated</text>')
    colocated = [row for row in rows if row.arm == "s0-colocated"]
    for panel_index, staleness in enumerate(STALENESS_LEVELS):
        panel_x = 35 + (panel_index % 2) * 600
        panel_y = 85 + (panel_index // 2) * 355
        elements.append(f'<text class="panel-title" x="{panel_x + panel_width / 2}" y="{panel_y + 18}" text-anchor="middle">max weight staleness = {staleness}</text>')
        axes, x_position, y_position = _panel_axes(
            x=panel_x,
            y=panel_y,
            width=panel_width,
            height=panel_height,
            min_step=min_step,
            max_step=max_step,
            y_bounds=y_bounds,
        )
        elements.extend(axes)
        for train_nodes, rollout_nodes in NODE_RATIOS:
            arm = f"s{staleness}-t{train_nodes}r{rollout_nodes}"
            arm_rows = [row for row in rows if row.arm == arm]
            elements.extend(
                _line_elements(
                    arm_rows,
                    color=COLORS[train_nodes],
                    dashed=False,
                    x_position=x_position,
                    y_position=y_position,
                )
            )
        elements.extend(
            _line_elements(
                colocated,
                color=COLORS[0],
                dashed=True,
                x_position=x_position,
                y_position=y_position,
            )
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _latest_complete_rows(rows: list[ResultRow]) -> list[ResultRow]:
    latest: list[ResultRow] = []
    for arm in ALL_ARMS:
        candidates = [
            row
            for row in rows
            if row.arm == arm and row.aime24 is not None and row.aime25 is not None and row.aime26 is not None
        ]
        if candidates:
            latest.append(max(candidates, key=lambda row: row.training_step))
    return latest


def _render_latest_scores(rows: list[ResultRow]) -> str:
    width, height = 1500, 720
    latest = _latest_complete_rows(rows)
    elements = _svg_header(width, height, "Latest complete AIME scores for each sweep arm")
    plot_x, plot_y, plot_width, plot_height = 70.0, 80.0, 1390.0, 500.0
    for score in range(0, 101, 20):
        axis_y = plot_y + plot_height * (100 - score) / 100
        elements.append(f'<line class="grid" x1="{plot_x}" y1="{axis_y}" x2="{plot_x + plot_width}" y2="{axis_y}"/>')
        elements.append(f'<text class="tick" x="{plot_x - 10}" y="{axis_y + 4}" text-anchor="end">{score}</text>')
    if not latest:
        elements.append(f'<text class="panel-title" x="{width / 2}" y="{height / 2}" text-anchor="middle">No complete three-task checkpoint results yet</text>')
        elements.append("</svg>")
        return "\n".join(elements) + "\n"
    group_width = plot_width / len(latest)
    bar_width = min(18.0, group_width / 4.0)
    for group_index, row in enumerate(latest):
        center = plot_x + group_width * (group_index + 0.5)
        for task_index, (task, score) in enumerate(
            (("aime24", row.aime24), ("aime25", row.aime25), ("aime26", row.aime26))
        ):
            bar_x = center + (task_index - 1) * bar_width - bar_width / 2
            bar_height = plot_height * score / 100.0
            elements.append(f'<rect x="{bar_x:.2f}" y="{plot_y + plot_height - bar_height:.2f}" width="{bar_width - 1:.2f}" height="{bar_height:.2f}" fill="{TASK_COLORS[task]}"/>')
        label_y = plot_y + plot_height + 18
        elements.append(f'<text class="tick" x="{center:.2f}" y="{label_y}" text-anchor="end" transform="rotate(-48 {center:.2f} {label_y})">{html.escape(row.arm)} @ {row.training_step}</text>')
    for index, task in enumerate(("aime24", "aime25", "aime26")):
        legend_x = 570 + index * 150
        elements.append(f'<rect x="{legend_x}" y="52" width="16" height="12" fill="{TASK_COLORS[task]}"/>')
        elements.append(f'<text class="label" x="{legend_x + 23}" y="63">{task.upper()}</text>')
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    aggregate_csv = args.aggregate_csv.resolve()
    output_dir = args.output_dir or aggregate_csv.parent / "figures"
    rows = _read_rows(aggregate_csv)
    _atomic_write(output_dir / "aime-mean-vs-step.svg", _render_learning_curves(rows))
    _atomic_write(output_dir / "latest-aime-by-arm.svg", _render_latest_scores(rows))
    obsolete_path = output_dir / "aime-macro-mean-vs-step.svg"
    if obsolete_path.exists():
        obsolete_path.unlink()
    print(output_dir)


if __name__ == "__main__":
    main()
