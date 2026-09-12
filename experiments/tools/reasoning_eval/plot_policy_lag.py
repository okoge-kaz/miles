#!/usr/bin/env python3
"""Plot schema-v1 policy lag without changing training or inventing missing data.

Delta RMS is sqrt(sum(delta**2) / response_token_count), not KL. Sensitivity
RMS uses the logger's fixed-reference pre/post coefficients, not parameter
gradient norms. Retention is a ratio of additive sums, not a ratio of RMSs.
This tool draws exact step-level values: no rank/run/RMS averaging, smoothing,
extrapolation, or interpolation across missing steps. Independent reruns keep
their namespace identity. Rollout diagnostics join train diagnostics only at
the same zero-based logged step; the displayed step is that index plus one.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "experiments/outputs/reasoning_eval/wandb-collapse-20260901"
DEFAULT_TRAINING = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/"
    "checkpoints/training/math/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/"
    "Qwen3-4B-Base-LR2e-5-Step4000/grpo-clip0.2-0.28-tis2.0/async/off-policy"
)
PREFIX = "policy_lag/"
POPULATIONS = ("all_response", "truncated", "non_truncated")
TREATMENTS = ("none", "zero-loss", "staleness-aware")
TITLES = ("No truncation treatment", "Zero loss on truncated", "Staleness-aware loss")
PALETTE = {8: "#A68F00", 12: "#999933", 16: "#E69F00", 24: "#CC79A7",
           28: "#6A3D9A", 32: "#0072B2", 40: "#009E73"}
COMPARATORS = ("staleness/total/mean", "rollout/raw_reward")
WEIGHTED = "loss_sensitivity_weighted_delta_rms_"
RETAINED = "loss_sensitivity_delta_sq_retained_fraction"


@dataclass(frozen=True)
class Cohort:
    namespace: str
    treatment: str
    queue_size: int
    staleness: tuple[int, ...]


COHORTS = (
    Cohort("hiso-policy-lag-v1-20260908-r1-no-treatment", "none", 6000, (8, 12, 16)),
    Cohort("hiso-policy-lag-v1-20260908-r1-zero-loss", "zero-loss", 6000, (8, 12, 16)),
    Cohort("zero-loss-trunc-s24-28-t1r7-step300-tbq6000-20260910-v1", "zero-loss", 6000, (24, 28)),
    Cohort("zero-loss-trunc-s32-40-t1r7-step300-tbq8000-20260908-v1", "zero-loss", 8000, (32, 40)),
    Cohort("staleness-aware-safe4-t1r7-cb1c041f", "staleness-aware", 6000, (28,)),
    Cohort("staleness-aware-loss-safe4-s32-40-t1r7-step300-tbq8000-20260908-v1",
           "staleness-aware", 8000, (32, 40)),
)


@dataclass(frozen=True)
class Run:
    cohort: Cohort
    staleness: int
    path: Path
    rows: dict[int, dict[str, float]]
    incomplete_tail_lines: int = 0


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def read_rows(path: Path) -> tuple[dict[int, dict[str, float]], int]:
    """Last appended value wins at a replayed step; never fill a missing step."""
    rows: dict[int, dict[str, float]] = {}
    incomplete = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not any(key in line for key in (PREFIX, *COMPARATORS)):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    incomplete += 1
                    continue
                raise
            axis = record.get("step_key")
            step = record.get("step")
            if axis not in ("train/step", "rollout/step") or not finite(step) or int(step) != step:
                continue
            values = {}
            for key, value in record.get("metrics", {}).items():
                if (axis == "train/step" and key.startswith(PREFIX)) or (
                    axis == "rollout/step" and key in COMPARATORS
                ):
                    values[key] = float(value) if finite(value) else math.nan
            rows.setdefault(int(step) + 1, {}).update(values)
    return {
        step: values for step, values in rows.items()
        if any(key.startswith(PREFIX) for key in values)
    }, incomplete


def discover_runs(root: Path) -> list[Run]:
    runs = []
    seen = set()
    for path in sorted(root.glob("*/*/dump/dashboard/metrics.jsonl")):
        name = path.parents[2].name
        match = re.match(r"s(\d+)-t1r7-(.+)", name)
        if match is None:
            continue
        age = int(match[1])
        cohort = next((c for c in COHORTS if match[2].startswith(c.namespace + "-")), None)
        if cohort is None or age not in cohort.staleness:
            continue
        if not name.endswith(f"-tbq{cohort.queue_size}"):
            raise ValueError(f"unexpected queue-size identity: {path}")
        identity = cohort.namespace, age
        if identity in seen:
            raise ValueError(f"ambiguous duplicate run: {identity}")
        seen.add(identity)
        rows, incomplete = read_rows(path)
        if rows:
            runs.append(Run(cohort, age, path, rows, incomplete))
    return runs


def measurement(row: dict[str, float], metric: str, population: str = "all_response") -> float:
    if metric in COMPARATORS:
        return row.get(metric, math.nan)
    base = f"{PREFIX}{population}/"
    value = row.get(base + metric, math.nan)
    if not finite(value) or not row.get(base + "response_token_count", 0) > 0:
        return math.nan
    if row.get(base + "nonfinite_delta_count") != 0:
        return math.nan
    if metric.startswith("loss_sensitivity"):
        if row.get(PREFIX + "loss_sensitivity_supported") != 1:
            return math.nan
        stages = (metric.rsplit("_", 1)[-1],) if metric.startswith(WEIGHTED) else ("pre", "post")
        for stage in stages:
            if row.get(base + f"nonfinite_loss_sensitivity_count_{stage}") != 0:
                return math.nan
            if metric.startswith(WEIGHTED) and row.get(
                base + f"loss_sensitivity_weighted_metrics_valid_{stage}"
            ) != 1:
                return math.nan
    return value


def segments(points: list[tuple[int, float]]) -> list[list[tuple[int, float]]]:
    """Break curves at invalid values or absent steps, including M_post=0 RMS."""
    result: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for step, value in sorted(points):
        if current and (not finite(value) or step != current[-1][0] + 1):
            result.append(current)
            current = []
        if finite(value):
            current.append((step, value))
    if current:
        result.append(current)
    return result


def upper_bound(values: list[float]) -> float:
    largest = max((v for v in values if finite(v)), default=0)
    if largest <= 0:
        return 1.0
    target = largest * 1.06
    power = 10 ** math.floor(math.log10(target))
    return next(scale * power for scale in (1, 2, 2.5, 3, 4, 5, 6, 8, 10) if scale * power >= target)


def canvas(width: int, height: int, runs: list[Run], labels: tuple[str, str] | None = None) -> list[str]:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222;font-size:15px}'
             '.heading{font-size:19px;font-weight:600}.title{font-size:17px}'
             '.grid{stroke:#ddd;stroke-width:.7}.curve{fill:none;stroke-width:1.7}'
             '.axis{stroke:#333;stroke-width:1}</style><rect width="100%" height="100%" fill="white"/>']
    ages = sorted({run.staleness for run in runs})
    legend_width = 72 * len(ages) + (260 if labels else 0)
    start = (width - legend_width) / 2
    for index, age in enumerate(ages):
        x = start + index * 72
        parts.append(f'<line x1="{x}" x2="{x + 26}" y1="17" y2="17" stroke="{PALETTE[age]}" stroke-width="2.5"/>'
                     f'<text x="{x + 33}" y="22">{age}</text>')
    if labels:
        for index, label in enumerate(labels):
            x = start + len(ages) * 72 + index * 130
            dash = ' stroke-dasharray="6 4"' if index == 0 else ''
            parts.append(f'<line x1="{x}" x2="{x + 26}" y1="17" y2="17" stroke="#333"{dash}/>'
                         f'<text x="{x + 33}" y="22">{html.escape(label)}</text>')
    return parts


def axes(parts: list[str], column: int, row: int, title: str, x_max: float, y_max: float,
         *, x_label: str = "Training step", tick_count: int = 4):
    left, top = column * 420 + 62, row * 240 + 99
    width, height = 340, 145
    x_map = lambda value: left + width * value / x_max
    y_map = lambda value: top + height * (1 - value / y_max)
    parts.append(f'<text class="title" x="{left + width / 2}" y="{top - 14}" text-anchor="middle">{html.escape(title)}</text>')
    for index in range(tick_count + 1):
        value = y_max * index / tick_count
        y = y_map(value)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + width}" y1="{y}" y2="{y}"/>'
                     f'<text x="{left - 8}" y="{y + 5}" text-anchor="end">{value:.3g}</text>')
    for index in range(5):
        value = x_max * index / 4
        parts.append(f'<text x="{x_map(value)}" y="{top + height + 22}" text-anchor="middle">{value:g}</text>')
    parts.append(f'<path class="axis" fill="none" d="M {left} {top} V {top + height} H {left + width}"/>'
                 f'<text x="{left + width / 2}" y="{top + height + 48}" text-anchor="middle">{html.escape(x_label)}</text>')
    return x_map, y_map


def draw_curve(parts: list[str], points: list[tuple[int, float]], age: int, x_map, y_map,
               *, dashed: bool = False) -> None:
    dash = ' stroke-dasharray="6 4"' if dashed else ''
    for segment in segments(points):
        if len(segment) == 1:
            step, value = segment[0]
            parts.append(f'<circle cx="{x_map(step):.2f}" cy="{y_map(value):.2f}" r="1.8" fill="{PALETTE[age]}"/>')
        else:
            coordinates = " ".join(f"{x_map(step):.2f},{y_map(value):.2f}" for step, value in segment)
            parts.append(f'<polyline class="curve" stroke="{PALETTE[age]}"{dash} points="{coordinates}"/>')


def points(run: Run, metric: str, population: str = "all_response") -> list[tuple[int, float]]:
    return [(step, measurement(row, metric, population)) for step, row in sorted(run.rows.items())]


def overview(runs: list[Run]) -> str:
    specs = (
        ("Realized staleness", (COMPARATORS[0],)),
        ("Policy lag RMS (nats)", ("delta_rms",)),
        ("Loss-sensitivity-weighted RMS (nats)", (WEIGHTED + "pre", WEIGHTED + "post")),
        ("Sensitivity x delta² retained", (RETAINED,)),
        ("Raw reward", (COMPARATORS[1],)),
    )
    parts = canvas(1260, 1265, runs, ("pre", "post"))
    for column, title in enumerate(TITLES):
        parts.append(f'<text class="heading" x="{column * 420 + 232}" y="56" text-anchor="middle">{title}</text>')
    for row_index, (title, metrics) in enumerate(specs):
        values = [value for run in runs for metric in metrics for _, value in points(run, metric)]
        y_max = 1.0 if row_index >= 3 else upper_bound(values)
        for column, treatment in enumerate(TREATMENTS):
            x_map, y_map = axes(parts, column, row_index, title, 300, y_max)
            for run in runs:
                if run.cohort.treatment != treatment:
                    continue
                for metric in metrics:
                    draw_curve(parts, points(run, metric), run.staleness, x_map, y_map,
                               dashed=metric == WEIGHTED + "pre")
    return "\n".join([*parts, "</svg>"])


def staleness_and_policy_lag_by_step(runs: list[Run]) -> str:
    """Align version age and token log-probability drift on the same step axis.

    Use separate rows (and units), shared scales across treatments, and the
    same run colors. Keep missing intervals empty, including late-start runs.
    """
    parts = canvas(1260, 545, runs)
    specs = (("Realized staleness", COMPARATORS[0]), ("Policy lag RMS (nats)", "delta_rms"))
    for column, title in enumerate(TITLES):
        parts.append(f'<text class="heading" x="{column * 420 + 232}" y="56" text-anchor="middle">{title}</text>')
    for row_index, (title, metric) in enumerate(specs):
        y_max = upper_bound([value for run in runs for _, value in points(run, metric)])
        for column, treatment in enumerate(TREATMENTS):
            x_map, y_map = axes(parts, column, row_index, title, 300, y_max)
            for run in runs:
                if run.cohort.treatment == treatment:
                    draw_curve(parts, points(run, metric), run.staleness, x_map, y_map)
    return "\n".join([*parts, "</svg>"])


def scatter(runs: list[Run]) -> str:
    parts = canvas(1260, 315, runs)
    y_max = upper_bound([value for run in runs for _, value in points(run, "delta_rms")])
    x_max = upper_bound([value for run in runs for _, value in points(run, COMPARATORS[0])])
    for column, treatment in enumerate(TREATMENTS):
        parts.append(f'<text class="heading" x="{column * 420 + 232}" y="56" text-anchor="middle">{TITLES[column]}</text>')
        x_map, y_map = axes(parts, column, 0, "Policy lag RMS (nats)", x_max, y_max,
                           x_label="Realized staleness")
        for run in runs:
            if run.cohort.treatment != treatment:
                continue
            for row in run.rows.values():
                x, y = measurement(row, COMPARATORS[0]), measurement(row, "delta_rms")
                if finite(x) and finite(y):
                    parts.append(f'<circle cx="{x_map(x):.2f}" cy="{y_map(y):.2f}" r="2.1" '
                                 f'fill="{PALETTE[run.staleness]}" opacity=".45"/>')
    return "\n".join([*parts, "</svg>"])


def populations(runs: list[Run], *, retention: bool = False) -> str:
    labels = ("sensitivity", "x delta²") if retention else None
    parts = canvas(1260, 785, runs, labels)
    metrics = ("loss_sensitivity_retained_fraction", RETAINED) if retention else ("delta_rms",)
    y_max = 1.0 if retention else upper_bound([
        value for run in runs for pop in POPULATIONS for _, value in points(run, "delta_rms", pop)
    ])
    for column, pop in enumerate(POPULATIONS):
        parts.append(f'<text class="heading" x="{column * 420 + 232}" y="56" text-anchor="middle">'
                     f'{pop.replace("_", " ").capitalize()}</text>')
        for row_index, treatment in enumerate(TREATMENTS):
            suffix = "retained" if retention else "RMS (nats)"
            title = f"{TITLES[row_index]} · {suffix}"
            x_map, y_map = axes(parts, column, row_index, title, 300, y_max)
            for run in runs:
                if run.cohort.treatment != treatment:
                    continue
                for metric in metrics:
                    draw_curve(parts, points(run, metric, pop), run.staleness, x_map, y_map,
                               dashed=retention and metric != RETAINED)
    return "\n".join([*parts, "</svg>"])


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_data(output: Path, runs: list[Run]) -> None:
    fields = sorted({key for run in runs for row in run.rows.values() for key in row})
    path = output / "policy-lag-step-metrics.csv"
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["namespace", "treatment", "queue_size", "staleness", "training_step", *fields])
        writer.writeheader()
        for run in runs:
            for step, row in sorted(run.rows.items()):
                writer.writerow({"namespace": run.cohort.namespace, "treatment": run.cohort.treatment,
                                 "queue_size": run.cohort.queue_size, "staleness": run.staleness,
                                 "training_step": step, **row})
    os.replace(temporary, path)
    coverage = []
    for cohort in COHORTS:
        for age in cohort.staleness:
            run = next((r for r in runs if r.cohort == cohort and r.staleness == age), None)
            coverage.append({"namespace": cohort.namespace, "staleness": age, "queue_size": cohort.queue_size,
                             "first_step": min(run.rows) if run else None,
                             "last_step": max(run.rows) if run else None, "logged_steps": len(run.rows) if run else 0,
                             "delta_valid_steps": sum(finite(measurement(r, "delta_rms")) for r in run.rows.values()) if run else 0,
                             "incomplete_tail_lines": run.incomplete_tail_lines if run else 0})
    atomic_write(output / "policy-lag-coverage.json", json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "coverage": coverage,
        "source": "local dashboard metrics.jsonl", "smoothing": "none", "step_alignment": "exact logged index + 1",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runs = discover_runs(args.training_root)
    if not runs:
        raise ValueError("no schema-v1 policy_lag records found")
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    renderers = {
        "policy-lag-and-reward-t1r7.svg": overview(runs),
        "staleness-and-policy-lag-by-step-t1r7.svg": staleness_and_policy_lag_by_step(runs),
        "policy-lag-vs-realized-staleness-t1r7.svg": scatter(runs),
        "policy-lag-by-population-t1r7.svg": populations(runs),
        "policy-lag-retention-by-population-t1r7.svg": populations(runs, retention=True),
    }
    for name, svg in renderers.items():
        atomic_write(figures / name, svg)
    write_data(args.output_dir, runs)
    print(f"runs={len(runs)} logged_steps={sum(len(run.rows) for run in runs)} figures={len(renderers)}")
    for run in runs:
        print(f"{run.cohort.treatment} S{run.staleness} tbq{run.cohort.queue_size}: {min(run.rows)}..{max(run.rows)}")


if __name__ == "__main__":
    main()
