#!/usr/bin/env python3
"""Create cross-cohort training and downstream diagnostic figures."""

from __future__ import annotations

import csv
import html
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/users/kfujii/evaluations/"
    "reasoning_eval/staleness-ratio-sweep"
)
TRAINING_STUDY_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/"
    "async-rl/checkpoints/training/math/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/"
    "Qwen3-4B-Base-LR2e-5-Step4000/grpo-clip0.2-0.28-tis2.0"
)
PROTOCOL = (
    "eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-"
    "thinking-t0.6-p0.95-k20-aime64-v1"
)
RESPONSE_QUANTILE_PATH = OUTPUT_ROOT / "response-length-quantiles.csv"
RESPONSE_QUANTILE_METRIC = "rollout/response_len/p10-p90"
GUESS_SAMPLE_PATH = OUTPUT_ROOT / "guess-sample-fractions.csv"
GUESS_PHRASE_PATH = OUTPUT_ROOT / "guess-phrase-token-probabilities.csv"
REWARD_CONDITIONED_PATH = OUTPUT_ROOT / "reward-conditioned-response-lengths.csv"
ARM_PATTERN = re.compile(r"^s(?P<staleness>\d+)-t(?P<train>\d+)r(?P<rollout>\d+)$")
ALLOWED_RATIOS = ((1, 7), (2, 6))
TRAINING_METRICS = (
    ("rollout/response_len/mean", "Response length (mean)", "tokens"),
    (RESPONSE_QUANTILE_METRIC, "Response length (p10–p90)", "tokens"),
    ("staleness/total/mean", "Staleness total mean", "staleness"),
    ("rollout/raw_reward", "Raw reward", "raw reward"),
)
AUXILIARY_TRAINING_METRICS = {
    "rollout/response_len/p10",
    "rollout/response_len/p90",
    "staleness/total/p10",
    "staleness/total/p90",
    "throughput/generated_tokens_per_second",
    "throughput/useful_tokens_per_second",
    "throughput/optimizer_updates_per_second",
}
DOWNSTREAM_METRICS = (
    ("aime24_percent", "AIME24 pass@1"),
    ("aime25_percent", "AIME25 pass@1"),
    ("aime26_percent", "AIME26 pass@1"),
    ("aime_macro_mean_percent", "AIME24/25/26 macro mean"),
)
AIME_MEAN_METRIC = "aime_macro_mean_percent"
EXCLUDED_RUN_NAMESPACE = (
    "hiso-zero-reward-trunc-s16-20-24-t2r6-20260828-v1"
)
EXCLUDED_RUN_ARM = "s24-t2r6"
EXCLUDED_RUN_START_STEP = 171
EXCLUDED_RUN_OBSERVED_END_STEP = 203
EXCLUDED_RUN_JOB_ID = "17601960"
EXCLUDED_RUN_WANDB_ID = "fqde65qe"


@dataclass(frozen=True)
class Source:
    namespace: str
    treatment: str
    max_response_len: int
    cohort: str
    history_path: Path | None
    aggregate_path: Path | None
    include_arms: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Identity:
    namespace: str
    treatment: str
    max_response_len: int
    cohort: str
    arm: str
    staleness: int
    trainer_nodes: int
    rollout_nodes: int

    @property
    def ratio_key(self) -> tuple[int, int]:
        return self.trainer_nodes, self.rollout_nodes

    @property
    def ratio_label(self) -> str:
        return "colocated (8:0)" if self.arm == "s0-colocated" else f"{self.trainer_nodes}:{self.rollout_nodes}"


def is_excluded_training_point(namespace: str, arm: str, training_step: int) -> bool:
    """Return whether a point descends from the low-throughput S=24 resume."""
    return (
        namespace == EXCLUDED_RUN_NAMESPACE
        and arm == EXCLUDED_RUN_ARM
        and training_step >= EXCLUDED_RUN_START_STEP
    )


def sources() -> list[Source]:
    original_namespace = "sr-20260819-212906"
    original_analysis = EVALUATION_ROOT / original_namespace / "analysis" / PROTOCOL / "full"
    generated_downstream = OUTPUT_ROOT / "downstream"
    definitions = (
        (
            original_namespace,
            "zero-reward",
            16384,
            "original-grid-tbq1000",
            original_analysis / "staleness" / "training-history.csv",
            original_analysis / "aggregate-results.csv",
        ),
        (
            "sr-20260826-141753-p1497131",
            "zero-reward",
            16384,
            "high-staleness-t1r7-tbq6000",
            OUTPUT_ROOT / "sr-20260826-141753-p1497131.csv",
            generated_downstream / "sr-20260826-141753-p1497131" / "aggregate-results.csv",
        ),
        (
            "zero-reward-s12-control-t1r7-cb1c041f",
            "zero-reward",
            16384,
            "zero-reward-s12-control-tbq6000",
            OUTPUT_ROOT / "zero-reward-s12-control-t1r7-cb1c041f.csv",
            generated_downstream
            / "zero-reward-s12-control-t1r7-cb1c041f"
            / "aggregate-results.csv",
        ),
        (
            "hiso-zero-reward-trunc-s16-20-24-t2r6-20260828-v1",
            "zero-reward",
            16384,
            "high-staleness-t2r6-tbq6000",
            OUTPUT_ROOT / "hiso-zero-reward-trunc-s16-20-24-t2r6-20260828-v1.csv",
            generated_downstream
            / "hiso-zero-reward-trunc-s16-20-24-t2r6-20260828-v1"
            / "aggregate-results.csv",
        ),
        (
            "hiso-zero-reward-trunc-total32k-s16-20-24-t1r7-20260828-v1",
            "zero-reward",
            32768,
            "total32k-t1r7-tbq6000",
            OUTPUT_ROOT
            / "hiso-zero-reward-trunc-total32k-s16-20-24-t1r7-20260828-v1.csv",
            generated_downstream
            / "hiso-zero-reward-trunc-total32k-s16-20-24-t1r7-20260828-v1"
            / "aggregate-results.csv",
        ),
        (
            "hiso-zero-loss-trunc-s8-16-20-r12-tbq6000-20260831-v1",
            "zero-loss",
            16384,
            "zero-loss-tbq6000",
            OUTPUT_ROOT / "hiso-zero-loss-trunc-s8-16-20-r12-tbq6000-20260831-v1.csv",
            generated_downstream
            / "hiso-zero-loss-trunc-s8-16-20-r12-tbq6000-20260831-v1"
            / "aggregate-results.csv",
        ),
        (
            "staleness-aware-safe4-t1r7-cb1c041f",
            "staleness-aware",
            16384,
            "zero-reward-staleness-aware-safe4-tbq6000",
            OUTPUT_ROOT / "staleness-aware-safe4-t1r7-cb1c041f.csv",
            generated_downstream
            / "staleness-aware-safe4-t1r7-cb1c041f"
            / "aggregate-results.csv",
        ),
        (
            "hiso-no-trunc-treatment-s8-16-r12-tbq6000-20260831-v1",
            "none",
            16384,
            "no-treatment-tbq6000",
            OUTPUT_ROOT / "hiso-no-trunc-treatment-s8-16-r12-tbq6000-20260831-v1.csv",
            generated_downstream
            / "hiso-no-trunc-treatment-s8-16-r12-tbq6000-20260831-v1"
            / "aggregate-results.csv",
        ),
        (
            "hiso-reward-off-trunc-coloc-s8-16-r12-20260827-v1",
            "none",
            16384,
            "no-treatment-colocated-reference",
            None,
            generated_downstream
            / "hiso-reward-off-trunc-coloc-s8-16-r12-20260827-v1"
            / "aggregate-results.csv",
            ("s0-colocated",),
        ),
    )
    return [Source(*definition) for definition in definitions]


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def identity(source: Source, arm: str) -> Identity:
    if arm == "s0-colocated":
        staleness, trainer_nodes, rollout_nodes = 0, 8, 0
    else:
        match = ARM_PATTERN.fullmatch(arm)
        if match is None:
            raise ValueError(f"unsupported arm: {arm}")
        staleness = int(match["staleness"])
        trainer_nodes = int(match["train"])
        rollout_nodes = int(match["rollout"])
    return Identity(
        namespace=source.namespace,
        treatment=source.treatment,
        max_response_len=source.max_response_len,
        cohort=source.cohort,
        arm=arm,
        staleness=staleness,
        trainer_nodes=trainer_nodes,
        rollout_nodes=rollout_nodes,
    )


def read_histories(
    configured_sources: list[Source],
) -> dict[Identity, dict[str, dict[int, float]]]:
    histories: dict[Identity, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    metric_names = (
        {metric for metric, _, _ in TRAINING_METRICS}
        | AUXILIARY_TRAINING_METRICS
        | {"rollout/truncated_ratio"}
    )
    for source in configured_sources:
        if source.history_path is None:
            continue
        if not source.history_path.is_file():
            raise FileNotFoundError(source.history_path)
        with source.history_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if source.include_arms is not None and row["arm"] not in source.include_arms:
                    continue
                series_identity = identity(source, row["arm"])
                step = int(row["training_step"])
                for metric in metric_names:
                    value = optional_float(row.get(metric))
                    if value is not None:
                        histories[series_identity][metric][step] = value
    overlay_local_histories(configured_sources, histories, metric_names)
    overlay_response_length_quantiles(configured_sources, histories)
    return {
        series_identity: {
            metric: {
                step: value
                for step, value in values.items()
                if not is_excluded_training_point(
                    series_identity.namespace, series_identity.arm, step
                )
            }
            for metric, values in metrics.items()
        }
        for series_identity, metrics in histories.items()
    }


def overlay_response_length_quantiles(
    configured_sources: list[Source],
    histories: dict[Identity, dict[str, dict[int, float]]],
) -> None:
    if not RESPONSE_QUANTILE_PATH.is_file():
        raise FileNotFoundError(RESPONSE_QUANTILE_PATH)
    sources_by_namespace = {source.namespace: source for source in configured_sources}
    with RESPONSE_QUANTILE_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source = sources_by_namespace.get(row["namespace"])
            if source is None:
                continue
            arm = row["arm"]
            if source.include_arms is not None and arm not in source.include_arms:
                continue
            series_identity = identity(source, arm)
            step = int(row["training_step"])
            histories[series_identity]["rollout/response_len/p10"][step] = float(row["p10"])
            histories[series_identity]["rollout/response_len/p90"][step] = float(row["p90"])


def overlay_local_histories(
    configured_sources: list[Source],
    histories: dict[Identity, dict[str, dict[int, float]]],
    metric_names: set[str],
) -> None:
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    step_pattern = re.compile(r'"step_key"\s*:\s*"rollout/step"\s*,\s*"step"\s*:\s*(\d+)')
    metric_patterns = {
        metric: re.compile(rf'"{re.escape(metric)}"\s*:\s*({number})')
        for metric in metric_names
    }
    staleness_count_pattern = re.compile(
        rf'"staleness/total/count_(\d+|ge_33)"\s*:\s*({number})'
    )
    async_root = TRAINING_STUDY_ROOT / "async" / "off-policy"
    colocated_root = TRAINING_STUDY_ROOT / "colocated"
    for source in configured_sources:
        paths = list(
            async_root.glob(
                f"max-weight-staleness-*-from-prefill/*-{source.namespace}-*/"
                "dump/dashboard/metrics.jsonl"
            )
        )
        paths.extend(
            path
            for path in colocated_root.rglob("metrics.jsonl")
            if f"s0-colocated-{source.namespace}-" in path.parents[2].name
        )
        for path in paths:
            run_name = path.parents[2].name
            arm_match = re.match(r"(s\d+-t\d+r\d+)", run_name)
            arm = arm_match.group(1) if arm_match is not None else "s0-colocated"
            if source.include_arms is not None and arm not in source.include_arms:
                continue
            series_identity = identity(source, arm)
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    step_match = step_pattern.search(line)
                    if step_match is None:
                        continue
                    training_step = int(step_match.group(1)) + 1
                    for metric, pattern in metric_patterns.items():
                        value_match = pattern.search(line)
                        if value_match is not None:
                            histories[series_identity][metric][training_step] = float(
                                value_match.group(1)
                            )
                    staleness_counts = [
                        (
                            33 if count_match.group(1) == "ge_33" else int(count_match.group(1)),
                            float(count_match.group(2)),
                        )
                        for count_match in staleness_count_pattern.finditer(line)
                    ]
                    total_count = sum(count for _, count in staleness_counts)
                    if total_count > 0.0:
                        threshold = total_count * 0.10
                        cumulative = 0.0
                        for staleness_value, count in sorted(staleness_counts):
                            cumulative += count
                            if cumulative >= threshold:
                                histories[series_identity]["staleness/total/p10"][
                                    training_step
                                ] = float(staleness_value)
                                break


def read_downstream(
    configured_sources: list[Source],
) -> dict[Identity, dict[str, dict[int, float]]]:
    scores: dict[Identity, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for source in configured_sources:
        if source.aggregate_path is None:
            continue
        if not source.aggregate_path.is_file():
            raise FileNotFoundError(source.aggregate_path)
        with source.aggregate_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if source.include_arms is not None and row["arm"] not in source.include_arms:
                    continue
                series_identity = identity(source, row["arm"])
                step = int(row["training_step"])
                if is_excluded_training_point(
                    series_identity.namespace, series_identity.arm, step
                ):
                    continue
                for metric, _ in DOWNSTREAM_METRICS:
                    value = optional_float(row.get(metric))
                    if value is not None:
                        scores[series_identity][metric][step] = value
    return {
        series_identity: {metric: dict(values) for metric, values in metrics.items()}
        for series_identity, metrics in scores.items()
        if any(values for values in metrics.values())
    }


def color(staleness: int) -> str:
    palette = {
        0: "#222222",
        1: "#0072B2",
        2: "#56B4E9",
        4: "#009E73",
        8: "#F0E442",
        12: "#999933",
        16: "#E69F00",
        20: "#D55E00",
        24: "#CC79A7",
        28: "#6A3D9A",
    }
    return palette.get(staleness, f"hsl({(staleness * 47) % 360},65%,42%)")


def rolling(values: dict[int, float], window: int = 5) -> list[tuple[float, float]]:
    ordered = sorted(values.items())
    return [
        (float(step), statistics.fmean(value for _, value in ordered[max(0, index - window + 1) : index + 1]))
        for index, (step, _) in enumerate(ordered)
    ]


def polyline(points: list[tuple[float, float]], x_map: Any, y_map: Any) -> str:
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in points)


def svg_canvas(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202020}"
        ".axis-label{font-size:13px}.tick{font-size:11px}.column-label{font-size:15px;font-weight:700}"
        ".metric-label{font-size:13px;font-weight:700}.grid{stroke:#e2e2e2;stroke-width:.8}"
        ".axis{stroke:#303030;stroke-width:1}.raw{fill:none;stroke-width:.9;opacity:.15}"
        ".smooth{fill:none;stroke-width:2.2}.reference{fill:none;stroke:#111;stroke-width:1.8;stroke-dasharray:7 4}"
        ".quantile-band{stroke:none;opacity:.04}.quantile-low{fill:none;stroke-width:1;opacity:.20}"
        ".quantile-high{fill:none;stroke-width:2.2}.reference-band{fill:#111;stroke:none;opacity:.05}"
        ".maxlen{stroke:#555;stroke-width:1.1;stroke-dasharray:4 4}"
        ".onset{stroke:#555;stroke-width:1;stroke-dasharray:3 3}"
        ".annotation{font-size:10px;fill:#8b0000;font-weight:600}"
        ".response-overlay{fill:none;stroke:#111;stroke-width:1.8;stroke-dasharray:7 3}"
        ".legend{font-size:11px}.endpoint{font-size:10px;font-weight:600}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def axis_ticks(lower: float, upper: float, count: int = 5) -> list[float]:
    return [lower + (upper - lower) * index / (count - 1) for index in range(count)]


def format_tick(value: float, span: float) -> str:
    if span <= 1.1:
        return f"{value:.2f}"
    if span < 20:
        return f"{value:.1f}"
    return f"{value:.0f}"


def training_bounds(
    metric: str,
    max_response_len: int,
    ratio: tuple[int, int],
    selected: dict[Identity, dict[str, dict[int, float]]],
) -> tuple[float, float]:
    if metric == RESPONSE_QUANTILE_METRIC:
        return 0.0, float(max_response_len) * 1.05
    if metric == "rollout/response_len/mean":
        if ratio == (2, 6):
            return 3_000.0, 11_000.0
        return 0.0, float(max_response_len) * 1.05
    if metric == "rollout/raw_reward":
        return 0.0, 1.0
    values = [
        value
        for series_identity, metrics in selected.items()
        if series_identity.ratio_key == ratio
        for value in metrics.get(metric, {}).values()
    ]
    maximum = max(values, default=1.0)
    padded = maximum * 1.05
    if padded <= 3.0:
        interval = 0.5
    elif padded <= 10.0:
        interval = 1.0
    elif padded <= 20.0:
        interval = 2.0
    else:
        interval = 5.0
    return 0.0, max(1.0, math.ceil(padded / interval) * interval)


def render_training_group(
    treatment: str,
    max_response_len: int,
    histories: dict[Identity, dict[str, dict[int, float]]],
    downstream: dict[Identity, dict[str, dict[int, float]]],
) -> str:
    selected = {
        series_identity: metrics
        for series_identity, metrics in histories.items()
        if series_identity.treatment == treatment
        and series_identity.max_response_len == max_response_len
        and series_identity.ratio_key in ALLOWED_RATIOS
    }
    selected_downstream = {
        series_identity: metrics
        for series_identity, metrics in downstream.items()
        if series_identity.treatment == treatment
        and series_identity.max_response_len == max_response_len
        and series_identity.ratio_key in ALLOWED_RATIOS
    }
    ratios = [
        ratio
        for ratio in ALLOWED_RATIOS
        if any(series_identity.ratio_key == ratio for series_identity in selected)
    ]
    if not ratios:
        raise ValueError(f"no supported train:rollout ratios for {treatment=} {max_response_len=}")
    reference_entry = next(
        (
            (series_identity, metrics)
            for series_identity, metrics in histories.items()
            if series_identity.arm == "s0-colocated"
            and series_identity.treatment == treatment
            and series_identity.max_response_len == max_response_len
        ),
        None,
    )
    downstream_reference_entry = next(
        (
            (series_identity, metrics)
            for series_identity, metrics in downstream.items()
            if series_identity.arm == "s0-colocated"
            and series_identity.treatment == treatment
            and series_identity.max_response_len == max_response_len
        ),
        None,
    )
    panel_width, panel_height = 520.0, 230.0
    column_gap, row_gap = 36.0, 28.0
    left, top = 78.0, 76.0
    row_count = len(TRAINING_METRICS) + 1
    width = int(left + len(ratios) * panel_width + (len(ratios) - 1) * column_gap + 25)
    height = int(
        top
        + row_count * panel_height
        + (row_count - 1) * row_gap
        + 30
    )
    staleness_levels = sorted({series_identity.staleness for series_identity in selected})
    max_step = 300
    elements = svg_canvas(width, height)
    for column, ratio in enumerate(ratios):
        panel_x = left + column * (panel_width + column_gap)
        elements.append(
            f'<text class="column-label" x="{panel_x + panel_width / 2:.1f}" y="24" '
            f'text-anchor="middle">Train:Rollout = {ratio[0]}:{ratio[1]}</text>'
        )
    for index, staleness in enumerate(staleness_levels):
        legend_x = left + index * 92.0
        legend_y = 53.0
        elements.extend(
            [
                f'<line x1="{legend_x:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 25:.1f}" y2="{legend_y:.1f}" stroke="{color(staleness)}" stroke-width="2.5"/>',
                f'<text class="legend" x="{legend_x + 31:.1f}" y="{legend_y + 4:.1f}">S={staleness}</text>',
            ]
        )
    if reference_entry is not None or downstream_reference_entry is not None:
        legend_x = left + len(staleness_levels) * 92.0 + 5.0
        elements.extend(
            [
                f'<line class="reference" x1="{legend_x:.1f}" y1="53" x2="{legend_x + 28:.1f}" y2="53"/>',
                f'<text class="legend" x="{legend_x + 35:.1f}" y="57">colocated reference</text>',
            ]
        )
    display_rows = (*TRAINING_METRICS, (AIME_MEAN_METRIC, "AIME24/25/26 mean", "pass@1 (%)"))
    for row_index, (metric, metric_title, y_label) in enumerate(display_rows):
        for column, ratio in enumerate(ratios):
            panel_x = left + column * (panel_width + column_gap)
            panel_y = top + row_index * (panel_height + row_gap)
            plot_x, plot_y = panel_x + 58.0, panel_y + 30.0
            plot_width, plot_height = panel_width - 174.0, panel_height - 78.0
            x_map = lambda value, base=plot_x: base + plot_width * (value - 1.0) / max(max_step - 1.0, 1.0)
            if metric == AIME_MEAN_METRIC:
                y_lower, y_upper = 0.0, 75.0
            else:
                y_lower, y_upper = training_bounds(metric, max_response_len, ratio, selected)
            span = y_upper - y_lower
            y_map = lambda value, base=plot_y: base + plot_height * (y_upper - value) / max(span, 1e-12)
            y_ticks = (0.0, 25.0, 50.0, 75.0) if metric == AIME_MEAN_METRIC else axis_ticks(y_lower, y_upper)
            for tick in y_ticks:
                y = y_map(tick)
                elements.extend(
                    [
                        f'<line class="grid" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}"/>',
                        f'<text class="tick" x="{plot_x - 8:.1f}" y="{y + 4:.1f}" text-anchor="end">{format_tick(tick, span)}</text>',
                    ]
                )
            for tick in sorted({1, max_step, *range(50, max_step + 1, 50)}):
                x = x_map(float(tick))
                elements.append(f'<text class="tick" x="{x:.1f}" y="{plot_y + plot_height + 18:.1f}" text-anchor="middle">{tick}</text>')
            elements.extend(
                [
                    f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 15:.1f}" text-anchor="middle">{html.escape(metric_title)}</text>',
                    f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                    f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                    f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height - 8:.1f}" text-anchor="middle">step</text>',
                    f'<text class="axis-label" transform="translate({panel_x + 15:.1f},{plot_y + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>',
                ]
            )
            if (
                metric in {"rollout/response_len/mean", RESPONSE_QUANTILE_METRIC}
                and y_lower <= max_response_len <= y_upper
            ):
                max_response_y = y_map(float(max_response_len))
                max_response_label = f"{max_response_len // 1024}K"
                elements.extend(
                    [
                        f'<line class="maxlen" x1="{plot_x:.1f}" y1="{max_response_y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{max_response_y:.1f}"/>',
                        f'<text class="tick" x="{plot_x + plot_width - 3:.1f}" y="{max_response_y - 5:.1f}" text-anchor="end">max response len = {max_response_label}</text>',
                    ]
                )
            if metric == AIME_MEAN_METRIC:
                endpoints: list[tuple[Identity, float, float, float]] = []
                for series_identity, metrics in sorted(
                    selected_downstream.items(), key=lambda item: item[0].staleness
                ):
                    values = metrics.get(metric, {})
                    if series_identity.ratio_key != ratio or not values:
                        continue
                    points = [(float(step), value) for step, value in sorted(values.items())]
                    elements.append(
                        f'<polyline class="smooth" stroke="{color(series_identity.staleness)}" points="{polyline(points, x_map, y_map)}"/>'
                    )
                    for step, value in points:
                        elements.append(
                            f'<circle cx="{x_map(step):.2f}" cy="{y_map(value):.2f}" r="2" fill="{color(series_identity.staleness)}"/>'
                        )
                    last_step, last_value = points[-1]
                    endpoints.append((series_identity, last_step, last_value, y_map(last_value)))
                if downstream_reference_entry is not None:
                    reference_identity, reference_metrics = downstream_reference_entry
                    reference_values = reference_metrics.get(metric, {})
                    if reference_values:
                        points = [
                            (float(step), value)
                            for step, value in sorted(reference_values.items())
                        ]
                        elements.append(
                            f'<polyline class="reference" points="{polyline(points, x_map, y_map)}"/>'
                        )
                        for step, value in points:
                            elements.append(
                                f'<circle cx="{x_map(step):.2f}" cy="{y_map(value):.2f}" r="2" fill="#111"/>'
                            )
                        last_step, last_value = points[-1]
                        endpoints.append(
                            (
                                reference_identity,
                                last_step,
                                last_value,
                                y_map(last_value),
                            )
                        )
                if not endpoints:
                    elements.append(
                        f'<text class="tick" x="{plot_x + plot_width / 2:.1f}" y="{plot_y + plot_height / 2:.1f}" text-anchor="middle">not evaluated</text>'
                    )
                else:
                    label_positions = spread_endpoint_labels(
                        [(series_identity, desired_y) for series_identity, _, _, desired_y in endpoints],
                        plot_y + 6.0,
                        plot_y + plot_height - 4.0,
                    )
                    label_x = plot_x + plot_width + 13.0
                    for series_identity, last_step, last_value, endpoint_y in endpoints:
                        label_y = label_positions[series_identity]
                        endpoint_label = (
                            "colocated"
                            if series_identity.arm == "s0-colocated"
                            else f"S={series_identity.staleness}"
                        )
                        elements.extend(
                            [
                                f'<line x1="{x_map(last_step) + 3:.1f}" y1="{endpoint_y:.1f}" x2="{label_x - 4:.1f}" y2="{label_y:.1f}" stroke="{color(series_identity.staleness)}" stroke-width=".8"/>',
                                f'<text class="endpoint" x="{label_x:.1f}" y="{label_y + 3.5:.1f}" style="fill:{color(series_identity.staleness)}">{endpoint_label}: {last_value:.1f}</text>',
                            ]
                        )
                continue
            if metric == RESPONSE_QUANTILE_METRIC:
                for series_identity, metrics in sorted(
                    selected.items(), key=lambda item: item[0].staleness
                ):
                    if series_identity.ratio_key != ratio:
                        continue
                    lower = dict(rolling(metrics.get("rollout/response_len/p10", {})))
                    upper = dict(rolling(metrics.get("rollout/response_len/p90", {})))
                    steps = sorted(set(lower) & set(upper))
                    if not steps:
                        continue
                    lower_points = [(float(step), lower[step]) for step in steps]
                    upper_points = [(float(step), upper[step]) for step in steps]
                    band_points = [*upper_points, *reversed(lower_points)]
                    series_color = color(series_identity.staleness)
                    elements.extend(
                        [
                            f'<polygon class="quantile-band" fill="{series_color}" points="{polyline(band_points, x_map, y_map)}"/>',
                            f'<polyline class="quantile-low" stroke="{series_color}" points="{polyline(lower_points, x_map, y_map)}"/>',
                            f'<polyline class="quantile-high" stroke="{series_color}" points="{polyline(upper_points, x_map, y_map)}"/>',
                        ]
                    )
                if reference_entry is not None:
                    _, reference_metrics = reference_entry
                    lower = dict(
                        rolling(reference_metrics.get("rollout/response_len/p10", {}))
                    )
                    upper = dict(
                        rolling(reference_metrics.get("rollout/response_len/p90", {}))
                    )
                    steps = sorted(set(lower) & set(upper))
                    if steps:
                        lower_points = [(float(step), lower[step]) for step in steps]
                        upper_points = [(float(step), upper[step]) for step in steps]
                        band_points = [*upper_points, *reversed(lower_points)]
                        elements.extend(
                            [
                                f'<polygon class="reference-band" points="{polyline(band_points, x_map, y_map)}"/>',
                                f'<polyline class="reference" points="{polyline(upper_points, x_map, y_map)}"/>',
                            ]
                        )
                continue
            for series_identity, metrics in sorted(selected.items(), key=lambda item: item[0].staleness):
                if series_identity.ratio_key != ratio or not metrics.get(metric):
                    continue
                raw = [(float(step), value) for step, value in sorted(metrics[metric].items())]
                smooth = rolling(metrics[metric])
                elements.extend(
                    [
                        f'<polyline class="raw" stroke="{color(series_identity.staleness)}" points="{polyline(raw, x_map, y_map)}"/>',
                        f'<polyline class="smooth" stroke="{color(series_identity.staleness)}" points="{polyline(smooth, x_map, y_map)}"/>',
                    ]
                )
            if (
                metric
                in {
                    "rollout/response_len/mean",
                    "rollout/response_len/p90",
                    "rollout/raw_reward",
                }
                and reference_entry is not None
            ):
                _, reference_metrics = reference_entry
                reference_values = reference_metrics.get(metric, {})
                if reference_values:
                    elements.append(
                        f'<polyline class="reference" points="{polyline(rolling(reference_values), x_map, y_map)}"/>'
                    )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def spread_endpoint_labels(
    labels: list[tuple[Identity, float]],
    top: float,
    bottom: float,
    gap: float = 12.0,
) -> dict[Identity, float]:
    ordered = sorted(labels, key=lambda item: item[1])
    placed: list[list[Any]] = []
    for series_identity, desired in ordered:
        y = max(top, desired)
        if placed:
            y = max(y, placed[-1][1] + gap)
        placed.append([series_identity, y])
    if placed and placed[-1][1] > bottom:
        placed[-1][1] = bottom
        for index in range(len(placed) - 2, -1, -1):
            placed[index][1] = min(placed[index][1], placed[index + 1][1] - gap)
    return {series_identity: y for series_identity, y in placed}


def read_guess_sample_fractions() -> dict[int, dict[int, float]]:
    if not GUESS_SAMPLE_PATH.is_file():
        raise FileNotFoundError(GUESS_SAMPLE_PATH)
    series: dict[int, dict[int, float]] = defaultdict(dict)
    with GUESS_SAMPLE_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arm_match = ARM_PATTERN.fullmatch(row["arm"])
            if arm_match is None or (int(arm_match["train"]), int(arm_match["rollout"])) != (1, 7):
                continue
            staleness = int(arm_match["staleness"])
            series[staleness][int(row["training_step"])] = (
                float(row["fraction_le_256"]) * 100.0
            )
    return {staleness: dict(values) for staleness, values in series.items()}


def read_max_response_length_fractions() -> dict[int, dict[int, float]]:
    if not GUESS_SAMPLE_PATH.is_file():
        raise FileNotFoundError(GUESS_SAMPLE_PATH)
    series: dict[int, dict[int, float]] = defaultdict(dict)
    with GUESS_SAMPLE_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arm_match = ARM_PATTERN.fullmatch(row["arm"])
            if arm_match is None or (int(arm_match["train"]), int(arm_match["rollout"])) != (1, 7):
                continue
            staleness = int(arm_match["staleness"])
            series[staleness][int(row["training_step"])] = (
                float(row["fraction_at_max_response_length"]) * 100.0
            )
    return {staleness: dict(values) for staleness, values in series.items()}


def render_guess_sample_fraction(
    series: dict[int, dict[int, float]],
    max_length_series: dict[int, dict[int, float]],
    histories: dict[Identity, dict[str, dict[int, float]]],
) -> str:
    staleness_levels = sorted(series)
    if not staleness_levels:
        raise ValueError("no guess-sample fraction series")
    columns = 4
    rows = math.ceil(len(staleness_levels) / columns)
    panel_width, panel_height = 330.0, 220.0
    column_gap, row_gap = 24.0, 28.0
    left, top = 58.0, 88.0
    width = int(left + columns * panel_width + (columns - 1) * column_gap + 65)
    grid_height = rows * panel_height + (rows - 1) * row_gap
    section_gap = 86.0
    max_length_top = top + grid_height + section_gap
    height = int(max_length_top + grid_height + 24)
    max_step = 300
    onset_threshold = 1.0
    max_response_len = 16_384.0
    response_y_upper = max_response_len * 1.05
    response_means = {
        series_identity.staleness: metrics["rollout/response_len/mean"]
        for series_identity, metrics in histories.items()
        if series_identity.treatment == "zero-reward"
        and series_identity.max_response_len == int(max_response_len)
        and series_identity.ratio_key == (1, 7)
        and series_identity.arm != "s0-colocated"
        and metrics.get("rollout/response_len/mean")
    }
    observed_peak = max(value for values in series.values() for value in values.values())
    full_y_upper = min(
        100.0,
        max(20.0, math.ceil(observed_peak / 20.0) * 20.0),
    )
    elements = svg_canvas(width, height)
    elements.extend(
        [
            f'<text class="column-label" x="{left:.1f}" y="18">Short-response fraction (≤256 tokens)</text>',
            f'<line class="raw" x1="{left:.1f}" y1="39" x2="{left + 24:.1f}" y2="39" stroke="#555"/>',
            f'<text class="legend" x="{left + 30:.1f}" y="43">per-step</text>',
            f'<line class="smooth" x1="{left + 100:.1f}" y1="39" x2="{left + 124:.1f}" y2="39" stroke="#555"/>',
            f'<text class="legend" x="{left + 130:.1f}" y="43">short-answer trailing-5</text>',
            f'<line class="response-overlay" x1="{left + 270:.1f}" y1="39" x2="{left + 298:.1f}" y2="39"/>',
            f'<text class="legend" x="{left + 306:.1f}" y="43">response length mean (right axis)</text>',
            f'<line class="onset" x1="{left + 505:.1f}" y1="30" x2="{left + 505:.1f}" y2="48"/>',
            f'<text class="legend" x="{left + 513:.1f}" y="43">first ≥1%</text>',
            f'<text class="legend" x="{left + 620:.1f}" y="43">short response: response length ≤256 tokens</text>',
            f'<text class="legend" x="{left:.1f}" y="65">left-axis zoom: S=1/2/4/8: 0–0.5% · S=12/16/20/24/28: pre-onset 0–1%</text>',
        ]
    )
    for index, staleness in enumerate(staleness_levels):
        row, column = divmod(index, columns)
        panel_x = left + column * (panel_width + column_gap)
        panel_y = top + row * (panel_height + row_gap)
        plot_x, plot_y = panel_x + 52.0, panel_y + 30.0
        plot_width, plot_height = panel_width - 68.0, panel_height - 76.0
        if staleness in {1, 2, 4, 8}:
            panel_y_upper = 0.5
        elif staleness in {12, 16, 20, 24, 28}:
            panel_y_upper = onset_threshold
        else:
            panel_y_upper = full_y_upper
        x_map = lambda value, base=plot_x: base + plot_width * (value - 1.0) / 299.0
        y_map = lambda value, base=plot_y: base + plot_height * (
            panel_y_upper - value
        ) / panel_y_upper
        response_y_map = lambda value, base=plot_y: base + plot_height * (
            response_y_upper - value
        ) / response_y_upper
        values = {
            step: value
            for step, value in series[staleness].items()
            if 1 <= step <= max_step
        }
        smooth = dict(rolling(values))
        onset_step = next(
            (int(step) for step, value in sorted(smooth.items()) if value >= onset_threshold),
            None,
        )
        if staleness in {1, 2, 4, 8}:
            panel_label = f"S={staleness} · peak {max(smooth.values()):.2f}%"
        elif staleness in {12, 16, 20, 24, 28} and onset_step is not None:
            panel_label = f"S={staleness} · pre-1% (onset {onset_step})"
        elif onset_step is not None:
            panel_label = f"S={staleness} · onset step {onset_step}"
        else:
            panel_label = f"S={staleness} · no ≥5% onset"
        clip_id = f"guess-panel-s{staleness}"
        elements.append(
            f'<defs><clipPath id="{clip_id}"><rect x="{plot_x:.1f}" y="{plot_y:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}"/></clipPath></defs>'
        )
        elements.append(
            f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 14:.1f}" text-anchor="middle">{panel_label}</text>'
        )
        for tick in axis_ticks(0.0, panel_y_upper):
            y = y_map(tick)
            elements.extend(
                [
                    f'<line class="grid" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}"/>',
                    f'<text class="tick" x="{plot_x - 7:.1f}" y="{y + 4:.1f}" text-anchor="end">{format_tick(tick, panel_y_upper)}</text>',
                ]
            )
        for tick in (1, 100, 200, 300):
            x = x_map(float(tick))
            elements.append(
                f'<text class="tick" x="{x:.1f}" y="{plot_y + plot_height + 17:.1f}" text-anchor="middle">{tick}</text>'
            )
        elements.extend(
            [
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="axis" x1="{plot_x + plot_width:.1f}" y1="{plot_y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="maxlen" x1="{plot_x:.1f}" y1="{response_y_map(max_response_len):.1f}" x2="{plot_x + plot_width:.1f}" y2="{response_y_map(max_response_len):.1f}"/>',
            ]
        )
        if panel_y_upper >= onset_threshold:
            elements.append(
                f'<line class="onset" x1="{plot_x:.1f}" y1="{y_map(onset_threshold):.1f}" x2="{plot_x + plot_width:.1f}" y2="{y_map(onset_threshold):.1f}"/>'
            )
        short_values = values
        short_smooth = smooth
        if staleness in {12, 16, 20, 24, 28} and onset_step is not None:
            short_values = {
                step: value for step, value in values.items() if step <= onset_step
            }
            short_smooth = {
                step: value for step, value in smooth.items() if step <= onset_step
            }
        elements.extend(
            [
                f'<polyline class="raw" clip-path="url(#{clip_id})" stroke="{color(staleness)}" points="{polyline([(float(step), value) for step, value in sorted(short_values.items())], x_map, y_map)}"/>',
                f'<polyline class="smooth" clip-path="url(#{clip_id})" stroke="{color(staleness)}" points="{polyline(list(short_smooth.items()), x_map, y_map)}"/>',
            ]
        )
        for tick, label in ((0.0, "0"), (8_192.0, "8K"), (16_384.0, "16K")):
            y = response_y_map(tick)
            elements.append(
                f'<text class="tick" x="{plot_x + plot_width + 6:.1f}" y="{y + 4:.1f}">{label}</text>'
            )
        response_values = {
            step: value
            for step, value in response_means.get(staleness, {}).items()
            if 1 <= step <= max_step
        }
        if response_values:
            elements.append(
                f'<polyline class="response-overlay" clip-path="url(#{clip_id})" points="{polyline(rolling(response_values), x_map, response_y_map)}"/>'
            )
        if onset_step is not None:
            onset_x = x_map(float(onset_step))
            onset_y = y_map(smooth[float(onset_step)])
            elements.extend(
                [
                    f'<line class="onset" x1="{onset_x:.1f}" y1="{plot_y:.1f}" x2="{onset_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                    f'<circle clip-path="url(#{clip_id})" cx="{onset_x:.1f}" cy="{onset_y:.1f}" r="3" fill="{color(staleness)}"/>',
                ]
            )
        if column == 0:
            elements.append(
                f'<text class="axis-label" transform="translate({panel_x + 13:.1f},{plot_y + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">rollout fraction (%)</text>'
            )
        if column == columns - 1:
            elements.append(
                f'<text class="axis-label" transform="translate({plot_x + plot_width + 42:.1f},{plot_y + plot_height / 2:.1f}) rotate(90)" text-anchor="middle">response length mean</text>'
            )
        if row == rows - 1:
            elements.append(
                f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height - 5:.1f}" text-anchor="middle">step</text>'
            )

    elements.extend(
        [
            f'<text class="column-label" x="{left:.1f}" y="{max_length_top - 51:.1f}">Exceed max response len fraction</text>',
            f'<line class="raw" x1="{left:.1f}" y1="{max_length_top - 29:.1f}" x2="{left + 24:.1f}" y2="{max_length_top - 29:.1f}" stroke="#555"/>',
            f'<text class="legend" x="{left + 30:.1f}" y="{max_length_top - 25:.1f}">per-step</text>',
            f'<line class="smooth" x1="{left + 100:.1f}" y1="{max_length_top - 29:.1f}" x2="{left + 124:.1f}" y2="{max_length_top - 29:.1f}" stroke="#555"/>',
            f'<text class="legend" x="{left + 130:.1f}" y="{max_length_top - 25:.1f}">trailing-5</text>',
            f'<line class="response-overlay" x1="{left + 215:.1f}" y1="{max_length_top - 29:.1f}" x2="{left + 243:.1f}" y2="{max_length_top - 29:.1f}"/>',
            f'<text class="legend" x="{left + 251:.1f}" y="{max_length_top - 25:.1f}">response length mean (right axis)</text>',
            f'<text class="legend" x="{left + 490:.1f}" y="{max_length_top - 25:.1f}">left axis: 0–20%; S=12/16/20/24/28 stop at first trailing-5 ≥20%</text>',
        ]
    )
    for index, staleness in enumerate(staleness_levels):
        row, column = divmod(index, columns)
        panel_x = left + column * (panel_width + column_gap)
        panel_y = max_length_top + row * (panel_height + row_gap)
        plot_x, plot_y = panel_x + 52.0, panel_y + 30.0
        plot_width, plot_height = panel_width - 68.0, panel_height - 76.0
        panel_max_length_y_upper = 20.0
        x_map = lambda value, base=plot_x: base + plot_width * (value - 1.0) / 299.0
        y_map = lambda value, base=plot_y: base + plot_height * (
            panel_max_length_y_upper - value
        ) / panel_max_length_y_upper
        response_y_map = lambda value, base=plot_y: base + plot_height * (
            response_y_upper - value
        ) / response_y_upper
        values = {
            step: value
            for step, value in max_length_series.get(staleness, {}).items()
            if 1 <= step <= max_step
        }
        smooth = dict(rolling(values))
        onset_step = next(
            (int(step) for step, value in sorted(smooth.items()) if value >= 20.0),
            None,
        )
        panel_label = (
            f"S={staleness} · pre-20% (onset {onset_step})"
            if staleness in {12, 16, 20, 24, 28} and onset_step is not None
            else f"S={staleness}"
        )
        clip_id = f"max-response-panel-s{staleness}"
        elements.append(
            f'<defs><clipPath id="{clip_id}"><rect x="{plot_x:.1f}" y="{plot_y:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}"/></clipPath></defs>'
        )
        elements.append(
            f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 14:.1f}" text-anchor="middle">{panel_label}</text>'
        )
        for tick in axis_ticks(0.0, panel_max_length_y_upper):
            y = y_map(tick)
            elements.extend(
                [
                    f'<line class="grid" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}"/>',
                    f'<text class="tick" x="{plot_x - 7:.1f}" y="{y + 4:.1f}" text-anchor="end">{format_tick(tick, panel_max_length_y_upper)}</text>',
                ]
            )
        for tick in (1, 100, 200, 300):
            x = x_map(float(tick))
            elements.append(
                f'<text class="tick" x="{x:.1f}" y="{plot_y + plot_height + 17:.1f}" text-anchor="middle">{tick}</text>'
            )
        elements.extend(
            [
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="axis" x1="{plot_x + plot_width:.1f}" y1="{plot_y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="maxlen" x1="{plot_x:.1f}" y1="{response_y_map(max_response_len):.1f}" x2="{plot_x + plot_width:.1f}" y2="{response_y_map(max_response_len):.1f}"/>',
            ]
        )
        plotted_values = values
        plotted_smooth = smooth
        if staleness in {12, 16, 20, 24, 28} and onset_step is not None:
            plotted_values = {
                step: value for step, value in values.items() if step <= onset_step
            }
            plotted_smooth = {
                step: value for step, value in smooth.items() if step <= onset_step
            }
        elements.extend(
            [
                f'<polyline class="raw" clip-path="url(#{clip_id})" stroke="{color(staleness)}" points="{polyline([(float(step), value) for step, value in sorted(plotted_values.items())], x_map, y_map)}"/>',
                f'<polyline class="smooth" clip-path="url(#{clip_id})" stroke="{color(staleness)}" points="{polyline(list(plotted_smooth.items()), x_map, y_map)}"/>',
            ]
        )
        response_values = {
            step: value
            for step, value in response_means.get(staleness, {}).items()
            if 1 <= step <= max_step
        }
        if response_values:
            elements.append(
                f'<polyline class="response-overlay" clip-path="url(#{clip_id})" points="{polyline(rolling(response_values), x_map, response_y_map)}"/>'
            )
        for tick, label in ((0.0, "0"), (8_192.0, "8K"), (16_384.0, "16K")):
            y = response_y_map(tick)
            elements.append(
                f'<text class="tick" x="{plot_x + plot_width + 6:.1f}" y="{y + 4:.1f}">{label}</text>'
            )
        if onset_step is not None and staleness in {12, 16, 20, 24, 28}:
            onset_x = x_map(float(onset_step))
            elements.append(
                f'<line class="onset" x1="{onset_x:.1f}" y1="{plot_y:.1f}" x2="{onset_x:.1f}" y2="{plot_y + plot_height:.1f}"/>'
            )
        if column == 0:
            elements.append(
                f'<text class="axis-label" transform="translate({panel_x + 13:.1f},{plot_y + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">rollout fraction (%)</text>'
            )
        if column == columns - 1:
            elements.append(
                f'<text class="axis-label" transform="translate({plot_x + plot_width + 42:.1f},{plot_y + plot_height / 2:.1f}) rotate(90)" text-anchor="middle">response length mean</text>'
            )
        if row == rows - 1:
            elements.append(
                f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height - 5:.1f}" text-anchor="middle">step</text>'
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def read_reward_conditioned_response_lengths() -> dict[int, dict[str, dict[int, float]]]:
    if not REWARD_CONDITIONED_PATH.is_file():
        raise FileNotFoundError(REWARD_CONDITIONED_PATH)
    series: dict[int, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    metric_columns = (
        "response_length_mean",
        "reward_1_response_length_mean",
        "reward_0_truncated_response_length_mean",
        "reward_0_non_truncated_response_length_mean",
        "reward_1_response_length_p10",
        "reward_1_response_length_p90",
        "reward_0_truncated_response_length_p10",
        "reward_0_truncated_response_length_p90",
        "reward_0_non_truncated_response_length_p10",
        "reward_0_non_truncated_response_length_p90",
        "reward_1_fraction",
        "reward_0_truncated_fraction",
        "reward_0_non_truncated_fraction",
    )
    with REWARD_CONDITIONED_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arm_match = ARM_PATTERN.fullmatch(row["arm"])
            if arm_match is None or (int(arm_match["train"]), int(arm_match["rollout"])) != (1, 7):
                continue
            staleness = int(arm_match["staleness"])
            step = int(row["training_step"])
            for metric in metric_columns:
                if row[metric] != "":
                    series[staleness][metric][step] = float(row[metric])
    return {
        staleness: {metric: dict(values) for metric, values in metrics.items()}
        for staleness, metrics in series.items()
    }


def linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of an empty population")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def read_guess_phrase_probabilities() -> dict[int, dict[int, list[float]]]:
    if not GUESS_PHRASE_PATH.is_file():
        raise FileNotFoundError(GUESS_PHRASE_PATH)
    grouped: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with GUESS_PHRASE_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[int(row["staleness"])][int(row["offset_from_guess"])].append(
                float(row["token_probability"])
            )
    return {
        staleness: {offset: list(values) for offset, values in offsets.items()}
        for staleness, offsets in grouped.items()
    }


def render_guess_phrase_probabilities(
    grouped: dict[int, dict[int, list[float]]],
) -> str:
    staleness_levels = sorted(grouped)
    columns = 2
    rows = math.ceil(len(staleness_levels) / columns)
    panel_width, panel_height = 450.0, 210.0
    column_gap, row_gap = 42.0, 28.0
    left, top = 72.0, 74.0
    width = int(left + columns * panel_width + column_gap + 30)
    height = int(top + rows * panel_height + (rows - 1) * row_gap + 26)
    elements = svg_canvas(width, height)
    elements.extend(
        [
            f'<text class="column-label" x="{left:.1f}" y="20">Sampled-token probability around explicit guess phrases</text>',
            f'<text class="legend" x="{left:.1f}" y="42">offset 0 = final token of I will guess / I guess / I\'ll guess / Let me guess · band = p10–p90 · line = median</text>',
        ]
    )
    for index, staleness in enumerate(staleness_levels):
        row, column = divmod(index, columns)
        panel_x = left + column * (panel_width + column_gap)
        panel_y = top + row * (panel_height + row_gap)
        plot_x, plot_y = panel_x + 50.0, panel_y + 30.0
        plot_width, plot_height = panel_width - 65.0, panel_height - 68.0
        x_map = lambda value, base=plot_x: base + plot_width * (value + 12.0) / 24.0
        y_map = lambda value, base=plot_y: base + plot_height * (1.0 - value)
        offsets = sorted(offset for offset in grouped[staleness] if -12 <= offset <= 12)
        p10 = [
            (float(offset), linear_quantile(grouped[staleness][offset], 0.10))
            for offset in offsets
        ]
        median = [
            (float(offset), linear_quantile(grouped[staleness][offset], 0.50))
            for offset in offsets
        ]
        p90 = [
            (float(offset), linear_quantile(grouped[staleness][offset], 0.90))
            for offset in offsets
        ]
        occurrence_count = len(grouped[staleness].get(0, ()))
        elements.append(
            f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 14:.1f}" text-anchor="middle">S={staleness} · n={occurrence_count:,}</text>'
        )
        for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
            y = y_map(tick)
            elements.extend(
                [
                    f'<line class="grid" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}"/>',
                    f'<text class="tick" x="{plot_x - 7:.1f}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>',
                ]
            )
        for tick in (-12, -8, -4, 0, 4, 8, 12):
            x = x_map(float(tick))
            elements.append(
                f'<text class="tick" x="{x:.1f}" y="{plot_y + plot_height + 17:.1f}" text-anchor="middle">{tick}</text>'
            )
        guess_x = x_map(0.0)
        band_points = p10 + list(reversed(p90))
        series_color = color(staleness)
        elements.extend(
            [
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="onset" x1="{guess_x:.1f}" y1="{plot_y:.1f}" x2="{guess_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<polygon fill="{series_color}" opacity="0.12" points="{polyline(band_points, x_map, y_map)}"/>',
                f'<polyline class="quantile-high" stroke="{series_color}" points="{polyline(median, x_map, y_map)}"/>',
            ]
        )
        if column == 0:
            elements.append(
                f'<text class="axis-label" transform="translate({panel_x + 12:.1f},{plot_y + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">realized token probability</text>'
            )
        if row == rows - 1:
            elements.append(
                f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height - 3:.1f}" text-anchor="middle">token offset</text>'
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def render_reward_conditioned_response_lengths(
    series: dict[int, dict[str, dict[int, float]]],
) -> str:
    staleness_levels = sorted(series)
    if not staleness_levels:
        raise ValueError("no reward-conditioned response-length series")
    panels = (
        ("Response length (mean)", "response_length_mean", None, "tokens"),
        ("Reward 1 response length (mean)", "reward_1_response_length_mean", None, "tokens"),
        (
            "Reward 0 · truncated response length (mean)",
            "reward_0_truncated_response_length_mean",
            None,
            "tokens",
        ),
        (
            "Reward 0 · non-truncated response length (mean)",
            "reward_0_non_truncated_response_length_mean",
            None,
            "tokens",
        ),
        (
            "Reward 1 response length (p10–p90)",
            "reward_1_response_length_p10",
            "reward_1_response_length_p90",
            "tokens",
        ),
        (
            "Reward 0 · truncated response length (p10–p90)",
            "reward_0_truncated_response_length_p10",
            "reward_0_truncated_response_length_p90",
            "tokens",
        ),
        (
            "Reward 0 · non-truncated response length (p10–p90)",
            "reward_0_non_truncated_response_length_p10",
            "reward_0_non_truncated_response_length_p90",
            "tokens",
        ),
        ("Sample fraction · Reward 1", "reward_1_fraction", None, "fraction"),
        (
            "Sample fraction · Reward 0 (truncated)",
            "reward_0_truncated_fraction",
            None,
            "fraction",
        ),
        (
            "Sample fraction · Reward 0 (non-truncated)",
            "reward_0_non_truncated_fraction",
            None,
            "fraction",
        ),
    )
    width = 1120
    left, right, top = 82.0, 32.0, 58.0
    panel_height, panel_gap = 142.0, 32.0
    height = int(top + len(panels) * panel_height + (len(panels) - 1) * panel_gap + 28)
    plot_x = left
    plot_width = width - left - right
    plot_height = panel_height - 34.0
    max_step = 300.0
    max_response_len = 16_384.0
    x_map = lambda value: plot_x + plot_width * (value - 1.0) / (max_step - 1.0)
    elements = svg_canvas(width, height)
    legend_x = left
    for staleness in staleness_levels:
        elements.extend(
            [
                f'<line class="smooth" x1="{legend_x:.1f}" y1="24" x2="{legend_x + 25:.1f}" y2="24" stroke="{color(staleness)}"/>',
                f'<text class="legend" x="{legend_x + 31:.1f}" y="28">S={staleness}</text>',
            ]
        )
        legend_x += 82.0
    elements.extend(
        [
            f'<line class="raw" x1="{legend_x + 5:.1f}" y1="24" x2="{legend_x + 30:.1f}" y2="24" stroke="#555"/>',
            f'<text class="legend" x="{legend_x + 36:.1f}" y="28">per-step</text>',
            f'<line class="smooth" x1="{legend_x + 103:.1f}" y1="24" x2="{legend_x + 128:.1f}" y2="24" stroke="#555"/>',
            f'<text class="legend" x="{legend_x + 134:.1f}" y="28">trailing-5 / p90</text>',
        ]
    )
    for panel_index, (label, lower_metric, upper_metric, unit) in enumerate(panels):
        panel_y = top + panel_index * (panel_height + panel_gap)
        plot_y = panel_y + 24.0
        y_upper = 1.0 if unit == "fraction" else 17_200.0
        y_for_panel = lambda value, base=plot_y, upper=y_upper: (
            base + plot_height * (upper - value) / upper
        )
        elements.append(
            f'<text class="metric-label" x="{plot_x:.1f}" y="{panel_y + 12:.1f}">{label}</text>'
        )
        ticks = (
            (0.0, 0.25, 0.50, 0.75, 1.0)
            if unit == "fraction"
            else (0.0, 4_000.0, 8_000.0, 12_000.0, 16_000.0)
        )
        for tick in ticks:
            y = y_for_panel(tick)
            tick_label = (
                f"{tick * 100:.0f}%" if unit == "fraction" else f"{int(tick / 1000)}K"
            )
            elements.extend(
                [
                    f'<line class="grid" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}"/>',
                    f'<text class="tick" x="{plot_x - 8:.1f}" y="{y + 4:.1f}" text-anchor="end">{tick_label}</text>',
                ]
            )
        elements.extend(
            [
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
            ]
        )
        if unit == "tokens":
            max_response_y = y_for_panel(max_response_len)
            elements.extend(
                [
                    f'<line class="maxlen" x1="{plot_x:.1f}" y1="{max_response_y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{max_response_y:.1f}"/>',
                    f'<text class="tick" x="{plot_x + plot_width - 3:.1f}" y="{max_response_y - 4:.1f}" text-anchor="end">max response len = 16K</text>',
                ]
            )
        for staleness in staleness_levels:
            values = {
                step: value
                for step, value in series[staleness].get(lower_metric, {}).items()
                if 1 <= step <= max_step
            }
            if not values:
                continue
            series_color = color(staleness)
            if upper_metric is None:
                elements.extend(
                    [
                        f'<polyline class="raw" stroke="{series_color}" points="{polyline([(float(step), value) for step, value in sorted(values.items())], x_map, y_for_panel)}"/>',
                        f'<polyline class="smooth" stroke="{series_color}" points="{polyline(rolling(values), x_map, y_for_panel)}"/>',
                    ]
                )
                continue
            upper_values = {
                step: value
                for step, value in series[staleness].get(upper_metric, {}).items()
                if 1 <= step <= max_step
            }
            shared_steps = sorted(set(values) & set(upper_values))
            if not shared_steps:
                continue
            lower_smooth = dict(rolling({step: values[step] for step in shared_steps}))
            upper_smooth = dict(rolling({step: upper_values[step] for step in shared_steps}))
            band_points = [
                (float(step), lower_smooth[float(step)]) for step in shared_steps
            ] + [
                (float(step), upper_smooth[float(step)]) for step in reversed(shared_steps)
            ]
            elements.extend(
                [
                    f'<polygon class="quantile-band" fill="{series_color}" points="{polyline(band_points, x_map, y_for_panel)}"/>',
                    f'<polyline class="quantile-low" stroke="{series_color}" points="{polyline(list(lower_smooth.items()), x_map, y_for_panel)}"/>',
                    f'<polyline class="quantile-high" stroke="{series_color}" points="{polyline(list(upper_smooth.items()), x_map, y_for_panel)}"/>',
                ]
            )
        if panel_index == len(panels) - 1:
            for tick in (1, 50, 100, 150, 200, 250, 300):
                x = x_map(float(tick))
                elements.append(
                    f'<text class="tick" x="{x:.1f}" y="{plot_y + plot_height + 18:.1f}" text-anchor="middle">{tick}</text>'
                )
            elements.append(
                f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{plot_y + plot_height + 34:.1f}" text-anchor="middle">step</text>'
            )
        elements.append(
            f'<text class="axis-label" transform="translate(18,{plot_y + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">{"sample fraction" if unit == "fraction" else "tokens"}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


RESPONSE_WAVE_BOUNDS = {
    "rollout/response_len/mean": {
        (1, 7): (0.0, 17_000.0),
        (2, 6): (3_000.0, 9_000.0),
    },
    "rollout/response_len/p90": {
        (1, 7): (0.0, 17_000.0),
        (2, 6): (6_000.0, 18_000.0),
    },
}


def render_response_wave_metric(
    metric: str,
    histories: dict[Identity, dict[str, dict[int, float]]],
) -> str:
    metric_title = (
        "Response length (mean)"
        if metric == "rollout/response_len/mean"
        else "Response length (p90)"
    )
    selected = {
        series_identity: metrics
        for series_identity, metrics in histories.items()
        if series_identity.treatment == "zero-reward"
        and series_identity.max_response_len == 16_384
        and series_identity.arm != "s0-colocated"
        and series_identity.ratio_key in ALLOWED_RATIOS
        and metrics.get(metric)
    }
    staleness_levels = sorted({
        series_identity.staleness for series_identity in selected
    })
    reference_entry = next(
        (
            (series_identity, metrics)
            for series_identity, metrics in histories.items()
            if series_identity.treatment == "zero-reward"
            and series_identity.max_response_len == 16_384
            and series_identity.arm == "s0-colocated"
            and metrics.get(metric)
        ),
        None,
    )
    panel_width, panel_height = 520.0, 142.0
    column_gap, row_gap = 36.0, 18.0
    left, top = 78.0, 72.0
    width = int(left + 2 * panel_width + column_gap + 25)
    height = int(
        top
        + len(staleness_levels) * panel_height
        + (len(staleness_levels) - 1) * row_gap
        + 30
    )
    max_step = 300
    elements = svg_canvas(width, height)
    for column, ratio in enumerate(ALLOWED_RATIOS):
        panel_x = left + column * (panel_width + column_gap)
        elements.append(
            f'<text class="column-label" x="{panel_x + panel_width / 2:.1f}" y="24" '
            f'text-anchor="middle">Train:Rollout = {ratio[0]}:{ratio[1]}</text>'
        )
    elements.extend(
        [
            f'<line class="raw" x1="{left:.1f}" y1="51" x2="{left + 25:.1f}" y2="51" stroke="#555"/>',
            f'<text class="legend" x="{left + 31:.1f}" y="55">per-step</text>',
            f'<line class="smooth" x1="{left + 102:.1f}" y1="51" x2="{left + 127:.1f}" y2="51" stroke="#555"/>',
            f'<text class="legend" x="{left + 133:.1f}" y="55">trailing-5 mean</text>',
        ]
    )
    if reference_entry is not None:
        elements.extend(
            [
                f'<line class="reference" x1="{left + 247:.1f}" y1="51" x2="{left + 275:.1f}" y2="51"/>',
                f'<text class="legend" x="{left + 282:.1f}" y="55">colocated reference</text>',
            ]
        )
    for row_index, staleness in enumerate(staleness_levels):
        for column, ratio in enumerate(ALLOWED_RATIOS):
            panel_x = left + column * (panel_width + column_gap)
            panel_y = top + row_index * (panel_height + row_gap)
            plot_x, plot_y = panel_x + 58.0, panel_y + 24.0
            plot_width, plot_height = panel_width - 82.0, panel_height - 52.0
            y_lower, y_upper = RESPONSE_WAVE_BOUNDS[metric][ratio]
            span = y_upper - y_lower
            x_map = lambda value, base=plot_x: base + plot_width * (value - 1.0) / 299.0
            y_map = lambda value, base=plot_y: base + plot_height * (y_upper - value) / span
            for tick in axis_ticks(y_lower, y_upper):
                y = y_map(tick)
                elements.extend(
                    [
                        f'<line class="grid" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}"/>',
                        f'<text class="tick" x="{plot_x - 8:.1f}" y="{y + 4:.1f}" text-anchor="end">{format_tick(tick, span)}</text>',
                    ]
                )
            for tick in (1, 100, 200, 300):
                x = x_map(float(tick))
                elements.append(
                    f'<text class="tick" x="{x:.1f}" y="{plot_y + plot_height + 16:.1f}" text-anchor="middle">{tick}</text>'
                )
            elements.extend(
                [
                    f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 13:.1f}" text-anchor="middle">{metric_title} · S={staleness}</text>',
                    f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                    f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                ]
            )
            if y_lower <= 16_384 <= y_upper:
                max_response_y = y_map(16_384.0)
                elements.extend(
                    [
                        f'<line class="maxlen" x1="{plot_x:.1f}" y1="{max_response_y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{max_response_y:.1f}"/>',
                        f'<text class="tick" x="{plot_x + plot_width - 3:.1f}" y="{max_response_y - 4:.1f}" text-anchor="end">max response len = 16K</text>',
                    ]
                )
            if column == 0:
                elements.append(
                    f'<text class="axis-label" transform="translate({panel_x + 15:.1f},{plot_y + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">tokens</text>'
                )
            if row_index == len(staleness_levels) - 1:
                elements.append(
                    f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height + 20:.1f}" text-anchor="middle">step</text>'
                )
            entry = next(
                (
                    (series_identity, metrics)
                    for series_identity, metrics in selected.items()
                    if series_identity.ratio_key == ratio
                    and series_identity.staleness == staleness
                ),
                None,
            )
            if entry is None:
                elements.append(
                    f'<text class="tick" x="{plot_x + plot_width / 2:.1f}" y="{plot_y + plot_height / 2:.1f}" text-anchor="middle">not run</text>'
                )
                continue
            series_identity, metrics = entry
            values = {
                step: value
                for step, value in metrics[metric].items()
                if 1 <= step <= max_step
            }
            raw_points = [(float(step), value) for step, value in sorted(values.items())]
            smooth_points = rolling(values)
            elements.extend(
                [
                    f'<polyline class="raw" stroke="{color(staleness)}" points="{polyline(raw_points, x_map, y_map)}"/>',
                    f'<polyline class="smooth" stroke="{color(staleness)}" points="{polyline(smooth_points, x_map, y_map)}"/>',
                ]
            )
            if reference_entry is not None:
                _, reference_metrics = reference_entry
                reference_values = {
                    step: value
                    for step, value in reference_metrics[metric].items()
                    if 1 <= step <= max_step
                }
                if reference_values:
                    elements.append(
                        f'<polyline class="reference" points="{polyline(rolling(reference_values), x_map, y_map)}"/>'
                    )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def render_staleness_quantiles_t2r6(
    histories: dict[Identity, dict[str, dict[int, float]]],
) -> str:
    selected = {
        series_identity: metrics
        for series_identity, metrics in histories.items()
        if series_identity.treatment == "zero-reward"
        and series_identity.max_response_len == 16_384
        and series_identity.arm != "s0-colocated"
        and series_identity.ratio_key == (2, 6)
    }
    display_metrics = (
        ("staleness/total/p10", "Staleness total p10"),
        ("staleness/total/mean", "Staleness total mean"),
        ("staleness/total/p90", "Staleness total p90"),
    )
    panel_width, panel_height = 800.0, 220.0
    row_gap = 24.0
    left, top = 78.0, 70.0
    width = int(left + panel_width + 25)
    height = int(top + len(display_metrics) * panel_height + 2 * row_gap + 30)
    max_step = 300
    elements = svg_canvas(width, height)
    for index, staleness in enumerate(sorted(identity.staleness for identity in selected)):
        legend_x = left + index * 99.0
        elements.extend(
            [
                f'<line x1="{legend_x:.1f}" y1="40" x2="{legend_x + 25:.1f}" y2="40" stroke="{color(staleness)}" stroke-width="2.5"/>',
                f'<text class="legend" x="{legend_x + 31:.1f}" y="44">S={staleness}</text>',
            ]
        )
    for row_index, (metric, metric_title) in enumerate(display_metrics):
        panel_x = left
        panel_y = top + row_index * (panel_height + row_gap)
        plot_x, plot_y = panel_x + 58.0, panel_y + 30.0
        plot_width, plot_height = panel_width - 82.0, panel_height - 78.0
        x_map = lambda value: plot_x + plot_width * (value - 1.0) / 299.0
        y_map = lambda value: plot_y + plot_height * (8.0 - value) / 8.0
        for tick in (0.0, 2.0, 4.0, 6.0, 8.0):
            y = y_map(tick)
            elements.extend(
                [
                    f'<line class="grid" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}"/>',
                    f'<text class="tick" x="{plot_x - 8:.1f}" y="{y + 4:.1f}" text-anchor="end">{tick:.0f}</text>',
                ]
            )
        for tick in sorted({1, max_step, *range(50, max_step + 1, 50)}):
            x = x_map(float(tick))
            elements.append(
                f'<text class="tick" x="{x:.1f}" y="{plot_y + plot_height + 18:.1f}" text-anchor="middle">{tick}</text>'
            )
        elements.extend(
            [
                f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + 15:.1f}" text-anchor="middle">{metric_title}</text>',
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                f'<text class="axis-label" transform="translate({panel_x + 15:.1f},{plot_y + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">staleness</text>',
                f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height - 8:.1f}" text-anchor="middle">step</text>',
            ]
        )
        for series_identity, metrics in sorted(
            selected.items(), key=lambda item: item[0].staleness
        ):
            values = {
                step: value
                for step, value in metrics.get(metric, {}).items()
                if 1 <= step <= max_step
            }
            if not values:
                continue
            raw_points = [(float(step), value) for step, value in sorted(values.items())]
            elements.extend(
                [
                    f'<polyline class="raw" stroke="{color(series_identity.staleness)}" points="{polyline(raw_points, x_map, y_map)}"/>',
                    f'<polyline class="smooth" stroke="{color(series_identity.staleness)}" points="{polyline(rolling(values), x_map, y_map)}"/>',
                ]
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def throughput_summary_t2r6(
    histories: dict[Identity, dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    metric_names = (
        "throughput/generated_tokens_per_second",
        "throughput/useful_tokens_per_second",
        "throughput/optimizer_updates_per_second",
    )
    columns = {
        "throughput/generated_tokens_per_second": "generated_tokens_per_second",
        "throughput/useful_tokens_per_second": "useful_tokens_per_second",
        "throughput/optimizer_updates_per_second": "optimizer_updates_per_hour",
    }
    rows: list[dict[str, Any]] = []
    for series_identity, metrics in sorted(
        histories.items(), key=lambda item: item[0].staleness
    ):
        if (
            series_identity.treatment != "zero-reward"
            or series_identity.max_response_len != 16_384
            or series_identity.arm == "s0-colocated"
            or series_identity.ratio_key != (2, 6)
        ):
            continue
        row: dict[str, Any] = {
            "namespace": series_identity.namespace,
            "cohort": series_identity.cohort,
            "arm": series_identity.arm,
            "max_weight_staleness": series_identity.staleness,
            "trainer_nodes": 2,
            "rollout_nodes": 6,
        }
        observed_steps: set[int] = set()
        for metric in metric_names:
            values = metrics.get(metric, {})
            observed_steps.update(values)
            post_warmup = [value for step, value in sorted(values.items()) if step >= 20]
            late = post_warmup[-20:]
            column = columns[metric]
            scale = 3_600.0 if metric.endswith("optimizer_updates_per_second") else 1.0
            row[f"median_{column}"] = (
                statistics.median(post_warmup) * scale if post_warmup else ""
            )
            row[f"late20_mean_{column}"] = (
                statistics.fmean(late) * scale if late else ""
            )
        row["first_step"] = min(observed_steps) if observed_steps else ""
        row["last_step"] = max(observed_steps) if observed_steps else ""
        rows.append(row)
    return rows


def centered_residuals(values: dict[int, float], window: int = 51) -> list[tuple[int, float]]:
    ordered = sorted(values.items())
    radius = window // 2
    residuals: list[tuple[int, float]] = []
    for index, (step, value) in enumerate(ordered):
        start = max(0, index - radius)
        end = min(len(ordered), index + radius + 1)
        trend = statistics.fmean(item_value for _, item_value in ordered[start:end])
        residuals.append((step, value - trend))
    return residuals


def wave_statistics(values: dict[int, float]) -> tuple[float | None, float | None, float | None]:
    residuals = centered_residuals(values)
    if len(residuals) < 24:
        return None, None, None
    residual_values = [value for _, value in residuals]
    residual_std = statistics.pstdev(residual_values)
    best_period: float | None = None
    best_amplitude = -1.0
    for period in range(8, 81):
        cosine = sum(
            value * math.cos(2.0 * math.pi * step / period)
            for step, value in residuals
        )
        sine = sum(
            value * math.sin(2.0 * math.pi * step / period)
            for step, value in residuals
        )
        amplitude = 2.0 * math.hypot(cosine, sine) / len(residuals)
        if amplitude > best_amplitude:
            best_amplitude = amplitude
            best_period = float(period)
    return residual_std, best_period, best_amplitude


def aligned_correlation(first: dict[int, float], second: dict[int, float]) -> float | None:
    shared_steps = sorted(set(first) & set(second))
    if len(shared_steps) < 2:
        return None
    first_values = [first[step] for step in shared_steps]
    second_values = [second[step] for step in shared_steps]
    first_mean = statistics.fmean(first_values)
    second_mean = statistics.fmean(second_values)
    covariance = sum(
        (first_value - first_mean) * (second_value - second_mean)
        for first_value, second_value in zip(first_values, second_values, strict=True)
    )
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first_values))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second_values))
    if first_scale == 0.0 or second_scale == 0.0:
        return None
    return covariance / (first_scale * second_scale)


def response_wave_summary(
    histories: dict[Identity, dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series_identity, metrics in sorted(
        histories.items(),
        key=lambda item: (item[0].trainer_nodes, item[0].staleness),
    ):
        if (
            series_identity.treatment != "zero-reward"
            or series_identity.max_response_len != 16_384
            or series_identity.arm == "s0-colocated"
            or series_identity.ratio_key not in ALLOWED_RATIOS
        ):
            continue
        mean_values = metrics.get("rollout/response_len/mean", {})
        p90_values = metrics.get("rollout/response_len/p90", {})
        if not mean_values or not p90_values:
            continue
        mean_smooth = [value for _, value in rolling(mean_values)]
        p90_smooth = [value for _, value in rolling(p90_values)]
        mean_std, mean_period, mean_amplitude = wave_statistics(mean_values)
        p90_std, p90_period, p90_amplitude = wave_statistics(p90_values)
        rows.append(
            {
                "namespace": series_identity.namespace,
                "arm": series_identity.arm,
                "max_weight_staleness": series_identity.staleness,
                "trainer_nodes": series_identity.trainer_nodes,
                "rollout_nodes": series_identity.rollout_nodes,
                "last_step": max(set(mean_values) | set(p90_values)),
                "response_mean_rolling5_min": min(mean_smooth),
                "response_mean_rolling5_max": max(mean_smooth),
                "response_mean_centered51_residual_std": mean_std,
                "response_mean_dominant_period_steps": mean_period,
                "response_mean_dominant_sinusoid_amplitude": mean_amplitude,
                "response_p90_rolling5_min": min(p90_smooth),
                "response_p90_rolling5_max": max(p90_smooth),
                "response_p90_centered51_residual_std": p90_std,
                "response_p90_dominant_period_steps": p90_period,
                "response_p90_dominant_sinusoid_amplitude": p90_amplitude,
                "mean_p90_stepwise_correlation": aligned_correlation(mean_values, p90_values),
            }
        )
    return rows


def window_mean(values: dict[int, float], *, first: bool, window: int = 20) -> float | None:
    ordered = [value for _, value in sorted(values.items())]
    if not ordered:
        return None
    selected = ordered[:window] if first else ordered[-window:]
    return statistics.fmean(selected)


def series_summary(histories: dict[Identity, dict[str, dict[int, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series_identity, metrics in sorted(
        histories.items(),
        key=lambda item: (
            item[0].treatment,
            item[0].max_response_len,
            item[0].trainer_nodes,
            item[0].staleness,
        ),
    ):
        reward_values = metrics.get("rollout/raw_reward", {})
        reward_smooth = rolling(reward_values)
        peak_step, peak_reward = max(reward_smooth, key=lambda item: item[1]) if reward_smooth else (None, None)
        steps = sorted({step for values in metrics.values() for step in values})
        early_reward = window_mean(reward_values, first=True)
        late_reward = window_mean(reward_values, first=False)
        late_response = window_mean(metrics.get("rollout/response_len/mean", {}), first=False)
        rows.append(
            {
                "namespace": series_identity.namespace,
                "cohort": series_identity.cohort,
                "treatment": series_identity.treatment,
                "rollout_max_response_len": series_identity.max_response_len,
                "arm": series_identity.arm,
                "max_weight_staleness": series_identity.staleness,
                "trainer_nodes": series_identity.trainer_nodes,
                "rollout_nodes": series_identity.rollout_nodes,
                "first_training_step": min(steps) if steps else "",
                "last_training_step": max(steps) if steps else "",
                "observed_steps": len(steps),
                "early20_raw_reward": early_reward if early_reward is not None else "",
                "late20_raw_reward": late_reward if late_reward is not None else "",
                "peak_rolling5_raw_reward": peak_reward if peak_reward is not None else "",
                "peak_reward_step": int(peak_step) if peak_step is not None else "",
                "late20_reward_drop_from_peak": (
                    peak_reward - late_reward
                    if peak_reward is not None and late_reward is not None
                    else ""
                ),
                "late20_response_len": late_response if late_response is not None else "",
                "late20_response_fraction_of_max": (
                    late_response / series_identity.max_response_len
                    if late_response is not None
                    else ""
                ),
                "late20_truncated_ratio": window_mean(
                    metrics.get("rollout/truncated_ratio", {}), first=False
                )
                or 0.0,
                "late20_staleness_total_mean": window_mean(
                    metrics.get("staleness/total/mean", {}), first=False
                )
                or 0.0,
            }
        )
    return rows


def downstream_values(
    scores: dict[Identity, dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series_identity, metrics in sorted(
        scores.items(),
        key=lambda item: (
            item[0].treatment,
            item[0].max_response_len,
            item[0].trainer_nodes,
            item[0].staleness,
        ),
    ):
        if (
            series_identity.ratio_key not in ALLOWED_RATIOS
            and series_identity.arm != "s0-colocated"
        ):
            continue
        steps = sorted({step for values in metrics.values() for step in values})
        for step in steps:
            rows.append(
                {
                    "namespace": series_identity.namespace,
                    "cohort": series_identity.cohort,
                    "treatment": series_identity.treatment,
                    "rollout_max_response_len": series_identity.max_response_len,
                    "arm": series_identity.arm,
                    "max_weight_staleness": series_identity.staleness,
                    "trainer_nodes": series_identity.trainer_nodes,
                    "rollout_nodes": series_identity.rollout_nodes,
                    "step": step,
                    "aime24_percent": metrics.get("aime24_percent", {}).get(step, ""),
                    "aime25_percent": metrics.get("aime25_percent", {}).get(step, ""),
                    "aime26_percent": metrics.get("aime26_percent", {}).get(step, ""),
                    "aime24_25_26_mean_percent": metrics.get(AIME_MEAN_METRIC, {}).get(step, ""),
                }
            )
    return rows


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    configured_sources = sources()
    histories = read_histories(configured_sources)
    downstream = read_downstream(configured_sources)
    figures = OUTPUT_ROOT / "figures"
    training_groups = sorted(
        {(series_identity.treatment, series_identity.max_response_len) for series_identity in histories}
    )
    generated: list[str] = []
    for treatment, max_response_len in training_groups:
        name = f"training-and-aime-{treatment}-maxlen{max_response_len}.svg"
        atomic_write(
            figures / name,
            render_training_group(treatment, max_response_len, histories, downstream),
        )
        generated.append(name)
    wave_figures = (
        (
            "rollout/response_len/mean",
            "zero-reward-response-length-mean-waves-maxlen16384.svg",
        ),
        (
            "rollout/response_len/p90",
            "zero-reward-response-length-p90-waves-maxlen16384.svg",
        ),
    )
    for metric, name in wave_figures:
        atomic_write(figures / name, render_response_wave_metric(metric, histories))
        generated.append(name)
    staleness_quantile_name = (
        "zero-reward-training-total-staleness-quantiles-t2r6-maxlen16384.svg"
    )
    atomic_write(
        figures / staleness_quantile_name,
        render_staleness_quantiles_t2r6(histories),
    )
    generated.append(staleness_quantile_name)
    guess_sample_name = "zero-reward-guess-sample-fraction-t1r7-maxlen16384.svg"
    atomic_write(
        figures / guess_sample_name,
        render_guess_sample_fraction(
            read_guess_sample_fractions(),
            read_max_response_length_fractions(),
            histories,
        ),
    )
    generated.append(guess_sample_name)
    reward_conditioned_name = (
        "zero-reward-response-length-by-reward-t1r7-maxlen16384.svg"
    )
    atomic_write(
        figures / reward_conditioned_name,
        render_reward_conditioned_response_lengths(
            read_reward_conditioned_response_lengths()
        ),
    )
    generated.append(reward_conditioned_name)
    guess_phrase_name = (
        "zero-reward-guess-phrase-token-probability-t1r7-maxlen16384.svg"
    )
    atomic_write(
        figures / guess_phrase_name,
        render_guess_phrase_probabilities(read_guess_phrase_probabilities()),
    )
    generated.append(guess_phrase_name)
    obsolete_names = (
        "training-staleness-and-raw-reward-zero-reward-maxlen16384.svg",
        "training-none-maxlen16384.svg",
        "training-zero-loss-maxlen16384.svg",
        "training-zero-reward-maxlen16384.svg",
        "training-zero-reward-maxlen32768.svg",
        "downstream-zero-reward-maxlen16384.svg",
        "downstream-zero-reward-maxlen32768.svg",
    )
    for obsolete_name in obsolete_names:
        (figures / obsolete_name).unlink(missing_ok=True)
    summary_rows = series_summary(histories)
    downstream_rows = downstream_values(downstream)
    wave_rows = response_wave_summary(histories)
    throughput_rows = throughput_summary_t2r6(histories)
    write_csv(OUTPUT_ROOT / "series-summary.csv", summary_rows)
    write_csv(OUTPUT_ROOT / "downstream-values.csv", downstream_rows)
    write_csv(OUTPUT_ROOT / "response-length-wave-summary.csv", wave_rows)
    write_csv(OUTPUT_ROOT / "zero-reward-t2r6-throughput-summary.csv", throughput_rows)
    lines = [
        "# Cross-cohort W&B and downstream diagnostics",
        "",
        "Each figure combines mean/p90 response length, raw reward, training total mean staleness, and AIME24/25/26 macro mean.",
        "Training curves use exact per-step W&B exports overlaid with newer local metrics.jsonl records; bold curves are trailing-5 means.",
        "AIME curves use completed NeMo Skills task outputs without smoothing; labels show the latest completed mean.",
        (
            f"Data-quality exclusion: {EXCLUDED_RUN_ARM} steps "
            f">={EXCLUDED_RUN_START_STEP} are excluded from every figure and "
            "derived table. The directly observed bad segment was steps "
            f"{EXCLUDED_RUN_START_STEP}-{EXCLUDED_RUN_OBSERVED_END_STEP} in "
            f"Slurm job {EXCLUDED_RUN_JOB_ID} / W&B run {EXCLUDED_RUN_WANDB_ID}; "
            "its median logged actor throughput was 87.9 TFLOP/s/GPU versus "
            "332.7/333.8 for concurrent S=16/S=20 controls. The exclusion is "
            "open-ended because later checkpoints inherit this optimizer "
            "trajectory. Raw extraction tables are deliberately preserved."
        ),
        "Only Train:Rollout=1:7 and 2:6 are plotted. Treatment-matched colocated response-length, reward, and AIME baselines are repeated in each applicable panel.",
        "The staleness-aware treatment keeps zero reward on truncated samples, keeps their base loss enabled, and scales the post-TIS objective above safe training staleness 4.",
        "The original S=8 grid used tbq1000 and is shown for provenance, not as a controlled S>=8 comparison.",
        "AIME panels include every currently complete three-task evaluation suite; partial task suites are excluded until AIME24/25/26 all finish.",
        "The two response-length wave figures use per-staleness small multiples, a fixed scale within each Train:Rollout column, and a treatment-matched colocated reference.",
        "The 2:6 staleness-quantile figure uses directly logged mean/p90 values and reconstructs p10 from the directly logged count histogram at each step.",
        "The 1:7 short-response figure defines a short response as at most 256 response tokens, zooms S=1/2/4/8 to 0-0.5%, and restricts S=12/16/20/24/28 to the pre-1% interval so its relation to response-length decline remains visible.",
        "The same figure separately shows response_length=16,384 as `Exceed max response len`, zoomed to 0-20% and overlaid with response-length mean; the dashboard parquets do not contain an explicit EOS or termination-reason field.",
        "The reward-conditioned response-length figure uses trained queue records with aligned scalar reward, response length, and generation status. It separates reward=1, reward=0 truncated, and reward=0 non-truncated response lengths and shows each population fraction.",
        "The guess-phrase probability figure shows the probability of each realized sampled token around explicit guess phrases; full-vocabulary logits were not saved, so it cannot recover probabilities of unchosen alternative tokens.",
        "The truncated-zero objective figure reconstructs `sum(response loss tokens * abs(group-normalized GRPO advantage))` for each saved trained batch. It plots the truncated-zero share of this pre-TIS absolute policy-gradient objective mass together with its exact loss-token share and mean response length; gray intervals have no GRPO objective mass because every group advantage is zero.",
        "Per-token train-policy/rollout-policy TIS weights and the final clipped objective were not saved, so this is a pre-TIS diagnostic rather than exact post-TIS loss attribution or a causal estimate.",
        "",
        "## Figures",
        "",
        *[f"- [figures/{name}](figures/{name})" for name in generated],
        "- [figures/truncated-zero-reward-objective-and-response-length-all-settings.svg](figures/truncated-zero-reward-objective-and-response-length-all-settings.svg)",
        "",
        "## Tables",
        "",
        "- [training-run-exclusions.csv](training-run-exclusions.csv)",
        "- [excluded-run-report.md](excluded-run-report.md)",
        "- [series-summary.csv](series-summary.csv)",
        "- [downstream-values.csv](downstream-values.csv)",
        "- [response-length-wave-summary.csv](response-length-wave-summary.csv)",
        "- [zero-reward-t2r6-throughput-summary.csv](zero-reward-t2r6-throughput-summary.csv)",
        "- [response-length-quantiles.csv](response-length-quantiles.csv)",
        "- [guess-sample-fractions.csv](guess-sample-fractions.csv)",
        "- [reward-conditioned-response-lengths.csv](reward-conditioned-response-lengths.csv)",
        "- [guess-phrase-token-probabilities.csv](guess-phrase-token-probabilities.csv)",
        "- [truncated-zero-reward-objective.csv](truncated-zero-reward-objective.csv)",
        "",
    ]
    atomic_write(OUTPUT_ROOT / "README.md", "\n".join(lines))
    print(
        f"series={len(histories)} downstream_series={len(downstream)} "
        f"downstream_rows={len(downstream_rows)} figures={len(generated)}"
    )


if __name__ == "__main__":
    main()
