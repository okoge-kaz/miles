#!/usr/bin/env python3
"""Plot realized staleness and raw reward for the 1:7 and 2:6 cohorts."""

from __future__ import annotations

import html
from pathlib import Path

import plot_cross_cohort as cross_cohort


OUTPUT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = (
    OUTPUT_ROOT
    / "figures"
    / "training-staleness-raw-reward-zero-reward-t1r7-t2r6-maxlen16384.svg"
)
STALENESS_ONLY_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "figures"
    / "training-staleness-zero-reward-t1r7-t2r6-maxlen16384.svg"
)
T1R7_STALENESS_ONLY_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "figures"
    / "training-staleness-zero-reward-t1r7-maxlen16384.svg"
)
T1R7_STALENESS_REWARD_OUTPUT_PATH = (
    OUTPUT_ROOT
    / "figures"
    / "training-staleness-raw-reward-zero-reward-t1r7-maxlen16384.svg"
)
T1R7_STALENESS_REWARD_SUMMARY_PATH = (
    OUTPUT_ROOT / "zero-reward-t1r7-staleness-reward-summary.csv"
)
TREATMENT = "zero-reward"
MAX_RESPONSE_LEN = 16_384
DISPLAY_METRICS = (
    ("staleness/total/mean", "Realized staleness", "staleness"),
    ("rollout/raw_reward", "Raw reward", "raw reward"),
)
STALENESS_UPPER_BY_RATIO = {
    (1, 7): 30.0,
    (2, 6): 6.0,
}
STALENESS_TICKS_BY_RATIO = {
    (1, 7): (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
    (2, 6): (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
}


def t1r7_staleness_reward_summary(
    histories: dict[cross_cohort.Identity, dict[str, dict[int, float]]],
) -> list[dict[str, object]]:
    """Return the late-window realized-staleness and reward table for 1:7."""
    rows = []
    for row in cross_cohort.series_summary(histories):
        if (
            row["treatment"] != TREATMENT
            or row["rollout_max_response_len"] != MAX_RESPONSE_LEN
            or row["trainer_nodes"] != 1
            or row["rollout_nodes"] != 7
        ):
            continue
        rows.append(
            {
                "arm": row["arm"],
                "max_weight_staleness": row["max_weight_staleness"],
                "last_training_step": row["last_training_step"],
                "late20_realized_staleness": row["late20_staleness_total_mean"],
                "late20_raw_reward": row["late20_raw_reward"],
            }
        )
    return rows


def render(
    histories: dict[cross_cohort.Identity, dict[str, dict[int, float]]],
    display_metrics: tuple[tuple[str, str, str], ...] = DISPLAY_METRICS,
    ratio_selection: tuple[tuple[int, int], ...] = cross_cohort.ALLOWED_RATIOS,
) -> str:
    """Render selected metrics and train-to-rollout ratios."""
    selected = {
        series_identity: metrics
        for series_identity, metrics in histories.items()
        if series_identity.treatment == TREATMENT
        and series_identity.max_response_len == MAX_RESPONSE_LEN
        and series_identity.ratio_key in ratio_selection
    }
    ratios = [
        ratio
        for ratio in ratio_selection
        if any(series_identity.ratio_key == ratio for series_identity in selected)
    ]
    if ratios != list(ratio_selection):
        raise ValueError(f"missing requested ratio histories: expected {ratio_selection}, got {ratios}")

    staleness_levels = sorted({series_identity.staleness for series_identity in selected})
    panel_width, panel_height = 590.0, 310.0
    column_gap, row_gap = 28.0, 12.0
    left = 0.0
    right = 8.0
    width = int(
        left
        + len(ratios) * panel_width
        + (len(ratios) - 1) * column_gap
        + right
    )
    legend_columns = min(len(staleness_levels), 9 if len(ratios) > 1 else 5)
    legend_rows = (len(staleness_levels) + legend_columns - 1) // legend_columns
    legend_row_height = 26.0
    heading_y = 52.0 + (legend_rows - 1) * legend_row_height
    top = 62.0 + (legend_rows - 1) * legend_row_height
    height = int(
        top
        + len(display_metrics) * panel_height
        + (len(display_metrics) - 1) * row_gap
        + 4.0
    )
    max_step = 300
    elements = cross_cohort.svg_canvas(width, height)
    elements.append(
        "<style>"
        ".axis-label{font-size:20px}"
        ".tick{font-size:17px}"
        ".column-label{font-size:22px}"
        ".metric-label{font-size:20px}"
        ".legend{font-size:18px}"
        ".axis{stroke-width:1.3}"
        ".grid{stroke-width:1}"
        ".raw{stroke-width:1.1;opacity:.16}"
        ".smooth{stroke-width:2.8}"
        "</style>"
    )
    for column, ratio in enumerate(ratios):
        panel_x = left + column * (panel_width + column_gap)
        elements.append(
            f'<text class="column-label" x="{panel_x + panel_width / 2:.1f}" '
            f'y="{heading_y:.1f}" text-anchor="middle">'
            f"Train:Rollout = {ratio[0]}:{ratio[1]}</text>"
        )

    legend_slot_width = 124.0 if legend_rows == 1 else 112.0
    for index, staleness in enumerate(staleness_levels):
        legend_row, legend_column = divmod(index, legend_columns)
        row_start = legend_row * legend_columns
        row_count = min(legend_columns, len(staleness_levels) - row_start)
        legend_start_x = (width - row_count * legend_slot_width) / 2.0
        legend_x = legend_start_x + legend_column * legend_slot_width + 5.0
        legend_y = 14.0 + legend_row * legend_row_height
        elements.extend(
            [
                f'<line x1="{legend_x:.1f}" y1="{legend_y:.1f}" '
                f'x2="{legend_x + 30:.1f}" y2="{legend_y:.1f}" '
                f'stroke="{cross_cohort.color(staleness)}" stroke-width="3"/>',
                f'<text class="legend" x="{legend_x + 39:.1f}" '
                f'y="{legend_y + 6:.1f}">S={staleness}</text>',
            ]
        )

    for row_index, (metric, metric_title, y_label) in enumerate(display_metrics):
        for column, ratio in enumerate(ratios):
            panel_x = left + column * (panel_width + column_gap)
            panel_y = top + row_index * (panel_height + row_gap)
            plot_x, plot_y = panel_x + 72.0, panel_y + 38.0
            plot_width, plot_height = panel_width - 86.0, panel_height - 101.0

            def x_map(
                value: float,
                base: float = plot_x,
                mapped_width: float = plot_width,
            ) -> float:
                return base + mapped_width * (value - 1.0) / (max_step - 1.0)

            y_lower, y_upper = (
                (0.0, STALENESS_UPPER_BY_RATIO[ratio])
                if metric == "staleness/total/mean"
                else (0.0, 1.0)
            )
            span = y_upper - y_lower

            def y_map(
                value: float,
                base: float = plot_y,
                mapped_height: float = plot_height,
                upper: float = y_upper,
                mapped_span: float = span,
            ) -> float:
                return base + mapped_height * (upper - value) / mapped_span

            y_ticks = (
                STALENESS_TICKS_BY_RATIO[ratio]
                if metric == "staleness/total/mean"
                else cross_cohort.axis_ticks(y_lower, y_upper)
            )
            for tick in y_ticks:
                y = y_map(tick)
                tick_label = (
                    f"{tick:g}"
                    if metric == "staleness/total/mean"
                    else cross_cohort.format_tick(tick, span)
                )
                elements.extend(
                    [
                        f'<line class="grid" x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_width:.1f}" y2="{y:.1f}"/>',
                        f'<text class="tick" x="{plot_x - 10:.1f}" '
                        f'y="{y + 6:.1f}" text-anchor="end">{tick_label}</text>',
                    ]
                )
            for tick in (1, 50, 100, 150, 200, 250, 300):
                x = x_map(float(tick))
                elements.append(
                    f'<text class="tick" x="{x:.1f}" '
                    f'y="{plot_y + plot_height + 24:.1f}" '
                    f'text-anchor="middle">{tick}</text>'
                )
            elements.extend(
                [
                    f'<text class="metric-label" x="{plot_x + plot_width / 2:.1f}" '
                    f'y="{panel_y + 22:.1f}" text-anchor="middle">'
                    f"{html.escape(metric_title)}</text>",
                    f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                    f'<line class="axis" x1="{plot_x:.1f}" y1="{plot_y + plot_height:.1f}" x2="{plot_x + plot_width:.1f}" y2="{plot_y + plot_height:.1f}"/>',
                    f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{panel_y + panel_height - 8:.1f}" text-anchor="middle">training step</text>',
                    f'<text class="axis-label" transform="translate({panel_x + 14:.1f},'
                    f'{plot_y + plot_height / 2:.1f}) rotate(-90)" '
                    f'text-anchor="middle">{html.escape(y_label)}</text>',
                ]
            )

            for series_identity, metrics in sorted(selected.items(), key=lambda item: item[0].staleness):
                if series_identity.ratio_key != ratio:
                    continue
                values = {step: value for step, value in metrics.get(metric, {}).items() if 1 <= step <= max_step}
                if not values:
                    continue
                raw_points = [(float(step), value) for step, value in sorted(values.items())]
                series_color = cross_cohort.color(series_identity.staleness)
                elements.extend(
                    [
                        f'<polyline class="raw" stroke="{series_color}" points="{cross_cohort.polyline(raw_points, x_map, y_map)}"/>',
                        f'<polyline class="smooth" stroke="{series_color}" points="{cross_cohort.polyline(cross_cohort.rolling(values), x_map, y_map)}"/>',
                    ]
                )

    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def main() -> None:
    histories = cross_cohort.read_histories(cross_cohort.sources())
    cross_cohort.atomic_write(OUTPUT_PATH, render(histories))
    cross_cohort.atomic_write(
        STALENESS_ONLY_OUTPUT_PATH,
        render(histories, display_metrics=(DISPLAY_METRICS[0],)),
    )
    cross_cohort.atomic_write(
        T1R7_STALENESS_ONLY_OUTPUT_PATH,
        render(
            histories,
            display_metrics=(DISPLAY_METRICS[0],),
            ratio_selection=((1, 7),),
        ),
    )
    cross_cohort.atomic_write(
        T1R7_STALENESS_REWARD_OUTPUT_PATH,
        render(histories, ratio_selection=((1, 7),)),
    )
    cross_cohort.write_csv(
        T1R7_STALENESS_REWARD_SUMMARY_PATH,
        t1r7_staleness_reward_summary(histories),
    )
    print(OUTPUT_PATH)
    print(STALENESS_ONLY_OUTPUT_PATH)
    print(T1R7_STALENESS_ONLY_OUTPUT_PATH)
    print(T1R7_STALENESS_REWARD_OUTPUT_PATH)
    print(T1R7_STALENESS_REWARD_SUMMARY_PATH)


if __name__ == "__main__":
    main()
