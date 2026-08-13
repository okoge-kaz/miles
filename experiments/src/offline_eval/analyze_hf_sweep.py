#!/usr/bin/env python3
"""Summarize and compare a nested AIME checkpoint-evaluation sweep.

Expected layout: EVAL_ROOT/SETTING[/SUBSETTING...]/step-N/aime{24,25,26}.jsonl.
Requires numpy, scipy, and matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BENCHMARKS = ("aime24", "aime25", "aime26")
STEP_PATTERN = re.compile(r"step-(\d+)$")


@dataclass(frozen=True)
class Evaluation:
    setting: str
    step: int
    values: dict[str, np.ndarray]
    truncated: dict[str, float]


def _load_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records.sort(key=lambda record: int(record["index"]))
    assert len(records) == 30, f"{path}: expected 30 records, found {len(records)}"
    assert [int(record["index"]) for record in records] == list(range(30)), (
        f"{path}: prompt indices are not exactly 0..29"
    )
    return records


def discover(eval_root: Path) -> list[Evaluation]:
    evaluations: list[Evaluation] = []
    for step_dir in sorted(eval_root.glob("**/step-*")):
        match = STEP_PATTERN.fullmatch(step_dir.name)
        if not step_dir.is_dir() or match is None:
            continue
        values: dict[str, np.ndarray] = {}
        truncated: dict[str, float] = {}
        for benchmark in BENCHMARKS:
            records = _load_records(step_dir / f"{benchmark}.jsonl")
            values[benchmark] = np.asarray([float(record["pass_rate"]) for record in records])
            fractions = [
                float(record["truncated_frac"])
                for record in records
                if record.get("truncated_frac") is not None
            ]
            truncated[benchmark] = float(np.mean(fractions)) if fractions else math.nan
        evaluations.append(
            Evaluation(
                setting=step_dir.parent.relative_to(eval_root).as_posix(),
                step=int(match.group(1)),
                values=values,
                truncated=truncated,
            )
        )
    assert evaluations, f"no complete evaluations found under {eval_root}"
    return sorted(evaluations, key=lambda item: (item.setting, item.step))


def _pooled(evaluation: Evaluation) -> np.ndarray:
    return np.concatenate([evaluation.values[name] for name in BENCHMARKS])


def _summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(stats.sem(values))


def write_score_tables(evaluations: list[Evaluation], out_dir: Path) -> None:
    with (out_dir / "scores_by_year.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("setting", "step", "benchmark", "n", "score", "se", "truncated_frac"))
        for evaluation in evaluations:
            for benchmark in BENCHMARKS:
                mean, se = _summary(evaluation.values[benchmark])
                writer.writerow(
                    (
                        evaluation.setting,
                        evaluation.step,
                        benchmark,
                        len(evaluation.values[benchmark]),
                        mean,
                        se,
                        evaluation.truncated[benchmark],
                    )
                )

    with (out_dir / "scores_mean.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("setting", "step", "n", "mean_score", "se"))
        for evaluation in evaluations:
            mean, se = _summary(_pooled(evaluation))
            writer.writerow((evaluation.setting, evaluation.step, 90, mean, se))


def _paired_test(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    assert left.shape == right.shape
    difference = left - right
    if np.all(difference == 0):
        return 0.0, 1.0
    result = stats.ttest_rel(left, right)
    p_value = float(result.pvalue) if math.isfinite(float(result.pvalue)) else 1.0
    return float(np.mean(difference)), p_value


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def pairwise_tests(evaluations: list[Evaluation]) -> list[dict]:
    lookup = {(item.setting, item.step): item for item in evaluations}
    settings = sorted({item.setting for item in evaluations})
    rows: list[dict] = []
    for step in sorted({item.step for item in evaluations}):
        present = [setting for setting in settings if (setting, step) in lookup]
        for benchmark in (*BENCHMARKS, "mean"):
            family: list[dict] = []
            for left_name, right_name in combinations(present, 2):
                left_eval = lookup[left_name, step]
                right_eval = lookup[right_name, step]
                left = _pooled(left_eval) if benchmark == "mean" else left_eval.values[benchmark]
                right = _pooled(right_eval) if benchmark == "mean" else right_eval.values[benchmark]
                effect, p_value = _paired_test(left, right)
                family.append(
                    {
                        "step": step,
                        "benchmark": benchmark,
                        "left": left_name,
                        "right": right_name,
                        "effect": effect,
                        "p_raw": p_value,
                    }
                )
            adjusted = _holm_adjust([row["p_raw"] for row in family])
            for row, p_holm in zip(family, adjusted, strict=True):
                row["p_holm"] = p_holm
                row["significant_0_05"] = p_holm < 0.05
                rows.append(row)
    return rows


def write_pairwise(rows: list[dict], out_dir: Path) -> None:
    fields = ("step", "benchmark", "left", "right", "effect", "p_raw", "p_holm", "significant_0_05")
    with (out_dir / "pairwise_significance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectories(evaluations: list[Evaluation], out_dir: Path) -> None:
    settings = sorted({item.setting for item in evaluations})
    groups = sorted({setting.rsplit("/", maxsplit=1)[0] for setting in settings})
    assert len(groups) == 4, f"expected four staleness groups, found {len(groups)}"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, groups, strict=True):
        for setting in (name for name in settings if name.startswith(f"{group}/")):
            points = [item for item in evaluations if item.setting == setting]
            steps = np.asarray([item.step for item in points])
            means, errors = zip(*[_summary(_pooled(item)) for item in points], strict=True)
            means_array = np.asarray(means) * 100
            confidence = np.asarray(errors) * 1.96 * 100
            (line,) = axis.plot(
                steps,
                means_array,
                marker="o",
                linestyle="-",
                linewidth=2.2,
                label=setting.rsplit("/", maxsplit=1)[1],
                zorder=3,
            )
            axis.fill_between(
                steps,
                means_array - confidence,
                means_array + confidence,
                color=line.get_color(),
                alpha=0.025,
                linewidth=0,
                zorder=1,
            )
        staleness = group.removeprefix("max-weight-staleness-").removesuffix("-from-prefill")
        axis.set_title(f"Max weight staleness = {staleness}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
    fig.supxlabel("Checkpoint step")
    fig.supylabel("Mean AIME 2024/2025/2026 score (%)")
    fig.suptitle("Off-policy checkpoint evaluation trajectories (mean ± 95% CI)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(out_dir / "mean_trajectory.png", dpi=180)
    fig.savefig(out_dir / "mean_trajectory.pdf")
    plt.close(fig)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _trajectory_analysis(evaluations: list[Evaluation]) -> list[str]:
    lines: list[str] = []
    for setting in sorted({item.setting for item in evaluations}):
        points = [item for item in evaluations if item.setting == setting]
        scores = np.asarray([_summary(_pooled(item))[0] for item in points])
        steps = np.asarray([item.step for item in points], dtype=float)
        best_index = int(np.argmax(scores))
        delta = scores[-1] - scores[0]
        slope = float(np.polyfit(steps, scores, 1)[0]) if len(points) > 1 else math.nan
        lines.append(
            f"- {setting}: first to last {_percent(scores[0])} to {_percent(scores[-1])} "
            f"(delta {100 * delta:+.1f} pp), peak {_percent(scores[best_index])} at step "
            f"{int(steps[best_index])}, linear slope {100 * slope:+.2f} pp/step."
        )
    return lines


def write_report(evaluations: list[Evaluation], tests: list[dict], out_dir: Path) -> None:
    lookup = {(item.setting, item.step): item for item in evaluations}
    settings = sorted({item.setting for item in evaluations})
    lines = [
        "# Hiso off-policy HF checkpoint evaluation",
        "",
        "Scores are avg@16 over 30 problems per AIME year. The mean pools all 90 "
        "problem-level avg@16 values; error bands use the standard error over problems.",
        "",
        "## Per-setting trajectories",
        "",
    ]
    for setting in settings:
        lines.extend(
            [
                f"### {setting}",
                "",
                "| step | AIME 2024 | AIME 2025 | AIME 2026 | mean |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for evaluation in (item for item in evaluations if item.setting == setting):
            scores = [_summary(evaluation.values[name])[0] for name in BENCHMARKS]
            mean = _summary(_pooled(evaluation))[0]
            lines.append(
                f"| {evaluation.step} | {_percent(scores[0])} | {_percent(scores[1])} "
                f"| {_percent(scores[2])} | {_percent(mean)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Combined mean trajectory",
            "",
            "| setting | step | mean | ±SE |",
            "|---|---:|---:|---:|",
        ]
    )
    for setting in settings:
        for step in sorted(item.step for item in evaluations if item.setting == setting):
            mean, se = _summary(_pooled(lookup[setting, step]))
            lines.append(f"| {setting} | {step} | {_percent(mean)} | {100 * se:.1f} pp |")

    significant = [
        row for row in tests if row["benchmark"] == "mean" and row["significant_0_05"]
    ]
    best = max(evaluations, key=lambda item: _summary(_pooled(item))[0])
    best_score = _summary(_pooled(best))[0]
    truncation_rates = [
        float(np.mean(list(evaluation.truncated.values()))) for evaluation in evaluations
    ]
    improving = sum(
        _summary(_pooled(points[-1]))[0] > _summary(_pooled(points[0]))[0]
        for setting in settings
        if (points := [item for item in evaluations if item.setting == setting])
    )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            f"- Best observed checkpoint: {best.setting} step {best.step} at "
            f"{_percent(best_score)}.",
            f"- {improving}/{len(settings)} settings improve from their first to last "
            "available checkpoint.",
            f"- Mean truncation fraction across the three benchmarks ranges from "
            f"{_percent(min(truncation_rates))} to {_percent(max(truncation_rates))} "
            "across checkpoints.",
            *_trajectory_analysis(evaluations),
            "",
            "## Pairwise significance",
            "",
            "Paired t-tests match the same 90 benchmark problems at each common step. "
            "Holm correction is applied within each step/benchmark family.",
            "",
        ]
    )
    if significant:
        lines.extend(
            [
                "| step | higher setting | lower setting | difference | Holm p |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for row in sorted(significant, key=lambda item: (item["step"], item["p_holm"])):
            if row["effect"] >= 0:
                higher, lower, effect = row["left"], row["right"], row["effect"]
            else:
                higher, lower, effect = row["right"], row["left"], -row["effect"]
            lines.append(
                f"| {row['step']} | {higher} | {lower} | {100 * effect:.1f} pp "
                f"| {row['p_holm']:.3g} |"
            )
    else:
        lines.append("No pairwise mean-score difference survives Holm correction at alpha=0.05.")
    lines.extend(
        [
            "",
            "These tests quantify evaluation uncertainty for the checkpoints that were run. "
            "There is one training run per setting, so they do not establish significance "
            "over training-seed variability.",
            "",
            "![Mean score trajectories](mean_trajectory.png)",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluations = discover(args.eval_root)
    tests = pairwise_tests(evaluations)
    write_score_tables(evaluations, args.out)
    write_pairwise(tests, args.out)
    plot_trajectories(evaluations, args.out)
    write_report(evaluations, tests, args.out)
    settings = {item.setting for item in evaluations}
    print(f"wrote analysis for {len(evaluations)} checkpoints across {len(settings)} settings to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
