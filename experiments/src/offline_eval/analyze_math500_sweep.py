#!/usr/bin/env python3
"""Analyze MATH-500 trajectories for a nested checkpoint sweep."""

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

EXPECTED_PROBLEMS = 500
STEP_PATTERN = re.compile(r"step-(\d+)$")


@dataclass(frozen=True)
class Evaluation:
    setting: str
    step: int
    values: np.ndarray
    truncated: float


def load_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records.sort(key=lambda record: int(record["index"]))
    assert len(records) == EXPECTED_PROBLEMS, (
        f"{path}: expected {EXPECTED_PROBLEMS} records, found {len(records)}"
    )
    assert [int(record["index"]) for record in records] == list(range(EXPECTED_PROBLEMS)), (
        f"{path}: prompt indices are not exactly 0..{EXPECTED_PROBLEMS - 1}"
    )
    return records


def discover(eval_root: Path) -> list[Evaluation]:
    evaluations: list[Evaluation] = []
    for step_dir in sorted(eval_root.glob("**/step-*")):
        match = STEP_PATTERN.fullmatch(step_dir.name)
        result = step_dir / "math500.jsonl"
        if not step_dir.is_dir() or match is None or not result.is_file():
            continue
        records = load_records(result)
        values = np.asarray([float(record["pass_rate"]) for record in records])
        fractions = [
            float(record["truncated_frac"])
            for record in records
            if record.get("truncated_frac") is not None
        ]
        evaluations.append(
            Evaluation(
                setting=step_dir.parent.relative_to(eval_root).as_posix(),
                step=int(match.group(1)),
                values=values,
                truncated=float(np.mean(fractions)) if fractions else math.nan,
            )
        )
    assert evaluations, f"no complete MATH-500 evaluations found under {eval_root}"
    return sorted(evaluations, key=lambda item: (item.setting, item.step))


def summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(stats.sem(values))


def holm_adjust(p_values: list[float]) -> list[float]:
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
        family: list[dict] = []
        for left_name, right_name in combinations(present, 2):
            left = lookup[left_name, step].values
            right = lookup[right_name, step].values
            difference = left - right
            if np.all(difference == 0):
                p_value = 1.0
            else:
                result = stats.ttest_rel(left, right)
                p_value = float(result.pvalue) if math.isfinite(float(result.pvalue)) else 1.0
            family.append(
                {
                    "step": step,
                    "left": left_name,
                    "right": right_name,
                    "effect": float(np.mean(difference)),
                    "p_raw": p_value,
                }
            )
        for row, adjusted in zip(
            family,
            holm_adjust([item["p_raw"] for item in family]),
            strict=True,
        ):
            row["p_holm"] = adjusted
            row["significant_0_05"] = adjusted < 0.05
            rows.append(row)
    return rows


def trajectory_tests(evaluations: list[Evaluation]) -> list[dict]:
    rows: list[dict] = []
    for setting in sorted({item.setting for item in evaluations}):
        points = [item for item in evaluations if item.setting == setting]
        first, last = points[0], points[-1]
        difference = last.values - first.values
        if np.all(difference == 0):
            p_value = 1.0
        else:
            result = stats.ttest_rel(last.values, first.values)
            p_value = float(result.pvalue) if math.isfinite(float(result.pvalue)) else 1.0
        rows.append(
            {
                "setting": setting,
                "first_step": first.step,
                "last_step": last.step,
                "effect": float(np.mean(difference)),
                "p_raw": p_value,
            }
        )
    for row, adjusted in zip(
        rows,
        holm_adjust([item["p_raw"] for item in rows]),
        strict=True,
    ):
        row["p_holm"] = adjusted
        row["significant_0_05"] = adjusted < 0.05
    return rows


def write_tables(
    evaluations: list[Evaluation],
    tests: list[dict],
    trends: list[dict],
    out_dir: Path,
) -> None:
    with (out_dir / "math500_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("setting", "step", "n", "score", "se", "ci95", "truncated_frac"))
        for evaluation in evaluations:
            mean, se = summary(evaluation.values)
            writer.writerow(
                (
                    evaluation.setting,
                    evaluation.step,
                    len(evaluation.values),
                    mean,
                    se,
                    1.96 * se,
                    evaluation.truncated,
                )
            )

    fields = ("step", "left", "right", "effect", "p_raw", "p_holm", "significant_0_05")
    with (out_dir / "math500_pairwise_significance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tests)

    trend_fields = (
        "setting",
        "first_step",
        "last_step",
        "effect",
        "p_raw",
        "p_holm",
        "significant_0_05",
    )
    with (out_dir / "math500_trajectory_significance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trend_fields)
        writer.writeheader()
        writer.writerows(trends)


def plot_trajectories(evaluations: list[Evaluation], out_dir: Path) -> None:
    settings = sorted({item.setting for item in evaluations})
    groups = sorted({setting.rsplit("/", maxsplit=1)[0] for setting in settings})
    assert len(groups) == 4, f"expected four staleness groups, found {len(groups)}"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, groups, strict=True):
        for setting in (name for name in settings if name.startswith(f"{group}/")):
            points = [item for item in evaluations if item.setting == setting]
            steps = np.asarray([item.step for item in points])
            means, errors = zip(*[summary(item.values) for item in points], strict=True)
            means_array = np.asarray(means) * 100
            confidence = np.asarray(errors) * 1.96 * 100
            (line,) = axis.plot(
                steps,
                means_array,
                marker="o",
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
    fig.supylabel("MATH-500 avg@16 score (%)")
    fig.suptitle("MATH-500 checkpoint evaluation trajectories (mean ± 95% CI)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(out_dir / "math500_trajectory.png", dpi=180)
    fig.savefig(out_dir / "math500_trajectory.pdf")
    plt.close(fig)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def load_aime_scores(path: Path | None) -> dict[tuple[str, int], float]:
    if path is None or not path.is_file():
        return {}
    with path.open() as handle:
        return {
            (row["setting"], int(row["step"])): float(row["mean_score"])
            for row in csv.DictReader(handle)
        }


def write_report(
    evaluations: list[Evaluation],
    tests: list[dict],
    trends: list[dict],
    aime_scores: dict[tuple[str, int], float],
    out_dir: Path,
) -> None:
    settings = sorted({item.setting for item in evaluations})
    best = max(evaluations, key=lambda item: summary(item.values)[0])
    best_score = summary(best.values)[0]
    significant = [row for row in tests if row["significant_0_05"]]
    significant_trends = [row for row in trends if row["significant_0_05"]]
    lines = [
        "# MATH-500 checkpoint evaluation",
        "",
        "Scores are avg@16 over the same 500 MATH-500 problems at every checkpoint. "
        "Confidence intervals use the standard error across problems.",
        "",
        "## Summary",
        "",
        f"- Best observed checkpoint: {best.setting} step {best.step} at {percent(best_score)}.",
        f"- {len(significant)} pairwise differences survive Holm correction at alpha=0.05.",
        f"- {len(significant_trends)} of {len(trends)} first-to-last improvements survive "
        "Holm correction at alpha=0.05.",
    ]
    if aime_scores:
        paired = [
            (summary(item.values)[0], aime_scores[item.setting, item.step])
            for item in evaluations
            if (item.setting, item.step) in aime_scores
        ]
        math_values, aime_values = np.asarray(paired).T
        correlation = stats.pearsonr(math_values, aime_values)
        lines.append(
            f"- Across {len(paired)} checkpoints, MATH-500 and pooled AIME24/25/26 "
            f"have Pearson r={correlation.statistic:.3f} (p={correlation.pvalue:.3g})."
        )
    lines.extend(["", "## Per-setting trajectories", ""])
    for setting in settings:
        points = [item for item in evaluations if item.setting == setting]
        first, last = points[0], points[-1]
        first_score = summary(first.values)[0]
        last_score = summary(last.values)[0]
        peak = max(points, key=lambda item: summary(item.values)[0])
        lines.extend(
            [
                f"### {setting}",
                "",
                "| step | score | ±95% CI | truncation |",
                "|---:|---:|---:|---:|",
            ]
        )
        for evaluation in points:
            mean, se = summary(evaluation.values)
            lines.append(
                f"| {evaluation.step} | {percent(mean)} | {100 * 1.96 * se:.2f} pp "
                f"| {percent(evaluation.truncated)} |"
            )
        lines.extend(
            [
                "",
                f"First-to-last change: {100 * (last_score - first_score):+.2f} pp; "
                f"peak {percent(summary(peak.values)[0])} at step {peak.step}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Pairwise significance",
            "",
            "Paired t-tests match the same 500 problems at each common checkpoint step. "
            "Holm correction is applied across all setting pairs within each step.",
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
                f"| {row['step']} | {higher} | {lower} | {100 * effect:.2f} pp "
                f"| {row['p_holm']:.3g} |"
            )
    else:
        lines.append("No pairwise difference survives Holm correction at alpha=0.05.")
    lines.extend(
        [
            "",
            "## First-to-last significance",
            "",
            "Paired t-tests compare the first and last checkpoint on the same 500 problems. "
            "Holm correction is applied across the 12 settings.",
            "",
            "| setting | steps | change | raw p | Holm p | significant |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in trends:
        lines.append(
            f"| {row['setting']} | {row['first_step']}→{row['last_step']} "
            f"| {100 * row['effect']:+.2f} pp | {row['p_raw']:.3g} "
            f"| {row['p_holm']:.3g} | {'yes' if row['significant_0_05'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The tests quantify evaluation uncertainty for these checkpoints. There is "
            "one training run per setting, so they do not establish significance across "
            "training seeds. MATH-500 may also overlap model pretraining or math finetuning "
            "corpora and should be interpreted together with the cleaner AIME evaluation.",
            "",
            "![MATH-500 trajectories](math500_trajectory.png)",
            "",
        ]
    )
    (out_dir / "math500_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--aime-scores", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluations = discover(args.eval_root)
    tests = pairwise_tests(evaluations)
    trends = trajectory_tests(evaluations)
    write_tables(evaluations, tests, trends, args.out)
    plot_trajectories(evaluations, args.out)
    write_report(evaluations, tests, trends, load_aime_scores(args.aime_scores), args.out)
    print(
        f"wrote MATH-500 analysis for {len(evaluations)} checkpoints across "
        f"{len({item.setting for item in evaluations})} settings to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
