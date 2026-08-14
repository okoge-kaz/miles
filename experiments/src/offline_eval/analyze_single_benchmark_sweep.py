#!/usr/bin/env python3
"""Analyze one problem-level benchmark across a nested checkpoint sweep."""

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

STEP_PATTERN = re.compile(r"step-(\d+)$")


@dataclass(frozen=True)
class Evaluation:
    setting: str
    step: int
    values: np.ndarray
    n_samples: int
    truncated: float


def load_records(path: Path, expected_problems: int, expected_samples: int) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records.sort(key=lambda record: int(record["index"]))
    assert len(records) == expected_problems, (
        f"{path}: expected {expected_problems} records, found {len(records)}"
    )
    assert [int(record["index"]) for record in records] == list(range(expected_problems)), (
        f"{path}: prompt indices are not exactly 0..{expected_problems - 1}"
    )
    assert all(int(record["n_samples"]) == expected_samples for record in records), (
        f"{path}: not every record has n_samples={expected_samples}"
    )
    assert all(len(record["rewards"]) == expected_samples for record in records), (
        f"{path}: reward vector length differs from n_samples"
    )
    return records


def discover(
    eval_root: Path,
    filename: str,
    expected_problems: int,
    expected_samples: int,
) -> list[Evaluation]:
    evaluations = []
    for step_dir in sorted(eval_root.glob("**/step-*")):
        match = STEP_PATTERN.fullmatch(step_dir.name)
        result = step_dir / filename
        if not step_dir.is_dir() or match is None or not result.is_file():
            continue
        records = load_records(result, expected_problems, expected_samples)
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
                n_samples=expected_samples,
                truncated=float(np.mean(fractions)) if fractions else math.nan,
            )
        )
    assert evaluations, f"no complete {filename} evaluations found under {eval_root}"
    return sorted(evaluations, key=lambda item: (item.setting, item.step))


def summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(stats.sem(values))


def monte_carlo_se(values: np.ndarray, n_samples: int) -> float:
    return float(np.sqrt(np.sum(values * (1 - values)) / (len(values) ** 2 * (n_samples - 1))))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_row(left: Evaluation, right: Evaluation) -> dict:
    difference = left.values - right.values
    if np.all(difference == 0):
        p_value = 1.0
    else:
        result = stats.ttest_rel(left.values, right.values)
        p_value = float(result.pvalue) if math.isfinite(float(result.pvalue)) else 1.0
    return {"effect": float(np.mean(difference)), "p_raw": p_value}


def pairwise_tests(evaluations: list[Evaluation]) -> list[dict]:
    lookup = {(item.setting, item.step): item for item in evaluations}
    settings = sorted({item.setting for item in evaluations})
    rows = []
    for step in sorted({item.step for item in evaluations}):
        present = [setting for setting in settings if (setting, step) in lookup]
        family = []
        for left_name, right_name in combinations(present, 2):
            row = paired_row(lookup[left_name, step], lookup[right_name, step])
            row.update({"step": step, "left": left_name, "right": right_name})
            family.append(row)
        for row, adjusted in zip(
            family, holm_adjust([item["p_raw"] for item in family]), strict=True
        ):
            row["p_holm"] = adjusted
            row["significant_0_05"] = adjusted < 0.05
            rows.append(row)
    return rows


def trajectory_tests(evaluations: list[Evaluation]) -> list[dict]:
    rows = []
    for setting in sorted({item.setting for item in evaluations}):
        points = [item for item in evaluations if item.setting == setting]
        row = paired_row(points[-1], points[0])
        row.update(
            {
                "setting": setting,
                "first_step": points[0].step,
                "last_step": points[-1].step,
            }
        )
        rows.append(row)
    for row, adjusted in zip(
        rows, holm_adjust([item["p_raw"] for item in rows]), strict=True
    ):
        row["p_holm"] = adjusted
        row["significant_0_05"] = adjusted < 0.05
    return rows


def write_tables(
    evaluations: list[Evaluation],
    pairs: list[dict],
    trends: list[dict],
    prefix: str,
    out_dir: Path,
) -> None:
    with (out_dir / f"{prefix}_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "setting",
                "step",
                "n_problems",
                "n_samples",
                "score",
                "problem_se",
                "monte_carlo_se",
                "truncated_frac",
            )
        )
        for evaluation in evaluations:
            mean, problem_se = summary(evaluation.values)
            writer.writerow(
                (
                    evaluation.setting,
                    evaluation.step,
                    len(evaluation.values),
                    evaluation.n_samples,
                    mean,
                    problem_se,
                    monte_carlo_se(evaluation.values, evaluation.n_samples),
                    evaluation.truncated,
                )
            )

    pair_fields = ("step", "left", "right", "effect", "p_raw", "p_holm", "significant_0_05")
    with (out_dir / f"{prefix}_pairwise_significance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pairs)

    trend_fields = (
        "setting",
        "first_step",
        "last_step",
        "effect",
        "p_raw",
        "p_holm",
        "significant_0_05",
    )
    with (out_dir / f"{prefix}_trajectory_significance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trend_fields)
        writer.writeheader()
        writer.writerows(trends)


def plot_trajectories(
    evaluations: list[Evaluation],
    benchmark_label: str,
    prefix: str,
    out_dir: Path,
) -> None:
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
    fig.supylabel(f"{benchmark_label} avg@{evaluations[0].n_samples} score (%)")
    fig.suptitle(f"{benchmark_label} trajectories (problem-resampling 95% CI)")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(out_dir / f"{prefix}_trajectory.png", dpi=180)
    fig.savefig(out_dir / f"{prefix}_trajectory.pdf")
    plt.close(fig)


def write_report(
    evaluations: list[Evaluation],
    pairs: list[dict],
    trends: list[dict],
    benchmark_label: str,
    prefix: str,
    out_dir: Path,
) -> None:
    best = max(evaluations, key=lambda item: summary(item.values)[0])
    significant_pairs = [row for row in pairs if row["significant_0_05"]]
    significant_trends = [row for row in trends if row["significant_0_05"]]
    problem_ci = [1.96 * summary(item.values)[1] for item in evaluations]
    mc_ci = [1.96 * monte_carlo_se(item.values, item.n_samples) for item in evaluations]
    lines = [
        f"# {benchmark_label} checkpoint evaluation",
        "",
        f"Scores are avg@{evaluations[0].n_samples} over the same "
        f"{len(evaluations[0].values)} problems at every checkpoint.",
        "",
        "## Summary",
        "",
        f"- Best checkpoint: {best.setting} step {best.step} at "
        f"{100 * summary(best.values)[0]:.2f}%.",
        f"- {len(significant_pairs)} setting differences survive Holm correction "
        "within checkpoint step.",
        f"- {len(significant_trends)}/{len(trends)} first-to-last changes survive Holm "
        "correction across settings.",
        f"- Mean problem-resampling 95% CI half-width: {100 * np.mean(problem_ci):.2f} pp.",
        f"- Mean fixed-problem Monte Carlo 95% CI half-width: {100 * np.mean(mc_ci):.2f} pp.",
        "",
        "## Per-setting trajectory",
        "",
        "| setting | first→last step | first score | last score | change | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lookup = {row["setting"]: row for row in trends}
    for setting in sorted(lookup):
        points = [item for item in evaluations if item.setting == setting]
        row = lookup[setting]
        lines.append(
            f"| {setting} | {row['first_step']}→{row['last_step']} "
            f"| {100 * summary(points[0].values)[0]:.2f}% "
            f"| {100 * summary(points[-1].values)[0]:.2f}% "
            f"| {100 * row['effect']:+.2f} pp | {row['p_holm']:.3g} |"
        )
    lines.extend(
        [
            "",
            "Paired tests match the same problems. They quantify evaluation uncertainty, "
            "not variability across training seeds; there is one training run per setting.",
            "",
            f"![{benchmark_label} trajectories]({prefix}_trajectory.png)",
            "",
        ]
    )
    (out_dir / f"{prefix}_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--expected-problems", required=True, type=int)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--benchmark-label", required=True)
    parser.add_argument("--prefix", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluations = discover(
        args.eval_root,
        args.filename,
        args.expected_problems,
        args.expected_samples,
    )
    pairs = pairwise_tests(evaluations)
    trends = trajectory_tests(evaluations)
    write_tables(evaluations, pairs, trends, args.prefix, args.out)
    plot_trajectories(evaluations, args.benchmark_label, args.prefix, args.out)
    write_report(
        evaluations,
        pairs,
        trends,
        args.benchmark_label,
        args.prefix,
        args.out,
    )
    print(
        f"wrote {args.benchmark_label} analysis for {len(evaluations)} checkpoints "
        f"across {len({item.setting for item in evaluations})} settings to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
