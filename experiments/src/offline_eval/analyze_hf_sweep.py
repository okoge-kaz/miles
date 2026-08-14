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
    base_values: dict[str, np.ndarray]
    truncated: dict[str, float]
    n_samples: int


def _load_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records.sort(key=lambda record: int(record["index"]))
    assert len(records) == 30, f"{path}: expected 30 records, found {len(records)}"
    assert [int(record["index"]) for record in records] == list(range(30)), (
        f"{path}: prompt indices are not exactly 0..29"
    )
    return records


def discover(eval_root: Path, extra_suffix: str | None = None) -> list[Evaluation]:
    evaluations: list[Evaluation] = []
    for step_dir in sorted(eval_root.glob("**/step-*")):
        match = STEP_PATTERN.fullmatch(step_dir.name)
        if not step_dir.is_dir() or match is None:
            continue
        values: dict[str, np.ndarray] = {}
        base_values: dict[str, np.ndarray] = {}
        truncated: dict[str, float] = {}
        observed_n_samples: set[int] = set()
        for benchmark in BENCHMARKS:
            records = _load_records(step_dir / f"{benchmark}.jsonl")
            sample_counts = {int(record["n_samples"]) for record in records}
            assert len(sample_counts) == 1, f"{benchmark}: inconsistent n_samples"
            base_n_samples = sample_counts.pop()
            if extra_suffix is None:
                base_values[benchmark] = np.asarray(
                    [float(record["pass_rate"]) for record in records]
                )
                combined_records = records
                evaluation_n_samples = base_n_samples
            elif base_n_samples == 32:
                for record in records:
                    assert len(record["rewards"]) == 32
                base_values[benchmark] = np.asarray(
                    [float(np.mean(record["rewards"][:16])) for record in records]
                )
                combined_records = records
                evaluation_n_samples = 32
            else:
                assert base_n_samples == 16, (
                    f"{benchmark}: expected a 16- or 32-sample base, found "
                    f"{base_n_samples}"
                )
                base_values[benchmark] = np.asarray(
                    [float(record["pass_rate"]) for record in records]
                )
                extra = _load_records(step_dir / f"{benchmark}{extra_suffix}.jsonl")
                assert all(int(record["n_samples"]) == 16 for record in extra)
                combined_records = []
                for base_record, extra_record in zip(records, extra, strict=True):
                    rewards = [*base_record["rewards"], *extra_record["rewards"]]
                    assert len(rewards) == 32
                    combined_records.append(
                        {
                            "pass_rate": float(np.mean(rewards)),
                            "truncated_frac": (
                                float(base_record["truncated_frac"])
                                + float(extra_record["truncated_frac"])
                            )
                            / 2,
                        }
                    )
                evaluation_n_samples = 32
            observed_n_samples.add(evaluation_n_samples)
            values[benchmark] = np.asarray(
                [float(record["pass_rate"]) for record in combined_records]
            )
            fractions = [
                float(record["truncated_frac"])
                for record in combined_records
                if record.get("truncated_frac") is not None
            ]
            truncated[benchmark] = float(np.mean(fractions)) if fractions else math.nan
        assert len(observed_n_samples) == 1, (
            f"{step_dir}: inconsistent effective n_samples: {observed_n_samples}"
        )
        evaluations.append(
            Evaluation(
                setting=step_dir.parent.relative_to(eval_root).as_posix(),
                step=int(match.group(1)),
                values=values,
                base_values=base_values,
                truncated=truncated,
                n_samples=evaluation_n_samples,
            )
        )
    assert evaluations, f"no complete evaluations found under {eval_root}"
    return sorted(evaluations, key=lambda item: (item.setting, item.step))


def _pooled(evaluation: Evaluation) -> np.ndarray:
    return np.concatenate([evaluation.values[name] for name in BENCHMARKS])


def _summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(stats.sem(values))


def _monte_carlo_se(values: np.ndarray, n_samples: int) -> float:
    """SE from generation sampling, conditional on the fixed benchmark questions."""
    assert n_samples > 1
    return float(np.sqrt(np.sum(values * (1 - values)) / (len(values) ** 2 * (n_samples - 1))))


def write_score_tables(evaluations: list[Evaluation], out_dir: Path) -> None:
    with (out_dir / "scores_by_year.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "setting",
                "step",
                "benchmark",
                "n_problems",
                "n_samples",
                "score",
                "problem_se",
                "monte_carlo_se",
                "truncated_frac",
            )
        )
        for evaluation in evaluations:
            for benchmark in BENCHMARKS:
                mean, se = _summary(evaluation.values[benchmark])
                writer.writerow(
                    (
                        evaluation.setting,
                        evaluation.step,
                        benchmark,
                        len(evaluation.values[benchmark]),
                        evaluation.n_samples,
                        mean,
                        se,
                        _monte_carlo_se(
                            evaluation.values[benchmark], evaluation.n_samples
                        ),
                        evaluation.truncated[benchmark],
                    )
                )

    with (out_dir / "scores_mean.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "setting",
                "step",
                "n_problems",
                "n_samples",
                "mean_score",
                "problem_se",
                "monte_carlo_se",
            )
        )
        for evaluation in evaluations:
            mean, se = _summary(_pooled(evaluation))
            writer.writerow(
                (
                    evaluation.setting,
                    evaluation.step,
                    90,
                    evaluation.n_samples,
                    mean,
                    se,
                    _monte_carlo_se(_pooled(evaluation), evaluation.n_samples),
                )
            )


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


def plot_trajectories(
    evaluations: list[Evaluation], out_dir: Path, ci_kind: str
) -> None:
    settings = sorted({item.setting for item in evaluations})
    groups = sorted({setting.rsplit("/", maxsplit=1)[0] for setting in settings})
    assert len(groups) == 4, f"expected four staleness groups, found {len(groups)}"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, groups, strict=True):
        for setting in (name for name in settings if name.startswith(f"{group}/")):
            points = [item for item in evaluations if item.setting == setting]
            steps = np.asarray([item.step for item in points])
            means = [_summary(_pooled(item))[0] for item in points]
            if ci_kind == "monte-carlo":
                errors = [_monte_carlo_se(_pooled(item), item.n_samples) for item in points]
            else:
                errors = [_summary(_pooled(item))[1] for item in points]
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
    ci_label = (
        "Monte Carlo 95% CI (fixed 90 questions)"
        if ci_kind == "monte-carlo"
        else "problem-resampling 95% CI"
    )
    fig.suptitle(f"AIME avg@{evaluations[0].n_samples} trajectories ({ci_label})")
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
        f"Scores are avg@{evaluations[0].n_samples} over 30 problems per AIME year. "
        "The mean pools all 90 problem-level values. Monte Carlo uncertainty from "
        "generation sampling and problem-resampling uncertainty are reported separately.",
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
    pooled = [_pooled(evaluation) for evaluation in evaluations]
    mc_ci = [
        1.96 * _monte_carlo_se(values, evaluation.n_samples)
        for values, evaluation in zip(pooled, evaluations, strict=True)
    ]
    problem_ci = [1.96 * _summary(values)[1] for values in pooled]
    base_mc_ci = [
        1.96
        * _monte_carlo_se(
            np.concatenate([evaluation.base_values[name] for name in BENCHMARKS]),
            16,
        )
        for evaluation in evaluations
    ]
    base_problem_ci = [
        1.96
        * _summary(np.concatenate([evaluation.base_values[name] for name in BENCHMARKS]))[1]
        for evaluation in evaluations
    ]
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
            f"- Mean Monte Carlo 95% CI half-width: avg@16 "
            f"{100 * np.mean(base_mc_ci):.2f} pp; avg@{evaluations[0].n_samples} "
            f"{100 * np.mean(mc_ci):.2f} pp "
            f"({100 * (1 - np.mean(mc_ci) / np.mean(base_mc_ci)):.1f}% reduction).",
            f"- Mean problem-resampling 95% CI half-width: avg@16 "
            f"{100 * np.mean(base_problem_ci):.2f} pp; avg@{evaluations[0].n_samples} "
            f"{100 * np.mean(problem_ci):.2f} pp. More generations do not materially "
            "increase the number or diversity of benchmark problems.",
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
    parser.add_argument(
        "--extra-suffix",
        help="combine each base file with <benchmark><suffix>.jsonl (for example _extra16)",
    )
    parser.add_argument(
        "--ci-kind",
        choices=("problem", "monte-carlo"),
        default="problem",
        help="uncertainty band shown in the trajectory plot",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluations = discover(args.eval_root, args.extra_suffix)
    tests = pairwise_tests(evaluations)
    write_score_tables(evaluations, args.out)
    write_pairwise(tests, args.out)
    plot_trajectories(evaluations, args.out, args.ci_kind)
    write_report(evaluations, tests, args.out)
    settings = {item.setting for item in evaluations}
    print(f"wrote analysis for {len(evaluations)} checkpoints across {len(settings)} settings to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
