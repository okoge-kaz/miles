"""Summarize local Lightning node-ratio dashboard metrics without torch or W&B.

    python -m experiments.tools.summarize_lightning_node_ratio \
        --plan-file experiments/outputs/lightning-node-ratio/review-plan.json \
        --output-dir experiments/outputs/lightning-node-ratio/summary

The first two completed training steps are excluded by default. Rollout drain
time is NOT model generation time. Pipeline token rates use sums of counts
divided by sums of their matching wall windows, not means of per-step rates.
"""

import argparse
import csv
import json
import math
import statistics
import warnings
from pathlib import Path

TIMINGS = {
    "step": "perf/step_time",
    "train": "perf/train_time",
    "actor_train": "perf/actor_train_time",
    "log_probs": "perf/log_probs_time",
    "train_wait": "perf/train_wait_time",
    "rollout_drain": "perf/rollout_time",
    "update_weights": "perf/update_weights_time",
    "save_model": "perf/save_model_time",
    "pipeline_window": "throughput/window_seconds",
}
EXTRA = (
    "perf/wait_time_ratio",
    "rollout/response_len/mean",
    "rollout/truncated_ratio",
    "staleness/total/mean",
    "staleness/total/max",
    "staleness/bound_exceeded_sample_frac",
    "rollout/fully_async/wasted_token_frac",
    "rollout/fully_async/stale_groups_recycled",
    "queue/depth_time_mean",
    "queue/rollout_backpressure_seconds",
    "queue/trainer_starvation_seconds",
)


def read_steps(dashboard: Path) -> dict[int, dict[str, float]]:
    paths = sorted((dashboard / "metrics").glob("*.jsonl"))
    if (legacy := dashboard / "metrics.jsonl").is_file():
        paths.insert(0, legacy)
    values = {}
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith("\n"):
                        warnings.warn(f"Ignoring unfinished final record: {path}:{line_number}", stacklevel=2)
                        continue
                    raise
                metrics = record["metrics"]
                step = record.get("step") if record.get("step_key") == "rollout/step" else metrics.get("rollout/step")
                if step is None:
                    continue
                for key, value in metrics.items():
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        continue
                    identity = (int(step), key)
                    timestamp = record["ts"]
                    if identity not in values or timestamp >= values[identity][0]:
                        values[identity] = (timestamp, value)
    steps = {}
    for (step, key), (_, value) in values.items():
        steps.setdefault(step, {})[key] = value
    return steps


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize_arm(arm: dict, *, warmup_steps: int) -> tuple[dict, list[dict]]:
    steps = read_steps(Path(arm["dashboard"]))
    completed = sorted(step for step, row in steps.items() if "perf/step_time" in row)
    selected = completed[warmup_steps:]
    summary = dict(
        name=arm["env"]["CONFIG_TAG"],
        staleness=arm["point"][0],
        train_nodes=arm["point"][1],
        rollout_nodes=arm["point"][2],
        max_running_requests=arm["env"].get("SGLANG_MAX_RUNNING_REQUESTS"),
        cuda_graph_max_bs=arm["env"].get("SGLANG_CUDA_GRAPH_MAX_BS"),
        async_max_concurrent_samples=arm["env"].get("ASYNC_MAX_CONCURRENT_SAMPLES"),
        completed_steps=len(completed),
        measured_steps=len(selected),
        status="ok" if selected else "insufficient_data",
    )
    identity = {
        key: summary[key]
        for key in (
            "name",
            "staleness",
            "train_nodes",
            "rollout_nodes",
            "max_running_requests",
            "cuda_graph_max_bs",
            "async_max_concurrent_samples",
        )
    }
    rows = [dict(identity, rollout_step=step, **steps[step]) for step in selected]
    for label, key in TIMINGS.items():
        samples = [steps[step][key] for step in selected if key in steps[step]]
        summary[f"{label}_count"] = len(samples)
        for statistic, function in (
            ("mean", statistics.mean),
            ("p50", statistics.median),
            ("p90", lambda data: percentile(data, 0.9)),
        ):
            summary[f"{label}_{statistic}_s"] = function(samples) if samples else None
    for key in EXTRA:
        samples = [steps[step][key] for step in selected if key in steps[step]]
        summary[f"{key}/step_mean"] = statistics.mean(samples) if samples else None
    for label, count, available in (
        ("generated_tokens", "throughput/generated_tokens", None),
        (
            "useful_tokens",
            "throughput/window_accepted_loss_tokens",
            "throughput/window_accepted_loss_tokens_available",
        ),
        ("optimizer_updates", "throughput/optimizer_updates", "throughput/optimizer_updates_available"),
    ):
        windows = [
            steps[step]
            for step in selected
            if count in steps[step]
            and steps[step].get("throughput/window_seconds", 0) > 0
            and (available is None or steps[step].get(available) == 1)
        ]
        seconds = sum(row["throughput/window_seconds"] for row in windows)
        summary[f"{label}_window_seconds"] = seconds
        summary[f"{label}_per_s"] = sum(row[count] for row in windows) / seconds if seconds else None
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be nonnegative")
    plan = json.loads(args.plan_file.read_text())
    summaries, steps = [], []
    for arm in plan["arms"]:
        summary, rows = summarize_arm(arm, warmup_steps=args.warmup_steps)
        summaries.append(summary)
        steps.extend(rows)
        print(f"{summary['name']}: {summary['status']}, {summary['measured_steps']} measured steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "steps.csv", steps)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "plan_file": str(args.plan_file.resolve()),
                "warmup_completed_steps": args.warmup_steps,
                "timing_keys": TIMINGS,
                "rollout_drain_is_generation_time": False,
                "rate_aggregation": "sum(counts) / sum(matching window_seconds)",
                "save_and_weight_sync": "not subtracted; include pipeline windows for end-to-end comparison",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output_dir / 'summary.csv'} and {args.output_dir / 'steps.csv'}")


if __name__ == "__main__":
    main()
