"""Audit real optimizer updates and prepared-batch continuity across a PBS restart."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.tools.replay_buffer_validation.log_parser import merge_step_records, parse_log


def _read_segment(path: Path, max_staleness: int) -> dict:
    text = path.read_text(errors="replace")
    assert "succeeded" in text, f"No successful Ray job termination in {path}"
    records = merge_step_records(parse_log(path))
    train = {r["step"]: r["metrics"] for r in records if "train/loss" in r["metrics"]}
    rollout = {r["step"]: r["metrics"] for r in records if "staleness/total/mean" in r["metrics"]}
    assert train, f"No optimizer metrics in {path}"
    for step, metrics in train.items():
        for key in ("train/loss", "train/grad_norm"):
            assert math.isfinite(metrics[key]), (path, step, key, metrics[key])
        assert metrics["train/grad_norm"] > 0, (path, step, "zero gradient")
        sample_metrics = rollout[step]
        assert 0 <= sample_metrics["staleness/total/max"] <= max_staleness, (path, step)
        assert sample_metrics["staleness/token_lag/exact/covered_response_token_frac"] == 1, (path, step)
        for key in (
            "staleness/token_lag/exact/invalid_segments",
            "staleness/token_lag/exact/invalid_samples",
            "staleness/token_lag/exact/loss_token/invalid_loss_masks",
        ):
            assert sample_metrics[key] == 0, (path, step, key)
    return {"path": str(path), "train": train, "rollout": rollout}


def _summarize(segment: dict) -> list[dict]:
    return [
        {
            "step": step,
            "loss": metrics["train/loss"],
            "grad_norm": metrics["train/grad_norm"],
            "raw_reward": segment["rollout"][step]["rollout/raw_reward"],
            "staleness_mean": segment["rollout"][step]["staleness/total/mean"],
            "staleness_max": segment["rollout"][step]["staleness/total/max"],
        }
        for step, metrics in sorted(segment["train"].items())
    ]


def _staleness_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key.startswith("staleness/")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-log", type=Path, required=True)
    parser.add_argument("--resume-log", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument(
        "--checkpoint-rollout-id",
        type=int,
        help="Selected checkpoint update, when resuming an archive or discarding later unsaved updates",
    )
    parser.add_argument("--max-staleness", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fresh = _read_segment(args.fresh_log, args.max_staleness)
    resumed = _read_segment(args.resume_log, args.max_staleness)
    first_resume = min(resumed["train"])
    saved_step = args.checkpoint_rollout_id
    if saved_step is None:
        saved_step = max(fresh["train"])
    assert saved_step in fresh["train"], "The saved optimizer update is absent from the fresh log"
    assert first_resume == saved_step + 1, "Resume did not start after the saved optimizer update"
    assert len(resumed["train"]) >= 2, "Need a prepared batch and a newly produced batch after resume"
    prepared = _staleness_metrics(fresh["rollout"][first_resume])
    restored = _staleness_metrics(resumed["rollout"][first_resume])
    assert prepared == restored, "Saved and resumed staleness metrics differ"
    resume = {
        key.removeprefix("resume/replay_buffer/"): value
        for key, value in resumed["rollout"][first_resume].items()
        if key.startswith("resume/replay_buffer/")
    }
    assert resume["warm_prepared_batch_hit"] == 1
    assert resume["prepared_batches_restored"] >= 1
    assert resume["inflight_groups_restored"] > 0
    assert resume["inflight_tokens_restored"] > 0
    assert resume["regenerated_active_groups"] == 0
    assert resume["current_applied_weight_version"] == resume["applied_weight_version_restored"] + 1
    report = {
        "fresh_log": fresh["path"],
        "resume_log": resumed["path"],
        "fresh_updates": _summarize(fresh),
        "resumed_updates": _summarize(resumed),
        "checkpoint_rollout_id": saved_step,
        "fresh_steps_replayed": sorted(step for step in fresh["train"] if step > saved_step),
        "prepared_step": first_resume,
        "identical_staleness_metrics": len(prepared),
        "resume": resume,
    }
    if args.baseline_log:
        baseline_records = merge_step_records(parse_log(args.baseline_log))
        baseline = next(r["metrics"] for r in baseline_records if "staleness/total/mean" in r["metrics"])
        previous_keys = set(_staleness_metrics(baseline))
        current_keys = set(prepared)
        report["baseline_schema"] = {
            "log": str(args.baseline_log),
            "previous_keys": len(previous_keys),
            "current_keys": len(current_keys),
            "removed": sorted(previous_keys - current_keys),
            "added": sorted(current_keys - previous_keys),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
