"""Audit the environment mix in actual trained multi-environment rollout dumps."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from experiments.src.reward_sets.math_code_stem import ALLOWED_VERIFIERS
from experiments.tools.replay_buffer_validation.log_parser import merge_step_records, parse_log


def trained_steps(logs: list[Path], max_staleness: int) -> list[int]:
    steps = set()
    for path in logs:
        assert "succeeded" in path.read_text(errors="replace"), f"Incomplete job: {path}"
        records = merge_step_records(parse_log(path))
        rollout = {r["step"]: r["metrics"] for r in records if "staleness/total/max" in r["metrics"]}
        for record in records:
            metrics = record["metrics"]
            if "train/loss" not in metrics:
                continue
            step = record["step"]
            assert math.isfinite(metrics["train/loss"]), (path, step)
            assert math.isfinite(metrics["train/grad_norm"]) and metrics["train/grad_norm"] > 0, (path, step)
            assert 0 <= rollout[step]["staleness/total/max"] <= max_staleness, (path, step)
            assert rollout[step]["staleness/token_lag/exact/covered_response_token_frac"] == 1, (path, step)
            steps.add(step)
    assert len(steps) >= 2, "Need multiple real optimizer updates"
    return sorted(steps)


def summarize_samples(samples: list[dict]) -> dict:
    rewards = [float(sample["reward"]) for sample in samples]
    assert all(math.isfinite(reward) and 0 <= reward <= 1 for reward in rewards)
    groups = defaultdict(list)
    for sample, reward in zip(samples, rewards, strict=True):
        groups[(sample["rollout_id"], sample["group_index"])].append(reward)
    return {
        "samples": len(samples),
        "groups": len(groups),
        "groups_with_reward_variation": sum(min(values) < max(values) for values in groups.values()),
        "positive_rewards": sum(reward > 0 for reward in rewards),
        "mean_reward": sum(rewards) / len(rewards),
        "statuses": dict(Counter(sample["status"] for sample in samples)),
        "response_tokens": sum(sample["response_length"] for sample in samples),
        "sources": dict(Counter(sample["metadata"]["source"] for sample in samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, nargs="+", required=True)
    parser.add_argument("--max-staleness", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps = trained_steps(args.log, args.max_staleness)
    by_verifier = defaultdict(list)
    batches = []
    for step in steps:
        path = args.checkpoint / "dump" / "rollout_data" / f"{step}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["rollout_id"] == step
        counts = Counter()
        for sample in data["samples"]:
            verifier = sample["metadata"]["verifier"]
            assert verifier in ALLOWED_VERIFIERS, (path, verifier)
            by_verifier[verifier].append(sample)
            counts[verifier] += 1
        batches.append({"step": step, "samples_by_verifier": dict(counts)})
    assert set(by_verifier) == ALLOWED_VERIFIERS, f"Unobserved verifiers: {ALLOWED_VERIFIERS - set(by_verifier)}"
    report = {
        "checkpoint": str(args.checkpoint),
        "logs": [str(path) for path in args.log],
        "trained_batches": batches,
        "verifiers": {name: summarize_samples(samples) for name, samples in sorted(by_verifier.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
