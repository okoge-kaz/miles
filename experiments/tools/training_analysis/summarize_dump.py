#!/usr/bin/env python3
"""Summarize reward, provenance, and staleness from Miles rollout dumps."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from miles.dashboard.dump_reader import DumpReader


DOMAIN_BY_VERIFIER = {
    "expert_action": "agentic",
    "ifeval_g": "instruction_following",
    "json_schema": "structured_output",
    "math": "math",
    "python_code": "code",
    "mcqa_regex": "stem",
    "reasoning_gym": "stem",
    "tau_bench_environment": "tau_bench",
}
REFERENCE_VERSION_KEY = "sample_staleness_reference_weight_version"
TRAIN_VERSION_KEY = "train_weight_version"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quarter_change(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    rewards_by_rollout: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["reward"] is not None:
            rewards_by_rollout[int(row["rollout_id"])].append(float(row["reward"]))
    rollout_means = [
        _mean(rewards_by_rollout[rollout_id])
        for rollout_id in sorted(rewards_by_rollout)
    ]
    rewards = [value for value in rollout_means if value is not None]
    if not rewards:
        return {"first_quarter_mean": None, "last_quarter_mean": None, "last_minus_first": None}
    window = max(1, len(rewards) // 4)
    first = _mean(rewards[:window])
    last = _mean(rewards[-window:])
    return {
        "first_quarter_mean": first,
        "last_quarter_mean": last,
        "last_minus_first": None if first is None or last is None else last - first,
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows if row["reward"] is not None]
    response_lengths = [float(row["response_length"]) for row in rows]
    truncation = [float(row["truncated"]) for row in rows]
    return {
        "samples": len(rows),
        "reward_mean": _mean(rewards),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "response_length_mean": _mean(response_lengths),
        "truncated_fraction": _mean(truncation),
        **_quarter_change(rows),
    }


def summarize(dump_dir: Path, *, requested_staleness_bound: int = 4) -> dict[str, Any]:
    reader = DumpReader(dump_dir, tensor_lru=1)
    rollout_ids = reader.rollout_ids().train
    rows = []
    segment_samples = 0
    exact_segment_coverage = 0
    exact_policy_segment_coverage = 0
    mixed_segment_versions = 0
    segment_covered_tokens = 0
    segment_response_tokens = 0
    segment_policy_tokens = 0
    staleness_values = []
    covered_train_rows = 0
    tau_done = []
    tau_turns = []
    tau_user_length_truncations = []
    for rollout_id in rollout_ids:
        joined = reader.load_joined(rollout_id)
        covered_train_rows += len(joined.train_rows)
        for sample in joined.samples:
            train_row = joined.train_rows.get(sample.index)
            reward = None
            if train_row is not None and isinstance(train_row.raw_reward, (int, float)):
                reward = float(train_row.raw_reward)
            elif isinstance(sample.reward, (int, float)):
                reward = float(sample.reward)
            verifier = str(sample.metadata.get("verifier") or "unknown")
            if verifier == "tau_bench_environment":
                if isinstance(sample.metadata.get("tau_done"), bool):
                    tau_done.append(float(sample.metadata["tau_done"]))
                if isinstance(sample.metadata.get("tau_turns"), int):
                    tau_turns.append(float(sample.metadata["tau_turns"]))
                truncations = sample.metadata.get("tau_user_length_truncations", 0)
                if isinstance(truncations, int):
                    tau_user_length_truncations.append(float(truncations))
            rows.append(
                {
                    "rollout_id": rollout_id,
                    "verifier": verifier,
                    "domain": DOMAIN_BY_VERIFIER.get(verifier, "unknown"),
                    "reward": reward,
                    "response_length": sample.response_length,
                    "truncated": sample.status.value == "truncated",
                }
            )
            segments = sample.response_weight_version_segments
            if segments:
                segment_samples += 1
                covered = sum(end - start for turn in segments for start, end, _version in turn)
                policy_tokens = (
                    int(sum(sample.loss_mask)) if sample.loss_mask is not None else sample.response_length
                )
                exact_segment_coverage += int(covered == sample.response_length)
                exact_policy_segment_coverage += int(covered == policy_tokens)
                segment_covered_tokens += covered
                segment_response_tokens += sample.response_length
                segment_policy_tokens += policy_tokens
                versions = {version for turn in segments for _start, _end, version in turn}
                mixed_segment_versions += int(len(versions) > 1)
            reference = sample.metadata.get(REFERENCE_VERSION_KEY)
            train_version = sample.metadata.get(TRAIN_VERSION_KEY)
            if isinstance(reference, int) and isinstance(train_version, int):
                staleness_values.append(float(train_version - reference))

    by_verifier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_verifier[row["verifier"]].append(row)
        by_domain[row["domain"]].append(row)
    finite_values = [row["reward"] for row in rows if row["reward"] is not None] + staleness_values
    return {
        "dump_dir": str(dump_dir),
        "rollout_ids": rollout_ids,
        "rollouts": len(rollout_ids),
        "samples": len(rows),
        "train_coverage": covered_train_rows / len(rows) if rows else 0.0,
        "by_domain": {key: _group_summary(value) for key, value in sorted(by_domain.items())},
        "by_verifier": {key: _group_summary(value) for key, value in sorted(by_verifier.items())},
        "sample_staleness": {
            "samples": len(staleness_values),
            "mean": _mean(staleness_values),
            "max": max(staleness_values) if staleness_values else None,
            "within_requested_bound": bool(staleness_values)
            and max(staleness_values) <= requested_staleness_bound,
            "requested_bound": requested_staleness_bound,
        },
        "tau_bench_environment": {
            "samples": len(tau_user_length_truncations),
            "done_fraction": _mean(tau_done),
            "turns_mean": _mean(tau_turns),
            "turns_max": max(tau_turns) if tau_turns else None,
            "user_length_truncation_samples": sum(value > 0 for value in tau_user_length_truncations),
            "user_length_truncation_events": sum(tau_user_length_truncations),
        },
        "response_weight_version_segments": {
            "samples_with_segments": segment_samples,
            "exact_coverage_samples": exact_segment_coverage,
            "exact_coverage_fraction": exact_segment_coverage / segment_samples if segment_samples else 0.0,
            "exact_policy_token_coverage_samples": exact_policy_segment_coverage,
            "exact_policy_token_coverage_fraction": (
                exact_policy_segment_coverage / segment_samples if segment_samples else 0.0
            ),
            "covered_response_token_fraction": (
                segment_covered_tokens / segment_response_tokens if segment_response_tokens else 0.0
            ),
            "covered_policy_token_fraction": (
                segment_covered_tokens / segment_policy_tokens if segment_policy_tokens else 0.0
            ),
            "mixed_version_samples": mixed_segment_versions,
        },
        "finite": all(math.isfinite(float(value)) for value in finite_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--requested-staleness-bound", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.dump_dir, requested_staleness_bound=args.requested_staleness_bound)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        partial = args.output.with_name(args.output.name + ".partial")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(rendered + "\n", encoding="utf-8")
        os.replace(partial, args.output)


if __name__ == "__main__":
    main()
