"""Check OCI async telemetry against raw rollout provenance, not just exit status.

Run inside the training image with --checkpoint /ckpt/training/... --steps 0 1 2.
Reports JSON; exits nonzero on missing steps, incomplete provenance, wrong lag,
queue-recycle bound violations, or non-finite training metrics. This checker is
scoped to the single-turn, one-optimizer-update-per-rollout validation recipe.
An inflight resume can split one response across multiple generation calls;
their token offsets restart at zero and their lengths must sum to the response.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch


def close(actual, expected, label):
    assert math.isfinite(actual), (label, actual)
    assert math.isclose(actual, expected, abs_tol=1e-6, rel_tol=1e-6), (label, actual, expected)


def validate_samples(samples, metrics, train_metrics, bound):
    version = metrics["fully_async/train_weight_version"]
    group_rows = defaultdict(list)
    sample_lags = []
    token_lag_sum = 0
    token_count = 0
    generation_calls = 0
    for sample in samples:
        metadata = sample["metadata"]
        first = min(sample["first_prefill_weight_versions"])
        last = max(sample["last_forward_weight_versions"])
        ready = metadata["group_ready_weight_version"]
        put = metadata["queue_put_weight_version"]
        drain = metadata["drain_weight_version"]
        trained = metadata["train_weight_version"]
        assert first <= last <= ready <= put <= drain <= trained == version
        assert trained - drain in (0, 1)
        close(metadata["sample_staleness_reference_weight_version"], first, "sample reference")
        sample_lags.append(trained - first)
        group_rows[sample["group_index"]].append((first, ready, drain, trained))
        turns = sample["response_weight_version_segments"]
        assert turns, "missing generation-call provenance"
        generation_calls += len(turns)
        covered = 0
        for call_segments in turns:
            cursor = 0
            for start, end, token_version in call_segments:
                assert start == cursor and start < end
                assert first <= token_version <= last <= trained
                token_lag_sum += (end - start) * (trained - token_version)
                token_count += end - start
                cursor = end
            covered += cursor
        assert covered == sample["response_length"], "incomplete exact-token coverage"

    phases = defaultdict(list)
    for group in group_rows.values():
        first = min(row[0] for row in group)
        ready, drain, trained = group[0][1:]
        assert all(row[1:] == (ready, drain, trained) for row in group)
        assert drain - first < bound and trained - first <= bound
        phases["pre_queue"].append(ready - first)
        phases["in_queue"].append(trained - ready)
        phases["total"].append(trained - first)
    for name, values in phases.items():
        close(metrics[f"staleness/{name}/mean"], sum(values) / len(values), name)
        close(metrics[f"staleness/{name}/max"], max(values), name)
    close(metrics["staleness/sample_lag/total/sequence_mean"], sum(sample_lags) / len(sample_lags), "sample lag")
    close(metrics["staleness/token_lag/exact/mean"], token_lag_sum / token_count, "token lag")
    close(metrics["staleness/token_lag/exact/num_tokens"], token_count, "token count")
    counts = Counter(sample_lags)
    for lag in range(bound + 1):
        close(
            train_metrics[f"sample_staleness/s_{lag}/consumed_sequence_mass"],
            counts[lag] / len(samples),
            "trainer bin",
        )
    return {
        "samples": len(samples),
        "generation_calls": generation_calls,
        "response_tokens": token_count,
        "sample_lag_counts": dict(counts),
    }


def analyze(checkpoint, steps, bound):
    all_metrics = defaultdict(dict)
    for line in (checkpoint / "dump/dashboard/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        all_metrics[(record["step_key"], record["step"])].update(record["metrics"])
    reports = []
    for step in steps:
        metrics = all_metrics[("rollout/step", step)]
        train_metrics = all_metrics[("train/step", step)]
        for key in (
            "staleness/sample_lag/provenance_sample_frac",
            "staleness/version_mix/train/provenance_sample_frac",
            "staleness/token_lag/exact/covered_response_token_frac",
            "staleness/token_lag/exact/loss_token/covered_loss_token_frac",
        ):
            close(metrics[key], 1, key)
        for key in ("invalid_segments", "invalid_turns", "invalid_samples"):
            close(metrics[f"staleness/token_lag/exact/{key}"], 0, key)
        close(metrics["rollout/fully_async/useful_rollout/accounting_error_tokens"], 0, "accounting")
        assert metrics["throughput/window_seconds"] > 0
        assert "queue/consumption/wall_wait_seconds/mean" in metrics
        for key in ("train/loss", "train/grad_norm"):
            assert math.isfinite(train_metrics[key]), (step, key)
        payload = torch.load(checkpoint / f"dump/rollout_data/{step}.pt", map_location="cpu", weights_only=False)
        assert payload["rollout_id"] == step
        evidence = validate_samples(payload["samples"], metrics, train_metrics, bound)
        reports.append(
            {
                "step": step,
                **evidence,
                **{
                    key: metrics[key]
                    for key in (
                        "fully_async/train_weight_version",
                        "staleness/total/mean",
                        "staleness/total/max",
                        "staleness/pre_queue/mean",
                        "staleness/in_queue/mean",
                        "staleness/token_lag/exact/mean",
                        "staleness/bound_exceeded_groups",
                        "staleness/version_mix/train/mixed_sample_frac",
                        "throughput/window_seconds",
                    )
                },
                "loss": train_metrics["train/loss"],
                "grad_norm": train_metrics["train/grad_norm"],
            }
        )
    assert any(row["staleness/total/max"] > 0 for row in reports), "no nonzero async lag observed"
    return {"checkpoint": str(checkpoint), "bound": bound, "passed": True, "steps": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--bound", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(analyze(args.checkpoint, args.steps, args.bound), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
