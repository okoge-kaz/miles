"""Check routing persistence in real Lightning checkpoint and training artifacts."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from miles.rollout.replay_buffer import load_replay_buffer, replay_buffer_path


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _identity(sample):
    return str((sample["group_index"], sample["index"], sample.get("retry_count", 0)))


def _fingerprints(sample, *, tokens=None, response=None):
    tokens = len(sample["tokens"]) if tokens is None else tokens
    response = sample["response_length"] if response is None else response
    return {
        "tokens": _digest(np.asarray(sample["tokens"][:tokens], dtype=np.int64)),
        "logprobs": _digest(np.asarray(sample["rollout_log_probs"][:response], dtype=np.float64)),
        "routes": _digest(sample["rollout_routed_experts"][: tokens - 1]),
    }


def _validate(sample, config):
    if sample["response_length"] == 0:
        return
    routes = sample["rollout_routed_experts"]
    assert routes.shape == (len(sample["tokens"]) - 1, config["num_hidden_layers"], config["num_experts_per_tok"])
    assert routes.dtype == np.int32
    layers = [i for i, kind in enumerate(config["hybrid_override_pattern"]) if kind == "E"]
    active = routes[:, layers]
    assert np.all((active >= 0) & (active < config["n_routed_experts"])), "Invalid MoE expert IDs"
    assert len(sample["rollout_log_probs"]) == sample["response_length"]


def capture(checkpoint, config):
    rollout_id = int((checkpoint / "latest_checkpointed_iteration.txt").read_text())
    # Read the saved fingerprint for artifact inspection; the training resume
    # separately compares it against the live configuration and dataset.
    fingerprint = torch.load(replay_buffer_path(checkpoint, rollout_id), weights_only=False, map_location="cpu")[
        "dataset_fingerprint"
    ]
    state = load_replay_buffer(checkpoint, rollout_id, expected_fingerprint=fingerprint)
    groups = [("ready", item["result"]) for item in state["ready_items"]]
    groups += [("prepared", group) for batch in state["prepared_batches"] for group in batch["samples"]]
    groups += [("inflight", item["generation_group"]) for item in state["inflight_items"]]
    samples = {}
    for location, group in groups:
        for sample in group:
            _validate(sample, config)
            if not sample["response_length"]:
                continue
            category = "inflight" if sample["status"] == "aborted" else "completed"
            samples[_identity(sample)] = {
                "category": category,
                "location": location,
                "total_tokens": len(sample["tokens"]),
                "response_tokens": sample["response_length"],
                "fingerprints": _fingerprints(sample),
            }
    assert {sample["category"] for sample in samples.values()} == {"completed", "inflight"}
    return {"rollout_id": rollout_id, "counts": state["snapshot_counts"], "samples": samples}


def _training_updates(checkpoint):
    history = defaultdict(dict)
    for line in (checkpoint / "dump/dashboard/metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["step_key"] == "train/step":
            history[row["step"]].update(row["metrics"])
    return {step: metrics for step, metrics in history.items() if "train/grad_norm" in metrics}


def check(checkpoint, config, reference, probe_job, expected_updates):
    updates = _training_updates(checkpoint)
    assert len(updates) >= expected_updates, len(updates)
    matches = defaultdict(set)
    for path in sorted((checkpoint / "dump/rollout_data").glob("[0-9]*.pt")):
        # Async writes prefetched batches too; count only completed optimizer updates.
        if int(path.stem) not in updates:
            continue
        data = torch.load(path, weights_only=False, map_location="cpu")
        groups = defaultdict(list)
        for sample in data["samples"]:
            _validate(sample, config)
            groups[sample["group_index"]].append(sample["reward"])
            saved = reference["samples"].get(_identity(sample))
            if saved is None or int(path.stem) <= reference["rollout_id"]:
                continue
            assert (
                _fingerprints(sample, tokens=saved["total_tokens"], response=saved["response_tokens"])
                == saved["fingerprints"]
            )
            matches[saved["category"]].add(_identity(sample))
        assert all(
            len(rewards) == 8 and set(rewards) == {0, 1} for rewards in groups.values()
        ), "Dynamic filtering failed"
    assert matches["completed"] and matches["inflight"], {key: len(value) for key, value in matches.items()}
    probes = list((checkpoint / "dump/replay_routing_probe").glob(f"{probe_job}-rank*.jsonl"))
    assert len(probes) == 8, probes
    expected_streams = config["hybrid_override_pattern"].count("E")
    for path in probes:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) >= 3, (path, len(rows))
        for row in rows:
            assert len(row["streams"]) == expected_streams
            for stream in row["streams"]:
                assert stream["forward_consumed"] == stream["backward_consumed"] == stream["microbatches"] > 0
    for metrics in updates.values():
        assert math.isfinite(metrics["train/grad_norm"]) and metrics["train/grad_norm"] > 0
        assert math.isfinite(metrics["train/loss"])
    return {
        "updates": len(updates),
        "training_ranks_verified": len(probes),
        "moe_streams_per_rank": expected_streams,
        "restored_samples_trained_with_identical_prefix": {key: len(value) for key, value in matches.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--probe-job")
    parser.add_argument("--expected-updates", type=int, default=6)
    args = parser.parse_args()
    config = json.loads(args.hf_config.read_text())
    evidence = args.checkpoint / "dump/r3-validation"
    evidence.mkdir(parents=True, exist_ok=True)
    reference_path = evidence / "saved-prefixes.json"
    if args.capture:
        result = capture(args.checkpoint, config)
        reference_path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "rollout_id": result["rollout_id"],
                    "counts": result["counts"],
                    "fingerprinted_samples": len(result["samples"]),
                }
            )
        )
    else:
        result = check(
            args.checkpoint, config, json.loads(reference_path.read_text()), args.probe_job, args.expected_updates
        )
        (evidence / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
