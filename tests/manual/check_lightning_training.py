"""Verify real Lightning RL updates, dynamic filtering, and complete HF exports."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def check_export(path, reference_keys):
    assert (path / ".complete").is_file(), path
    config = json.loads((path / "config.json").read_text())
    assert config["num_nextn_predict_layers"] == 0
    index = json.loads((path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == reference_keys
    expected = defaultdict(set)
    for key, shard in index["weight_map"].items():
        expected[shard].add(key)
    for shard, keys in expected.items():
        with safe_open(path / shard, framework="pt", device="cpu") as tensors:
            assert set(tensors.keys()) == keys, shard


def check_weights_changed(reference_path, export_path, reference_index):
    exported = json.loads((export_path / "model.safetensors.index.json").read_text())
    candidates = sorted(key for key in reference_index["weight_map"] if key.endswith("up_proj.weight"))
    keys = candidates[:: max(1, len(candidates) // 16)][:16]
    changed = 0
    for key in keys:
        with safe_open(reference_path / reference_index["weight_map"][key], framework="pt", device="cpu") as original:
            before = original.get_slice(key)[:64, :64]
        with safe_open(export_path / exported["weight_map"][key], framework="pt", device="cpu") as trained:
            after = trained.get_slice(key)[:64, :64]
        assert torch.isfinite(after).all(), key
        changed += int((before != after).sum())
    assert changed > 0, "No optimizer update visible in sampled HF weights"
    print(
        json.dumps(
            {
                "reference": reference_path.name,
                "export": export_path.name,
                "sampled_expert_matrices": len(keys),
                "changed_weight_elements": changed,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-reference", type=Path, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--minimum-steps", type=int, default=2)
    args = parser.parse_args()
    history = defaultdict(dict)
    for line in (args.checkpoint / "dump/dashboard/metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["step_key"] == "train/step":
            history[row["step"]].update(row["metrics"])
    updates = {step: metrics for step, metrics in history.items() if "train/grad_norm" in metrics}
    assert len(updates) >= args.minimum_steps, updates
    for step, metrics in sorted(updates.items()):
        assert math.isfinite(metrics["train/grad_norm"]) and metrics["train/grad_norm"] > 0, (step, metrics)
        assert math.isfinite(metrics["train/loss"]), (step, metrics)
        learning_rates = [value for key, value in metrics.items() if key.startswith("train/lr-")]
        assert learning_rates and all(math.isfinite(value) and value > 0 for value in learning_rates)
        print(json.dumps({"step": step, "grad_norm": metrics["train/grad_norm"], "loss": metrics["train/loss"]}))
    rollouts = sorted((args.checkpoint / "dump/rollout_data").glob("[0-9]*.pt"))
    assert len(rollouts) >= args.minimum_steps
    for path in rollouts:
        payload = torch.load(path, weights_only=False, map_location="cpu")
        groups = defaultdict(list)
        for sample in payload["samples"]:
            groups[sample["group_index"]].append(sample["reward"])
        assert len(groups) == args.groups, (path, len(groups))
        assert all(len(rewards) == args.n_samples and set(rewards) == {0, 1} for rewards in groups.values()), groups
        print(
            json.dumps({"rollout": path.stem, "accepted_groups": len(groups), "group_rewards": list(groups.values())})
        )
    reference = json.loads((args.hf_reference / "model.safetensors.index.json").read_text())
    exports = sorted((args.checkpoint / "hf").glob("[0-9]*"), key=lambda path: int(path.name))
    assert len(exports) >= args.minimum_steps, exports
    for path in exports:
        check_export(path, set(reference["weight_map"]))
    check_weights_changed(args.hf_reference, exports[-1], reference)
    if len(exports) > 1:
        previous_index = json.loads((exports[-2] / "model.safetensors.index.json").read_text())
        check_weights_changed(exports[-2], exports[-1], previous_index)
    tracker = (args.checkpoint / "latest_checkpointed_iteration.txt").read_text().strip()
    # Saved iterations are zero-based rollout IDs, including optimizer state.
    assert int(tracker) >= args.minimum_steps - 1
    print(json.dumps({"updates": len(updates), "complete_hf_exports": len(exports), "latest_iteration": tracker}))


if __name__ == "__main__":
    main()
