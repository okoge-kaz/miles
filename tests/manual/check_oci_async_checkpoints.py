"""Verify retained async replay/HF artifacts using CPU-only libraries.

Checks every replay checksum and HF shard header, and compares a prepared batch
with the resumed rollout. Weight-value checks are explicitly small probes, not
a full-model finite-value scan. No checkpoint or data files are modified.
"""

import argparse
import json
import pickle
import struct
from pathlib import Path

import torch
from safetensors import safe_open

from miles.rollout.replay_buffer import load_replay_buffer


def check_hf_headers(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    expected = index["weight_map"]
    actual = {}
    tensor_bytes = 0
    for shard in sorted(set(expected.values())):
        path = root / shard
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        offsets = sorted(value["data_offsets"] for value in entries.values())
        assert offsets[0][0] == 0
        assert all(left[1] == right[0] for left, right in zip(offsets, offsets[1:]))
        assert path.stat().st_size == 8 + header_size + offsets[-1][1]
        tensor_bytes += offsets[-1][1]
        for key in entries:
            assert key not in actual
            actual[key] = shard
    assert actual == expected
    assert tensor_bytes == index["metadata"]["total_size"]
    return {"shards": len(set(expected.values())), "tensors": len(actual), "tensor_bytes": tensor_bytes}


def weight_probe(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    name = "model.layers.0.self_attn.q_proj.weight"
    with safe_open(root / index["weight_map"][name], framework="pt", device="cpu") as tensors:
        probe = tensors.get_slice(name)[:128, :128].float().clone()
    assert torch.isfinite(probe).all()
    return probe


def check_artifacts(root, hf_base):
    assert int((root / "latest_checkpointed_iteration.txt").read_text()) == 5
    raw = torch.load(root / "rollout/replay_buffer_2.pt", map_location="cpu", weights_only=False)
    fingerprint = raw["dataset_fingerprint"]
    reports = {}
    replay_two = None
    for step in range(6):
        state = load_replay_buffer(root, step, expected_fingerprint=fingerprint)
        reports[step] = {"applied_version": state["applied_weight_version"], "counts": state["snapshot_counts"]}
        assert state["applied_weight_version"] == step + 1
        if step == 2:
            replay_two = state
        with (root / f"iter_{step:07d}/.metadata").open("rb") as stream:
            metadata = pickle.load(stream)
        keys = metadata.state_dict_metadata
        assert any(key.startswith("optimizer.") for key in keys), "missing optimizer state"
        assert any(key.startswith("rng_state") for key in keys), "missing RNG state"
    prepared = replay_two["prepared_batches"]
    assert len(prepared) == 1 and prepared[0]["rollout_id"] == 3
    saved_samples = [sample for group in prepared[0]["samples"] for sample in group]
    resumed = torch.load(root / "dump/rollout_data/3.pt", map_location="cpu", weights_only=False)["samples"]
    assert len(saved_samples) == len(resumed) == 16
    fields = (
        "index",
        "group_index",
        "retry_count",
        "tokens",
        "response",
        "response_length",
        "reward",
        "rollout_log_probs",
        "first_prefill_weight_versions",
        "last_forward_weight_versions",
        "min_forward_weight_versions",
        "max_forward_weight_versions",
        "response_weight_version_segments",
    )
    for before, after in zip(saved_samples, resumed, strict=True):
        for field in fields:
            assert before[field] == after[field], (field, before["index"])
    base_probe = weight_probe(hf_base)
    hf = {}
    previous_probe = base_probe
    for step in (2, 5):
        directory = root / f"hf/{step}"
        hf[step] = check_hf_headers(directory)
        probe = weight_probe(directory)
        changed = int((probe != previous_probe).sum())
        assert changed > 0, "weight probe did not change across training"
        hf[step]["probe_changed_elements_from_previous"] = changed
        previous_probe = probe
    return {
        "passed": True,
        "replay_checkpoints": reports,
        "prepared_batch_samples_preserved": 16,
        "hf": hf,
        "weight_probe_shape": [128, 128],
        "full_model_finite_scan": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-base", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_artifacts(args.checkpoint, args.hf_base), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
