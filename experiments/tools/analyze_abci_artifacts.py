"""Verify finalized validation checkpoints using the real replay-buffer loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from miles.rollout.replay_buffer import load_replay_buffer

PREPARED_SAMPLE_FIELDS = (
    "index",
    "group_index",
    "prompt",
    "tokens",
    "response",
    "response_length",
    "label",
    "reward",
    "weight_versions",
    "first_prefill_weight_versions",
    "min_forward_weight_versions",
    "max_forward_weight_versions",
    "last_forward_weight_versions",
    "response_weight_versions",
    "response_weight_version_segments",
    "rollout_log_probs",
    "status",
    "retry_count",
)


def verify_prepared_batches(checkpoint: Path, state: dict) -> list[dict]:
    reports = []
    for prepared in state["prepared_batches"]:
        rollout_id = prepared["rollout_id"]
        path = checkpoint / "dump" / "rollout_data" / f"{rollout_id}.pt"
        dumped = torch.load(path, map_location="cpu", weights_only=False)["samples"]
        by_index = {sample["index"]: sample for sample in dumped}
        saved = [sample for group in prepared["samples"] for sample in group]
        assert len(saved) == len(dumped) == len(by_index), (path, "sample count mismatch")
        assert len({sample["index"] for sample in saved}) == len(saved), (path, "duplicate sample index")
        for sample in saved:
            restored = by_index[sample["index"]]
            for field in PREPARED_SAMPLE_FIELDS:
                assert sample[field] == restored[field], (path, sample["index"], field)
            # Dumps precede train-data conversion, which expands an omitted
            # loss mask into ones before the prepared batch is checkpointed.
            saved_mask = sample["loss_mask"]
            dumped_mask = restored["loss_mask"]
            if saved_mask is None:
                saved_mask = [1] * sample["response_length"]
            if dumped_mask is None:
                dumped_mask = [1] * restored["response_length"]
            assert saved_mask == dumped_mask, (path, sample["index"], "loss_mask")
        reports.append(
            {
                "rollout_id": rollout_id,
                "samples": len(saved),
                "response_tokens": sum(sample["response_length"] for sample in saved),
                "exact_match_fields": list(PREPARED_SAMPLE_FIELDS),
                "loss_masks_equivalent": True,
            }
        )
    return reports


def verify_replay(checkpoint: Path) -> list[dict]:
    reports = []
    paths = sorted(
        (checkpoint / "rollout").glob("replay_buffer_*.pt"), key=lambda path: int(path.stem.rsplit("_", 1)[1])
    )
    assert paths, f"No replay snapshots in {checkpoint}"
    for path in paths:
        iteration = int(path.stem.rsplit("_", 1)[1])
        stored = torch.load(path, map_location="cpu", weights_only=False)
        # The training restart checks the fingerprint against its live config.
        # This offline audit checks payload integrity, including external tensors.
        fingerprint = stored["dataset_fingerprint"]
        del stored
        state = load_replay_buffer(checkpoint, iteration, expected_fingerprint=fingerprint)
        manifest = json.loads(Path(f"{path}.sha256.json").read_text())
        reports.append(
            {
                "iteration": iteration,
                "bytes": manifest["size"],
                "parts": len(manifest["parts"]),
                "checksums_verified": True,
                "snapshot_counts": state["snapshot_counts"],
                "prepared_batches": verify_prepared_batches(checkpoint, state),
            }
        )
        del state
    return reports


def verify_hf_exports(checkpoint: Path) -> list[dict]:
    reports = []
    for index in sorted((checkpoint / "hf").glob("*/model.safetensors.index.json")):
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = set(weight_map.values())
        assert shards and weight_map, index
        for shard in shards:
            path = index.parent / shard
            assert path.is_file() and path.stat().st_size > 0, path
        assert (index.parent / "config.json").is_file(), index
        reports.append({"iteration": int(index.parent.name), "nonempty_shards": len(shards)})
    assert reports, f"No indexed HF exports in {checkpoint}"
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    latest = int((args.checkpoint / "latest_checkpointed_iteration.txt").read_text())
    model_iterations = sorted(
        int(path.name.removeprefix("iter_"))
        for path in args.checkpoint.glob("iter_*")
        if (path / ".metadata").is_file()
    )
    assert latest in model_iterations, "Latest distributed model metadata is missing"
    replay = verify_replay(args.checkpoint)
    missing_replay = set(model_iterations) - {entry["iteration"] for entry in replay}
    assert not missing_replay, f"Retained model iterations lack verified replay snapshots: {sorted(missing_replay)}"
    report = {
        "checkpoint": str(args.checkpoint),
        "latest_iteration": latest,
        "model_iterations": model_iterations,
        "model_replay_pairs_verified": True,
        "replay": replay,
        "hf_exports": verify_hf_exports(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
