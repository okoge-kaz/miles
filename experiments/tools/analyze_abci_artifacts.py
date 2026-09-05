"""Verify finalized validation checkpoints using the real replay-buffer loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from miles.rollout.replay_buffer import load_replay_buffer


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
    assert (args.checkpoint / f"iter_{latest:07d}").is_dir()
    replay = verify_replay(args.checkpoint)
    assert latest in {entry["iteration"] for entry in replay}, "Latest model lacks a verified replay snapshot"
    report = {
        "checkpoint": str(args.checkpoint),
        "latest_iteration": latest,
        "replay": replay,
        "hf_exports": verify_hf_exports(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
