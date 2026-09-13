"""Download pinned Lightning BF16 weights and create the RL policy view.

Args:
    --hf-root: Parent directory for the original and -policy HF checkpoints.
    --policy-only: Create the policy view from an existing original download.

Example:
    python -m scripts.download.checkpoints.nemotron_3_5_lightning_30b_a3b --hf-root /ckpt/hf
"""

import argparse
import json
import math
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

MODEL_NAME = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
MODEL_REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"


def stage_model(root: Path) -> None:
    destination = root / MODEL_NAME
    snapshot_download(repo_id=f"nvidia/{MODEL_NAME}", revision=MODEL_REVISION, local_dir=destination)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    expected = {}
    for key, shard in index["weight_map"].items():
        expected.setdefault(shard, set()).add(key)
    for shard, keys in expected.items():
        with safe_open(destination / shard, framework="pt", device="cpu") as tensors:
            assert keys == set(tensors.keys()), f"Tensor index mismatch in {shard}"
    report = {
        "repo_id": f"nvidia/{MODEL_NAME}",
        "revision": MODEL_REVISION,
        "shards": len(expected),
        "tensors": len(index["weight_map"]),
        "weight_bytes": sum((destination / shard).stat().st_size for shard in expected),
    }
    (destination / "staging_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / ".download_complete").write_text(MODEL_REVISION + "\n")
    print(json.dumps(report), flush=True)


def prepare_policy_view(root: Path) -> None:
    """Exclude unused MTP heads from the RL view and subsequent strict HF exports.

    The full downloaded model is retained. Relative symlinks share its BF16
    shards, while the policy's config and index describe only trained weights.
    """
    source = root / MODEL_NAME
    destination = root / f"{MODEL_NAME}-policy"
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    config.update(num_nextn_predict_layers=0, mtp_layers_block_type=[])
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = {key: shard for key, shard in index["weight_map"].items() if not key.startswith("mtp.")}
    policy_shards = set(weight_map.values())
    byte_count = 0
    for shard in policy_shards:
        with safe_open(source / shard, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key in weight_map:
                    tensor = tensors.get_slice(key)
                    byte_count += math.prod(tensor.get_shape()) * {"BF16": 2, "F32": 4}[tensor.get_dtype()]
    excluded = {"config.json", "model.safetensors.index.json", "staging_manifest.json", ".download_complete"}
    for path in source.iterdir():
        if path.is_file() and path.name not in excluded:
            target = destination / path.name
            if path.suffix == ".safetensors" and path.name not in policy_shards:
                if target.is_symlink() and target.resolve() == path.resolve():
                    target.unlink()
                continue
            if not target.exists():
                target.symlink_to(Path("..") / MODEL_NAME / path.name)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": byte_count}, "weight_map": weight_map}, indent=2) + "\n"
    )
    report = {
        "source_revision": MODEL_REVISION,
        "policy_tensors": len(weight_map),
        "policy_bytes": byte_count,
        "excluded_mtp_tensors": len(index["weight_map"]) - len(weight_map),
    }
    (destination / "staging_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / ".download_complete").write_text(MODEL_REVISION + "\n")
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--policy-only", action="store_true", help="Build the RL view from an existing full download")
    args = parser.parse_args()
    if not args.policy_only:
        stage_model(args.hf_root)
    prepare_policy_view(args.hf_root)


if __name__ == "__main__":
    main()
