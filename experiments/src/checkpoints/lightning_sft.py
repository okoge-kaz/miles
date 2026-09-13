"""Stage a local Lightning SFT policy in native HF format without altering its source.

Copies policy shards and tokenizer assets, translates the legacy hybrid pattern,
and excludes unused MTP heads from the policy index. No download is performed.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
)
LAYER_TYPES = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}


def native_policy_config(config: dict) -> dict:
    result = dict(config)
    if pattern := result.get("hybrid_override_pattern"):
        layers = [LAYER_TYPES[character] for character in pattern]
        if result.get("layers_block_type") not in (None, layers):
            raise ValueError("Legacy and native hybrid layer descriptions disagree")
        result["layers_block_type"] = layers
    if len(result["layers_block_type"]) != result["num_hidden_layers"]:
        raise ValueError("Hybrid layer count differs from num_hidden_layers")
    result.pop("auto_map", None)
    result.update(num_nextn_predict_layers=0, mtp_layers_block_type=[], mtp_hybrid_override_pattern="")
    return result


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_size < path.stat().st_size:
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(stream.read(header_size))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if 8 + header_size + max(value["data_offsets"][1] for value in tensors.values()) != path.stat().st_size:
        raise ValueError(f"Incomplete safetensors file: {path}")
    return tensors


def copy_with_hash(source: Path, destination: Path) -> str:
    before = source.stat()
    digest = hashlib.sha256()
    temporary = destination.with_name(destination.name + ".partial")
    with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
        while chunk := input_stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            output_stream.write(chunk)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed while copying: {source}")
    if temporary.stat().st_size != before.st_size:
        raise ValueError(f"Incomplete copy: {temporary}")
    temporary.replace(destination)
    return digest.hexdigest()


def stage_policy(source: Path, destination: Path, *, source_id: str) -> dict:
    if source.resolve() == destination.resolve():
        raise ValueError("Destination must differ from the original SFT checkpoint")
    config_bytes = (source / "config.json").read_bytes()
    index_bytes = (source / "model.safetensors.index.json").read_bytes()
    identity = {
        "source": source_id,
        "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
    }
    marker = destination / ".download_complete"
    if marker.exists():
        report = json.loads((destination / "staging_manifest.json").read_text())
        if any(report.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Destination contains a different source: {destination}")
        for name, checksum in report["copied_sha256"].items():
            with (destination / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                    raise ValueError(f"Staged file changed: {name}")
        return report
    index = json.loads(index_bytes)
    weights = {key: shard for key, shard in index["weight_map"].items() if not key.startswith("mtp.")}
    headers = {name: read_header(source / name) for name in sorted(set(index["weight_map"].values()))}
    for name, header in headers.items():
        if set(header) != {key for key, shard in index["weight_map"].items() if shard == name}:
            raise ValueError(f"Tensor index differs from shard header: {name}")
    policy_bytes = sum(
        headers[shard][key]["data_offsets"][1] - headers[shard][key]["data_offsets"][0]
        for key, shard in weights.items()
    )
    policy_parameters = sum(math.prod(headers[shard][key]["shape"]) for key, shard in weights.items())
    config = native_policy_config(json.loads(config_bytes))
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    names = sorted(set(weights.values())) + [name for name in TOKENIZER_FILES if (source / name).is_file()]
    for name in names:
        print(f"Copying SFT asset: {name}", flush=True)
        copied[name] = copy_with_hash(source / name, destination / name)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": policy_bytes}, "weight_map": weights}, indent=2) + "\n"
    )
    for name in ("config.json", "model.safetensors.index.json"):
        copied[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    report = dict(
        identity,
        policy_tensors=len(weights),
        policy_parameters=policy_parameters,
        policy_bytes=policy_bytes,
        excluded_mtp_tensors=len(index["weight_map"]) - len(weights),
        policy_shards=len(set(weights.values())),
        copied_sha256=copied,
    )
    (destination / "staging_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    marker.write_text(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-id", help="original host path to record when --source is a container mount")
    args = parser.parse_args()
    report = stage_policy(args.source, args.destination, source_id=args.source_id or str(args.source.resolve()))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
