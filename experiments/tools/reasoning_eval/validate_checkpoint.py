#!/usr/bin/env python3
"""Validate a Qwen3 Hugging Face checkpoint without loading its tensors."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_JSON_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)
REQUIRED_FILES = (*REQUIRED_JSON_FILES, "chat_template.jinja", "tokenizer.json")
MAX_HEADER_BYTES = 100 << 20
EXPECTED_ARCHITECTURE = {
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
}


@dataclass(frozen=True)
class CheckpointInfo:
    """Validated checkpoint dimensions and storage size."""

    vocabulary_size: int
    embedding_rows: int
    context_length: int
    shard_count: int
    tensor_count: int
    tensor_bytes: int


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_safetensor_header(path: Path) -> tuple[dict[str, Any], int]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        encoded_size = stream.read(8)
        if len(encoded_size) != 8:
            raise ValueError(f"truncated safetensors length prefix: {path}")
        (header_size,) = struct.unpack("<Q", encoded_size)
        if header_size <= 0 or header_size > min(MAX_HEADER_BYTES, file_size - 8):
            raise ValueError(f"invalid safetensors header size in {path}: {header_size}")
        encoded_header = stream.read(header_size)
    header = json.loads(encoded_header)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object: {path}")
    return header, 8 + header_size


def _validate_shard(path: Path) -> tuple[dict[str, Any], int]:
    header, data_start = _read_safetensor_header(path)
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError(f"safetensors shard has no tensors: {path}")
    ranges: list[tuple[int, int, str]] = []
    for name, tensor in tensors.items():
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid data_offsets for {name} in {path}")
        begin, end = offsets
        if not isinstance(begin, int) or not isinstance(end, int) or begin < 0 or end < begin:
            raise ValueError(f"invalid tensor byte range for {name} in {path}")
        ranges.append((begin, end, name))
    ranges.sort()
    expected_begin = 0
    for begin, end, name in ranges:
        if begin != expected_begin:
            raise ValueError(f"non-contiguous tensor data before {name} in {path}")
        expected_begin = end
    expected_file_size = data_start + expected_begin
    if path.stat().st_size != expected_file_size:
        raise ValueError(
            f"incomplete or oversized shard {path}: "
            f"size={path.stat().st_size}, expected={expected_file_size}"
        )
    return tensors, expected_begin


def validate_checkpoint(path: Path) -> CheckpointInfo:
    """Validate checkpoint metadata, index mappings, and complete shard sizes."""
    for name in REQUIRED_FILES:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise FileNotFoundError(path / name)
    parsed_json = {name: _read_json(path / name) for name in REQUIRED_JSON_FILES}
    config = parsed_json["config.json"]
    if config.get("model_type") != "qwen3":
        raise ValueError(f"expected model_type=qwen3 in {path / 'config.json'}")
    for name, expected in EXPECTED_ARCHITECTURE.items():
        if config.get(name) != expected:
            raise ValueError(
                f"expected Qwen3-4B {name}={expected}, got {config.get(name)!r} "
                f"in {path / 'config.json'}"
            )
    vocabulary_size = int(config["vocab_size"])
    context_length = int(config["max_position_embeddings"])
    if vocabulary_size <= 0 or context_length <= 0:
        raise ValueError(f"invalid Qwen3 dimensions in {path / 'config.json'}")

    index = parsed_json["model.safetensors.index.json"]
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"weight_map is missing from {path / 'model.safetensors.index.json'}")
    shard_names = set(weight_map.values())
    if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shard_names):
        raise ValueError(f"invalid shard names in {path / 'model.safetensors.index.json'}")

    shard_tensors: dict[str, dict[str, Any]] = {}
    tensor_bytes = 0
    for shard_name in sorted(shard_names):
        shard_tensors[shard_name], shard_bytes = _validate_shard(path / shard_name)
        tensor_bytes += shard_bytes
    for tensor_name, shard_name in weight_map.items():
        if tensor_name not in shard_tensors[shard_name]:
            raise ValueError(f"{tensor_name} is absent from indexed shard {shard_name}")

    embedding_name = "model.embed_tokens.weight"
    embedding_shard = weight_map.get(embedding_name)
    if not isinstance(embedding_shard, str):
        raise ValueError(f"{embedding_name} is absent from the checkpoint index")
    embedding_shape = shard_tensors[embedding_shard][embedding_name].get("shape")
    if not isinstance(embedding_shape, list) or len(embedding_shape) != 2:
        raise ValueError(f"invalid embedding shape in {embedding_shard}")
    embedding_rows = int(embedding_shape[0])
    if embedding_rows < vocabulary_size:
        raise ValueError(f"embedding rows {embedding_rows} are below vocab_size {vocabulary_size}")
    return CheckpointInfo(
        vocabulary_size=vocabulary_size,
        embedding_rows=embedding_rows,
        context_length=context_length,
        shard_count=len(shard_names),
        tensor_count=len(weight_map),
        tensor_bytes=tensor_bytes,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        info = validate_checkpoint(args.checkpoint.resolve())
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"invalid checkpoint {args.checkpoint}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if not args.quiet:
        print(
            f"valid Qwen3 checkpoint: {args.checkpoint} "
            f"vocab={info.vocabulary_size}/{info.embedding_rows} "
            f"context={info.context_length} shards={info.shard_count} "
            f"tensors={info.tensor_count} bytes={info.tensor_bytes}"
        )


if __name__ == "__main__":
    main()
