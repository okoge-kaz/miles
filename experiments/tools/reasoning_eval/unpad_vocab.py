#!/usr/bin/env python3
"""Drop Megatron vocabulary padding from an exported Hugging Face checkpoint.

The Miles Qwen3-4B export records the tokenizer vocabulary size in config.json,
but the embedding tensor can retain Megatron's padded row count. vLLM rejects
that mismatch. This tool creates a runtime checkpoint whose embedding has the
configured number of rows. Unchanged safetensor shards are hard-linked where
possible and absolute-symlinked otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any


EMBEDDING_NAME = "model.embed_tokens.weight"
COPY_CHUNK_BYTES = 64 << 20


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        (header_size,) = struct.unpack("<Q", stream.read(8))
        return json.loads(stream.read(header_size)), 8 + header_size


def _tensor_data_size(path: Path) -> int:
    header, _ = _read_header(path)
    return sum(
        int(tensor["data_offsets"][1]) - int(tensor["data_offsets"][0])
        for name, tensor in header.items()
        if name != "__metadata__"
    )


def _rewrite_shard(*, source: Path, destination: Path, tensor_name: str, keep_rows: int) -> None:
    header, data_start = _read_header(source)
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    rows, columns = tensors[tensor_name]["shape"]
    begin, end = tensors[tensor_name]["data_offsets"]
    item_size = (end - begin) // (rows * columns)
    shortened_length = keep_rows * columns * item_size

    ordered_names = sorted(tensors, key=lambda name: tensors[name]["data_offsets"][0])
    output_header: dict[str, Any] = {}
    if "__metadata__" in header:
        output_header["__metadata__"] = header["__metadata__"]
    cursor = 0
    copy_plan: list[tuple[int, int]] = []
    for name in ordered_names:
        tensor_begin, tensor_end = tensors[name]["data_offsets"]
        length = shortened_length if name == tensor_name else tensor_end - tensor_begin
        shape = [keep_rows, columns] if name == tensor_name else tensors[name]["shape"]
        output_header[name] = {
            "dtype": tensors[name]["dtype"],
            "shape": shape,
            "data_offsets": [cursor, cursor + length],
        }
        copy_plan.append((tensor_begin, length))
        cursor += length

    encoded_header = json.dumps(output_header, separators=(",", ":")).encode()
    encoded_header += b" " * ((-len(encoded_header)) % 8)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
        output_stream.write(struct.pack("<Q", len(encoded_header)))
        output_stream.write(encoded_header)
        for tensor_begin, length in copy_plan:
            input_stream.seek(data_start + tensor_begin)
            remaining = length
            while remaining:
                chunk = input_stream.read(min(remaining, COPY_CHUNK_BYTES))
                if not chunk:
                    raise EOFError(f"{source} ended {remaining} bytes early")
                output_stream.write(chunk)
                remaining -= len(chunk)
    temporary.replace(destination)


def _link_or_copy_metadata(*, source: Path, destination: Path, rewritten_shard: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if item.name == rewritten_shard:
            continue
        if item.name.endswith(".safetensors"):
            try:
                os.link(item, target)
            except OSError:
                target.symlink_to(item.resolve())
        elif item.is_file():
            shutil.copy2(item, target)


def unpad_checkpoint(*, source: Path, destination: Path) -> bool:
    """Create an unpadded runtime checkpoint and return whether it was needed."""
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    vocabulary_size = int(config["vocab_size"])
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_name = index["weight_map"][EMBEDDING_NAME]
    shard_path = source / shard_name
    header, _ = _read_header(shard_path)
    embedding_rows = int(header[EMBEDDING_NAME]["shape"][0])

    if embedding_rows == vocabulary_size:
        print(f"{source}: embedding already has {embedding_rows} rows")
        return False
    if embedding_rows < vocabulary_size:
        raise ValueError(
            f"{source}: embedding has {embedding_rows} rows, fewer than config vocab_size={vocabulary_size}"
        )

    _, columns = header[EMBEDDING_NAME]["shape"]
    begin, end = header[EMBEDDING_NAME]["data_offsets"]
    item_size = (end - begin) // (embedding_rows * columns)
    removed_bytes = (embedding_rows - vocabulary_size) * columns * item_size
    original_total_size = index.get("metadata", {}).get("total_size")
    if original_total_size is None:
        original_total_size = sum(
            _tensor_data_size(source / name) for name in set(index["weight_map"].values())
        )

    _link_or_copy_metadata(source=source, destination=destination, rewritten_shard=shard_name)
    _rewrite_shard(
        source=shard_path,
        destination=destination / shard_name,
        tensor_name=EMBEDDING_NAME,
        keep_rows=vocabulary_size,
    )
    index["weight_map"] = dict(index["weight_map"])
    index["metadata"] = dict(index.get("metadata", {}))
    index["metadata"]["total_size"] = int(original_total_size) - removed_bytes
    index_path_out = destination / index_path.name
    index_path_out.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    after, _ = _read_header(destination / shard_name)
    print(
        f"{source} -> {destination}: {EMBEDDING_NAME} "
        f"{embedding_rows} -> {after[EMBEDDING_NAME]['shape'][0]} rows"
    )
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    unpad_checkpoint(source=args.source.resolve(), destination=args.destination)


if __name__ == "__main__":
    main()
