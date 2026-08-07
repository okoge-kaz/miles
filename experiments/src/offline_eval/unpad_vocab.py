#!/usr/bin/env python3
"""Drop Megatron's vocabulary padding from an exported HF checkpoint.

    experiments/src/offline_eval/unpad_vocab.py <src-hf-dir> <dst-dir>

`--vocab-size 151936` is padded to `padded_vocab_size` 152064 by
`_vocab_size_with_padding` (`megatron_utils/arguments.py:35`), and
`megatron.bridge`'s `save_hf_pretrained` writes the padded tensor while leaving
`config.json` at the true 151936. sglang then refuses to load it:

    AssertionError: self.org_vocab_size=151936 ... loaded_weight.shape[0]=152064

Setting `vocab_size` to 152064 instead would load, and would be wrong. The model
ties its output projection to the embedding (`tie_word_embeddings: true`), so the
padding rows become 128 extra logits. They are not zero -- measured at 1.65e-09
against 8e-2 for real rows -- which puts them at a logit of about 0. Against a
peaked distribution that is ~4e-5 of the mass per token, and over a 6k-token
response it is a coin-flip whether at least one sampled id is outside the
tokenizer's range.

So truncate. Only the shard holding the embedding is rewritten; the others are
hard-linked where the filesystem allows it and symlinked otherwise; either way
the reader must be able to see the source tree, which is why run_eval.sbatch
takes EXTRA_MOUNTS.

No torch: safetensors is read and written as bytes, which keeps this runnable
outside the training container.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from pathlib import Path

EMBED = "model.embed_tokens.weight"


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n)), 8 + n


def rewrite_shard(src: Path, dst: Path, name: str, keep_rows: int) -> None:
    """Copy `src` to `dst` with tensor `name` truncated to its first `keep_rows`."""
    header, data_start = read_header(src)
    meta = {k: v for k, v in header.items() if k != "__metadata__"}
    rows, cols = meta[name]["shape"]
    itemsize = (meta[name]["data_offsets"][1] - meta[name]["data_offsets"][0]) // (rows * cols)
    new_len = keep_rows * cols * itemsize

    # Offsets are relative to the start of the data block and must stay ordered
    # and contiguous, so recompute every one rather than patching the shrunk pair.
    order = sorted(meta, key=lambda k: meta[k]["data_offsets"][0])
    out_header: dict = {}
    if "__metadata__" in header:
        out_header["__metadata__"] = header["__metadata__"]
    cursor = 0
    plan = []
    for k in order:
        begin, end = meta[k]["data_offsets"]
        length = new_len if k == name else end - begin
        shape = [keep_rows, cols] if k == name else meta[k]["shape"]
        out_header[k] = {"dtype": meta[k]["dtype"], "shape": shape, "data_offsets": [cursor, cursor + length]}
        plan.append((begin, length))
        cursor += length

    blob = json.dumps(out_header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8  # the header is 8-byte aligned
    blob += b" " * pad

    tmp = dst.with_suffix(dst.suffix + ".partial")
    with src.open("rb") as fi, tmp.open("wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for begin, length in plan:
            fi.seek(data_start + begin)
            remaining = length
            while remaining:
                chunk = fi.read(min(remaining, 64 << 20))
                if not chunk:
                    raise EOFError(f"{src} ended {remaining} bytes early")
                fo.write(chunk)
                remaining -= len(chunk)
    tmp.rename(dst)


def main(src_dir: str, dst_dir: str) -> int:
    src, dst = Path(src_dir), Path(dst_dir)
    vocab = json.loads((src / "config.json").read_text())["vocab_size"]
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_name = index["weight_map"][EMBED]
    header, _ = read_header(src / shard_name)
    rows, _cols = header[EMBED]["shape"]

    if rows == vocab:
        print(f"{src}: already {rows} rows, nothing to do")
        return 0
    if rows < vocab:
        print(f"{src}: embedding has {rows} rows, fewer than config's {vocab}", file=sys.stderr)
        return 1

    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if item.name == shard_name:
            continue
        if item.name.endswith(".safetensors"):
            # A hard link where the filesystem allows it, a symlink otherwise.
            # Either way the shard is not copied: they are 4 GB each and
            # unchanged. A symlink points at an absolute host path outside this
            # directory, so whatever reads it has to see that path too -- pass
            # the source root to run_eval.sbatch as EXTRA_MOUNTS.
            try:
                os.link(item, target)
            except OSError:
                target.symlink_to(item.resolve())
        else:
            shutil.copy2(item, target)

    rewrite_shard(src / shard_name, dst / shard_name, EMBED, vocab)

    index["weight_map"] = dict(index["weight_map"])
    index["metadata"] = dict(index.get("metadata", {}))
    index["metadata"]["total_size"] = sum(
        (dst / n).stat().st_size if not (dst / n).is_symlink() else os.stat(dst / n).st_size
        for n in set(index["weight_map"].values())
    )
    (dst / index_path.name).write_text(json.dumps(index, indent=2))

    after, _ = read_header(dst / shard_name)
    print(f"{src} -> {dst}: {EMBED} {rows} -> {after[EMBED]['shape'][0]} rows")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
