#!/usr/bin/env python3
"""Normalize .sif filenames into one symlink farm.

The Apptainer sandbox provider takes a single ``container_formatter``
template, but the prebuilt images use per-publisher prefixes
(``xingyaoww_*`` for SWE-Gym, ``swebench_*`` for Verified, ``namanjain12_*``
for R2E-Gym). One template cannot produce all of them, so this creates
``<link-dir>/<instance_id>.sif`` symlinks pointing at whatever the real file
is called. Symlinks cost no disk and leave the originals untouched.

Usage:
    python link_sifs.py --sif-dir /lustre/.../sif --link-dir /lustre/.../sif_by_id \
        --input data/train_swegym.jsonl --input data/eval_verified_full.jsonl
"""

import argparse
import json
from pathlib import Path

from preflight import build_sif_index, find_sif


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sif-dir", required=True, type=Path)
    parser.add_argument("--link-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, action="append", help="Miles prompt-data JSONL (repeatable)")
    args = parser.parse_args()

    args.link_dir.mkdir(parents=True, exist_ok=True)
    index = build_sif_index(args.sif_dir)

    linked = skipped = missing = 0
    for path in args.input:
        with open(path) as f:
            for line in f:
                instance_id = json.loads(line)["metadata"]["instance_id"]
                link = args.link_dir / f"{instance_id}.sif"
                if link.is_symlink() or link.exists():
                    skipped += 1
                    continue
                target = find_sif(instance_id, index)
                if target is None:
                    missing += 1
                    continue
                link.symlink_to(target.resolve())
                linked += 1

    print(f"linked {linked}, already present {skipped}, missing {missing}")
    print(f"container_formatter: {args.link_dir}/{{instance_id}}.sif")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
