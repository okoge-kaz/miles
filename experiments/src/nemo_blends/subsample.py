"""Cut a prompt JSONL down to a measurable size, deterministically.

Measuring a pass rate costs 8 generations per prompt, so a full component is not
always affordable: knowledge-mcqa is 685,573 prompts, which at the ~0.6 prompt/s
this cluster sustains would be roughly 300 GPU-hours for one policy. Breadth
across components buys more than exhaustiveness within one, so each gets a fixed
budget instead.

Selection is by hash of the line index, not `random.sample` and not the first N:

  * the first N is ordered-biased -- these files arrive grouped by source
    dataset and, per the blend README, sorted easiest-first;
  * a hash is reproducible without carrying a seed *and* stable under growth:
    re-running with a larger --size keeps every prompt the smaller run picked,
    so an existing measurement stays valid and only the new prompts need
    generating.
"""

import argparse
import hashlib
import json
from pathlib import Path


def keep(index: int, size: int, total: int, salt: str) -> bool:
    """Deterministic membership test, monotone in `size`."""
    digest = hashlib.sha256(f"{salt}:{index}".encode()).digest()
    # Map the hash into [0, total) and keep the lowest `size` slots. Monotone in
    # size, so growing the budget only ever adds prompts.
    return int.from_bytes(digest[:8], "big") % total < size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, required=True, help="approximate number of prompts to keep")
    ap.add_argument("--salt", default="", help="changes the selection; leave empty for the canonical subset")
    args = ap.parse_args()

    src = Path(args.input)
    total = sum(1 for line in src.open() if line.strip())
    if args.size >= total:
        raise SystemExit(f"--size {args.size} >= {total} rows in {src}; nothing to subsample")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src.open() as fin, out.open("w") as fout:
        index = 0
        for line in fin:
            if not line.strip():
                continue
            if keep(index, args.size, total, args.salt):
                # `source_index` is what ties a measurement back to the full file.
                row = json.loads(line)
                row.setdefault("metadata", {})["source_index"] = index
                fout.write(json.dumps(row) + "\n")
                kept += 1
            index += 1

    print(f"{src.name}: {total} -> {kept} rows (target {args.size}) -> {out}")


if __name__ == "__main__":
    main()
