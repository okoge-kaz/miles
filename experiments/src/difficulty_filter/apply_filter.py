"""Annotate a prompt file with measured pass rates, and optionally cut a window from it.

Pure bookkeeping: no GPU, no generation. Re-selecting a different window from an
existing measurement runs in seconds, which is why `measure_pass_rate.py` records
the pass rate rather than a keep/drop decision.

Annotations are keyed by policy and **merged**, never overwritten, so one dataset
accumulates every model it has been measured with:

    "difficulty": {
      "Qwen3-4B-Instruct-2507": {"pass_rate": 0.375, "n_correct": 3, "n_samples": 8,
                                 "truncated_frac": 0.0, "rm_type": "math",
                                 "max_new_tokens": 24576, "temperature": 1.0},
      "Qwen3-4B":               {"pass_rate": 0.875, ...}
    }

Keying by policy is not decoration. A pass rate is a property of the (prompt,
policy, sampling-params) triple, so a bare `pass_rate` field would silently mean
something different after the next model is measured. The sampling parameters
travel with it for the same reason -- a rate measured at a 8k generation budget
is not comparable to one measured at 24k.

    # annotate the whole file, and cut a window in the same pass
    python -m experiments.src.difficulty_filter.apply_filter \
        --prompt-data       /data/dapo-math-17k/dapo-math-17k.jsonl \
        --pass-rates        /data/difficulty/dapo-math-17k.Qwen3-4B-Instruct-2507.passrate.jsonl \
        --output-annotated  /data/dapo-math-17k/dapo-math-17k.annotated.jsonl \
        --output            /data/dapo-math-17k/dapo-math-17k.p20-80.jsonl \
        --pass-rate-min 0.2 --pass-rate-max 0.8

    # add a second model to the same annotated file (feed it back in)
    python -m experiments.src.difficulty_filter.apply_filter \
        --prompt-data       /data/dapo-math-17k/dapo-math-17k.annotated.jsonl \
        --pass-rates        /data/difficulty/dapo-math-17k.Qwen3-4B.passrate.jsonl \
        --output-annotated  /data/dapo-math-17k/dapo-math-17k.annotated.jsonl.new
"""

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from experiments.src.difficulty_filter.pass_rate import (  # noqa: E402
    DEFAULT_PASS_RATE_MAX,
    DEFAULT_PASS_RATE_MIN,
    PassRateRecord,
    pass_rate_in_window,
    summarize,
)

DIFFICULTY_KEY = "difficulty"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--prompt-data",
        required=True,
        help="the JSONL or parquet that was measured (JSONL may already be annotated)",
    )
    # Optional: an annotated file already carries difficulty[policy].pass_rate,
    # so re-cutting a window from it needs no measurement file at all.
    p.add_argument("--pass-rates", default=None, help="measure_pass_rate.py output; omit to read existing tags")
    p.add_argument("--output-annotated", default=None, help="every prompt, with this policy's tag merged in")
    p.add_argument("--output", default=None, help="only the prompts inside the window")
    p.add_argument("--pass-rate-min", type=float, default=DEFAULT_PASS_RATE_MIN)
    p.add_argument("--pass-rate-max", type=float, default=DEFAULT_PASS_RATE_MAX)
    p.add_argument("--policy", default=None, help="tag key; defaults to the measurement's .meta.json")
    p.add_argument(
        "--drop-unmeasured",
        action="store_true",
        help="drop prompts with no measurement instead of failing (use when the sweep was cut short)",
    )
    args = p.parse_args()
    if not args.output_annotated and not args.output:
        p.error("nothing to do: pass --output-annotated and/or --output")
    if not args.pass_rates and not args.policy:
        p.error("--policy is required when reading pass rates from existing tags (no --pass-rates)")
    if not args.pass_rates and args.output_annotated:
        p.error("--output-annotated needs --pass-rates: there is no new measurement to merge")
    return args


def meta_path(pass_rates_path):
    return os.path.splitext(pass_rates_path)[0] + ".meta.json"


def load_meta(pass_rates_path):
    """Sampling parameters recorded beside the measurement, if present.

    Absent for a file produced before the sidecar existed; the annotation then
    carries only what can be derived from the records themselves.
    """
    path = meta_path(pass_rates_path)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_records(path):
    records = {}
    record_fields = {field.name for field in fields(PassRateRecord)}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            # Search-R1 measurements also retain agentic-cost fields (search
            # calls, observation length, statuses). Filtering needs only the
            # common pass-rate record, so preserve those in the measurement
            # artifact and ignore them here.
            r = PassRateRecord(**{key: value for key, value in payload.items() if key in record_fields})
            records[r.index] = r
    return records


def build_tag(record, meta):
    tag = {
        "pass_rate": record.pass_rate,
        "n_correct": record.n_correct,
        "n_samples": record.n_samples,
        "truncated_frac": record.truncated_frac,
    }
    # Only the parameters that change what a pass rate *means* are copied in;
    # the sidecar keeps the full record.
    for key in (
        "rm_type",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "max_context_length",
        "max_new_tokens_per_turn",
        "max_turns",
        "search_topk",
        "format_score",
        "zero_reward_on_truncated",
    ):
        if key in meta:
            tag[key] = meta[key]
    return tag


def iter_prompt_rows(path):
    suffix = Path(path).suffix
    if suffix == ".jsonl":
        with open(path) as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".parquet":
        if pq is None:
            raise SystemExit("pyarrow is required to read parquet prompt data")
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            yield from batch.to_pylist()
        return
    raise SystemExit(f"unsupported prompt data {path!r}; expected .jsonl or .parquet")


def main():
    args = parse_args()

    meta = load_meta(args.pass_rates) if args.pass_rates else {}
    policy = args.policy or meta.get("policy")
    if not policy:
        raise SystemExit(f"no policy name: pass --policy, or place a meta sidecar at {meta_path(args.pass_rates)}")

    records = load_records(args.pass_rates) if args.pass_rates else None
    print(f"policy       {policy}")
    print(f"source       {args.pass_rates or f'existing {DIFFICULTY_KEY}[{policy}] tags in --prompt-data'}")
    if records is not None:
        print(f"measurements {len(records)}")
    if meta:
        print(f"meta         {json.dumps({k: meta[k] for k in sorted(meta)})}")
    print(f"window       [{args.pass_rate_min}, {args.pass_rate_max}]")
    if records is not None:
        for k, v in summarize(list(records.values()), args.pass_rate_min, args.pass_rate_max).items():
            print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")

    for path in (args.output_annotated, args.output):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    annotated = open(args.output_annotated, "w") if args.output_annotated else None
    filtered = open(args.output, "w") if args.output else None
    n_annotated = n_kept = n_unmeasured = 0
    try:
        for i, row in enumerate(iter_prompt_rows(args.prompt_data)):
            if records is not None:
                record = records.get(i)
                if record is None:
                    n_unmeasured += 1
                    if not args.drop_unmeasured:
                        raise SystemExit(
                            f"prompt index {i} has no measurement; finish the sweep or pass --drop-unmeasured"
                        )
                    continue
                # Merge rather than assign: re-running for another policy must
                # not discard the tags already on the row.
                row.setdefault(DIFFICULTY_KEY, {})[policy] = build_tag(record, meta)
                pass_rate = record.pass_rate
            else:
                tag = row.get(DIFFICULTY_KEY, {}).get(policy)
                if tag is None:
                    n_unmeasured += 1
                    if not args.drop_unmeasured:
                        raise SystemExit(
                            f"prompt index {i} has no {DIFFICULTY_KEY}[{policy}] tag; "
                            f"pass --pass-rates to measure it in, or --drop-unmeasured"
                        )
                    continue
                pass_rate = tag["pass_rate"]

            if annotated:
                annotated.write(json.dumps(row) + "\n")
                n_annotated += 1
            if filtered and pass_rate_in_window(pass_rate, args.pass_rate_min, args.pass_rate_max):
                filtered.write(json.dumps(row) + "\n")
                n_kept += 1
    finally:
        for f in (annotated, filtered):
            if f:
                f.close()

    if annotated:
        print(f"\nannotated {n_annotated} prompts -> {args.output_annotated}")
    if filtered:
        print(f"kept      {n_kept} prompts -> {args.output}")
    if n_unmeasured:
        print(f"skipped   {n_unmeasured} unmeasured prompts")


if __name__ == "__main__":
    main()
