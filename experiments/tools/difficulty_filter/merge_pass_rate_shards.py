"""Validate and merge contiguous pass-rate shards into one measurement."""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-data", required=True)
    return parser.parse_args()


def meta_path(pass_rates_path):
    return os.path.splitext(pass_rates_path)[0] + ".meta.json"


def count_prompts(path):
    with open(path) as source:
        return sum(1 for line in source if line.strip())


def load_meta(path):
    with open(meta_path(path)) as source:
        return json.load(source)


def measurement_meta(meta):
    """Drop range-only fields before comparing sampling provenance."""
    return {key: value for key, value in meta.items() if key not in {"start_index", "end_index"}}


def merge(inputs, output, prompt_data):
    inputs = [os.fspath(path) for path in inputs]
    output = os.fspath(output)
    prompt_data = os.fspath(prompt_data)
    total = count_prompts(prompt_data)
    records = {}
    metas = []
    ranges = []

    for path in inputs:
        meta = load_meta(path)
        metas.append(meta)
        ranges.append((meta.get("start_index", 0), meta.get("end_index")))
        with open(path) as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                index = record["index"]
                if index in records:
                    raise SystemExit(f"duplicate prompt index {index} in {path}")
                records[index] = record

    reference = measurement_meta(metas[0])
    for path, meta in zip(inputs[1:], metas[1:], strict=True):
        if measurement_meta(meta) != reference:
            raise SystemExit(f"measurement metadata differs in {path}")

    sorted_ranges = sorted(ranges)
    expected_start = 0
    for start, end in sorted_ranges:
        if start != expected_start or end is None or end <= start:
            raise SystemExit(f"shard ranges do not form a contiguous partition: {sorted_ranges}")
        expected_start = end
    if expected_start != total:
        raise SystemExit(f"shard ranges end at {expected_start}, but prompt data has {total} rows")

    expected_indices = set(range(total))
    missing = expected_indices - records.keys()
    extra = records.keys() - expected_indices
    if missing or extra:
        raise SystemExit(
            f"incomplete shard merge: records={len(records)} expected={total} "
            f"missing={len(missing)} extra={len(extra)}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w") as destination:
        for index in range(total):
            destination.write(json.dumps(records[index]) + "\n")
    os.replace(temporary, output)

    merged_meta = dict(reference)
    merged_meta["start_index"] = 0
    merged_meta["end_index"] = total
    temporary_meta = meta_path(output) + ".tmp"
    with open(temporary_meta, "w") as destination:
        json.dump(merged_meta, destination, indent=2)
    os.replace(temporary_meta, meta_path(output))
    print(f"merged {len(inputs)} shards / {total} prompts -> {output}")


def main():
    args = parse_args()
    merge(args.inputs, args.output, args.prompt_data)


if __name__ == "__main__":
    main()
