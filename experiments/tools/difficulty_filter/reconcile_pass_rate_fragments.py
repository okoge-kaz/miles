"""Merge completed range fragments into a resumable pass-rate output."""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--fragments", nargs="+", required=True)
    parser.add_argument(
        "--mark-complete",
        action="store_true",
        help="create .measured/.complete markers when the primary range is exact",
    )
    return parser.parse_args()


def meta_path(pass_rates_path):
    return os.path.splitext(pass_rates_path)[0] + ".meta.json"


def load_meta(path):
    with open(meta_path(path)) as source:
        return json.load(source)


def measurement_meta(meta):
    """Drop the source range before comparing sampling provenance."""
    return {key: value for key, value in meta.items() if key not in {"start_index", "end_index"}}


def load_records(path, start_index, end_index):
    records = {}
    malformed = 0
    duplicates = 0
    with open(path) as source:
        lines = [(line_number, line) for line_number, line in enumerate(source, start=1) if line.strip()]
        for position, (line_number, line) in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if position != len(lines) - 1:
                    raise SystemExit(f"malformed non-trailing record in {path}:{line_number}")
                malformed += 1
                continue

            index = record.get("index")
            if not isinstance(index, int):
                raise SystemExit(f"invalid index in {path}:{line_number}: {index!r}")
            if index < start_index or index >= end_index:
                raise SystemExit(
                    f"index {index} in {path}:{line_number} is outside "
                    f"declared range [{start_index}, {end_index})"
                )
            n_samples = record.get("n_samples")
            rewards = record.get("rewards")
            if not isinstance(n_samples, int) or not isinstance(rewards, list) or len(rewards) != n_samples:
                raise SystemExit(f"invalid reward shape in {path}:{line_number}")
            if index in records:
                duplicates += 1
            records[index] = record

    return records, malformed, duplicates


def reconcile(primary, fragments, mark_complete=False):
    primary = os.fspath(primary)
    fragments = [os.fspath(path) for path in fragments]
    primary_meta = load_meta(primary)
    start_index = primary_meta.get("start_index", 0)
    end_index = primary_meta.get("end_index")
    if end_index is None or end_index <= start_index:
        raise SystemExit(f"primary has an invalid range: [{start_index}, {end_index})")

    records, malformed, duplicates = load_records(primary, start_index, end_index)
    primary_count = len(records)
    reference = measurement_meta(primary_meta)

    for path in fragments:
        fragment_meta = load_meta(path)
        if measurement_meta(fragment_meta) != reference:
            raise SystemExit(f"measurement metadata differs in {path}")
        fragment_start = fragment_meta.get("start_index", 0)
        fragment_end = fragment_meta.get("end_index")
        if (
            fragment_end is None
            or fragment_start < start_index
            or fragment_end > end_index
            or fragment_end <= fragment_start
        ):
            raise SystemExit(
                f"fragment {path} range [{fragment_start}, {fragment_end}) is not inside "
                f"primary range [{start_index}, {end_index})"
            )
        fragment_records, fragment_malformed, fragment_duplicates = load_records(
            path, fragment_start, fragment_end
        )
        malformed += fragment_malformed
        duplicates += fragment_duplicates
        for index, record in fragment_records.items():
            records.setdefault(index, record)

    temporary = primary + ".reconcile.tmp"
    with open(temporary, "w") as destination:
        for index in sorted(records):
            destination.write(json.dumps(records[index]) + "\n")
    os.replace(temporary, primary)

    expected = end_index - start_index
    missing = expected - len(records)
    if missing < 0:
        raise SystemExit(f"reconciled output has {len(records)} records, expected at most {expected}")
    complete = missing == 0 and set(records) == set(range(start_index, end_index))
    if mark_complete and complete:
        for marker in (primary + ".measured", primary + ".complete"):
            with open(marker, "a"):
                pass

    print(
        f"reconciled primary={primary_count} total={len(records)} expected={expected} "
        f"missing={missing} malformed_dropped={malformed} duplicates={duplicates} "
        f"complete={complete} -> {primary}"
    )
    return complete


def main():
    args = parse_args()
    reconcile(args.primary, args.fragments, args.mark_complete)


if __name__ == "__main__":
    main()
