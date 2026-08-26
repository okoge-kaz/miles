"""Fail-loud structural audit for converted Miles JSONL datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _audit_row(row: dict[str, Any], line_number: int, require_eval_only: bool) -> tuple[str, str]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"line {line_number}: prompt must be a non-empty message list")
    for message in prompt:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"line {line_number}: invalid chat message")
    if row.get("label") is None:
        raise ValueError(f"line {line_number}: missing label")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"line {line_number}: metadata must be an object")
    verifier = metadata.get("verifier")
    source = metadata.get("source")
    if not verifier or not source:
        raise ValueError(f"line {line_number}: metadata.source and metadata.verifier are required")
    if require_eval_only and metadata.get("eval_only") is not True:
        raise ValueError(f"line {line_number}: benchmark row is not marked eval_only")
    tools = row.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"line {line_number}: tools must be a list")
    if "_hf_placeholder" in row:
        raise ValueError(f"line {line_number}: unresolved placeholder")
    return str(source), str(verifier)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    verifiers: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {}
    rows = 0
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            source, verifier = _audit_row(row, line_number, args.require_eval_only)
            sources[source] += 1
            verifiers[verifier] += 1
            verifier_samples = samples.setdefault(verifier, [])
            if len(verifier_samples) < args.samples_per_verifier:
                verifier_samples.append(row)
            rows += 1
    if args.expected_rows is not None and rows != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {rows}")
    missing = sorted(set(args.require_verifiers or []) - verifiers.keys())
    if missing:
        raise ValueError(f"required verifiers absent from data: {', '.join(missing)}")
    if args.sample_output:
        args.sample_output.parent.mkdir(parents=True, exist_ok=True)
        with args.sample_output.open("w", encoding="utf-8") as handle:
            for verifier in sorted(samples):
                for row in samples[verifier]:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "input": str(args.input),
        "rows": rows,
        "sources": dict(sorted(sources.items())),
        "verifiers": dict(sorted(verifiers.items())),
        "eval_only_required": args.require_eval_only,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--require-verifiers", nargs="*", default=[])
    parser.add_argument("--require-eval-only", action="store_true")
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--samples-per-verifier", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = audit(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
