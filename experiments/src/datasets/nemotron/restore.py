"""Restore Nano blend math placeholders from already-downloaded parquet files.

This is an offline equivalent of NVIDIA's ``create_nanov3_jsonl.py``. It reads
only the parquet rows referenced by placeholders, so it does not redownload the
source datasets or materialize millions of unused Python objects.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import expand_paths

DAPO_DATASET = "nano_v3_sft_profiled_dapo17k"
SKYWORK_DATASET = "nano_v3_sft_profiled_skywork_no_omni"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def _placeholder_indices(path: Path) -> dict[str, set[int]]:
    indices: dict[str, set[int]] = defaultdict(set)
    for row in _iter_jsonl(path):
        placeholder = row.get("_hf_placeholder")
        dataset = row.get("dataset")
        if placeholder and dataset in {DAPO_DATASET, SKYWORK_DATASET}:
            indices[str(dataset)].add(int(placeholder["row"]))
    return indices


def _read_selected_parquet(paths: list[Path], indices: set[int]) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Nano restoration requires pyarrow") from exc

    wanted = sorted(indices)
    selected: dict[int, dict[str, Any]] = {}
    wanted_offset = 0
    global_offset = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65_536):
            batch_end = global_offset + batch.num_rows
            local_indices = []
            global_indices = []
            while wanted_offset < len(wanted) and wanted[wanted_offset] < batch_end:
                index = wanted[wanted_offset]
                if index >= global_offset:
                    local_indices.append(index - global_offset)
                    global_indices.append(index)
                wanted_offset += 1
            if local_indices:
                rows = batch.take(pa.array(local_indices, type=pa.int64())).to_pylist()
                selected.update(zip(global_indices, rows, strict=True))
            global_offset = batch_end
    missing = indices - selected.keys()
    if missing:
        examples = ", ".join(str(index) for index in sorted(missing)[:5])
        raise IndexError(f"{len(missing)} placeholder indices exceed the parquet rows; first: {examples}")
    return selected


def _extract_path(value: Any, path: list[Any]) -> Any:
    for key in path:
        value = value[key] if isinstance(key, int) else value.get(key)
    return value


def _answer(raw: Any) -> Any:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            decoded = json.loads(text)
            return decoded[0] if decoded else ""
        return text
    if isinstance(raw, list):
        return raw[0] if raw else ""
    return raw


def _strip_dapo_prompt(text: str) -> str:
    prefix = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    )
    suffix = 'Remember to put your answer on its own line after "Answer:".'
    if prefix not in text or suffix not in text:
        raise ValueError("DAPO source prompt no longer matches NVIDIA's restoration template")
    return text[text.index(prefix) + len(prefix) : text.rfind(suffix)]


def _restore_question(row: dict[str, Any], source_row: dict[str, Any]) -> tuple[str, str]:
    dataset = row["dataset"]
    question = str(_extract_path(source_row, ["prompt", 0, "content"]))
    template = row["_hf_placeholder"]["question_template"]
    if dataset == DAPO_DATASET:
        stripped = _strip_dapo_prompt(question)
        if template.get("prefix"):
            full_question = f"{template['prefix']}{stripped}".removesuffix("\n\n")
        elif template.get("suffix"):
            full_question = f"{stripped}{template['suffix']}"
        else:
            raise ValueError(f"unknown DAPO question template: {template}")
        return full_question, full_question
    full_question = str(template["template"]).replace("{question}", question)
    return question, full_question


def _restore_row(row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    question, full_question = _restore_question(row, source_row)
    restored = dict(row)
    restored.pop("_hf_placeholder")
    restored["question"] = question
    restored["expected_answer"] = _answer(_extract_path(source_row, ["reward_model", "ground_truth"]))
    restored["responses_create_params"] = {"input": [{"role": "user", "content": full_question}]}
    return restored


def restore(args: argparse.Namespace) -> dict[str, Any]:
    indices = _placeholder_indices(args.input)
    sources = {
        DAPO_DATASET: _read_selected_parquet(expand_paths(args.dapo), indices[DAPO_DATASET]),
        SKYWORK_DATASET: _read_selected_parquet(expand_paths(args.skywork), indices[SKYWORK_DATASET]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    counts = defaultdict(int)
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for row in _iter_jsonl(args.input):
                placeholder = row.get("_hf_placeholder")
                dataset = row.get("dataset")
                if placeholder and dataset in sources:
                    source_row = sources[str(dataset)][int(placeholder["row"])]
                    row = _restore_row(row, source_row)
                    counts[str(dataset)] += 1
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["total"] += 1
        os.replace(partial, args.output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {"output": str(args.output), **dict(sorted(counts.items()))}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dapo", nargs="+", required=True, help="DAPO parquet paths or globs")
    parser.add_argument("--skywork", nargs="+", required=True, help="Skywork math parquet paths or globs")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(restore(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
