#!/usr/bin/env python3
"""Convert the official OlympiadBench text-only English math subset to miles JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

EXPECTED_ROWS = 674
EXPECTED_SOURCE_SHA256 = "7c17ca2662b3ea77cddcb66a64b04ad9b53d3d32fa372d444a4884739eb1638d"
PROMPT_TEMPLATE = """Solve the following olympiad math problem step by step. The last line of your response must be of the form Answer: \\boxed{{$Answer}}.

If the problem has multiple answers, put all required answers inside the same box, separated by commas. Include no units inside the box.

{question}

Remember to put your complete answer on its own line after \"Answer:\"."""


def convert(source: Path, output: Path) -> None:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == EXPECTED_SOURCE_SHA256, (
        f"unexpected source SHA256: expected {EXPECTED_SOURCE_SHA256}, found {source_hash}"
    )
    rows = pq.read_table(source).to_pylist()
    assert len(rows) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, found {len(rows)}"

    converted = []
    for row in rows:
        assert row["modality"] == "Text-only"
        assert row["difficulty"] == "Competition"
        assert row["question_type"] == "Open-ended"
        assert row["subject"] == "Math"
        assert row["language"] == "English"
        assert all(row[f"image_{index}"] is None for index in range(1, 10))
        assert len(row["final_answer"]) == 1

        question = row["question"]
        if row["context"]:
            question = f"{row['context']}\n\n{question}"
        converted.append(
            {
                "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
                "label": row["final_answer"][0],
                "metadata": {
                    "source": "OlympiadBench/OE_TO_maths_en_COMP",
                    "olympiadbench_id": row["id"],
                    "subfield": row["subfield"],
                    "answer_type": row["answer_type"],
                    "is_multiple_answer": row["is_multiple_answer"],
                    "unit": row["unit"],
                    "precision": float(row["error"]) if row["error"] else 1e-8,
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(f"wrote {len(converted)} rows to {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
