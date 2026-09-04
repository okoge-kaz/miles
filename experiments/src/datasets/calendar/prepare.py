"""Convert Nemotron Calendar rows to Miles deterministic-RL JSONL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import read_rows, write_jsonl
from experiments.src.environments.calendar.verifier import build_calendar_solution, score_calendar_response
from experiments.src.protocols.openai_responses import to_chat_messages


def adapt_calendar(row: dict[str, Any], *, eval_only: bool) -> dict[str, Any]:
    params = row.get("responses_create_params") or {}
    prompt = to_chat_messages(params.get("input") or [])
    expected_state = row.get("exp_cal_state")
    if not prompt or not isinstance(expected_state, dict) or not expected_state:
        raise ValueError("Calendar row is missing conversation or exp_cal_state")
    reference = build_calendar_solution(expected_state)
    if score_calendar_response(reference, expected_state) != 1.0:
        raise ValueError("Calendar reference solution did not pass the local verifier")
    return {
        "prompt": prompt,
        "label": json.dumps(expected_state, ensure_ascii=False, separators=(",", ":")),
        "metadata": {
            "source": "nemotron-calendar-v2",
            "verifier": "calendar_constraints",
            "expected_calendar_state": expected_state,
            "eval_only": eval_only,
            "runtime_dependency": "miles-local-calendar-verifier",
        },
    }


def _convert(path: Path, *, eval_only: bool) -> Iterator[dict[str, Any]]:
    for row in read_rows([path]):
        yield adapt_calendar(row, eval_only=eval_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    counts = {
        "train": write_jsonl(args.train_output, _convert(args.train_input, eval_only=False)),
        "validation": write_jsonl(
            args.validation_output,
            _convert(args.validation_input, eval_only=True),
        ),
    }
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
