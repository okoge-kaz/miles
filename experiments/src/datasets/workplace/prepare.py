"""Convert Nemotron Workplace Assistant rows to standalone Miles env JSONL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import read_rows, write_jsonl
from experiments.src.environments.workplace.runtime import WORKPLACE_RESOURCE_COMMIT
from experiments.src.protocols.openai_responses import to_chat_messages, to_chat_tools
from experiments.src.protocols.workplace import WORKPLACE_INTERACTION_MODE


def adapt_workplace(row: dict[str, Any], *, eval_only: bool) -> dict[str, Any]:
    params = row.get("responses_create_params") or {}
    prompt = to_chat_messages(params.get("input") or [])
    tools = to_chat_tools(params.get("tools") or [])
    expected_actions = row.get("ground_truth")
    if not prompt or not tools or not isinstance(expected_actions, list):
        raise ValueError("Workplace row is missing prompt, tools, or ground truth")
    return {
        "prompt": prompt,
        "label": json.dumps(expected_actions, ensure_ascii=False, separators=(",", ":")),
        "tools": tools,
        "metadata": {
            "source": "nemotron-workplace-assistant",
            "verifier": "workplace_environment",
            "interaction_mode": WORKPLACE_INTERACTION_MODE,
            "conversation_turns": 1,
            "stateful_environment": True,
            "expected_actions": expected_actions,
            "task_id": row.get("id"),
            "category": row.get("category"),
            "eval_only": eval_only,
            "workplace_resource_commit": WORKPLACE_RESOURCE_COMMIT,
            "runtime_dependency": "pinned-standalone-resource-modules-no-nemo-gym-server",
        },
    }


def _convert(path: Path, *, eval_only: bool) -> Iterator[dict[str, Any]]:
    for row in read_rows([path]):
        yield adapt_workplace(row, eval_only=eval_only)


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
