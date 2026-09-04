from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.src.datasets.common.io import read_rows
from experiments.src.datasets.nemotron.adapters import adapt_conv_tooluse_pivot
from experiments.src.datasets.tool_call_pivot.prepare import prepare
from experiments.src.evaluators.tool_call_pivot import EvaluationResult, _calls_from_message, _metrics


def _row(source: str, index: int, *, kind: str = "function_call") -> dict:
    if kind == "function_call":
        expected = {
            "type": "function_call",
            "name": "lookup",
            "arguments": json.dumps({"index": index}),
        }
    else:
        expected = {"type": "message", "content": f"answer {index}"}
    return {
        "prompt": [{"role": "user", "content": f"question {index}"}],
        "label": kind,
        "metadata": {
            "source": source,
            "verifier": "expert_action",
            "trajectory_id": index,
            "expected_action": expected,
            "expected_kind": kind,
            "interaction_mode": "static_single_turn_pivot",
            "stateful_environment": False,
        },
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {"index": {"type": "integer"}}},
                },
            }
        ],
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_conversational_tool_use_pivot_has_distinct_source_identity() -> None:
    raw = {
        "trajectory_id": 7,
        "responses_create_params": {
            "input": [{"role": "user", "content": "Look up order 7"}],
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
        },
        "expected_action": {
            "type": "function_call",
            "name": "lookup",
            "arguments": '{"index":7}',
        },
    }

    converted = adapt_conv_tooluse_pivot(raw)

    assert converted is not None
    assert converted["metadata"]["source"] == "conv-tooluse-pivot"
    assert converted["metadata"]["expected_kind"] == "function_call"
    assert converted["metadata"]["interaction_mode"] == "static_single_turn_pivot"
    assert converted["metadata"]["stateful_environment"] is False
    assert converted["tools"][0]["function"]["name"] == "lookup"


def test_prepare_balances_sources_excludes_messages_and_has_no_overlap(tmp_path: Path):
    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    rows_a = [_row("a", index) for index in range(8)] + [_row("a", 99, kind="message")]
    rows_b = [_row("b", index) for index in range(8)] + [_row("b", 99, kind="message")]
    _write(source_a, rows_a)
    _write(source_b, rows_b)
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    summary = prepare(
        argparse.Namespace(
            source=[("a", source_a), ("b", source_b)],
            train_output=train,
            eval_output=evaluation,
            train_per_source=5,
            eval_per_source=2,
        )
    )

    train_rows = list(read_rows([train]))
    eval_rows = list(read_rows([evaluation]))
    train_ids = {row["metadata"]["split_fingerprint"] for row in train_rows}
    eval_ids = {row["metadata"]["split_fingerprint"] for row in eval_rows}
    assert summary["train_counts"] == {"a": 5, "b": 5}
    assert summary["eval_counts"] == {"a": 2, "b": 2}
    assert summary["interaction_mode"] == "static_single_turn_pivot"
    assert summary["stateful_environment"] is False
    assert summary["audit"]["a"]["message"] == 1
    assert train_ids.isdisjoint(eval_ids)
    assert all(row["metadata"]["expected_kind"] == "function_call" for row in train_rows)
    assert all(row["metadata"]["interaction_mode"] == "static_single_turn_pivot" for row in train_rows)
    assert all(row["metadata"]["stateful_environment"] is False for row in train_rows)
    assert all(row["metadata"]["eval_only"] for row in eval_rows)


def test_prepare_deduplicates_identical_rows(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [_row("only", index) for index in range(5)]
    _write(source, [rows[0], rows[0], *rows[1:]])
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    summary = prepare(
        argparse.Namespace(
            source=[("only", source)],
            train_output=train,
            eval_output=evaluation,
            train_per_source=3,
            eval_per_source=1,
        )
    )

    assert summary["audit"]["only"]["duplicate"] == 1
    assert len(list(read_rows([train]))) == 3
    assert len(list(read_rows([evaluation]))) == 1


def test_prepare_can_use_all_remaining_eligible_rows(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [_row("only", index) for index in range(8)]
    _write(source, rows)
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    summary = prepare(
        argparse.Namespace(
            source=[("only", source)],
            train_output=train,
            eval_output=evaluation,
            train_per_source=None,
            eval_per_source=2,
        )
    )

    assert summary["train_counts"] == {"only": 6}
    assert summary["eval_counts"] == {"only": 2}
    assert summary["train_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert summary["eval_sha256"] == hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert len(list(read_rows([train]))) == 6
    assert len(list(read_rows([evaluation]))) == 2


def test_evaluator_accepts_structured_and_tagged_tool_calls():
    structured = {
        "tool_calls": [
            {"function": {"name": "lookup", "arguments": '{"index":3}'}}
        ]
    }
    tagged = {
        "content": '<tool_call>{"name":"lookup","arguments":{"index":3}}</tool_call>'
    }
    assert _calls_from_message(structured) == [{"name": "lookup", "arguments": {"index": 3}}]
    assert _calls_from_message(tagged) == [{"name": "lookup", "arguments": {"index": 3}}]


def test_evaluator_reports_exact_accuracy_per_source():
    results = [
        EvaluationResult(0, "a", "x", True, True, True, 1, "lookup", ["lookup"], 10, None),
        EvaluationResult(1, "a", "y", False, True, False, 1, "lookup", ["lookup"], 11, None),
        EvaluationResult(2, "b", "z", False, False, False, 0, "lookup", [], 0, "error"),
    ]
    metrics = _metrics(results)
    assert metrics["interaction_mode"] == "static_single_turn_pivot"
    assert metrics["stateful_environment"] is False
    assert metrics["overall"]["exact_action_accuracy"] == 1 / 3
    assert metrics["by_source"]["a"]["tool_name_accuracy"] == 1.0
    assert metrics["by_source"]["b"]["error_rate"] == 1.0
