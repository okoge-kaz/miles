from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.src.datasets.common.audit import audit
from experiments.src.datasets.common.merge import (
    balanced_merge_training_files,
    merge_training_files,
)
from experiments.src.datasets.nemotron.adapters import (
    adapt_competitive_coding,
    adapt_conv_tooluse,
    adapt_dapo_math,
    adapt_gpqa,
    adapt_instruction_following,
    adapt_knowledge_mcqa,
    adapt_livecodebench,
    adapt_nano,
    adapt_reasoning_gym,
    adapt_skywork_or1_code,
    adapt_structured_outputs,
)
from experiments.src.datasets.nemotron.convert import convert_nano, convert_one_dataset
from experiments.src.datasets.nemotron.restore import _restore_row
from experiments.src.protocols.openai_responses import (
    expected_action_signature,
    to_chat_messages,
    to_chat_tools,
)


def _training_row(source: str, verifier: str, content: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": content}],
        "label": "A",
        "metadata": {"source": source, "verifier": verifier},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_responses_api_translation_keeps_reproducible_tool_fields():
    messages = to_chat_messages(
        [
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {"type": "reasoning", "content": "private"},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "weather",
                "arguments": '{"city":"LA"}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ]
    )
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "weather"
    assert messages[2]["tool_call_id"] == "c1"

    tools = to_chat_tools(
        [{"type": "function", "name": "weather", "parameters": {"type": "object"}}]
    )
    assert tools == [
        {
            "type": "function",
            "function": {"name": "weather", "parameters": {"type": "object"}},
        }
    ]
    assert expected_action_signature(
        {"name": "send_email", "arguments": {"to": "a@example.com"}}
    ) == {
        "kind": "function_call",
        "name": "send_email",
        "arguments": {"to": "a@example.com"},
    }


def test_static_adapters_preserve_verifier_inputs():
    mcqa = adapt_knowledge_mcqa(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "q"}]},
            "expected_answer": "C",
            "options": [{"A": "a", "B": "b", "C": "c", "D": "d"}],
            "template_metadata": {"output_regex": r"Selected Option -> ([A-D])"},
        }
    )
    assert mcqa is not None
    assert mcqa["metadata"]["verifier"] == "mcqa_regex"
    assert mcqa["metadata"]["valid_letters"] == ["A", "B", "C", "D"]

    competitive = adapt_competitive_coding(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "add"}]},
            "verifier_metadata": {"unit_tests": {"inputs": ["1 2\n"], "outputs": ["3\n"]}},
        }
    )
    assert competitive is not None
    assert competitive["metadata"]["verifier"] == "python_code"
    assert competitive["metadata"]["unit_tests"]["inputs"] == ["1 2\n"]

    harness = {
        "entry_point": "Solution().add",
        "test_code": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    }
    skywork = adapt_skywork_or1_code(
        {
            "prompt": [{"role": "user", "content": "add"}],
            "reward_model": {"ground_truth": json.dumps(harness)},
        }
    )
    assert skywork is not None
    assert skywork["metadata"]["unit_tests"] == harness

    instruction = adapt_instruction_following(
        {
            "prompt": "Mention cat.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["cat"]}],
        }
    )
    assert instruction is not None
    assert instruction["metadata"]["verifier"] == "ifeval_g"

    reasoning = adapt_reasoning_gym(
        {
            "question": "Who?",
            "answer": "Alice",
            "metadata": {"source_dataset": "logic"},
        }
    )
    assert reasoning is not None
    assert reasoning["metadata"]["source_dataset"] == "logic"

    schema = json.dumps({"type": "object"})
    structured = adapt_structured_outputs(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "json"}]},
            "schema_str": schema,
        }
    )
    assert structured is not None
    assert structured["metadata"]["verifier"] == "json_schema"

    math = adapt_dapo_math(
        {
            "prompt": [{"role": "user", "content": "1+1?"}],
            "reward_model": {"ground_truth": "2"},
        }
    )
    assert math is not None
    assert math["metadata"]["verifier"] == "math"
    assert math["prompt"][0]["content"].endswith(r"Answer: \boxed{...}`.")


def test_eval_adapters_are_explicitly_held_out():
    gpqa = adapt_gpqa(
        {
            "problem": "Which option is correct?",
            "A": "wrong one",
            "B": "truth",
            "C": "wrong two",
            "D": "wrong three",
            "expected_answer": "B",
        }
    )
    assert gpqa is not None
    assert gpqa["label"] == "B"
    assert gpqa["metadata"]["eval_only"] is True

    livecodebench = adapt_livecodebench(
        {
            "question_content": "Add two integers.",
            "question_id": "q1",
            "public_test_cases": "[]",
            "private_test_cases": "[]",
        }
    )
    assert livecodebench is not None
    assert livecodebench["metadata"]["eval_only"] is True
    assert livecodebench["metadata"]["verifier"] == "livecodebench"


def test_expert_action_conversion_preserves_exact_action_and_tools():
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "search"}],
            "tools": [{"name": "search", "parameters": {"type": "object"}}],
        },
        "expected_action": {
            "type": "function_call",
            "name": "search",
            "arguments": '{"q":"miles"}',
        },
    }
    converted = adapt_conv_tooluse(row)
    assert converted is not None
    assert converted["metadata"]["verifier"] == "expert_action"
    assert converted["metadata"]["expected_action"] == row["expected_action"]
    assert converted["tools"][0]["function"]["name"] == "search"


def test_nano_routes_environment_and_unverifiable_rows_out_of_static_training():
    workbench = {
        "dataset": "nano_v3_sft_profiled_workbench",
        "responses_create_params": {
            "input": [{"role": "user", "content": "send mail"}],
            "tools": [{"name": "send_email", "parameters": {"type": "object"}}],
        },
        "ground_truth": [{"name": "send_email", "arguments": {"to": "x"}}],
    }
    converted, route = adapt_nano(workbench)
    assert route == "environment"
    assert converted is not None
    assert converted["metadata"]["support_status"] == "requires_environment"

    missing_answer = {
        "dataset": "nano_v3_sft_profiled_skywork_no_omni",
        "responses_create_params": {"input": [{"role": "user", "content": "q"}]},
        "expected_answer": "",
    }
    converted, route = adapt_nano(missing_answer)
    assert route == "unverifiable"
    assert converted is not None
    assert converted["metadata"]["verifier"] == "missing_ground_truth"


def test_converter_and_audit_write_the_miles_contract(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "converted.jsonl"
    _write_jsonl(
        source,
        [
            {
                "prompt": [{"role": "user", "content": "1+1?"}],
                "reward_model": {"ground_truth": "2"},
            }
        ],
    )
    summary = convert_one_dataset(
        argparse.Namespace(dataset="dapo-math", input=[str(source)], output=output, limit=None)
    )
    assert summary["kept"] == 1
    assert summary["verifiers"] == {"math": 1}
    report = audit(
        argparse.Namespace(
            input=output,
            expected_rows=1,
            require_verifiers=["math"],
            require_eval_only=False,
            sample_output=None,
            samples_per_verifier=1,
        )
    )
    assert report["rows"] == 1
    assert report["verifiers"] == {"math": 1}


def test_nano_converter_splits_all_routes_atomically(tmp_path: Path):
    source = tmp_path / "nano.jsonl"
    ready = tmp_path / "ready.jsonl"
    environment = tmp_path / "environment.jsonl"
    unverifiable = tmp_path / "unverifiable.jsonl"
    _write_jsonl(
        source,
        [
            {
                "dataset": "nano_v3_sft_profiled_dapo17k",
                "responses_create_params": {"input": [{"role": "user", "content": "1+1?"}]},
                "expected_answer": "2",
            },
            {
                "dataset": "nano_v3_sft_profiled_workbench",
                "responses_create_params": {"input": [{"role": "user", "content": "mail"}]},
                "ground_truth": [],
            },
            {
                "dataset": "nano_v3_sft_profiled_skywork_no_omni",
                "responses_create_params": {"input": [{"role": "user", "content": "unknown"}]},
                "expected_answer": "",
            },
        ],
    )
    summary = convert_nano(
        argparse.Namespace(
            input=[str(source)],
            output=ready,
            environment_output=environment,
            unverifiable_output=unverifiable,
            limit=None,
        )
    )
    assert (summary["ready"], summary["requires_environment"], summary["unverifiable"]) == (1, 1, 1)
    assert all(path.read_text(encoding="utf-8").count("\n") == 1 for path in (ready, environment, unverifiable))
    assert not list(tmp_path.glob("*.partial"))


def test_training_merges_are_balanced_deterministic_and_reject_eval_rows(tmp_path: Path):
    inputs = []
    for source in ("math", "code", "stem"):
        path = tmp_path / f"{source}.jsonl"
        _write_jsonl(path, [_training_row(source, source, f"{source}-{index}") for index in range(5)])
        inputs.append(str(path))

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    summary = balanced_merge_training_files(inputs, first, rows_per_input=2, seed=7)
    balanced_merge_training_files(inputs, second, rows_per_input=2, seed=7)
    assert summary["rows"] == 6
    assert set(summary["rows_by_input"].values()) == {2}
    assert first.read_bytes() == second.read_bytes()

    benchmark = tmp_path / "benchmark.jsonl"
    eval_row = _training_row("gpqa", "gpqa", "held out")
    eval_row["metadata"]["eval_only"] = True
    _write_jsonl(benchmark, [eval_row])
    with pytest.raises(ValueError, match="eval_only"):
        merge_training_files([inputs[0], str(benchmark)], first)
    assert first.read_bytes() == second.read_bytes()
    assert not first.with_name(first.name + ".partial").exists()


def test_restore_row_reconstructs_published_nano_math_prompt():
    row = {
        "dataset": "nano_v3_sft_profiled_skywork_no_omni",
        "_hf_placeholder": {"question_template": {"template": "Solve: {question}"}},
    }
    source = {
        "prompt": [{"role": "user", "content": "1+1?"}],
        "reward_model": {"ground_truth": "2"},
    }
    restored = _restore_row(row, source)
    assert "_hf_placeholder" not in restored
    assert restored["question"] == "1+1?"
    assert restored["expected_answer"] == "2"
    assert restored["responses_create_params"]["input"][0]["content"] == "Solve: 1+1?"
