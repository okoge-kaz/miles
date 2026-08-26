from __future__ import annotations

import asyncio
import base64
import json
import pickle
import threading
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

from experiments.src.environments.competitive_programming import verifier as code_exec
from experiments.src.environments.instruction_following import verifier as ifeval_g
from experiments.src.datasets.common.merge import balanced_merge_training_files, merge_training_files
from experiments.src.datasets.nemotron.adapters import (
    adapt_competitive_coding,
    adapt_dapo_math,
    adapt_gpqa,
    adapt_ifbench,
    adapt_knowledge_mcqa,
    adapt_nano,
    adapt_skywork_or1_code,
)
from experiments.src.evaluators.livecodebench import decode_private_tests
from experiments.src.protocols.openai_responses import expected_action_signature, to_chat_messages, to_chat_tools
from experiments.src.reward_sets import all_domains as rewards
from experiments.src.reward_sets.all_domains import blend_reward
from experiments.tools.training_analysis.summarize_log import summarize
from experiments.tools.training_analysis import summarize_dump
from miles.utils.types import Sample


def _reward(sample: Sample) -> float:
    return asyncio.run(blend_reward(SimpleNamespace(), sample))


def test_responses_api_translation_preserves_reproducible_tool_fields():
    messages = to_chat_messages(
        [
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {"type": "reasoning", "content": "private"},
            {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": '{"city":"LA"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        ]
    )
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "weather"
    assert messages[2]["tool_call_id"] == "c1"
    tools = to_chat_tools([{"type": "function", "name": "weather", "parameters": {"type": "object"}}])
    assert tools == [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]


def test_expected_action_accepts_nemo_ground_truth_shape_without_type():
    signature = expected_action_signature({"name": "send_email", "arguments": {"to": "a@example.com"}})
    assert signature == {
        "kind": "function_call",
        "name": "send_email",
        "arguments": {"to": "a@example.com"},
    }


def test_mcqa_adapter_uses_per_row_output_regex():
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "q"}]},
        "expected_answer": "C",
        "options": [{"A": "a", "B": "b", "C": "c", "D": "d"}],
        "template_metadata": {
            "template_id": "generated",
            "output_regex": r"Selected Option\s*->\s*([A-Za-z0-9])",
        },
    }
    converted = adapt_knowledge_mcqa(row)
    assert converted is not None
    assert converted["metadata"]["verifier"] == "mcqa_regex"
    correct = Sample(response="reasoning\nSelected Option -> C", label="C", metadata=converted["metadata"])
    wrong = Sample(response="Selected Option -> B", label="C", metadata=converted["metadata"])
    assert _reward(correct) == 1.0
    assert _reward(wrong) == 0.0


def test_structured_output_reward_validates_schema():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1}},
            "required": ["count"],
            "additionalProperties": False,
        }
    )
    metadata = {"verifier": "json_schema", "schema_str": schema}
    assert _reward(Sample(response='```json\n{"count":2}\n```', label=schema, metadata=metadata)) == 1.0
    assert _reward(Sample(response='{"count":0}', label=schema, metadata=metadata)) == 0.0


def test_ifbench_adapter_is_eval_only_and_preserves_verifier_inputs():
    row = {
        "key": "17",
        "prompt": "Use the word miles exactly once.",
        "instruction_id_list": ["count:keywords_multiple"],
        "kwargs": [{"keyword1": "miles"}],
    }
    converted = adapt_ifbench(row)
    assert converted is not None
    assert converted["prompt"] == [{"role": "user", "content": row["prompt"]}]
    assert converted["metadata"] == {
        "source": "ifbench",
        "verifier": "ifbench",
        "instruction_id_list": row["instruction_id_list"],
        "prompt_text": row["prompt"],
        "kwargs": row["kwargs"],
        "record_id": "17",
        "eval_only": True,
    }


def test_ifeval_normalizes_only_count_increment_singleton_keywords(monkeypatch):
    class CountIncrementChecker:
        def __init__(self, instruction_id):
            self.id = instruction_id

        def get_instruction_args_keys(self):
            return ["keyword1", "keyword2"]

        def build_description(self, keyword1, keyword2):
            assert isinstance(keyword1, str)
            assert isinstance(keyword2, str)
            self.keyword1 = keyword1
            self.keyword2 = keyword2

        def check_following(self, response):
            return response.count(self.keyword1) == 1 and response.count(self.keyword2) == 2

    registry = SimpleNamespace(
        INSTRUCTION_DICT={"count:count_increment_word": CountIncrementChecker}
    )
    monkeypatch.setattr(ifeval_g, "_load_registry", lambda: registry)
    metadata = {
        "instruction_id_list": ["count:count_increment_word"],
        "kwargs": [{"keyword1": ["help"], "keyword2": ["dump"]}],
    }
    sample = Sample(response="help dump dump", metadata=metadata)
    assert ifeval_g.validate_ifeval_metadata(metadata) == 1
    assert ifeval_g.score_ifeval_sample(sample) == 1.0

    malformed = dict(metadata, kwargs=[{"keyword1": ["help", "assist"], "keyword2": ["dump"]}])
    assert ifeval_g.score_ifeval_sample(Sample(response=sample.response, metadata=malformed)) == 0.0


def test_ifeval_preserves_genuine_list_arguments(monkeypatch):
    class KeywordChecker:
        def __init__(self, instruction_id):
            self.id = instruction_id

        def get_instruction_args_keys(self):
            return ["keywords"]

        def build_description(self, keywords):
            assert keywords == ["cat", "dog"]
            self.keywords = keywords

        def check_following(self, response):
            return all(keyword in response for keyword in self.keywords)

    registry = SimpleNamespace(INSTRUCTION_DICT={"keywords:existence": KeywordChecker})
    monkeypatch.setattr(ifeval_g, "_load_registry", lambda: registry)
    metadata = {
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["cat", "dog"]}],
    }
    assert ifeval_g.score_ifeval_sample(Sample(response="cat and dog", metadata=metadata)) == 1.0


def test_ifeval_matches_official_fractional_reward_and_removes_thinking(monkeypatch):
    class ContainsKeyword:
        def __init__(self, instruction_id):
            self.id = instruction_id

        def get_instruction_args_keys(self):
            return ["keyword"]

        def build_description(self, keyword):
            self.keyword = keyword

        def check_following(self, response):
            return self.keyword in response

    class NoComma:
        def __init__(self, instruction_id):
            self.id = instruction_id

        def get_instruction_args_keys(self):
            return []

        def build_description(self):
            pass

        def check_following(self, response):
            return "," not in response

    registry = SimpleNamespace(
        INSTRUCTION_DICT={
            "keywords:frequency": ContainsKeyword,
            "punctuation:no_comma": NoComma,
        }
    )
    monkeypatch.setattr(ifeval_g, "_load_registry", lambda: registry)
    metadata = {
        "instruction_id_list": ["keywords:frequency", "punctuation:no_comma"],
        "kwargs": [{"keyword": "cat"}, None],
    }
    response = "<|assistant|><think>reasoning, has commas</think><answer>dog</answer>"
    assert ifeval_g.score_ifeval_sample(Sample(response=response, metadata=metadata)) == 0.5


def test_reasoning_gym_does_not_credit_answer_mentioned_only_in_reasoning(monkeypatch):
    monkeypatch.setenv("REASONING_GYM_ALLOW_EXACT_FALLBACK", "1")
    metadata = {"verifier": "reasoning_gym"}
    assert _reward(Sample(response="work\nAnswer: Richard", label="Richard", metadata=metadata)) == 1.0
    wrong = Sample(response="Richard may be relevant.\nAnswer: Alice", label="Richard", metadata=metadata)
    assert _reward(wrong) == 0.0


def test_tool_reward_requires_exact_single_call_and_arguments():
    metadata = {
        "verifier": "expert_action",
        "expected_action": {"type": "function_call", "name": "search", "arguments": '{"q":"miles"}'},
    }
    correct = '<tool_call>{"name":"search","arguments":{"q":"miles"}}</tool_call>'
    extra = '<tool_call>{"name":"search","arguments":{"q":"miles","page":2}}</tool_call>'
    assert _reward(Sample(response=correct, metadata=metadata)) == 1.0
    assert _reward(Sample(response=extra, metadata=metadata)) == 0.0


def test_competitive_coding_adapter_keeps_tests_and_prompt():
    converted = adapt_competitive_coding(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "add"}]},
            "verifier_metadata": {"unit_tests": {"inputs": ["1 2\n"], "outputs": ["3\n"]}},
        }
    )
    assert converted is not None
    assert converted["metadata"]["unit_tests"]["inputs"] == ["1 2\n"]
    assert converted["metadata"]["verifier"] == "python_code"


def test_skywork_code_adapter_parses_published_ground_truth():
    tests = {"inputs": ["1 2\n"], "outputs": ["3\n"], "fn_name": None}
    converted = adapt_skywork_or1_code(
        {
            "prompt": [{"role": "user", "content": "add"}],
            "reward_model": {"ground_truth": json.dumps(tests)},
        }
    )
    assert converted is not None
    assert converted["metadata"]["unit_tests"] == tests
    assert converted["metadata"]["verifier"] == "python_code"


def test_skywork_code_adapter_and_verifier_support_published_harness(monkeypatch):
    tests = {
        "entry_point": "Solution().add",
        "import_prefix": "from typing import *\n",
        "test_code": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    }
    converted = adapt_skywork_or1_code(
        {
            "prompt": [{"role": "user", "content": "add"}],
            "reward_model": {"ground_truth": json.dumps(tests)},
        }
    )
    assert converted is not None
    assert converted["metadata"]["unit_tests"] == tests
    monkeypatch.setattr(code_exec, "SANDBOX_BACKEND", "process")
    correct = "class Solution:\n    def add(self, left, right):\n        return left + right"
    wrong = "class Solution:\n    def add(self, left, right):\n        return 0"
    assert code_exec.run_tests(correct, tests, timeout=2) == 1.0
    assert code_exec.run_tests(wrong, tests, timeout=2) == 0.0


def test_dapo_math_adapter_requests_boxed_final_answer():
    converted = adapt_dapo_math(
        {
            "prompt": [{"role": "user", "content": "What is 1 + 1?"}],
            "reward_model": {"ground_truth": "2"},
        }
    )
    assert converted is not None
    assert converted["prompt"][0]["content"].endswith(r"Answer: \boxed{...}`.")
    assert converted["metadata"]["verifier"] == "math"


def test_gpqa_adapter_is_eval_only_and_reward_discriminates():
    row = {
        "Question": "Which option is correct?",
        "Correct Answer": "truth",
        "Incorrect Answer 1": "wrong one",
        "Incorrect Answer 2": "wrong two",
        "Incorrect Answer 3": "wrong three",
    }
    first = adapt_gpqa(row)
    second = adapt_gpqa(row)
    assert first == second
    assert first is not None
    assert first["metadata"]["eval_only"] is True
    label = first["label"]
    wrong = next(letter for letter in "ABCD" if letter != label)
    assert _reward(Sample(response=f"Answer: {label}", label=label, metadata=first["metadata"])) == 1.0
    assert _reward(Sample(response=f"Answer: {wrong}", label=label, metadata=first["metadata"])) == 0.0


def test_gpqa_adapter_accepts_nemo_skills_preprocessed_rows():
    row = {
        "problem": "Which option is correct?",
        "A": "wrong one",
        "B": "truth",
        "C": "wrong two",
        "D": "wrong three",
        "expected_answer": "B",
        "explanation": "Because B is true.",
        "subset_for_metrics": "Physics",
        "difficulty": "Hard graduate level",
    }
    converted = adapt_gpqa(row)
    assert converted is not None
    assert converted["label"] == "B"
    assert converted["metadata"]["choices"] == ["wrong one", "truth", "wrong two", "wrong three"]
    assert converted["metadata"]["source_format"] == "nemo-skills-preprocessed"
    assert converted["metadata"]["eval_only"] is True
    assert row["explanation"] not in converted["prompt"][0]["content"]
    assert _reward(Sample(response="Answer: B", label="B", metadata=converted["metadata"])) == 1.0
    assert _reward(Sample(response="Answer: A", label="B", metadata=converted["metadata"])) == 0.0


def test_code_exec_known_program_discriminates_without_weakening_default(monkeypatch):
    monkeypatch.setattr(code_exec, "SANDBOX_BACKEND", "process")
    tests = {"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]}
    correct = "a, b = map(int, input().split())\nprint(a + b)"
    wrong = "print(0)"
    assert code_exec.run_tests(correct, tests, timeout=2) == 1.0
    assert code_exec.run_tests(wrong, tests, timeout=2) == 0.0


def test_blend_reward_batches_code_execution_under_one_shared_limit(monkeypatch):
    calls = []

    async def fake_code_reward(args, samples, **kwargs):
        calls.append(samples)
        return [1.0] * len(samples)

    monkeypatch.setitem(rewards._HANDLERS, "python_code", fake_code_reward)
    samples = [
        Sample(response="pass", label="tests", metadata={"verifier": "python_code"}),
        Sample(response="pass", label="tests", metadata={"verifier": "python_code"}),
        Sample(
            response="Answer: A",
            label="A",
            metadata={"verifier": "mcqa_regex", "valid_letters": ["A", "B"]},
        ),
    ]
    assert asyncio.run(blend_reward(SimpleNamespace(), samples)) == [1.0, 1.0, 1.0]
    assert calls == [samples[:2]]


def test_scalar_code_rewards_share_one_execution_limit(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_score(sample):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return 1.0

    monkeypatch.setattr(code_exec, "CONCURRENCY", 2)
    monkeypatch.setattr(code_exec, "_score_one", fake_score)
    code_exec._LOOP_SEMAPHORES.clear()
    sample = Sample(response="pass", metadata={"verifier": "python_code"})

    async def run_scalar_contract():
        return await asyncio.gather(*(code_exec.code_exec_reward(None, sample) for _ in range(8)))

    assert asyncio.run(run_scalar_contract()) == [1.0] * 8
    assert max_active == 2


def test_nano_routes_workbench_away_from_static_training():
    workbench = {
        "dataset": "nano_v3_sft_profiled_workbench",
        "responses_create_params": {
            "input": [{"role": "user", "content": "send mail"}],
            "tools": [{"name": "send_email", "parameters": {"type": "object"}}],
        },
        "ground_truth": [{"name": "send_email", "arguments": {"to": "x"}}],
        "environment_name": "workbench",
    }
    converted, route = adapt_nano(workbench)
    assert route == "environment"
    assert converted is not None
    assert converted["metadata"]["verifier"] == "nemo_gym_environment"
    assert converted["tools"][0]["function"]["name"] == "send_email"


def test_nano_rejects_unrestored_math_placeholder():
    row = {
        "dataset": "nano_v3_sft_profiled_dapo17k",
        "_hf_placeholder": {"row": 1},
    }
    converted, route = adapt_nano(row)
    assert converted is None
    assert route == "ready"


def test_livecodebench_private_test_decoder_handles_published_encoding():
    tests = [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
    payload = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert decode_private_tests(payload) == tests
    assert decode_private_tests(json.dumps(tests)) == tests


def test_training_merge_rejects_eval_only_and_is_atomic(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "benchmark.jsonl"
    output = tmp_path / "merged.jsonl"
    train.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "train"}],
                "label": "A",
                "metadata": {"source": "train", "verifier": "gpqa"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "benchmark"}],
                "label": "B",
                "metadata": {"source": "benchmark", "verifier": "gpqa", "eval_only": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = merge_training_files([str(train)], output, expected_rows=1, require_verifiers=["gpqa"])
    assert summary["rows"] == 1
    assert output.is_file()

    try:
        merge_training_files([str(train), str(benchmark)], output, expected_rows=2)
    except ValueError as exc:
        assert "eval_only" in str(exc)
    else:
        raise AssertionError("eval_only benchmark row was accepted into a training blend")
    assert json.loads(output.read_text(encoding="utf-8"))["prompt"][0]["content"] == "train"
    assert not output.with_name(output.name + ".partial").exists()


def test_balanced_training_merge_is_equal_deterministic_and_atomic(tmp_path: Path):
    inputs = []
    for source in ("math", "code", "stem"):
        path = tmp_path / f"{source}.jsonl"
        rows = [
            {
                "prompt": [{"role": "user", "content": f"{source}-{index}"}],
                "label": str(index),
                "metadata": {"source": source, "verifier": source},
            }
            for index in range(7)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        inputs.append(str(path))

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    summary = balanced_merge_training_files(inputs, first, rows_per_input=3, seed=7)
    balanced_merge_training_files(inputs, second, rows_per_input=3, seed=7)
    assert summary["rows"] == 9
    assert set(summary["rows_by_input"].values()) == {3}
    assert first.read_bytes() == second.read_bytes()
    assert not first.with_name(first.name + ".partial").exists()


def test_training_summary_measures_reward_change_and_staleness(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "sglang_enable_response_weight_version_segments .. True\n"
        "rollout 0: {'rollout/rewards': -0.1, 'rollout/raw_reward': 0.25, "
        "'rollout/truncated': 0.1, 'rollout/sample_staleness': 0.0}\n"
        "step 0: {'train/optimizer_step_applied': 1.0}\n"
        "ft cls=actor fn=update_weights phase=end ok=true elapsed_s=1.0\n"
        "rollout 1: {'rollout/rewards': 0.1, 'rollout/raw_reward': 0.75, "
        "'rollout/truncated': 0.0, 'rollout/sample_staleness': 2.0}\n",
        encoding="utf-8",
    )
    summary = summarize(log)
    assert summary["reward"]["last_minus_first"] == 0.5
    assert summary["reward"]["mean"] == 0.5
    assert summary["reward"]["normalized_mean"] == 0.0
    assert summary["reward"]["source"] == "raw_reward"
    assert summary["optimizer_steps_applied"] == 1
    assert summary["sample_staleness"]["within_requested_bound_4"] is True
    assert summary["response_weight_version_segments"] is True


def test_dump_summary_groups_domains_and_checks_exact_segments(monkeypatch, tmp_path: Path):
    code_sample = Sample(
        index=1,
        tokens=[1, 2, 3],
        response_length=3,
        reward=1.0,
        status=Sample.Status.COMPLETED,
        metadata={
            "verifier": "python_code",
            "sample_staleness_reference_weight_version": 2,
            "train_weight_version": 4,
        },
        response_weight_version_segments=[[[0, 1, 2], [1, 3, 3]]],
    )
    math_sample = Sample(
        index=2,
        tokens=[1, 2],
        response_length=2,
        reward=0.0,
        status=Sample.Status.TRUNCATED,
        metadata={
            "verifier": "math",
            "sample_staleness_reference_weight_version": 3,
            "train_weight_version": 4,
        },
    )
    train_rows = {
        1: SimpleNamespace(raw_reward=1.0),
        2: SimpleNamespace(raw_reward=0.0),
    }

    class FakeReader:
        def __init__(self, *args, **kwargs):
            pass

        def rollout_ids(self):
            return SimpleNamespace(train=[0])

        def load_joined(self, rollout_id):
            assert rollout_id == 0
            return SimpleNamespace(samples=[code_sample, math_sample], train_rows=train_rows)

    monkeypatch.setattr(summarize_dump, "DumpReader", FakeReader)
    summary = summarize_dump.summarize(tmp_path)
    assert summary["by_domain"]["code"]["reward_mean"] == 1.0
    assert summary["by_domain"]["math"]["truncated_fraction"] == 1.0
    assert summary["sample_staleness"]["max"] == 2.0
    assert summary["sample_staleness"]["within_requested_bound"] is True
    assert summary["response_weight_version_segments"]["exact_coverage_fraction"] == 1.0
    assert summary["response_weight_version_segments"]["exact_policy_token_coverage_fraction"] == 1.0
    assert summary["response_weight_version_segments"]["mixed_version_samples"] == 1


def test_dump_summary_separates_masked_observations_from_policy_tokens(monkeypatch, tmp_path: Path):
    sample = Sample(
        index=1,
        tokens=[1, 2, 3],
        response_length=3,
        reward=1.0,
        loss_mask=[1, 0, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "verifier": "tau_bench_environment",
            "tau_done": True,
            "tau_turns": 2,
            "tau_user_length_truncations": 1,
        },
        response_weight_version_segments=[[[0, 2, 1]]],
    )

    class FakeReader:
        def __init__(self, *args, **kwargs):
            pass

        def rollout_ids(self):
            return SimpleNamespace(train=[0])

        def load_joined(self, rollout_id):
            assert rollout_id == 0
            return SimpleNamespace(samples=[sample], train_rows={})

    monkeypatch.setattr(summarize_dump, "DumpReader", FakeReader)
    summary = summarize_dump.summarize(tmp_path)
    segments = summary["response_weight_version_segments"]
    assert segments["exact_coverage_fraction"] == 0.0
    assert segments["exact_policy_token_coverage_fraction"] == 1.0
    assert segments["covered_response_token_fraction"] == 2 / 3
    assert segments["covered_policy_token_fraction"] == 1.0
    tau = summary["tau_bench_environment"]
    assert tau["done_fraction"] == 1.0
    assert tau["turns_mean"] == 2.0
    assert tau["user_length_truncation_samples"] == 1
    assert tau["user_length_truncation_events"] == 1.0
