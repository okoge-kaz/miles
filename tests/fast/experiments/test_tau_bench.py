import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.src.datasets.tau_bench import prepare
from experiments.src.datasets.tau_bench.prepare import partition_reward_verified_rows, reservoir_sample
from experiments.src.evaluators import tau_bench as tau_evaluator
from experiments.src.environments.tau_bench import generator as tau_generator
from experiments.src.environments.tau_bench.generator import (
    STOP_MARKER,
    _extract_local_user_response,
    append_tool_observation,
    append_user_observation,
    build_user_system_prompt,
)
from experiments.src.environments.tau_bench.user_simulator import (
    DEFAULT_GEMINI_MODEL,
    GeminiRequestError,
    UserGeneration,
    build_gemini_payload,
    generate_gemini_user,
    parse_gemini_response,
    require_gemini_api_key,
)
from miles.utils.types import Sample


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert kwargs.get("return_dict") is False
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode()) if tokenize else text

    def decode(self, token_ids):
        return bytes(token_ids).decode()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_user_observation_is_loss_masked_and_budgeted():
    sample = Sample(tokens=[1, 2], response="", response_length=0, loss_mask=[], rollout_log_probs=[])
    tokenizer = FakeTokenizer()

    assert append_user_observation(sample, tokenizer, "hello", max_response_len=100)
    assert sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)
    assert set(sample.loss_mask) == {0}
    assert set(sample.rollout_log_probs) == {0.0}

    previous_length = sample.response_length
    assert not append_user_observation(sample, tokenizer, "too long", max_response_len=previous_length)
    assert sample.response_length == previous_length
    assert sample.status == Sample.Status.TRUNCATED


def test_tool_observation_is_loss_masked_and_budgeted():
    sample = Sample(tokens=[1, 2], response="", response_length=0, loss_mask=[], rollout_log_probs=[])
    tokenizer = FakeTokenizer()
    message = {
        "role": "tool",
        "name": "lookup",
        "content": "result",
        "tool_call_id": "tau_test",
    }

    assert append_tool_observation(sample, tokenizer, message, max_response_len=100)
    assert sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)
    assert set(sample.loss_mask) == {0}

    previous_length = sample.response_length
    assert not append_tool_observation(sample, tokenizer, message, max_response_len=previous_length)
    assert sample.response_length == previous_length
    assert sample.status == Sample.Status.TRUNCATED


def test_local_user_prompt_keeps_stop_marker_private():
    prompt = build_user_system_prompt("Return an item after confirmation.")
    assert STOP_MARKER in prompt
    assert "Return an item after confirmation." in prompt
    assert "do not call tools" in prompt


def test_local_user_length_finish_keeps_partial_message_and_records_truncation():
    sample = Sample(metadata={})
    output = {
        "text": "I need help with my order",
        "meta_info": {"finish_reason": {"type": "length"}},
    }

    assert _extract_local_user_response(output, sample) == "I need help with my order"
    assert sample.metadata["tau_user_length_truncations"] == 1


def test_local_user_abort_is_rejected():
    sample = Sample(metadata={})
    output = {
        "text": "partial",
        "meta_info": {"finish_reason": {"type": "abort"}},
    }

    with pytest.raises(RuntimeError, match="aborted"):
        _extract_local_user_response(output, sample)


def test_gemini_payload_maps_chat_roles_and_disables_thinking():
    payload = build_gemini_payload(
        [
            {"role": "system", "content": "private task"},
            {"role": "user", "content": "agent greeting"},
            {"role": "assistant", "content": "customer request"},
            {"role": "user", "content": "agent response"},
        ],
        max_output_tokens=512,
        temperature=0.7,
        top_p=0.95,
        seed=42,
    )

    assert payload["systemInstruction"] == {"parts": [{"text": "private task"}]}
    assert [content["role"] for content in payload["contents"]] == ["user", "model", "user"]
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert payload["generationConfig"]["seed"] == 42


def test_gemini_response_parses_text_usage_and_stop_marker():
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "hidden reasoning"},
                        {"text": f"Done. {STOP_MARKER}"},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"candidatesTokenCount": 7},
    }

    result = parse_gemini_response(response)

    assert result.text == STOP_MARKER
    assert result.output_tokens == 7
    assert result.finish_reason == "STOP"


def test_gemini_key_is_process_only_and_does_not_read_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=dotenv-secret\n")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiRequestError, match="process/job environment"):
        require_gemini_api_key()


def test_tau_ray_submission_does_not_serialize_gemini_key():
    repository = Path(__file__).parents[3]
    train_script = repository / (
        "experiments/scripts/tau_bench/async/nemotron3-agentic-retail/"
        "qwen3-4b-instruct-2507/train.sh"
    )
    script = train_script.read_text(encoding="utf-8")

    assert "GEMINI_API_KEY" not in script
    assert '--runtime-env-json="${RUNTIME_ENV_JSON}"' in script
    assert "env_vars[" not in script


async def test_gemini_request_keeps_key_out_of_url_and_payload(monkeypatch):
    secret = "test-gemini-secret"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    async def post_json(url, payload, headers):
        calls.append((url, payload, headers))
        return {
            "candidates": [{"content": {"parts": [{"text": "I need help."}]}, "finishReason": "STOP"}],
            "usageMetadata": {"candidatesTokenCount": 4},
        }

    result = await generate_gemini_user(
        [
            {"role": "system", "content": "private task"},
            {"role": "user", "content": "How can I help?"},
        ],
        post_json=post_json,
        model=DEFAULT_GEMINI_MODEL,
        max_retries=0,
    )

    assert result.text == "I need help."
    assert len(calls) == 1
    url, payload, headers = calls[0]
    assert secret not in url
    assert secret not in json.dumps(payload)
    assert headers == {"Content-Type": "application/json", "x-goog-api-key": secret}


async def test_gemini_request_retries_rate_limit_without_exposing_key(monkeypatch):
    secret = "test-gemini-secret"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    attempts = 0

    async def post_json(url, payload, headers):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise GeminiRequestError("rate limited", status_code=429)
        return {"candidates": [{"content": {"parts": [{"text": "Retry worked."}]}}]}

    result = await generate_gemini_user(
        [
            {"role": "system", "content": "private task"},
            {"role": "user", "content": "How can I help?"},
        ],
        post_json=post_json,
        max_retries=1,
        retry_backoff=0,
    )

    assert result.text == "Retry worked."
    assert attempts == 2


async def test_rl_gemini_backend_dispatches_to_external_user(monkeypatch):
    calls = []

    async def fake_generate(input, messages, sample, *, user_turn):
        calls.append((input, messages, sample, user_turn))
        return "customer reply"

    monkeypatch.setattr(tau_generator, "_generate_gemini_user", fake_generate)
    rollout_input = SimpleNamespace(args=SimpleNamespace(tau_user_backend="gemini"))
    messages = [{"role": "system", "content": "private"}, {"role": "user", "content": "hello"}]
    sample = Sample()

    result = await tau_generator._generate_user(
        rollout_input,
        "http://local-policy/generate",
        messages,
        sample,
        user_turn=3,
    )

    assert result == "customer reply"
    assert calls == [(rollout_input, messages, sample, 3)]


async def test_evaluator_gemini_backend_dispatches_to_external_user(monkeypatch):
    calls = []

    async def fake_generate(messages, **kwargs):
        calls.append((messages, kwargs))
        return UserGeneration(text="customer reply", output_tokens=6, finish_reason="STOP")

    monkeypatch.setattr(tau_evaluator, "generate_gemini_user", fake_generate)
    args = SimpleNamespace(
        user_backend="gemini",
        user_model=DEFAULT_GEMINI_MODEL,
        user_max_tokens=512,
        user_temperature=0.7,
        user_top_p=0.95,
        user_request_timeout=120.0,
        user_max_retries=4,
        user_retry_backoff=1.0,
    )
    messages = [{"role": "system", "content": "private"}, {"role": "user", "content": "hello"}]

    text, tokens = await tau_evaluator._user_completion(
        args,
        object(),
        asyncio.Semaphore(1),
        messages,
        seed=42,
    )

    assert (text, tokens) == ("customer reply", 6)
    assert calls[0][0] == messages
    assert calls[0][1]["model"] == DEFAULT_GEMINI_MODEL
    assert calls[0][1]["seed"] == 42


async def test_evaluator_local_user_disables_thinking(monkeypatch):
    calls = []

    async def fake_completion(*args, **kwargs):
        calls.append((args, kwargs))
        return {"content": "customer reply"}, 4

    monkeypatch.setattr(tau_evaluator, "_completion", fake_completion)
    args = SimpleNamespace(
        user_backend="local-policy",
        endpoint="http://local-policy/v1/chat/completions",
        model="local-policy",
        user_max_tokens=512,
        user_temperature=0.7,
        user_top_p=0.95,
    )
    messages = [{"role": "system", "content": "private"}, {"role": "user", "content": "hello"}]

    text, tokens = await tau_evaluator._user_completion(
        args,
        object(),
        asyncio.Semaphore(1),
        messages,
        seed=42,
    )

    assert (text, tokens) == ("customer reply", 4)
    assert calls[0][1]["enable_thinking"] is False


def test_reservoir_sample_is_deterministic_and_rejects_eval_rows(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_rows(source, [{"prompt": str(index), "metadata": {"verifier": "expert_action"}} for index in range(20)])
    first = reservoir_sample(source, count=5, seed=42)
    second = reservoir_sample(source, count=5, seed=42)
    assert first == second
    assert len({row["prompt"] for row in first}) == 5

    contaminated = tmp_path / "contaminated.jsonl"
    _write_rows(contaminated, [{"prompt": "held out", "metadata": {"eval_only": True}}])
    with pytest.raises(ValueError, match="eval_only"):
        reservoir_sample(contaminated, count=1, seed=42)


def test_reward_verification_rejects_no_op_positive_tasks(monkeypatch):
    class FakeTask:
        def __init__(self, index):
            self.index = index
            self.actions = [f"action-{index}"]

        def model_dump(self):
            return {"index": self.index, "actions": self.actions}

    tasks = [FakeTask(0), FakeTask(1)]
    env = type("FakeEnv", (), {"tasks": tasks})()
    rows = [
        {
            "prompt": str(index),
            "metadata": {
                "tau_commit": prepare.TAU_COMMIT,
                "tau_task_index": index,
                "tau_task_sha256": prepare._task_digest(prepare._task_dict(task)),
            },
        }
        for index, task in enumerate(tasks)
    ]

    def fake_reward(unused_env, task_index, actions):
        if actions:
            return 1.0
        return float(task_index == 1)

    monkeypatch.setattr(prepare, "_official_reward", fake_reward)
    verified, rejected = partition_reward_verified_rows(rows, env)

    assert [row["metadata"]["tau_task_index"] for row in verified] == [0]
    assert verified[0]["metadata"]["tau_reward_verified"] is True
    assert verified[0]["metadata"]["tau_reward_audit"] == {
        "no_op_reward": 0.0,
        "ground_truth_reward": 1.0,
    }
    assert rejected == [{"task_index": 1, "no_op_reward": 1.0, "ground_truth_reward": 1.0}]
