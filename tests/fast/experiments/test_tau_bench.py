from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.evaluators.tau_bench import _tau_action
from experiments.src.environments.tau_bench.generator import (
    PolicyAction,
    PolicyActionParseError,
    PolicyTurnResponse,
    TauOverhead,
    _add_arguments,
    _finish_policy_turn,
    _generate_tau,
    _generate_turn,
    _log_overhead,
    _parse_policy_action,
    append_user_observation,
)
from experiments.src.environments.tau_bench.runtime import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MODEL,
    TauContinuationError,
    TauEnvironmentError,
    TauSession,
    TauUserConfig,
    _external_agent_observations,
    verify_tau_runtime,
)
from experiments.src.environments.tau_bench.task_identity import TAU_VERIFIER
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser
from miles.utils.types import Sample

REPO_ROOT = Path(__file__).resolve().parents[3]


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert kwargs.get("return_dict") is False
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode()) if tokenize else text

    def decode(self, token_ids):
        return bytes(token_ids).decode()

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


class ThinkingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert kwargs.get("return_dict") is False
        parts = []
        for message in messages:
            if message["role"] == "assistant":
                reasoning = message.get("reasoning_content", "")
                parts.append(f"<assistant><think>{reasoning}</think>{message['content']}</assistant>")
            else:
                parts.append(f"<{message['role']}>{message['content']}</{message['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        text = "".join(parts)
        return list(text.encode()) if tokenize else text


def test_user_observation_is_loss_masked_and_budgeted() -> None:
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


def test_user_observation_supports_thinking_chat_template() -> None:
    sample = Sample(tokens=[1, 2], response="", response_length=0, loss_mask=[], rollout_log_probs=[])

    assert append_user_observation(sample, ThinkingTokenizer(), "hello", max_response_len=100)

    assert bytes(sample.tokens[2:]).decode() == "<user>hello</user><assistant>"
    assert set(sample.loss_mask) == {0}


def test_tau_user_config_reads_only_process_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("NVIDIA_INFERENCE_API_KEY=dotenv-secret\n")
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="process environment"):
        TauUserConfig(provider="nvidia", model=DEFAULT_NVIDIA_MODEL).litellm_config()


def test_tau_nvidia_user_config_maps_to_litellm_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "test-secret")
    monkeypatch.delenv("NVIDIA_INFERENCE_BASE_URL", raising=False)
    model, config = TauUserConfig(provider="nvidia", model=DEFAULT_NVIDIA_MODEL, seed=7).litellm_config()
    assert model == f"openai/{DEFAULT_NVIDIA_MODEL}"
    assert config["api_base"] == DEFAULT_NVIDIA_BASE_URL
    assert config["api_key"] == "test-secret"
    assert config["seed"] == 7

    monkeypatch.setenv("NVIDIA_INFERENCE_BASE_URL", "")
    _, empty_base_config = TauUserConfig(
        provider="nvidia",
        model=DEFAULT_NVIDIA_MODEL,
    ).litellm_config()
    assert empty_base_config["api_base"] == DEFAULT_NVIDIA_BASE_URL


def test_tau_gemini_user_config_uses_official_simulator_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    model, config = TauUserConfig(provider="gemini", model=DEFAULT_GEMINI_MODEL).litellm_config()
    assert model == f"gemini/{DEFAULT_GEMINI_MODEL}"
    assert config["api_key"] == "test-secret"


def test_tau_runtime_version_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _package: "0.9.9")
    with pytest.raises(RuntimeError, match="version mismatch"):
        verify_tau_runtime()


def test_tau_session_sanitizes_environment_reset_failure() -> None:
    session = object.__new__(TauSession)
    session._environment = SimpleNamespace(reset=lambda: (_ for _ in ()).throw(TimeoutError("secret")))

    with pytest.raises(TauEnvironmentError) as exception:
        session.reset()

    assert exception.value.operation == "reset"
    assert exception.value.reason == "TimeoutError"
    assert "secret" not in str(exception.value)


def test_tau_continuation_restore_failure_is_non_retryable_and_sanitized() -> None:
    session = object.__new__(TauSession)
    session._continuation = {"saved": True}
    session._environment = SimpleNamespace(
        reset=lambda: (_ for _ in ()).throw(TimeoutError("secret")),
    )

    with pytest.raises(TauContinuationError) as exception:
        session.reset()

    assert exception.value.operation == "restore"
    assert exception.value.reason == "TimeoutError"
    assert "secret" not in str(exception.value)


def test_tau_session_detects_official_early_termination() -> None:
    session = object.__new__(TauSession)
    session._environment = SimpleNamespace(
        reset=lambda: (None, {}),
        _simulation_done=SimpleNamespace(is_set=lambda: True),
    )

    with pytest.raises(TauEnvironmentError, match="ended before the first policy turn"):
        session.reset()


def test_tau_session_does_not_retry_local_integration_failure() -> None:
    session = object.__new__(TauSession)
    session._environment = SimpleNamespace(
        reset=lambda: (None, {}),
        _simulation_done=SimpleNamespace(is_set=lambda: False),
    )
    session._history = lambda: (_ for _ in ()).throw(RuntimeError("local integration bug"))

    with pytest.raises(RuntimeError, match="local integration bug") as exception:
        session.reset()

    assert not isinstance(exception.value, TauEnvironmentError)


def test_tau_session_rejects_terminal_step_without_verifier_artifacts() -> None:
    session = object.__new__(TauSession)
    session._environment = SimpleNamespace(step=lambda _action: (None, 0.0, True, False, {}))

    with pytest.raises(TauEnvironmentError, match="without reward or trajectory data"):
        session.step("hello")


def test_tau_session_filters_policy_echoes_regardless_of_delta_position() -> None:
    messages = [
        SimpleNamespace(role="assistant", content="first policy echo"),
        SimpleNamespace(role="user", content="new user message"),
        SimpleNamespace(role="assistant", content="late policy echo"),
        SimpleNamespace(role="tool", content="new tool result"),
    ]

    observations = _external_agent_observations(messages)

    assert [(message.role, message.content) for message in observations] == [
        ("user", "new user message"),
        ("tool", "new tool result"),
    ]


def test_tau_session_close_skips_finished_official_simulation() -> None:
    session = object.__new__(TauSession)
    session._terminated = False
    session._environment = SimpleNamespace(
        _simulation_done=SimpleNamespace(is_set=lambda: True),
        step=lambda _action: (_ for _ in ()).throw(AssertionError("step must not be called")),
    )

    session.close()

    assert session._terminated is True


def test_tau_action_parser_emits_tau_three_tool_call_json() -> None:
    call = SimpleNamespace(id="call-1", name="lookup", parameters='{"query":"item"}')
    parser = SimpleNamespace(parse_non_stream=lambda _response: ("", [call]))
    action = _parse_policy_action(parser, "ignored", turn=2)
    assert json.loads(action.tau_action) == {
        "id": "call-1",
        "name": "lookup",
        "arguments": {"query": "item"},
        "requestor": "assistant",
    }
    assert action.message["tool_calls"][0]["function"]["name"] == "lookup"


def test_tau_action_parser_normalizes_tool_parser_type_error() -> None:
    class InvalidNameParser:
        def parse_non_stream(self, _response):
            raise TypeError("unhashable type: 'dict'")

    with pytest.raises(
        PolicyActionParseError,
        match="tool-call parser raised TypeError: unhashable type: 'dict'",
    ):
        _parse_policy_action(InvalidNameParser(), "malformed tool call", turn=2)


def test_tau_action_parser_contains_sglang_qwen_name_object_failure() -> None:
    parser = create_tool_call_parser(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up an item",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "qwen25",
    )
    response = '<tool_call>\n{"name":{"lookup":{}},"arguments":{}}\n</tool_call>'

    with pytest.raises(PolicyActionParseError):
        _parse_policy_action(parser, response, turn=2)


@pytest.mark.parametrize("name", ({"lookup": {}}, ["lookup"], None, ""))
def test_tau_action_parser_rejects_non_string_tool_name(name) -> None:
    call = SimpleNamespace(id="call-1", name=name, parameters="{}")
    parser = SimpleNamespace(parse_non_stream=lambda _response: ("", [call]))

    with pytest.raises(PolicyActionParseError, match="tool name must be a non-empty string"):
        _parse_policy_action(parser, "ignored", turn=2)


def test_tau_policy_parser_failure_is_confined_to_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidNameParser:
        def parse_non_stream(self, _response):
            raise TypeError("unhashable type: 'dict'")

    async def fake_update(_args, _sample, _payload, _output, *, update_loss_mask):
        assert update_loss_mask is True

    monkeypatch.setattr(
        "experiments.src.environments.tau_bench.generator.update_sample_from_response",
        fake_update,
    )
    malformed_response = "x" * 1100 + '<tool_call>\n{"name":{"lookup":{}}}\n</tool_call>'
    response = PolicyTurnResponse(
        payload={},
        output={
            "text": malformed_response,
            "meta_info": {"finish_reason": {"type": "stop"}},
        },
        halt_reason=None,
        halt_status=None,
        started_at=1.0,
        finished_at=2.0,
    )
    sample = Sample(index=17)
    input = GenerateFnInput(
        state=SimpleNamespace(args=SimpleNamespace()),
        sample=sample,
        sampling_params={},
        evaluation=False,
    )

    action, halt_reason, response_text = asyncio.run(
        _finish_policy_turn(
            input,
            sample,
            InvalidNameParser(),
            turn=3,
            response=response,
            response_prefix="",
        )
    )

    assert action is None
    assert halt_reason == "action"
    assert response_text == malformed_response
    assert sample.status == Sample.Status.FAILED
    assert sample.reward == 0.0
    assert sample.metadata["tau_policy_action_error"] == {
        "turn": 4,
        "error_type": "TypeError",
        "reason": "tool-call parser raised TypeError: unhashable type: 'dict'",
        "response_tail": malformed_response[-1024:],
    }


def test_tau_overhead_is_opt_in_and_classifies_environment_waits() -> None:
    disabled = TauOverhead(enabled=False)
    sample = Sample(index=1)
    _log_overhead(sample, disabled)
    assert "tau_overhead" not in sample.metadata

    overhead = TauOverhead(enabled=True)
    overhead.record_reset(1.0)
    overhead.record_step(PolicyAction("hello", {"role": "assistant", "content": "hello"}), 2.0)
    overhead.record_step(
        PolicyAction(
            "tool",
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "lookup"}}],
            },
        ),
        3.0,
    )
    overhead.record_step(
        PolicyAction(
            "done",
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "done"}}],
            },
        ),
        4.0,
    )
    overhead.record_close(5.0)
    _log_overhead(sample, overhead)

    assert sample.metadata["tau_overhead"] == {
        "reset_seconds": 1.0,
        "tool_wait_seconds": 3.0,
        "user_simulator_wait_seconds": 2.0,
        "terminal_wait_seconds": 4.0,
        "close_seconds": 5.0,
        "environment_total_seconds": 15.0,
        "tool_steps": 1,
        "user_simulator_steps": 1,
        "terminal_steps": 1,
    }


def test_tau_overhead_cli_defaults_off() -> None:
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    assert parser.parse_args([]).tau_log_overhead is False
    assert parser.parse_args([]).tau_overlap_db_restore_with_prefill is False
    assert parser.parse_args(["--tau-log-overhead"]).tau_log_overhead is True
    assert (
        parser.parse_args(
            ["--tau-overlap-db-restore-with-prefill"]
        ).tau_overlap_db_restore_with_prefill
        is True
    )


def test_tau_overhead_reports_hidden_db_restore_time() -> None:
    overhead = TauOverhead(enabled=True)
    overhead.record_reset(
        5.0,
        policy_request_seconds=4.0,
        overlap_seconds=3.0,
    )

    summary = overhead.summary()

    assert summary["resume_overlap_attempts"] == 1
    assert summary["resume_policy_request_seconds"] == 4.0
    assert summary["resume_db_prefill_overlap_seconds"] == 3.0
    assert summary["resume_db_restore_unhidden_seconds"] == 2.0
    assert summary["environment_unhidden_seconds"] == 2.0


def test_tau_environment_failure_is_aborted_without_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSession:
        def __init__(self, _metadata, _user, *, max_steps, continuation):
            assert max_steps == 200
            assert continuation is None

        def reset(self):
            raise TauEnvironmentError("reset", "ProviderTimeout")

        def close(self):
            return None

    monkeypatch.setattr("experiments.src.environments.tau_bench.generator.TauSession", FailingSession)
    args = SimpleNamespace(
        partial_rollout=False,
        rollout_seed=1,
        tau_log_overhead=False,
        tau_max_steps=200,
        tau_user_max_retries=4,
        tau_user_max_tokens=512,
        tau_user_model=DEFAULT_NVIDIA_MODEL,
        tau_user_provider="nvidia",
        tau_user_request_timeout=120.0,
        tau_user_temperature=0.7,
        tau_user_top_p=0.95,
    )
    state = SimpleNamespace(args=args)
    sample = Sample(
        index=3,
        metadata={"verifier": TAU_VERIFIER},
        tokens=[1, 2, 3],
        response="old",
        response_length=3,
        loss_mask=[1, 1, 1],
        rollout_log_probs=[-0.1, -0.2, -0.3],
        weight_versions=[5],
        first_prefill_weight_versions=[5],
        min_forward_weight_versions=[5],
        max_forward_weight_versions=[5],
        last_forward_weight_versions=[5],
        response_weight_versions=["5"],
        response_weight_version_segments=[[[0, 3, 5]]],
        non_generation_time=7.0,
    )
    input = GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False)

    result = asyncio.run(_generate_tau(input)).samples

    assert isinstance(result, Sample)
    assert result.status == Sample.Status.ABORTED
    assert result.reward is None
    assert result.metadata["tau_infrastructure_error"] == {
        "operation": "reset",
        "reason": "ProviderTimeout",
    }
    assert result.tokens == []
    assert result.response_length == 0
    assert result.loss_mask is None
    assert result.rollout_log_probs is None
    assert result.weight_versions == []
    assert result.first_prefill_weight_versions == []
    assert result.response_weight_versions == []
    assert result.response_weight_version_segments == []
    assert "tau_overhead" not in result.metadata
    assert result.non_generation_time == 0.0


def test_tau_policy_engine_abort_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(*_args, **_kwargs):
        return {"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}

    async def fake_update(_args, sample, _payload, _output, *, update_loss_mask):
        assert update_loss_mask is True
        sample.status = Sample.Status.ABORTED

    monkeypatch.setattr(
        "experiments.src.environments.tau_bench.generator.compute_request_payload",
        lambda *_args, **_kwargs: ({}, None),
    )
    monkeypatch.setattr("experiments.src.environments.tau_bench.generator.post", fake_post)
    monkeypatch.setattr(
        "experiments.src.environments.tau_bench.generator.update_sample_from_response",
        fake_update,
    )
    args = SimpleNamespace(rollout_max_response_len=64, sglang_router_policy="random")
    input = GenerateFnInput(
        state=SimpleNamespace(args=args),
        sample=Sample(),
        sampling_params={},
        evaluation=False,
    )
    sample = Sample(tokens=[1])

    action, halt_reason, response_text = asyncio.run(
        _generate_turn(input, sample, SimpleNamespace(), "url", 0)
    )

    assert action is None
    assert halt_reason == "abort"
    assert response_text == ""
    assert sample.status == Sample.Status.ABORTED
    assert sample.reward is None


def test_evaluator_converts_openai_tool_call_to_tau_three_action() -> None:
    message = {
        "content": None,
        "tool_calls": [
            {
                "id": "call-2",
                "function": {"name": "lookup", "arguments": '{"query":"item"}'},
            }
        ],
        "_miles_finish_reason": "tool_calls",
    }
    assert json.loads(_tau_action(message, turn=0)) == {
        "id": "call-2",
        "name": "lookup",
        "arguments": {"query": "item"},
        "requestor": "assistant",
    }


def test_tau2_training_and_tau3_evaluation_are_explicitly_separated() -> None:
    tau_async = REPO_ROOT / "experiments/scripts/tau_bench/async"
    recipe = tau_async / "areal-tau2/qwen3-4b-agentic-sft-953"
    run_script = (recipe / "run.sbatch").read_text(encoding="utf-8")
    train_script = (recipe / "train.sh").read_text(encoding="utf-8")
    eval_script = (REPO_ROOT / "experiments/scripts/tau_bench/evaluate.sbatch").read_text(
        encoding="utf-8"
    )

    assert "/data/areal-tau2-data/miles-tau2-rl-train.jsonl" in run_script
    assert "stateful_multi_turn_user_simulator_environment" in run_script
    assert "experiments.src.environments.areal_tau2.generator.generate" in run_script
    assert ': "${USE_REPLAY_BUFFER:=1}"' in run_script
    assert ': "${REPLAY_BUFFER_TYPE:=inflight}"' in run_script
    assert '--use-replay-buffer --replay-buffer-type "${REPLAY_BUFFER_TYPE}"' in train_script
    assert '--custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"' in train_script
    assert "/data/tau-bench/tau3-test-miles.jsonl" in eval_script
    assert "/data/tau-bench/tau3-train-miles.jsonl" not in eval_script
    assert "/data/tau-bench/tau3-base-miles.jsonl" not in eval_script
    assert "tau1-" not in eval_script
    assert "local-policy" not in eval_script
