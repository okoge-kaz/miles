import json
from collections import Counter
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web

from experiments.src.datasets.tau2.prepare import adapt_tau2
from experiments.src.nemo_gym.bridge import client
from experiments.src.nemo_gym.bridge.payloads import verify_payload
from experiments.src.nemo_gym.bridge.reward import reward


@asynccontextmanager
async def verifier_server(replies):
    calls = []

    async def verify(request):
        calls.append(await request.json())
        status, data = replies[min(len(calls) - 1, len(replies) - 1)]
        return web.json_response(data, status=status)

    app = web.Application()
    app.router.add_post("/verify", verify)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}", calls
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "bad_reply",
    [
        (503, {}),
        (302, {"reward": 0}),
        (200, {"reward": 0, "mask_sample": True}),
        (200, {"reward": 0, "_ng_failure_class": "tau2_agent_error"}),
        (
            200,
            {
                "reward": 0,
                "mask_sample": True,
                "failure_kind": "judge_failed",
                "failure_reason": "judge_invalid_response",
            },
        ),
    ],
)
async def test_remote_failures_retry_then_remain_unscored(bad_reply, monkeypatch):
    async with verifier_server([bad_reply]) as (url, calls):
        monkeypatch.setattr(client, "endpoints", lambda: {"services": {"test": url}})
        diagnostics = {}
        assert await client.request("test", "/verify", {"task": "x"}, attempts=2, diagnostics=diagnostics) is None
        if bad_reply[1].get("failure_reason") == "judge_invalid_response":
            assert diagnostics["failure_kind"] == "judge_invalid_response"
        assert len(calls) == 2


async def test_valid_zero_after_http_retry_is_a_scored_answer(monkeypatch):
    async with verifier_server([(502, {}), (200, {"reward": 0})]) as (url, calls):
        monkeypatch.setattr(client, "endpoints", lambda: {"services": {"test": url}})
        assert await client.request("test", "/verify", {}, attempts=2) == {"reward": 0}
        assert len(calls) == 2


@pytest.mark.parametrize(
    ("response", "answer"),
    [
        ("<think>Answer: A</think>Answer: B", "Answer: B"),
        ("Answer: A</think>Answer: B", "Answer: B"),
        ("<think>Answer: A</think>I cannot decide.", "I cannot decide."),
        ("<think>Answer: A", ""),
        ("<think>Answer: A</think>", ""),
        ("<think>A</think>draft<think>B</think>Answer: C", "Answer: C"),
        ("<think>A</think><think>Answer: B", ""),
        ("Answer: B", "Answer: B"),
    ],
)
def test_mcqa_grades_final_answer_without_mutating_training_response(response, answer):
    sample = SimpleNamespace(
        prompt="Choose one.",
        response=response,
        label="A",
        metadata={"verifier": "mcqa_regex", "valid_letters": ["A", "B", "C"]},
    )
    _, body = verify_payload(sample)
    assert body["response"]["output"][0]["content"][0]["text"] == answer
    assert sample.response == response


@pytest.mark.parametrize(
    "metadata,label,answer",
    [
        (
            {"verifier": "python_code", "unit_tests": {"inputs": [""], "outputs": ["1"]}},
            "1 test",
            "```python\nprint(1)\n```",
        ),
        ({"verifier": "json_schema"}, '{"type":"object"}', '{"answer":1}'),
        ({"verifier": "nemo_gym_equivalence_judge"}, "1", "1"),
        (
            {
                "verifier": "ifeval_g",
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{}],
                "prompt_text": "Reply.",
            },
            "",
            "final answer",
        ),
    ],
)
def test_other_text_verifiers_receive_only_final_answer(metadata, label, answer):
    sample = SimpleNamespace(
        prompt="Reply.", response=f"<think>private draft</think>{answer}", label=label, metadata=metadata
    )
    _, body = verify_payload(sample)
    assert body["response"]["output"][0]["content"][0]["text"] == answer


@pytest.mark.parametrize(
    ("response", "answer"),
    [
        ("<think>Paris</think><answer>London</answer><|im_end|>", "London"),
        ("<answer>Paris</answer><answer>Paris and London</answer>", "Paris and London"),
        ("PARIS.<|endoftext|>", "PARIS."),
        ("<answer> </answer>", ""),
    ],
)
def test_qa_sends_only_final_answer_and_keeps_references_out_of_prompt(response, answer):
    sample = SimpleNamespace(
        prompt="Where?",
        response=response,
        label=["private reference", "alias"],
        metadata={"verifier": "qa_exact_match"},
    )
    service, body = verify_payload(sample)
    assert service == "hotpotqa_qa"
    assert json.loads(body["response"]["output"][0]["content"][0]["text"]) == {"answer": answer}
    assert body["responses_create_params"]["input"] == [{"role": "user", "content": "Where?"}]
    assert body["expected_answer"] == "private reference"


async def test_qa_uses_native_exact_match_and_accepts_a_later_alias(monkeypatch):
    from miles.rollout.rm_hub import async_rm
    from miles.utils.types import RewardSpec, Sample

    replies = [(200, {"reward": 1, "answer_em": 0}), (200, {"reward": 1, "answer_em": 1})]
    sample = Sample(
        prompt="Where?",
        response="<answer>NYC</answer>",
        label=["New York City", "NYC"],
        metadata={"verifier": "qa_exact_match"},
        reward_spec=RewardSpec(custom_rm_path="experiments.src.nemo_gym.bridge.reward.reward"),
    )
    async with verifier_server(replies) as (url, calls):
        monkeypatch.setattr(client, "endpoints", lambda: {"services": {"hotpotqa_qa": url}})
        assert await async_rm(SimpleNamespace(custom_rm_path=None, rm_type=None), sample) == 1
        assert [call["expected_answer"] for call in calls] == sample.label
        assert sample.metadata["nemo_gym_result"]["reward"] == 1
    async with verifier_server([replies[0]]) as (url, _):
        monkeypatch.setattr(client, "endpoints", lambda: {"services": {"hotpotqa_qa": url}})
        sample.label = "Paris"
        sample.response = "Paris and London"
        assert await reward(None, sample) == 0
        assert sample.metadata["nemo_gym_result"]["reward"] == 0


@pytest.mark.parametrize("failure", [(503, {}), (200, {"reward": 1})])
async def test_qa_partial_alias_failure_stays_unscored(failure, monkeypatch):
    replies = [(200, {"reward": 1, "answer_em": 0}), failure]
    async with verifier_server(replies) as (url, calls):
        monkeypatch.setattr(client, "endpoints", lambda: {"services": {"hotpotqa_qa": url}})
        sample = SimpleNamespace(
            prompt="Where?", response="NYC", label=["New York City", "NYC"], metadata={"verifier": "qa_exact_match"}
        )
        assert await reward(None, sample) is None
        assert [call["expected_answer"] for call in calls] == ["New York City", "NYC", "NYC", "NYC"]
        record = sample.metadata["nemo_gym_result"]
        assert record["status"] == "unscored" and record["reward"] is None
        assert record["failure_kind"] == ("http_error" if failure[0] == 503 else "invalid_reward")


def test_reasoning_metadata_and_reference_do_not_enter_the_policy_prompt():
    sample = SimpleNamespace(
        prompt="question",
        response="<answer>42</answer>",
        metadata={
            "verifier": "reasoning_gym",
            "question": "question",
            "reference_answer": "42",
            "source_dataset": "task",
            "verifier_metadata": {"secret": "state"},
        },
    )
    service, body = verify_payload(sample)
    assert service == "reasoning_gym"
    assert body["metadata"] == {"secret": "state", "source_dataset": "task"}
    assert body["answer"] == "42"
    assert body["responses_create_params"]["input"] == [{"role": "user", "content": "question"}]


def test_tau_dataset_preserves_native_request_and_marks_test_rows():
    row = {
        "config": {"domain": "retail"},
        "task": {"id": "1", "evaluation_criteria": {"reward_basis": ["DB"]}},
        "responses_create_params": {"input": [{"role": "system", "content": "policy"}]},
    }
    converted = adapt_tau2(row, split="test")
    assert converted["metadata"]["eval_only"] is True
    assert converted["metadata"]["nemo_gym"]["request"]["config"]["task_split_name"] == "test"
    assert "task_split_name" not in row["config"]
    with pytest.raises(ValueError, match="explicit official"):
        adapt_tau2(row, split="base")


async def test_agent_retries_whole_episode_but_accepts_a_valid_zero(monkeypatch):
    from experiments.src.nemo_gym.bridge.generate import generate
    from miles.rollout.generate_hub import agentic_tool_call
    from miles.utils.types import Sample

    attempts = []
    from experiments.src.nemo_gym.bridge import generate as generate_module

    events = Counter()
    monkeypatch.setattr(generate_module, "EVENTS", events)

    async def episode(input):
        sample = Sample(
            status=Sample.Status.COMPLETED, metadata={"nemo_gym_result": {"status": "unscored", "reward": None}}
        )
        attempts.append(sample)
        if len(attempts) == 2:
            sample.metadata["nemo_gym_result"] = {"status": "scored", "reward": 0.0}
        sample.metadata["nemo_gym_events"] = {"tau2/request_count": 1}
        return SimpleNamespace(samples=[sample, Sample(status=sample.status, metadata=dict(sample.metadata))])

    monkeypatch.setattr(agentic_tool_call, "generate", episode)
    input = SimpleNamespace(sample=Sample(metadata={"verifier": "nemo_gym_tau2"}))
    result = await generate(input)
    assert len(attempts) == 2
    assert result.samples[0] is attempts[-1]
    assert result.samples[0].metadata["nemo_gym_result"]["reward"] == 0
    assert events == {"tau2/request_count": 2}
    assert all("nemo_gym_events" not in sample.metadata for sample in result.samples)


@pytest.mark.parametrize("mode", ["subproc", "inline"])
@pytest.mark.parametrize("scored", [False, True])
async def test_agent_returns_events_across_process_boundary(tmp_path, monkeypatch, mode, scored):
    from experiments.src.nemo_gym.bridge.agent import run
    from miles.rollout.agentic.agent_function import call_agent_function

    async def create_session(request):
        return web.json_response({"rollout_id": "test-episode"})

    async def delete_session(request):
        return web.json_response({})

    async def episode(request):
        return web.json_response({"reward": 0.0} if scored else {"error": "failed"})

    app = web.Application()
    app.router.add_post("/sessions", create_session)
    app.router.add_delete("/sessions/{id}", delete_session)
    app.router.add_post("/run", episode)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    manifest = tmp_path / "endpoints.json"
    manifest.write_text(json.dumps({"schema": 1, "policy_bridge": url, "services": {"tau2": url}}))
    monkeypatch.setenv("NEMO_GYM_ENDPOINTS", str(manifest))
    client.endpoints.cache_clear()
    before = client.EVENTS.copy()
    try:
        result = await call_agent_function(
            run,
            mode=mode,
            base_url="http://unused-policy",
            prompt="test",
            request_kwargs={},
            metadata={"verifier": "nemo_gym_tau2", "nemo_gym": {"service": "tau2", "request": {}}},
        )
        assert result["nemo_gym_events"]["tau2/request_count"] == 1
        outcome = "scored" if scored else "unscored"
        assert result["nemo_gym_events"][f"tau2/{outcome}_count"] == 1
        assert result["nemo_gym_result"]["status"] == outcome
        assert client.EVENTS == before  # Inline mode must not also count in the parent.
    finally:
        client.endpoints.cache_clear()
        await runner.cleanup()


async def test_exhausted_agent_failure_is_unscored_and_not_requeued_as_aborted(monkeypatch):
    from experiments.src.nemo_gym.bridge.generate import generate
    from miles.rollout.generate_hub import agentic_tool_call
    from miles.utils.types import Sample

    attempts = []

    async def episode(input):
        attempts.append(1)
        return SimpleNamespace(samples=[Sample(status=Sample.Status.ABORTED, metadata={})])

    monkeypatch.setattr(agentic_tool_call, "generate", episode)
    result = await generate(SimpleNamespace(sample=Sample(metadata={"verifier": "nemo_gym_tau2"})))
    assert len(attempts) == 3
    assert result.samples[0].status == Sample.Status.FAILED
    assert result.samples[0].reward is None


@pytest.mark.parametrize("kind", ["execution_time_limit", "compilation_deadline", "sandbox_start_failed"])
async def test_code_failure_reasons_survive_http_without_becoming_zero(kind, monkeypatch):
    retryable = kind == "sandbox_start_failed"
    failure = (503, {"detail": {"failure_kind": kind, "retryable": retryable}})
    async with verifier_server([failure]) as (url, calls):
        monkeypatch.setattr(client, "endpoints", lambda: {"services": {"code_gen": url}})
        diagnostics = {}
        assert await client.request("code_gen", "/verify", {}, attempts=2, diagnostics=diagnostics) is None
        assert len(calls) == (2 if retryable else 1)
        assert diagnostics == {
            "status": "unscored",
            "service": "code_gen",
            "reward": None,
            "attempts": len(calls),
            "failure_kind": kind,
            "http_status": 503,
        }


async def test_miles_dispatch_keeps_nemo_scoring_failures_missing(monkeypatch):
    from miles.rollout.rm_hub import batched_async_rm
    from miles.utils.types import RewardSpec, Sample

    async with verifier_server([(200, {"reward": 1}), (200, {"reward": 0}), (503, {})]) as (url, calls):
        monkeypatch.setattr(client, "endpoints", lambda: {"services": {"math_with_judge": url}})
        samples = [
            Sample(
                prompt="one plus one",
                response=response,
                label="2",
                metadata={"verifier": "nemo_gym_math_judge"},
                reward_spec=RewardSpec(custom_rm_path="experiments.src.nemo_gym.bridge.reward.reward"),
            )
            for response in ("2", "3", "unknown")
        ]
        result = await batched_async_rm(SimpleNamespace(custom_rm_path=None, rm_type=None), samples)
        assert result == [1, 0, None]
        assert samples[-1].metadata["nemo_gym_result"]["failure_kind"] == "http_error"


def test_judge_payload_preserves_verifier_options_without_leaking_labels():
    native = {
        "responses_create_params": {"input": [{"role": "user", "content": "Question"}]},
        "expected_answer": "private reference",
        "template_metadata": {"output_regex": "Answer: (.*)"},
    }
    sample = SimpleNamespace(
        prompt=native["responses_create_params"]["input"],
        label=native["expected_answer"],
        response="Candidate",
        metadata={"verifier": "nemo_gym_equivalence_judge", "nemo_gym": {"request": native}},
    )
    service, payload = verify_payload(sample)
    assert service == "equivalence_llm_judge"
    assert payload["template_metadata"] == native["template_metadata"]
    assert payload["expected_answer"] == "private reference"
    assert payload["responses_create_params"]["input"] == [{"role": "user", "content": "Question"}]
    assert "response" not in native


def test_mcqa_rejects_broken_reference_or_regex_before_scoring():
    sample = SimpleNamespace(
        prompt="choose",
        response="Answer: B",
        label="B",
        metadata={"verifier": "mcqa_regex", "valid_letters": ["A", "B"], "output_regex": "["},
    )
    import re

    with pytest.raises(re.error):
        verify_payload(sample)
    sample.metadata["output_regex"] = "Answer: ([AB])"
    sample.label = "Z"
    with pytest.raises(ValueError, match="reference letter"):
        verify_payload(sample)


def test_structured_output_preserves_native_schema_and_response():
    native = {
        "responses_create_params": {"input": [{"role": "user", "content": "Return a record."}]},
        "schema_str": '{"type": "object", "properties": {"count": {"type": "integer"}}}',
        "schema_type": "json",
    }
    sample = SimpleNamespace(
        prompt=native["responses_create_params"]["input"],
        label=native["schema_str"],
        metadata={"verifier": "json_schema", "schema_type": "json", "nemo_gym": {"request": native}},
    )
    candidate = '```json\n{"count": 3}\n```'
    sample.response = candidate
    service, body = verify_payload(sample)
    assert service == "structured_outputs"
    assert body["schema_str"] == native["schema_str"] and body["schema_type"] == "json"
    assert body["response"]["output"][0]["content"][0]["text"] == candidate
    assert body["responses_create_params"] == native["responses_create_params"]
    with pytest.raises(ValueError, match="text responses only"):
        sample.metadata["nemo_gym"]["request"]["response_mode"] = "tool_call"
        verify_payload(sample)


@pytest.mark.parametrize(
    "error_type,expected,attempts", [("parse_error", 0.0, 1), ("schema_error", None, 1), ("api_error", None, 3)]
)
async def test_structured_output_failures_distinguish_bad_answer_from_unscorable_task(
    error_type, expected, attempts, monkeypatch
):
    from miles.rollout.rm_hub import async_rm
    from miles.utils.types import RewardSpec, Sample

    sample = Sample(
        prompt="Return JSON",
        response="not JSON",
        label='{"type": "object"}',
        metadata={"verifier": "json_schema"},
        reward_spec=RewardSpec(custom_rm_path="experiments.src.nemo_gym.bridge.reward.reward"),
    )
    async with verifier_server([(200, {"reward": 0, "error_type": error_type})]) as (url, calls):
        monkeypatch.setattr(client, "endpoints", lambda: {"services": {"structured_outputs": url}})
        assert await async_rm(SimpleNamespace(custom_rm_path=None, rm_type=None), sample) == expected
        assert len(calls) == attempts
        record = sample.metadata["nemo_gym_result"]
        if error_type == "parse_error":
            assert record["status"] == "scored" and record["scoring_reason"] == "parse_error"
        else:
            assert record["status"] == "unscored"
            assert record["failure_kind"] == ("invalid_schema" if error_type == "schema_error" else "model_api_error")
