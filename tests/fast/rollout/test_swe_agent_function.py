"""Fail-closed response handling for Harbor-backed agent rollouts."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from experiments.src.environments.swe.result import HarborSWEOutcome
from experiments.src.environments.swe.timeouts import TRIAL_REQUEST_TIMEOUT_SEC
from miles.rollout.harbor.auth import (
    derive_harbor_cancel_bearer,
    derive_harbor_flush_bearer,
    derive_harbor_run_bearer,
)


_MODULE_PATH = Path(__file__).parents[3] / "examples" / "swe-agent" / "swe_agent_function.py"
_SPEC = importlib.util.spec_from_file_location("miles_swe_agent_function", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_validated_trial_metadata = _MODULE._validated_trial_metadata


def test_rollout_http_client_does_not_trust_proxy_environment(monkeypatch) -> None:
    client = MagicMock()
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(_MODULE.httpx, "AsyncClient", constructor)
    monkeypatch.setattr(
        _MODULE.httpx,
        "AsyncHTTPTransport",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(_MODULE, "_agent_server_client", None)

    assert _MODULE._get_agent_server_client() is client
    assert constructor.call_args.kwargs["trust_env"] is False


def test_rollout_client_uses_the_bounded_trial_request_timeout(monkeypatch) -> None:
    monkeypatch.delenv("SWE_TRIAL_REQUEST_TIMEOUT_SEC", raising=False)
    assert _MODULE._request_timeout_seconds() == TRIAL_REQUEST_TIMEOUT_SEC
    assert _MODULE._DEFAULT_REQUEST_TIMEOUT_SEC == TRIAL_REQUEST_TIMEOUT_SEC
    assert _MODULE._MIN_REQUEST_TIMEOUT_SEC == TRIAL_REQUEST_TIMEOUT_SEC

    monkeypatch.setenv(
        "SWE_TRIAL_REQUEST_TIMEOUT_SEC",
        str(TRIAL_REQUEST_TIMEOUT_SEC - 1),
    )
    with pytest.raises(ValueError, match="bounded end-to-end trial contract"):
        _MODULE._request_timeout_seconds()


def test_submitted_verifier_result_is_graded() -> None:
    metadata = _validated_trial_metadata(
        {
            "reward": 0,
            "exit_status": "Submitted",
            "eval_report": {"reward": 0},
            "agent_metrics": {"turns": 3},
        }
    )

    assert metadata == {
        "reward": 0.0,
        "exit_status": "Submitted",
        "eval_report": {"reward": 0.0, "resolved": False},
        "agent_metrics": {"turns": 3},
    }


def test_agent_metrics_are_finite_bounded_and_allowlisted() -> None:
    metadata = _validated_trial_metadata(
        {
            "reward": 1,
            "exit_status": "Submitted",
            "eval_report": {"reward": 1},
            "agent_metrics": {
                "agent_run_time": 12.5,
                "cost_usd": 0.25,
                "n_input_tokens": 1_024,
                "n_output_tokens": 256,
                "total_tool_time": 3.0,
                "turns": 4,
                "arbitrary": 9,
                "nested": {"secret": "value"},
            },
        }
    )

    assert metadata is not None
    assert metadata["agent_metrics"] == {
        "agent_run_time": 12.5,
        "cost_usd": 0.25,
        "n_input_tokens": 1_024,
        "n_output_tokens": 256,
        "total_tool_time": 3.0,
        "turns": 4,
    }


def test_rollout_and_offline_evaluation_share_result_validation() -> None:
    response = {
        "reward": 1,
        "exit_status": "Submitted",
        "eval_report": {
            "reward": 1,
            "passed_count": 3,
            "expected_test_ids": ["private::test"],
        },
        "agent_metrics": {
            "n_cache_tokens": 128,
            "n_input_tokens": 256,
            "total_tool_time": 1.5,
            "private": {"secret": True},
        },
    }

    outcome = HarborSWEOutcome.from_mapping(response)

    assert _validated_trial_metadata(response) == {
        "reward": outcome.reward,
        "exit_status": outcome.exit_status,
        "eval_report": outcome.eval_report,
        "agent_metrics": outcome.agent_metrics,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("total_tool_time", "3"),
        ("total_tool_time", float("nan")),
        ("total_tool_time", float("inf")),
        ("total_tool_time", -1),
        ("total_tool_time", 86_401),
        ("n_input_tokens", 1.5),
        ("n_output_tokens", True),
        ("turns", {"value": 3}),
        ("cost_usd", 1_000_001),
    ],
)
def test_invalid_agent_metric_is_dropped(key: str, value: object) -> None:
    metadata = _validated_trial_metadata(
        {
            "reward": 0,
            "exit_status": "Submitted",
            "eval_report": {"reward": 0},
            "agent_metrics": {key: value},
        }
    )

    assert metadata is not None
    assert metadata["agent_metrics"] == {}


@pytest.mark.parametrize(
    "response",
    [
        {"reward": 0, "exit_status": "AgentError", "eval_report": {}},
        {"reward": 0, "exit_status": "TimeLimitExceeded", "eval_report": {}},
        {"reward": 0, "exit_status": "Submitted", "eval_report": {}},
        {"reward": 0, "exit_status": "Submitted", "eval_report": {"reward": 1}},
        {"reward": -0.1, "exit_status": "Submitted", "eval_report": {"reward": -0.1}},
        {"reward": float("nan"), "exit_status": "Submitted", "eval_report": {"reward": 0}},
        {"reward": True, "exit_status": "Submitted", "eval_report": {"reward": 1}},
    ],
)
def test_ungraded_or_invalid_trial_is_aborted(response: dict) -> None:
    assert _validated_trial_metadata(response) is None


def test_run_forwards_private_manifest_digest_as_explicit_binding(monkeypatch) -> None:
    digest = "a" * 64
    run_secret = "r" * 32
    post = AsyncMock(
        return_value={
            "reward": 1,
            "exit_status": "Submitted",
            "eval_report": {"reward": 1},
            "agent_metrics": {},
        }
    )
    monkeypatch.setattr(_MODULE, "_post_agent_server", post)
    monkeypatch.setenv("HARBOR_RUN_SECRET", run_secret)
    monkeypatch.setenv("HARBOR_CLIENT_ID", "client-1")

    output = asyncio.run(
        _MODULE.run(
            base_url="http://127.0.0.1:30000",
            prompt="fix it",
            metadata={
                "instance_id": "task-1",
                "agent_name": "terminus-2",
                "swe_task": {"task_digest": digest},
            },
        )
    )

    assert output.aborted is False
    assert post.await_args.args[1]["task_digest"] == digest
    assert post.await_args.args[1]["client_id"] == "client-1"
    request_id = post.await_args.args[1]["request_id"]
    assert isinstance(request_id, str) and len(request_id) == 32
    expected_bearer = derive_harbor_run_bearer(
        run_secret,
        instance_id="task-1",
        client_id="client-1",
        request_id=request_id,
    )
    assert post.await_args.kwargs == {"bearer_token": expected_bearer}
    assert run_secret not in post.await_args.args[1].values()


def test_run_timeout_cancels_the_exact_client_session_task(monkeypatch) -> None:
    run_secret = "r" * 32
    post = AsyncMock(
        side_effect=[
            asyncio.TimeoutError(),
            {"cancelled": 1},
        ]
    )
    monkeypatch.setattr(_MODULE, "_post_agent_server", post)
    monkeypatch.setattr(_MODULE, "_request_timeout_seconds", lambda: 1)
    monkeypatch.setattr(_MODULE.secrets, "token_hex", lambda _size: "1" * 32)
    monkeypatch.setenv("HARBOR_RUN_SECRET", run_secret)
    monkeypatch.setenv("HARBOR_CLIENT_ID", "client-1")

    output = asyncio.run(
        _MODULE.run(
            base_url="http://127.0.0.1:30000",
            prompt="fix it",
            metadata={
                "instance_id": "task-1",
                "agent_name": "terminus-2",
                "session_server_instance_id": "session-1",
                "swe_task": {"task_digest": "a" * 64},
            },
        )
    )

    assert output.aborted is True
    assert post.await_count == 2
    cancel_call = post.await_args_list[1]
    assert cancel_call.args == (
        "http://localhost:11000/cancel",
        {
            "client_id": "client-1",
            "instance_id": "task-1",
            "request_id": "1" * 32,
            "session_server_instance_id": "session-1",
        },
    )
    assert cancel_call.kwargs == {
        "bearer_token": derive_harbor_cancel_bearer(
            run_secret,
            client_id="client-1",
            request_id="1" * 32,
            instance_id="task-1",
            session_server_instance_id="session-1",
        )
    }


def test_run_aborts_before_dispatch_without_run_authentication(monkeypatch) -> None:
    post = AsyncMock()
    monkeypatch.setattr(_MODULE, "_post_agent_server", post)
    monkeypatch.delenv("HARBOR_RUN_SECRET", raising=False)

    output = asyncio.run(
        _MODULE.run(
            base_url="http://127.0.0.1:30000",
            prompt="fix it",
            metadata={
                "instance_id": "task-1",
                "agent_name": "terminus-2",
                "swe_task": {"task_digest": "a" * 64},
            },
        )
    )

    assert output.aborted is True
    post.assert_not_awaited()


def test_abort_requires_and_sends_rollout_authentication(monkeypatch) -> None:
    post = AsyncMock(return_value={"cancelled": 1})
    monkeypatch.setattr(_MODULE, "post", post)
    monkeypatch.setenv("AGENT_SERVER_URL", "http://127.0.0.1:11000")
    monkeypatch.setenv("HARBOR_RUN_SECRET", "r" * 32)

    asyncio.run(
        _MODULE.abort(SimpleNamespace(session_server_instance_id="session-1"))
    )

    expected_bearer = derive_harbor_flush_bearer(
        "r" * 32,
        session_server_instance_id="session-1",
    )
    assert post.await_args.kwargs["headers"] == {
        "Authorization": f"Bearer {expected_bearer}"
    }
    assert post.await_args.args[1] == {"session_server_instance_id": "session-1"}


def test_abort_flushes_the_real_driver_instance_map(monkeypatch) -> None:
    post = AsyncMock(return_value={"cancelled": 1})
    run_secret = "r" * 32
    monkeypatch.setattr(_MODULE, "post", post)
    monkeypatch.setenv("AGENT_SERVER_URL", "http://127.0.0.1:11000")
    monkeypatch.setenv("HARBOR_RUN_SECRET", run_secret)

    asyncio.run(
        _MODULE.abort(
            SimpleNamespace(
                session_server_instance_ids={31001: "session-b", 31000: "session-a"}
            )
        )
    )

    post.assert_has_awaits(
        [
            call(
                "http://127.0.0.1:11000/flush",
                {"session_server_instance_id": instance_id},
                max_retries=3,
                headers={
                    "Authorization": "Bearer "
                    + derive_harbor_flush_bearer(
                        run_secret,
                        session_server_instance_id=instance_id,
                    )
                },
            )
            for instance_id in ("session-a", "session-b")
        ],
        any_order=True,
    )


@pytest.mark.parametrize(
    "instance_ids",
    [
        {31000: "duplicate", 31001: "duplicate"},
        {0: "session-a"},
        {31000: "bad instance"},
        [(31000, "session-a")],
    ],
)
def test_abort_rejects_the_entire_invalid_driver_inventory(
    monkeypatch,
    instance_ids,
) -> None:
    post = AsyncMock()
    monkeypatch.setattr(_MODULE, "post", post)
    monkeypatch.setenv("AGENT_SERVER_URL", "http://127.0.0.1:11000")
    monkeypatch.setenv("HARBOR_RUN_SECRET", "r" * 32)

    asyncio.run(
        _MODULE.abort(SimpleNamespace(session_server_instance_ids=instance_ids))
    )

    post.assert_not_awaited()


def test_abort_refuses_to_dispatch_without_rollout_authentication(monkeypatch) -> None:
    post = AsyncMock()
    monkeypatch.setattr(_MODULE, "post", post)
    monkeypatch.setenv("AGENT_SERVER_URL", "http://127.0.0.1:11000")
    monkeypatch.delenv("HARBOR_RUN_SECRET", raising=False)

    asyncio.run(
        _MODULE.abort(SimpleNamespace(session_server_instance_id="session-1"))
    )

    post.assert_not_awaited()
