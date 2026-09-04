"""Scoped-bearer tests for the authenticated Harbor agent server."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from miles.rollout.harbor.auth import (
    derive_harbor_cancel_bearer,
    derive_harbor_drain_bearer,
    derive_harbor_flush_bearer,
    derive_harbor_health_bearer,
    derive_harbor_run_bearer,
)


def _load_readiness_module():
    path = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "experimental"
        / "swe-agent-harbor-e2b"
        / "wait_for_agent_server.py"
    )
    spec = importlib.util.spec_from_file_location("miles_harbor_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_offline_bearer_is_bound_to_one_task() -> None:
    master = "m" * 32

    first = derive_harbor_run_bearer(master, instance_id="task-a")
    second = derive_harbor_run_bearer(master, instance_id="task-b")

    assert first != second
    assert len(first) == 64


def test_rollout_bearer_binds_both_task_and_session() -> None:
    master = "m" * 32

    first = derive_harbor_run_bearer(
        master,
        instance_id="task-a",
        session_server_instance_id="session-a",
    )
    same_session_other_task = derive_harbor_run_bearer(
        master,
        instance_id="task-b",
        session_server_instance_id="session-a",
    )
    other_session = derive_harbor_run_bearer(
        master,
        instance_id="task-a",
        session_server_instance_id="session-b",
    )

    assert first != same_session_other_task
    assert first != other_session


def test_rollout_bearer_is_bound_to_the_job_client() -> None:
    master = "m" * 32

    first = derive_harbor_run_bearer(
        master,
        instance_id="task-a",
        session_server_instance_id="session-a",
        client_id="client-a",
        request_id="1" * 32,
    )
    other_client = derive_harbor_run_bearer(
        master,
        instance_id="task-a",
        session_server_instance_id="session-a",
        client_id="client-b",
        request_id="1" * 32,
    )

    assert first != other_client


def test_cancel_and_drain_capabilities_are_distinct_and_scoped() -> None:
    master = "m" * 32
    cancel = derive_harbor_cancel_bearer(
        master,
        client_id="client-a",
        request_id="1" * 32,
        session_server_instance_id="session-a",
        instance_id="task-a",
    )
    other_task = derive_harbor_cancel_bearer(
        master,
        client_id="client-a",
        request_id="2" * 32,
        session_server_instance_id="session-a",
        instance_id="task-b",
    )
    drain = derive_harbor_drain_bearer(master, client_id="client-a")
    other_client = derive_harbor_drain_bearer(master, client_id="client-b")

    assert len({cancel, other_task, drain, other_client}) == 4


def test_flush_bearer_is_distinct_and_bound_only_to_one_session() -> None:
    master = "m" * 32

    flush = derive_harbor_flush_bearer(
        master,
        session_server_instance_id="session-a",
    )
    same = derive_harbor_flush_bearer(
        master,
        session_server_instance_id="session-a",
    )
    other = derive_harbor_flush_bearer(
        master,
        session_server_instance_id="session-b",
    )
    run = derive_harbor_run_bearer(
        master,
        instance_id="task-a",
        session_server_instance_id="session-a",
    )

    assert flush == same
    assert flush != other
    assert flush != run


def test_authenticated_readiness_probe_uses_distinct_health_bearer() -> None:
    module = _load_readiness_module()
    master = "m" * 32
    observed = {}
    expected = ("a" * 64, "b" * 64, "c" * 64, 7)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def read(_limit):
            return json.dumps(
                {
                    "status": "ok",
                    "ready": True,
                    "task_set_sha256": expected[0],
                    "task_binding_sha256": expected[1],
                    "task_runtime_sha256": expected[2],
                    "task_count": expected[3],
                }
            ).encode()

    def opener(request, *, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.get_header("Authorization")
        observed["timeout"] = timeout
        return Response()

    module.wait_until_ready(
        "http://server.internal:11000",
        master,
        expected_task_set_sha256=expected[0],
        expected_task_binding_sha256=expected[1],
        expected_task_runtime_sha256=expected[2],
        expected_task_count=expected[3],
        timeout_seconds=30,
        opener=opener,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )

    assert observed == {
        "url": "http://server.internal:11000/health",
        "authorization": f"Bearer {derive_harbor_health_bearer(master)}",
        "timeout": 5.0,
    }
    assert observed["authorization"] != (
        "Bearer "
        + derive_harbor_run_bearer(
            master,
            instance_id="task-a",
            session_server_instance_id="session-a",
        )
    )


def test_readiness_probe_rejects_credentialed_or_unbounded_configuration() -> None:
    module = _load_readiness_module()
    expected = {
        "expected_task_set_sha256": "a" * 64,
        "expected_task_binding_sha256": "b" * 64,
        "expected_task_runtime_sha256": "c" * 64,
        "expected_task_count": 1,
    }

    with pytest.raises(ValueError, match="credential-free"):
        module.wait_until_ready(
            "http://user:password@server.internal:11000",
            "m" * 32,
            **expected,
            timeout_seconds=30,
        )
    with pytest.raises(ValueError, match="readiness timeout"):
        module.wait_until_ready(
            "http://server.internal:11000",
            "m" * 32,
            **expected,
            timeout_seconds=0,
        )


def test_readiness_probe_rejects_stale_task_binding() -> None:
    module = _load_readiness_module()

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def read(_limit):
            return json.dumps(
                {
                    "status": "ok",
                    "ready": True,
                    "task_set_sha256": "a" * 64,
                    "task_binding_sha256": "c" * 64,
                    "task_runtime_sha256": "d" * 64,
                    "task_count": 1,
                }
            ).encode()

    clock_values = iter((0.0, 0.5, 1.0, 1.0))
    with pytest.raises(TimeoutError):
        module.wait_until_ready(
            "http://server.internal:11000",
            "m" * 32,
            expected_task_set_sha256="a" * 64,
            expected_task_binding_sha256="b" * 64,
            expected_task_runtime_sha256="d" * 64,
            expected_task_count=1,
            timeout_seconds=1,
            opener=lambda *_args, **_kwargs: Response(),
            clock=lambda: next(clock_values),
            sleeper=lambda _seconds: None,
        )


@pytest.mark.parametrize(
    ("master", "instance_id", "session_id"),
    [
        ("short", "task", None),
        ("m" * 32 + "\n", "task", None),
        ("m" * 32, "../task", None),
        ("m" * 32, "task", "session:other"),
    ],
)
def test_bearer_rejects_unsafe_secret_or_binding(
    master: str,
    instance_id: str,
    session_id: str | None,
) -> None:
    with pytest.raises(ValueError):
        derive_harbor_run_bearer(
            master,
            instance_id=instance_id,
            session_server_instance_id=session_id,
        )


@pytest.mark.parametrize(
    ("master", "session_id"),
    [
        ("short", "session"),
        ("m" * 32 + "\n", "session"),
        ("m" * 32, "session:other"),
    ],
)
def test_flush_bearer_rejects_unsafe_secret_or_session(
    master: str,
    session_id: str,
) -> None:
    with pytest.raises(ValueError):
        derive_harbor_flush_bearer(
            master,
            session_server_instance_id=session_id,
        )
