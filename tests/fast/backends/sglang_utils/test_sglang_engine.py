import time
from unittest.mock import ANY, MagicMock, call

import pytest
import requests


def test_wait_server_healthy_rejects_listener_owned_by_another_process(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import _wait_server_healthy

    response = MagicMock(status_code=200)
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = response
    monkeypatch.setattr(requests, "Session", MagicMock(return_value=session))
    monkeypatch.setattr(time, "sleep", MagicMock())
    is_server_owner = MagicMock(side_effect=[False, True])

    _wait_server_healthy(
        base_url="http://127.0.0.1:22000",
        api_key=None,
        is_process_alive=lambda: True,
        is_server_owner=is_server_owner,
    )

    assert session.get.call_args_list == [
        call("http://127.0.0.1:22000/health_generate", headers=ANY),
        call("http://127.0.0.1:22000/health_generate", headers=ANY),
        call("http://127.0.0.1:22000/flush_cache", headers=ANY),
    ]


def test_process_tree_listener_check_includes_descendants(monkeypatch):
    pytest.importorskip("sglang")
    import psutil

    from miles.backends.sglang_utils.sglang_engine import _process_tree_listens_on_port

    root = MagicMock()
    child = MagicMock()
    root.children.return_value = [child]
    root.net_connections.return_value = []
    child.net_connections.return_value = [
        MagicMock(status=psutil.CONN_LISTEN, laddr=MagicMock(port=22021)),
    ]
    monkeypatch.setattr(psutil, "Process", MagicMock(return_value=root))

    assert _process_tree_listens_on_port(root_pid=1234, port=22021)


def test_begin_weight_update_sends_authoritative_version():
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine._make_request = MagicMock(return_value=[True, "Success"])

    result = engine.begin_weight_update(selector="all", weight_version="11")

    assert result == [True, "Success"]
    engine._make_request.assert_called_once_with(
        "begin_weight_update",
        {"selector": "all", "weight_version": "11"},
    )


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )
