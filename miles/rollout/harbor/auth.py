"""Task- and session-scoped authentication for the Harbor agent server."""

from __future__ import annotations

import hashlib
import hmac
import re

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_SAFE_SESSION_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_RUN_TOKEN_CONTEXT = b"miles-harbor-run-task-v2\0"
_CANCEL_TOKEN_CONTEXT = b"miles-harbor-cancel-task-v1\0"
_DRAIN_TOKEN_CONTEXT = b"miles-harbor-drain-client-v1\0"
_FLUSH_TOKEN_CONTEXT = b"miles-harbor-flush-session-v1\0"
_HEALTH_TOKEN_CONTEXT = b"miles-harbor-health-v1\0"
_MIN_SECRET_LENGTH = 32
_MAX_SECRET_LENGTH = 4096


def _validated_master_secret(master_secret: str) -> bytes:
    if (
        not isinstance(master_secret, str)
        or not _MIN_SECRET_LENGTH <= len(master_secret) <= _MAX_SECRET_LENGTH
        or "\r" in master_secret
        or "\n" in master_secret
    ):
        raise ValueError("Harbor run master secret is missing or invalid")
    return master_secret.encode()


def derive_harbor_run_bearer(
    master_secret: str,
    *,
    instance_id: str,
    session_server_instance_id: str | None = None,
    client_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Derive a bearer usable only for one task in one optional session.

    The process-environment master secret never crosses the HTTP boundary. A
    live rollout token binds both the opaque session instance and task ID, so a
    captured token cannot start another admitted task in the same session.
    Offline evaluation requests without a session remain scoped to one task.
    """
    secret = _validated_master_secret(master_secret)
    if _SAFE_TASK_ID.fullmatch(instance_id) is None:
        raise ValueError("instance_id is unsafe")
    binding = _task_binding(
        instance_id=instance_id,
        session_server_instance_id=session_server_instance_id,
        client_id=client_id,
        request_id=request_id,
    )
    return hmac.new(
        secret,
        _RUN_TOKEN_CONTEXT + binding.encode(),
        hashlib.sha256,
    ).hexdigest()


def derive_harbor_cancel_bearer(
    master_secret: str,
    *,
    instance_id: str,
    client_id: str,
    request_id: str,
    session_server_instance_id: str | None = None,
) -> str:
    """Derive a capability for cancelling one task within one client scope."""

    secret = _validated_master_secret(master_secret)
    binding = _task_binding(
        instance_id=instance_id,
        session_server_instance_id=session_server_instance_id,
        client_id=client_id,
        request_id=request_id,
    )
    return hmac.new(
        secret,
        _CANCEL_TOKEN_CONTEXT + binding.encode(),
        hashlib.sha256,
    ).hexdigest()


def derive_harbor_drain_bearer(master_secret: str, *, client_id: str) -> str:
    """Derive a capability that cancels only one job/client inventory."""

    secret = _validated_master_secret(master_secret)
    _validate_client_id(client_id)
    return hmac.new(
        secret,
        _DRAIN_TOKEN_CONTEXT + f"client:{client_id}".encode(),
        hashlib.sha256,
    ).hexdigest()


def _task_binding(
    *,
    instance_id: str,
    session_server_instance_id: str | None,
    client_id: str | None,
    request_id: str | None,
) -> str:
    if _SAFE_TASK_ID.fullmatch(instance_id) is None:
        raise ValueError("instance_id is unsafe")
    fields = []
    if (client_id is None) != (request_id is None):
        raise ValueError("client_id and request_id must be supplied together")
    if client_id is not None and request_id is not None:
        _validate_client_id(client_id)
        fields.append(f"client:{client_id}")
        if _SAFE_REQUEST_ID.fullmatch(request_id) is None:
            raise ValueError("request_id is unsafe")
        fields.append(f"request:{request_id}")
    if session_server_instance_id is not None:
        if _SAFE_SESSION_INSTANCE_ID.fullmatch(session_server_instance_id) is None:
            raise ValueError("session_server_instance_id is unsafe")
        fields.append(f"session:{session_server_instance_id}")
    fields.append(f"task:{instance_id}")
    return "\0".join(fields)


def _validate_client_id(client_id: str) -> None:
    if _SAFE_SESSION_INSTANCE_ID.fullmatch(client_id) is None:
        raise ValueError("client_id is unsafe")


def derive_harbor_flush_bearer(
    master_secret: str,
    *,
    session_server_instance_id: str,
) -> str:
    """Derive the distinct session-only capability accepted by ``/flush``."""

    secret = _validated_master_secret(master_secret)
    if _SAFE_SESSION_INSTANCE_ID.fullmatch(session_server_instance_id) is None:
        raise ValueError("session_server_instance_id is unsafe")
    return hmac.new(
        secret,
        _FLUSH_TOKEN_CONTEXT + f"session:{session_server_instance_id}".encode(),
        hashlib.sha256,
    ).hexdigest()


def derive_harbor_health_bearer(master_secret: str) -> str:
    """Derive the read-only capability accepted by the readiness endpoint."""

    return hmac.new(
        _validated_master_secret(master_secret),
        _HEALTH_TOKEN_CONTEXT,
        hashlib.sha256,
    ).hexdigest()
