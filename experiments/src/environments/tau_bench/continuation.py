"""Serializable continuation contract for stateful Tau inflight replay."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

TAU_CONTINUATION_KEY = "tau_inflight_continuation"
TAU_CONTINUATION_SCHEMA_VERSION = 1


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        value = deepcopy(message)
    else:
        value = message.model_dump(mode="json")
    value.pop("timestamp", None)
    value.pop("turn_idx", None)
    return value


def message_history_digest(messages: list[Any]) -> str:
    """Hash semantic message content without regenerated timestamp fields."""

    return _json_digest([_message_dict(message) for message in messages])


def build_continuation(
    session_state: dict[str, Any],
    *,
    policy_turns: int,
    turn_start_response_length: int,
    policy_response_prefix: str,
    policy_prefix_response_tokens: int,
) -> dict[str, Any]:
    """Combine a safe Tau environment boundary with an interrupted policy turn."""

    continuation = {
        "schema_version": TAU_CONTINUATION_SCHEMA_VERSION,
        **deepcopy(session_state),
        "policy_turns": policy_turns,
        "turn_start_response_length": turn_start_response_length,
        "policy_response_prefix": policy_response_prefix,
        "policy_prefix_response_tokens": policy_prefix_response_tokens,
    }
    validate_continuation(continuation)
    return continuation


def validate_continuation(value: Any) -> dict[str, Any]:
    """Fail closed unless a continuation contains a coherent event-log snapshot."""

    if not isinstance(value, dict):
        raise ValueError("Tau inflight continuation must be an object")
    if value.get("schema_version") != TAU_CONTINUATION_SCHEMA_VERSION:
        raise ValueError("Tau inflight continuation has an unsupported schema")

    history = value.get("message_history")
    if not isinstance(history, list) or not history:
        raise ValueError("Tau inflight continuation requires a non-empty message history")
    expected_history_digest = value.get("message_history_sha256")
    if expected_history_digest != message_history_digest(history):
        raise ValueError("Tau inflight continuation message-history digest mismatch")

    for key in (
        "orchestrator_step_count",
        "orchestrator_num_errors",
        "policy_turns",
        "turn_start_response_length",
        "policy_prefix_response_tokens",
    ):
        item = value.get(key)
        if type(item) is not int or item < 0:
            raise ValueError(f"Tau inflight continuation {key} must be a non-negative integer")
    if not isinstance(value.get("policy_response_prefix"), str):
        raise ValueError("Tau inflight continuation policy_response_prefix must be text")
    for key in ("agent_db_hash", "user_db_hash"):
        if value.get(key) is not None and not isinstance(value[key], str):
            raise ValueError(f"Tau inflight continuation {key} must be text or null")
    return value


def task_with_continuation(task: Any, continuation: dict[str, Any]) -> Any:
    """Inject the saved event log into a copy of an official Tau task."""

    continuation = validate_continuation(continuation)
    task_data = task.model_dump(mode="json")
    initial_state = dict(task_data.get("initial_state") or {})
    initial_state["message_history"] = deepcopy(continuation["message_history"])
    task_data["initial_state"] = initial_state
    return type(task).model_validate(task_data)


__all__ = [
    "TAU_CONTINUATION_KEY",
    "TAU_CONTINUATION_SCHEMA_VERSION",
    "build_continuation",
    "message_history_digest",
    "task_with_continuation",
    "validate_continuation",
]
