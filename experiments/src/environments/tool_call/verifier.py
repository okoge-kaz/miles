"""Deterministic exact-match verifier for static function-call rows."""

from __future__ import annotations

import json
import re
from typing import Any

from experiments.src.protocols.openai_responses import expected_action_signature

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def normalize_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"__unparsed__": value}
    return value if isinstance(value, dict) else {}


def parse_tool_calls(response: str) -> list[dict[str, Any]]:
    calls = []
    for blob in _TOOL_CALL_PATTERN.findall(str(response or "")):
        try:
            call = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if call.get("name"):
            calls.append(
                {"name": call["name"], "arguments": normalize_arguments(call.get("arguments"))}
            )
    return calls


def arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if set(expected) != set(actual):
        return False
    for key, wanted in expected.items():
        received = actual[key]
        if wanted == received:
            continue
        scalar_types = (str, int, float, bool)
        if isinstance(wanted, scalar_types) and isinstance(received, scalar_types):
            if str(wanted).strip().lower() == str(received).strip().lower():
                continue
        return False
    return True


def score_tool_call_sample(sample: Any) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    expected = metadata.get("expected_action")
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            return 0.0
    signature = expected_action_signature(expected)
    if signature is None:
        return 0.0
    emitted = parse_tool_calls(sample.response)
    if signature["kind"] == "message":
        return 1.0 if not emitted and str(sample.response or "").strip() else 0.0
    if len(emitted) != 1 or emitted[0]["name"] != signature["name"]:
        return 0.0
    return 1.0 if arguments_match(signature["arguments"], emitted[0]["arguments"]) else 0.0
