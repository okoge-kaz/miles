"""Verify Workplace Assistant action trajectories against the official state checker."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from experiments.src.environments.workplace.runtime import _load_resource_functions


def _official_action(action: dict[str, Any]) -> dict[str, str]:
    arguments = action.get("arguments") or {}
    if isinstance(arguments, str):
        json.loads(arguments)
        arguments_text = arguments
    elif isinstance(arguments, dict):
        arguments_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    else:
        raise ValueError("Workplace tool arguments must be an object or JSON object string")
    return {"name": str(action.get("name") or ""), "arguments": arguments_text}


def score_action_trajectory(predicted: list[dict[str, Any]], expected: list[dict[str, Any]]) -> float:
    """Compare final database state after predicted and ground-truth trajectories."""

    if not isinstance(predicted, list) or not isinstance(expected, list):
        return 0.0
    try:
        predicted_actions = [_official_action(action) for action in predicted]
        expected_actions = [_official_action(action) for action in expected]
        _, is_correct = _load_resource_functions()
        return float(bool(is_correct(predicted_actions, expected_actions, None)))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, subprocess.SubprocessError):
        return 0.0
