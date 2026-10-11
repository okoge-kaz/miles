from __future__ import annotations

import string
from typing import Any

from experiments.src.datasets.common.openai_responses import to_chat_messages
from experiments.src.nemo_gym.bridge.payloads import code_metadata


def _chat_from_params(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    params = row.get("responses_create_params") or {}
    messages = to_chat_messages(params.get("input"))
    messages = [
        message
        for message in messages
        if message.get("content") or message.get("tool_calls") or message.get("role") == "tool"
    ]
    return messages or None


def _base_metadata(row: dict[str, Any], *, source: str, verifier: str) -> dict[str, Any]:
    metadata = {"source": source, "verifier": verifier}
    for key in (
        "dataset",
        "pass_rate",
        "pass_rate_total",
        "pass_rate_passed",
        "trajectory_id",
        "uuid",
        "id",
    ):
        if row.get(key) is not None:
            metadata[key] = row[key]
    if row.get("agent_ref"):
        metadata["agent_ref"] = row["agent_ref"]
    return metadata


def adapt_knowledge_mcqa(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    label = str(row.get("expected_answer") or "").strip().upper()
    if not prompt or not label:
        return None
    valid_letters = sorted(
        {
            key.upper()
            for option in row.get("options") or []
            if isinstance(option, dict)
            for key, value in option.items()
            if value is not None
        }
    )
    # Some published rows collapse every choice into option A while retaining a
    # different reference letter. Do not invent choices or train on that label.
    if (
        len(valid_letters) < 2
        or any(len(letter) != 1 or letter not in string.ascii_uppercase for letter in valid_letters)
        or len(label) != 1
        or label not in valid_letters
    ):
        return None
    template = row.get("template_metadata") or {}
    metadata = _base_metadata(row, source="knowledge-mcqa", verifier="mcqa_regex")
    metadata.update(
        {
            "valid_letters": valid_letters,
            "output_regex": template.get("output_regex"),
            "template_id": template.get("template_id"),
            "reward_profiles": row.get("reward_profiles") or [],
        }
    )
    return {"prompt": prompt, "label": label, "metadata": metadata}


def adapt_competitive_coding(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    unit_tests = (row.get("verifier_metadata") or {}).get("unit_tests")
    if not prompt or unit_tests is None:
        return None
    code_metadata({"unit_tests": unit_tests})
    metadata = _base_metadata(row, source="competitive-coding", verifier="python_code")
    metadata.update(
        {
            "unit_tests": unit_tests,
            "problem_id": (row.get("verifier_metadata") or {}).get("problem_id"),
        }
    )
    return {
        "prompt": prompt,
        "label": f"{len(unit_tests['inputs'])} tests",
        "metadata": metadata,
    }


ADAPTERS = {
    "competitive-coding": adapt_competitive_coding,
    "knowledge-mcqa": adapt_knowledge_mcqa,
}
