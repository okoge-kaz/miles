"""Deterministic JSON Schema reward for structured-output rows."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema


def _extract_json(text: str) -> Any:
    text = str(text or "").strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced[-1]] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for start, character in enumerate(candidate):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[start:])
                return value
            except json.JSONDecodeError:
                continue
    return None


def score_structured_output_sample(sample: Any) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    schema_value = metadata.get("schema_str") or sample.label
    try:
        schema = json.loads(schema_value) if isinstance(schema_value, str) else schema_value
    except (json.JSONDecodeError, TypeError):
        return 0.0
    instance = _extract_json(sample.response)
    if instance is None or not isinstance(schema, dict):
        return 0.0
    if "definitions" in schema and "$defs" not in schema:
        schema = {**schema, "$defs": schema["definitions"]}
    try:
        jsonschema.validate(instance=instance, schema=schema)
    except Exception:  # noqa: BLE001 - generated schemas can be malformed
        return 0.0
    return 1.0
