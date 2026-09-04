"""Environment-independent deterministic scorers shared by reward sets."""

from __future__ import annotations

import re
from typing import Any

from miles.rollout.rm_hub.gpqa import compute_gpqa_reward
from miles.rollout.rm_hub.math_utils import grade_answer_verl


def score_math_sample(sample: Any) -> float:
    return float(grade_answer_verl(sample.response, sample.label))


def score_gpqa_sample(sample: Any) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return float(compute_gpqa_reward(sample.response, sample.label, metadata=metadata))


def score_mcqa_regex_sample(sample: Any) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    pattern = metadata.get("output_regex")
    expected = str(sample.label or "").strip().upper()
    if not expected:
        return 0.0
    if not pattern:
        return score_gpqa_sample(sample)
    try:
        matches = re.findall(str(pattern), str(sample.response or ""), flags=re.IGNORECASE)
    except re.error:
        return 0.0
    if not matches:
        return 0.0
    extracted = matches[-1]
    if isinstance(extracted, tuple):
        extracted = next((value for value in extracted if value), "")
    extracted = str(extracted).strip().upper()
    valid_letters = {str(letter).upper() for letter in metadata.get("valid_letters") or []}
    if valid_letters and extracted not in valid_letters:
        return 0.0
    return 1.0 if extracted == expected else 0.0
