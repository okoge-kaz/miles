"""Outcome and format rewards for Search-R1 trajectories."""

import re
import string
from typing import Any


_ASSISTANT_PATTERN = re.compile(r"<\|im_start\|>assistant\s*")
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TAG_PATTERN = re.compile(r"(</?(?:think|search|information|answer)>)")


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _is_exact_match(prediction: str, targets: list[str]) -> bool:
    normalized_prediction = _normalize_answer(prediction)
    return any(normalized_prediction == _normalize_answer(target) for target in targets)


def _extract_generated_answer(solution: str) -> str | None:
    matches = list(_ANSWER_PATTERN.finditer(solution))
    # The Search-R1 prompt itself contains one literal <answer>...</answer>
    # example. A generated answer is therefore the final match after that one.
    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def _is_valid_sequence(solution: str) -> bool:
    assistant_match = _ASSISTANT_PATTERN.search(solution)
    if assistant_match is None:
        return False

    state = "start"
    transitions = {
        ("start", "<think>"): "in_think",
        ("information", "<think>"): "in_think",
        ("in_think", "</think>"): "after_think",
        ("after_think", "<search>"): "in_search",
        ("in_search", "</search>"): "after_search",
        ("after_search", "<information>"): "in_information",
        ("in_information", "</information>"): "information",
        ("after_think", "<answer>"): "in_answer",
        ("in_answer", "</answer>"): "end",
    }
    content_states = {"in_think", "in_search", "in_information", "in_answer"}
    whitespace_states = {"start", "after_think", "after_search", "information"}
    for part in _TAG_PATTERN.split(solution[assistant_match.end() :]):
        if not part.strip():
            continue
        if _TAG_PATTERN.fullmatch(part):
            state = transitions.get((state, part), "invalid")
        elif state not in content_states and (state not in whitespace_states or part.strip()):
            state = "invalid"
        if state == "invalid":
            return False
    return state == "end"


def compute_search_r1_reward(
    *,
    prompt: str,
    response: str,
    label: dict[str, Any],
    format_score: float = 0.0,
) -> float:
    """Score the final tagged answer, matching the original Search-R1 EM reward."""
    if not 0.0 <= format_score <= 1.0:
        raise ValueError(f"Search-R1 format score must be between 0 and 1, got {format_score}")
    try:
        targets = label["ground_truth"]["target"]
    except (KeyError, TypeError) as error:
        raise ValueError("Search-R1 labels must contain ground_truth.target") from error
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, list) or not all(isinstance(target, str) for target in targets):
        raise ValueError("Search-R1 ground_truth.target must be a string or list of strings")

    solution = prompt + response
    valid_format = _is_valid_sequence(solution)
    answer = _extract_generated_answer(solution)
    if answer is None:
        return format_score if valid_format else 0.0
    if _is_exact_match(answer, targets):
        return 1.0 if valid_format else 1.0 - format_score
    return format_score if valid_format else 0.0
