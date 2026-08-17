"""Strict single-turn terminal-answer reward for staleness experiments.

Use this through ``--custom-rm-path``.  It deliberately does not change the
built-in ``math`` or ``deepscaler`` rewards, because multi-turn trajectories
can legitimately contain more than one assistant answer.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.rollout.recycle_compute_metrics import SELECTION_METRICS_KEY


_ANSWER_LINE = re.compile(r"^[ \t]*Answer[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)
_BOXED_PREFIX = r"\boxed{"
_REJECTED_STATUSES = {"aborted", "failed", "truncated"}
_TERMINAL_PREFIX = re.compile(r"[ \t]*\$?[ \t]*")
_TERMINAL_SUFFIX = re.compile(
    r"[ \t]*(?:\$[ \t]*)?(?:\.[ \t]*)?"
    r"(?:(?:✅|✔️|☑️|🎉|🚀)[ \t]*)?\s*"
)
_TOKENIZER_CONFIG_FILES = (
    "tokenizer_config.json",
    "generation_config.json",
    "config.json",
    "special_tokens_map.json",
)
_STRICT_METRIC_PREFIX = "strict_math/"


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _token_strings(value):
    tokens = []
    for item in _as_list(value):
        if isinstance(item, str):
            tokens.append(item)
        elif isinstance(item, dict) and isinstance(item.get("content"), str):
            tokens.append(item["content"])
    return tokens


def _token_ids(value):
    ids = []
    for item in _as_list(value):
        if isinstance(item, int):
            ids.append(item)
        elif isinstance(item, str) and item.isdigit():
            ids.append(int(item))
    return ids


def _read_json(path):
    try:
        with path.open() as input_file:
            return json.load(input_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _added_token_decoder(tokenizer_config):
    decoder = {}
    for token_id, value in tokenizer_config.get("added_tokens_decoder", {}).items():
        content = value.get("content") if isinstance(value, dict) else value
        if isinstance(content, str) and str(token_id).isdigit():
            decoder[int(token_id)] = content
    return decoder


@lru_cache(maxsize=None)
def eos_tokens_from_config(tokenizer_path: str | Path) -> tuple[str, ...]:
    """Read every configured EOS string without assuming a model vocabulary."""
    path = Path(tokenizer_path)
    is_config_file = path.is_file() or path.suffix == ".json"
    directory = path.parent if is_config_file else path
    explicit_path = path if is_config_file else None
    config_paths = [directory / name for name in _TOKENIZER_CONFIG_FILES]
    if explicit_path is not None and explicit_path not in config_paths:
        config_paths.insert(0, explicit_path)

    configs = [_read_json(config_path) for config_path in config_paths]
    tokenizer_config = _read_json(directory / "tokenizer_config.json")
    decoder = _added_token_decoder(tokenizer_config)
    tokens = []
    eos_ids = []
    for config in configs:
        tokens.extend(_token_strings(config.get("eos_token")))
        eos_ids.extend(_token_ids(config.get("eos_token_id")))

    unresolved_ids = [token_id for token_id in eos_ids if token_id not in decoder]
    if unresolved_ids:
        tokenizer_json = _read_json(directory / "tokenizer.json")
        for item in tokenizer_json.get("added_tokens", []):
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                decoder[item["id"]] = item.get("content")
        vocabulary = tokenizer_json.get("model", {}).get("vocab", {})
        if isinstance(vocabulary, dict):
            for token, token_id in vocabulary.items():
                if isinstance(token, str) and isinstance(token_id, int):
                    decoder[token_id] = token
        elif isinstance(vocabulary, list):
            for token_id, item in enumerate(vocabulary):
                token = item[0] if isinstance(item, list) and item else item
                if isinstance(token, str):
                    decoder[token_id] = token
    tokens.extend(decoder.get(token_id) for token_id in eos_ids)
    return tuple(dict.fromkeys(token for token in tokens if token))


def _tokens_from_tokenizer(tokenizer):
    if tokenizer is None:
        return ()
    tokens = _token_strings(getattr(tokenizer, "eos_token", None))
    eos_ids = _token_ids(getattr(tokenizer, "eos_token_id", None))
    if eos_ids and hasattr(tokenizer, "convert_ids_to_tokens"):
        converted = tokenizer.convert_ids_to_tokens(eos_ids)
        tokens.extend(_token_strings(converted))
    return tuple(dict.fromkeys(token for token in tokens if token))


def resolve_eos_tokens(args=None, tokenizer=None) -> tuple[str, ...]:
    """Resolve EOS strings from an injected tokenizer or Miles checkpoint args."""
    tokens = list(_tokens_from_tokenizer(tokenizer))
    for attribute in ("tokenizer_path", "hf_checkpoint"):
        path = getattr(args, attribute, None) if args is not None else None
        if path:
            tokens.extend(eos_tokens_from_config(path))
    return tuple(dict.fromkeys(token for token in tokens if token))


def _strip_terminal_eos_tokens(response: str, eos_tokens=()) -> str:
    response = response.rstrip()
    while True:
        matching_tokens = [
            token for token in eos_tokens if token and response.endswith(token)
        ]
        if not matching_tokens:
            return response
        token = max(matching_tokens, key=len)
        response = response[: -len(token)].rstrip()


def _balanced_boxed_span(response: str, start: int) -> tuple[int, int] | None:
    r"""Return a ``\boxed{...}`` span at ``start``, including nested braces."""
    if not response.startswith(_BOXED_PREFIX, start):
        return None

    depth = 0
    for index in range(start + len(r"\boxed"), len(response)):
        character = response[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
            if depth < 0:
                return None
    return None


def strict_terminal_answer_error(response: str, eos_tokens=()) -> str | None:
    r"""Return why ``...\nAnswer: \boxed{...}`` is invalid, or ``None``."""
    response = _strip_terminal_eos_tokens(str(response or ""), eos_tokens)
    answer_markers = list(_ANSWER_LINE.finditer(response))
    if not answer_markers:
        return "missing_answer_marker"
    if len(answer_markers) > 1:
        return "multiple_answer_markers"

    marker = answer_markers[0]
    box_start = response.find(_BOXED_PREFIX, marker.end())
    if box_start < 0:
        return "missing_terminal_box"
    boxed_span = _balanced_boxed_span(response, box_start)
    if boxed_span is None:
        return "malformed_terminal_box"

    box_start, box_end = boxed_span
    if _TERMINAL_PREFIX.fullmatch(response[marker.end() : box_start]) is None:
        return "box_not_on_answer_line"
    if _TERMINAL_SUFFIX.fullmatch(response[box_end:]) is None:
        return "trailing_content"
    return None


def has_strict_terminal_answer(response: str, eos_tokens=()) -> bool:
    r"""Check ``...\nAnswer: \boxed{...}`` with nothing after the answer."""
    return strict_terminal_answer_error(response, eos_tokens) is None


def _status_value(sample) -> str | None:
    status = getattr(sample, "status", None)
    if status is None:
        return None
    value = getattr(status, "value", status)
    return str(value).lower()


def score_strict_math_sample(sample, eos_tokens=()) -> float:
    """Score one sample and attach fixed-cardinality math/format diagnostics."""
    response = str(sample.response or "")
    status_rejected = _status_value(sample) in _REJECTED_STATUSES
    format_error = strict_terminal_answer_error(response, eos_tokens)
    math_reward = float(bool(grade_answer_verl(response, sample.label)))
    format_valid = not status_rejected and format_error is None
    strict_reward = float(format_valid and bool(math_reward))

    metadata = getattr(sample, "metadata", None)
    if metadata is None:
        metadata = {}
        sample.metadata = metadata
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Sample metadata must be a mapping, got {type(metadata).__name__}")
    selection_metrics = metadata.setdefault(SELECTION_METRICS_KEY, {})
    if not isinstance(selection_metrics, dict):
        raise RuntimeError(
            f"{SELECTION_METRICS_KEY} metadata must be a mapping, "
            f"got {type(selection_metrics).__name__}"
        )
    selection_metrics.update(
        {
            f"{_STRICT_METRIC_PREFIX}math_reward": math_reward,
            f"{_STRICT_METRIC_PREFIX}strict_reward": strict_reward,
            f"{_STRICT_METRIC_PREFIX}format_valid": float(format_valid),
            f"{_STRICT_METRIC_PREFIX}multiple_answer_markers": float(
                format_error == "multiple_answer_markers"
            ),
            f"{_STRICT_METRIC_PREFIX}reward_disagreement": float(math_reward != strict_reward),
        }
    )
    return strict_reward


async def strict_math_reward(args, sample_or_samples, **kwargs):
    """Accept both Miles custom-RM contracts: one Sample or a list."""
    eos_tokens = resolve_eos_tokens(args, kwargs.get("tokenizer"))
    if isinstance(sample_or_samples, list):
        return [
            score_strict_math_sample(sample, eos_tokens)
            for sample in sample_or_samples
        ]
    return score_strict_math_sample(sample_or_samples, eos_tokens)


def build_preflight_probes(label, metadata):
    """Supply known-good and known-bad responses to verifier preflight."""
    del metadata
    label = str(label)
    if label.startswith(r"\boxed{") and label.endswith("}"):
        label = label[len(_BOXED_PREFIX) : -1]
    correct = f"Work.\n\nAnswer: \\boxed{{{label}}}"
    wrong = "Work.\n\nAnswer: \\boxed{-987654321}"
    return correct, wrong
