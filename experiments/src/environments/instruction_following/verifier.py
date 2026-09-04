"""Official Open-Instruct IFEvalG instruction-following verifier."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REPO_PATH = Path("/data/open-instruct")
DEFAULT_DEPS_PATH = Path("/data/open-instruct-deps")
MODULE_NAME = "open_instruct.IFEvalG.instructions_registry"

_LOCK = threading.Lock()
_REGISTRY = None

# The original Nemotron-RL-instruction_following release stores these two
# scalar keyword arguments as singleton lists in every affected row.  NVIDIA's
# later curated release calls out this exact count_increment_word formatting
# inconsistency.  Keep the compatibility rule constraint-specific: other
# IFEvalG arguments such as ``keywords`` and ``forbidden_words`` are real lists.
_SINGLETON_LIST_ARGUMENTS = {
    "count:count_increment_word": frozenset({"keyword1", "keyword2"}),
}


def _load_registry():
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    with _LOCK:
        if _REGISTRY is not None:
            return _REGISTRY
        repo_path = Path(os.environ.get("OPEN_INSTRUCT_PATH", str(DEFAULT_REPO_PATH)))
        deps_path = Path(os.environ.get("OPEN_INSTRUCT_DEPS_PATH", str(DEFAULT_DEPS_PATH)))
        if not (repo_path / "open_instruct" / "IFEvalG" / "instructions_registry.py").is_file():
            raise ImportError(
                f"IFEvalG registry not found at {repo_path}. Run stage_nemotron_rl_datasets.sh "
                "or set OPEN_INSTRUCT_PATH to a pinned allenai/open-instruct checkout."
            )
        repo_string = str(repo_path)
        if deps_path.is_dir():
            deps_string = str(deps_path)
            if deps_string not in sys.path:
                # Prefer the image's tested dependency set; this directory fills
                # only packages the immutable image does not carry.
                sys.path.append(deps_string)
            os.environ.setdefault("NLTK_DATA", str(deps_path / "nltk_data"))
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        try:
            _REGISTRY = importlib.import_module(MODULE_NAME)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"IFEvalG dependency {exc.name!r} is unavailable. Run "
                "experiments/setup/environments/prepare_ifeval_dependencies.sbatch or set "
                "OPEN_INSTRUCT_DEPS_PATH to its pinned Python dependency directory."
            ) from exc
        logger.info("loaded %d IFEvalG instruction ids", len(_REGISTRY.INSTRUCTION_DICT))
        return _REGISTRY


def _normalize_checker_kwargs(instruction_id: str, checker_kwargs: Any) -> dict[str, Any]:
    if checker_kwargs is None:
        normalized = {}
    elif isinstance(checker_kwargs, dict):
        normalized = {key: value for key, value in checker_kwargs.items() if value is not None}
    else:
        raise TypeError(f"IFEvalG kwargs for {instruction_id} must be a mapping or null")

    for key in _SINGLETON_LIST_ARGUMENTS.get(instruction_id, ()):
        value = normalized.get(key)
        if not isinstance(value, list):
            continue
        if len(value) != 1 or not isinstance(value[0], str) or not value[0].strip():
            raise ValueError(f"IFEvalG {instruction_id}.{key} must be a non-empty singleton string list")
        normalized[key] = value[0]
    return normalized


def _build_checkers(metadata: dict[str, Any]) -> list[Any]:
    instruction_ids = metadata.get("instruction_id_list") or []
    kwargs_list = metadata.get("kwargs") or []
    if not isinstance(instruction_ids, list) or not instruction_ids:
        raise ValueError("IFEvalG instruction_id_list must be a non-empty list")
    if not isinstance(kwargs_list, list) or len(kwargs_list) != len(instruction_ids):
        raise ValueError("IFEvalG kwargs must have one entry per instruction id")

    prompt_text = str(metadata.get("prompt_text") or "")
    registry = _load_registry()
    checkers = []
    for index, instruction_id in enumerate(instruction_ids):
        checker_class = registry.INSTRUCTION_DICT.get(instruction_id)
        if checker_class is None:
            raise KeyError(f"unknown IFEvalG instruction id {instruction_id}")
        checker = checker_class(instruction_id)
        checker_kwargs = _normalize_checker_kwargs(instruction_id, kwargs_list[index])
        argument_keys = checker.get_instruction_args_keys() if hasattr(checker, "get_instruction_args_keys") else []
        if "prompt" in (argument_keys or []) and "prompt" not in checker_kwargs:
            checker_kwargs["prompt"] = prompt_text
        checker.build_description(**checker_kwargs)
        checkers.append(checker)
    return checkers


def validate_ifeval_metadata(metadata: dict[str, Any]) -> int:
    """Construct every checker for one row, raising on incompatible metadata."""
    return len(_build_checkers(metadata))


def _remove_thinking_section(response: str) -> str:
    """Match Open-Instruct's IFEvalVerifier response normalization."""
    response = response.replace("<|assistant|>", "").strip()
    response = response.split("</think>")[-1]
    response = response.replace("<answer>", "").replace("</answer>", "")
    return response.strip()


def score_ifeval_sample(sample: Any) -> float:
    """Apply the pinned Open-Instruct per-constraint mean reward."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    instruction_ids = metadata.get("instruction_id_list") or []
    response = _remove_thinking_section(str(sample.response or ""))
    if not instruction_ids or not response.strip():
        return 0.0
    try:
        checkers = _build_checkers(metadata)
    except Exception as exc:  # noqa: BLE001 - malformed metadata must not stop training
        logger.warning("IFEvalG checker construction raised %s", type(exc).__name__)
        return 0.0
    rewards = []
    for checker in checkers:
        try:
            rewards.append(float(bool(checker.check_following(response))))
        except Exception as exc:  # noqa: BLE001 - a malformed row must not stop training
            logger.warning("IFEvalG checker %s raised %s", checker.id, type(exc).__name__)
            return 0.0
    return sum(rewards) / len(rewards)


async def ifeval_reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    if isinstance(sample_or_samples, list):
        return [score_ifeval_sample(sample) for sample in sample_or_samples]
    return score_ifeval_sample(sample_or_samples)


def build_preflight_probes(label: Any, metadata: dict[str, Any]) -> None:
    """Arbitrary combined constraints do not have a generic correct response."""
    return None
