"""Pinned official Reasoning Gym environment scoring for NVIDIA's task blend."""

from __future__ import annotations

import importlib
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

DEFAULT_DEPS_PATH = Path("/data/reasoning-gym-deps")

_LOCK = threading.Lock()
_PACKAGE = None
_EXTRACT_ANSWER = None


def _load_package():
    global _EXTRACT_ANSWER, _PACKAGE
    if os.environ.get("REASONING_GYM_ALLOW_EXACT_FALLBACK") == "1":
        return None, None
    if _PACKAGE is not None:
        return _PACKAGE, _EXTRACT_ANSWER
    with _LOCK:
        if _PACKAGE is not None:
            return _PACKAGE, _EXTRACT_ANSWER
        deps_path = Path(os.environ.get("REASONING_GYM_DEPS_PATH", str(DEFAULT_DEPS_PATH)))
        if deps_path.is_dir() and str(deps_path) not in sys.path:
            # Keep the image's CUDA-tested NumPy/SymPy stack authoritative; the
            # staged directory supplies Reasoning Gym and packages that are absent.
            sys.path.append(str(deps_path))
        try:
            _PACKAGE = importlib.import_module("reasoning_gym")
            _EXTRACT_ANSWER = importlib.import_module("reasoning_gym.utils").extract_answer
        except (AttributeError, ModuleNotFoundError) as exc:
            raise ImportError(
                "the official Reasoning Gym verifier is unavailable; run "
                "experiments/setup/environments/prepare_reasoning_gym_dependencies.sbatch "
                "or set "
                "REASONING_GYM_DEPS_PATH"
            ) from exc
        return _PACKAGE, _EXTRACT_ANSWER


def _normalize(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^\s*(?:final\s+)?answer\s*[:\-]\s*", "", text)
    text = text.strip().strip("`*_ \t\n")
    boxed = re.fullmatch(r"\\boxed\{(.*)\}", text, flags=re.DOTALL)
    if boxed:
        text = boxed.group(1)
    return re.sub(r"\s+", " ", text).strip()


def _fallback_exact(sample: Any) -> float:
    expected = _normalize(sample.label)
    response = str(sample.response or "")
    if not expected or not response.strip():
        return 0.0
    candidates = [next((line for line in reversed(response.splitlines()) if line.strip()), "")]
    candidates.extend(re.findall(r"\\boxed\{([^{}]*)\}", response))
    markers = re.findall(r"(?:final\s+)?answer\s*[:\-]\s*([^\n]+)", response, flags=re.IGNORECASE)
    candidates.extend(markers[-1:])
    return 1.0 if any(_normalize(candidate) == expected for candidate in candidates) else 0.0


def _extract_response(response: str, extract_answer) -> str:
    extracted = extract_answer(response, tag_name="answer")
    if extracted is not None:
        return extracted
    boxed = re.search(r"\\boxed\{([^}]+)\}", response)
    return boxed.group(1).strip() if boxed else response.strip()


def score_reasoning_gym_sample(sample: Any) -> float:
    """Use the task-specific scorer, including non-unique planning answers."""
    package, extract_answer = _load_package()
    if package is None:
        return _fallback_exact(sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_name = metadata.get("source_dataset")
    entry_metadata = metadata.get("verifier_metadata") or {}
    question = str(metadata.get("question") or "")
    if not task_name or not question:
        return 0.0
    model_answer = _extract_response(str(sample.response or ""), extract_answer)
    entry = {
        "question": question,
        "answer": metadata.get("reference_answer"),
        "metadata": entry_metadata,
    }
    try:
        score_fn = package.get_score_answer_fn(str(task_name))
        return float(score_fn(answer=model_answer, entry=entry))
    except Exception:  # noqa: BLE001 - a malformed sample must score zero, not stop training
        return 0.0


async def reasoning_gym_reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    """Expose the Miles scalar/batch reward contract for this environment."""

    if isinstance(sample_or_samples, list):
        return [score_reasoning_gym_sample(sample) for sample in sample_or_samples]
    return score_reasoning_gym_sample(sample_or_samples)


def build_preflight_probes(label: Any, metadata: dict[str, Any]) -> tuple[str, str] | None:
    if label in (None, ""):
        return None
    return f"Answer: {label}", "Answer: definitely-wrong"
