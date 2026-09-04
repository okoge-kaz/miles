from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_IFBENCH_REPO = Path("/data/ifbench")
DEFAULT_IFBENCH_DEPS = Path("/data/ifbench-deps")
PINNED_IFBENCH_COMMIT = "db69a6f05689830b0068b8f1529ebcfd2f3b164c"
PINNED_DEPS_MARKER = ".miles-ifbench-0.2.0-db69a6f"


def _ensure_ifbench_repo() -> Path:
    """Validate and expose the staged IFBench checkout without mutating it."""

    repo_path = Path(os.environ.get("IFBENCH_REPO_PATH", str(DEFAULT_IFBENCH_REPO)))
    if not (repo_path / "evaluation_lib.py").is_file():
        raise ImportError(
            f"IFBench is not staged at {repo_path}. Run "
            "experiments/setup/environments/prepare_ifbench.sbatch or set "
            "IFBENCH_REPO_PATH."
        )

    deps_path = Path(os.environ.get("IFBENCH_DEPS_PATH", str(DEFAULT_IFBENCH_DEPS)))
    marker_path = deps_path / PINNED_DEPS_MARKER
    nltk_data_path = deps_path / "nltk_data"
    if not marker_path.is_file() or not nltk_data_path.is_dir():
        raise ImportError(
            f"pinned IFBench dependencies are not staged at {deps_path}. Run "
            "experiments/setup/environments/prepare_ifbench.sbatch or set "
            "IFBENCH_DEPS_PATH."
        )

    try:
        revision = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ImportError(f"could not validate official IFBench checkout: {repo_path}") from exc
    if revision != PINNED_IFBENCH_COMMIT:
        raise ImportError(
            f"IFBench checkout is at {revision}, expected pinned commit {PINNED_IFBENCH_COMMIT}"
        )
    if tracked_status:
        raise ImportError(f"IFBench checkout has modified tracked files: {repo_path}")

    deps_str = str(deps_path)
    if deps_str not in sys.path:
        sys.path.append(deps_str)
    os.environ.setdefault("NLTK_DATA", str(nltk_data_path))

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath is None:
        os.environ["PYTHONPATH"] = repo_str
    elif repo_str not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str, current_pythonpath])

    return repo_path


@cache
def _load_evaluation_lib():
    repo_path = _ensure_ifbench_repo()
    try:
        module = importlib.import_module("evaluation_lib")
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"IFBench dependency {exc.name!r} is unavailable. Run "
            "experiments/setup/environments/prepare_ifbench.sbatch or set "
            "IFBENCH_DEPS_PATH."
        ) from exc
    module_path = Path(module.__file__).resolve()
    if module_path != (repo_path / "evaluation_lib.py").resolve():
        raise ImportError(f"loaded IFBench evaluator from unexpected path: {module_path}")
    return module


JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    """Ensure instruction identifiers are clean strings."""

    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_kwargs_list(
    raw_kwargs: Any,
    num_instructions: int,
) -> list[KwargsDict]:
    """Convert stored kwargs into the list structure expected by IFBench."""

    if isinstance(raw_kwargs, list):
        processed: list[KwargsDict] = []
        for entry in raw_kwargs:
            if isinstance(entry, dict):
                processed.append(dict(entry))
            else:
                processed.append({})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    # Remove explicit None values to match official preprocessing.
    sanitized: list[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized


def _build_input_example(metadata: JsonDict, evaluation_lib: Any | None = None) -> Any | None:
    if evaluation_lib is None:
        evaluation_lib = _load_evaluation_lib()
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list") or [])
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    prompt_text = metadata.get("prompt_text")
    if prompt_text is None:
        prompt_text = ""
    else:
        prompt_text = str(prompt_text)

    raw_kwargs = metadata.get("kwargs")
    kwargs_list = _coerce_kwargs_list(raw_kwargs, len(instruction_ids))

    return evaluation_lib.InputExample(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def compute_ifbench_reward(response: str, label: Any, metadata: JsonDict | None = None) -> float:
    """Score a model response using the official IFBench rules."""

    if metadata is None:
        logger.debug("No metadata provided for IFBench scoring.")
        return 0.0

    if response is None:
        return 0.0

    evaluation_lib = _load_evaluation_lib()
    inp = _build_input_example(metadata, evaluation_lib=evaluation_lib)
    if inp is None:
        return 0.0

    prompt_to_response = {inp.prompt: str(response or "")}
    output = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    return 1.0 if output.follow_all_instructions else 0.0
