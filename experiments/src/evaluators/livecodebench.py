"""Miles reward wrapper around LiveCodeBench's pinned official evaluator.

LiveCodeBench is evaluation-only. Calling this reward requires the explicit
``LCB_ALLOW_LOCAL_EXECUTION=1`` acknowledgement and should happen on a
disposable CPU evaluation worker, not inside policy training.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import pickle
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

from experiments.src.environments.competitive_programming.verifier import extract_code

DEFAULT_REPO_PATH = Path("/data/livecodebench-code")
PINNED_REPO_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"


class _RestrictedUnpickler(pickle.Unpickler):
    """LCB private tests contain only JSON text; reject every pickle global."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"pickle global is forbidden: {module}.{name}")


def decode_private_tests(value: str) -> list[dict[str, Any]]:
    """Decode either plain JSON or LCB's base64+zlib+pickle JSON payload."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        compressed = base64.b64decode(value.encode("utf-8"), validate=True)
        serialized = zlib.decompress(compressed)
        decoded = _RestrictedUnpickler(io.BytesIO(serialized)).load()
        if isinstance(decoded, bytes):
            decoded = decoded.decode("utf-8")
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
    if not isinstance(decoded, list) or not all(isinstance(test, dict) for test in decoded):
        raise ValueError("LiveCodeBench private tests are not a list of objects")
    return decoded


def _evaluation_sample(metadata: dict[str, Any]) -> dict[str, str]:
    public = json.loads(metadata["public_test_cases"])
    private = decode_private_tests(metadata["private_test_cases"])
    if not isinstance(public, list):
        raise ValueError("LiveCodeBench public tests are not a list")
    lcb_metadata = metadata.get("lcb_metadata") or "{}"
    lcb_metadata = json.loads(lcb_metadata) if isinstance(lcb_metadata, str) else lcb_metadata
    test_cases = [*public, *private]
    payload = {
        "inputs": [test["input"] for test in test_cases],
        "outputs": [test["output"] for test in test_cases],
        "fn_name": (lcb_metadata or {}).get("func_name"),
    }
    return {"input_output": json.dumps(payload)}


def _load_official_metrics():
    repo_path = Path(os.environ.get("LCB_REPO_PATH", str(DEFAULT_REPO_PATH)))
    module_path = repo_path / "lcb_runner" / "evaluation" / "compute_code_generation_metrics.py"
    if not module_path.is_file():
        raise ImportError(
            f"official LiveCodeBench evaluator not found at {repo_path}; stage the pinned repository "
            "or set LCB_REPO_PATH"
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ImportError(f"could not validate official LiveCodeBench checkout: {repo_path}") from error
    if revision != PINNED_REPO_COMMIT:
        raise ImportError(
            f"LiveCodeBench checkout is at {revision}, expected pinned commit {PINNED_REPO_COMMIT}"
        )
    if status:
        raise ImportError(f"LiveCodeBench checkout has uncommitted files: {repo_path}")
    repo_string = str(repo_path)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    return codegen_metrics


def _score_samples(samples: list[Any]) -> list[float]:
    if os.environ.get("LCB_ALLOW_LOCAL_EXECUTION") != "1":
        raise RuntimeError(
            "LiveCodeBench executes generated code. Run on a disposable CPU worker and set "
            "LCB_ALLOW_LOCAL_EXECUTION=1; never enable it in policy training."
        )
    evaluation_samples = []
    generations = []
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        if metadata.get("eval_only") is not True or metadata.get("verifier") != "livecodebench":
            raise ValueError("LiveCodeBench reward accepts only rows explicitly marked eval_only")
        evaluation_samples.append(_evaluation_sample(metadata))
        generations.append([extract_code(sample.response)])
    metrics = _load_official_metrics()
    _, results, _ = metrics(
        evaluation_samples,
        generations,
        k_list=[1],
        num_process_evaluate=min(int(os.environ.get("LCB_EVAL_PROCESSES", "8")), len(samples)),
        timeout=int(os.environ.get("LCB_TEST_TIMEOUT", "6")),
        debug=False,
    )
    rewards = []
    for index in range(len(samples)):
        test_results = results[index][0]
        rewards.append(1.0 if test_results and all(result is True for result in test_results) else 0.0)
    return rewards


async def livecodebench_reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    samples = sample_or_samples if isinstance(sample_or_samples, list) else [sample_or_samples]
    rewards = await asyncio.to_thread(_score_samples, samples)
    return rewards if isinstance(sample_or_samples, list) else rewards[0]


def build_preflight_probes(label: Any, metadata: dict[str, Any]) -> None:
    return None
