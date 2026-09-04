from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LOADER = REPO_ROOT / "experiments/common/load_job_env.sh"


def _run_loader(
    env_file: Path,
    *keys: str,
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = """
set -euo pipefail
source "$1"
shift
env_file=$1
shift
load_job_env "$env_file" "$@"
printf 'key=%s\\nmodel=%s\\n' "${NVIDIA_INFERENCE_API_KEY:-}" "${TAU_USER_MODEL:-}"
"""
    environment = dict(os.environ)
    environment.pop("NVIDIA_INFERENCE_API_KEY", None)
    environment.pop("TAU_USER_MODEL", None)
    environment.update(environment_overrides or {})
    return subprocess.run(
        ["bash", "-c", command, "_", str(LOADER), str(env_file), *keys],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_job_dotenv_loader_reads_only_allowlisted_values_without_evaluation(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# job credentials\n"
        "export NVIDIA_INFERENCE_API_KEY='fake-secret'\n"
        'TAU_USER_MODEL="google/example-model"\n'
        f"GEMINI_API_KEY='$(touch {marker})'\n"
        f"UNRELATED=$(touch {marker})\n",
        encoding="utf-8",
    )

    result = _run_loader(
        env_file,
        "NVIDIA_INFERENCE_API_KEY",
        "TAU_USER_MODEL",
        "GEMINI_API_KEY",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "key=fake-secret\nmodel=google/example-model\n"
    assert not marker.exists()


def test_job_dotenv_loader_preserves_explicit_job_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NVIDIA_INFERENCE_API_KEY=dotenv-secret\n"
        "TAU_USER_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    result = _run_loader(
        env_file,
        "NVIDIA_INFERENCE_API_KEY",
        "TAU_USER_MODEL",
        environment_overrides={
            "NVIDIA_INFERENCE_API_KEY": "scheduler-secret",
            "TAU_USER_MODEL": "scheduler-model",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "key=scheduler-secret\nmodel=scheduler-model\n"


def test_job_dotenv_loader_rejects_malformed_or_ambiguous_values(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.env"
    malformed.write_text("not an assignment\n", encoding="utf-8")
    whitespace = tmp_path / "whitespace.env"
    whitespace.write_text("TAU_USER_MODEL=unquoted value\n", encoding="utf-8")

    malformed_result = _run_loader(malformed, "TAU_USER_MODEL")
    whitespace_result = _run_loader(whitespace, "TAU_USER_MODEL")

    assert malformed_result.returncode != 0
    assert "invalid dotenv assignment" in malformed_result.stderr
    assert whitespace_result.returncode != 0
    assert "unquoted whitespace" in whitespace_result.stderr
