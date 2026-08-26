"""Apply an admitted model patch, then the trusted hidden SWE-Gym patch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class InfrastructureError(RuntimeError):
    """The trusted base image or hidden patch is invalid."""


class RejectedPatch(ValueError):
    """The model patch does not apply to the admitted base."""


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_COUNT": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "HOME": "/opt/miles-swe/root-home",
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "XDG_CONFIG_HOME": "/opt/miles-swe/root-home/xdg",
    }


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repo),
            *arguments,
        ],
        env=_git_environment(),
        capture_output=True,
        check=False,
    )


def _require_git(repo: Path, *arguments: str, reason: str) -> None:
    completed = _git(repo, *arguments)
    if completed.returncode != 0:
        raise InfrastructureError(reason)


def prepare(repo: Path, model_patch: Path, test_patch: Path, base_commit: str) -> None:
    """Reset and apply patches in the only order that hides tests from the agent."""
    _require_git(repo, "reset", "--hard", base_commit, reason="cannot reset verifier base")
    hidden_check = _git(repo, "apply", "--check", "--binary", str(test_patch))
    if hidden_check.returncode != 0:
        raise InfrastructureError("hidden SWE-Gym patch does not apply to the pristine base")
    if model_patch.stat().st_size:
        model_check = _git(repo, "apply", "--check", "--binary", str(model_patch))
        if model_check.returncode != 0:
            raise RejectedPatch("model patch does not apply to the admitted base")
        _require_git(
            repo,
            "apply",
            "--binary",
            str(model_patch),
            reason="model patch changed after its applicability check",
        )
    hidden_after_model = _git(repo, "apply", "--check", "--binary", str(test_patch))
    if hidden_after_model.returncode != 0:
        raise RejectedPatch("model patch conflicts with isolated hidden tests")
    _require_git(
        repo,
        "apply",
        "--binary",
        str(test_patch),
        reason="hidden SWE-Gym patch changed after its applicability check",
    )


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: swe_gym_prepare.py REPO MODEL_PATCH TEST_PATCH BASE", file=sys.stderr)
        return 2
    try:
        prepare(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    except RejectedPatch as exc:
        print(f"SWE-Gym model patch rejected: {exc}", file=sys.stderr)
        return 10
    except (InfrastructureError, OSError) as exc:
        print(f"SWE-Gym patch preparation infrastructure error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
