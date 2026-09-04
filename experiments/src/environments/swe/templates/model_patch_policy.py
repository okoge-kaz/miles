"""Validate an untrusted model patch against a private oracle path policy."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "miles-swe-model-path-policy-v1"
_EVAL_SCHEMA_VERSION = "miles-swe-model-path-policy-v2"
_REGULAR_MODES = {b"100644", b"100755"}
_MODE_LINE = re.compile(rb"^(?:new file mode|deleted file mode) ([0-7]{6})$")
_UNSUPPORTED_PREFIXES = (
    b"rename from ",
    b"rename to ",
    b"copy from ",
    b"copy to ",
    b"old mode ",
    b"new mode ",
)
_EVAL_DENIED_BASENAMES = {
    ".gitattributes",
    ".gitmodules",
    "dockerfile",
    "makefile",
    "conftest.py",
    "environment.yml",
    "package-lock.json",
    "package.json",
    "poetry.lock",
    "pyproject.toml",
    "pytest.ini",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "sitecustomize.py",
    "tox.ini",
    "usercustomize.py",
    "uv.lock",
}
_EVAL_DENIED_COMPONENTS = {
    ".circleci",
    ".devcontainer",
    ".github",
    ".gitlab",
    "test",
    "tests",
    "testing",
}


class InfrastructureError(RuntimeError):
    """Trusted verifier configuration or Git execution failed."""


class RejectedPatch(ValueError):
    """The model patch violates the private policy."""


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise InfrastructureError(f"{name} must be a list of non-empty strings")
    return value


def _validate_path(path: str) -> None:
    if path.startswith("/") or "\\" in path or "\0" in path or "\n" in path or any(component in {"", ".", ".."} for component in path.split("/")):
        raise RejectedPatch(f"unsafe patch path: {path!r}")


def _validate_patch_features(content: bytes) -> None:
    for line in content.splitlines():
        if line == b"GIT binary patch" or line.startswith((b"Binary files ", b"literal ", b"delta ")):
            raise RejectedPatch("binary patches are not supported")
        if line.startswith(_UNSUPPORTED_PREFIXES):
            raise RejectedPatch("rename/copy/mode-changing patches are not supported")
        mode = _MODE_LINE.fullmatch(line)
        if mode is not None and mode.group(1) not in _REGULAR_MODES:
            raise RejectedPatch("symlink/submodule/non-regular patch paths are not supported")


def _eval_path_is_denied(path: str) -> bool:
    components = path.lower().split("/")
    basename = components[-1]
    return (
        any(component in _EVAL_DENIED_COMPONENTS for component in components)
        or basename in _EVAL_DENIED_BASENAMES
        or basename.startswith("test_")
        or basename.endswith(("_test.py", ".ini", ".toml", ".yaml", ".yml"))
    )


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


def _patch_paths(repo: Path, patch: Path) -> set[str]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repo),
            "apply",
            "--numstat",
            "-z",
            "--binary",
            str(patch),
        ],
        env=_git_environment(),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RejectedPatch("model patch is not a parseable Git patch")
    paths: set[str] = set()
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            raise RejectedPatch("model patch has unsupported rename/path metadata")
        try:
            path = fields[2].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RejectedPatch("model patch contains a non-UTF-8 path") from exc
        _validate_path(path)
        paths.add(path)
    return paths


def validate(repo: Path, patch: Path, policy_path: Path) -> set[str]:
    """Return validated model paths or raise a typed failure."""
    if not patch.is_file() or patch.is_symlink():
        raise InfrastructureError("model patch artifact is absent or not a regular file")
    if not policy_path.is_file() or policy_path.is_symlink():
        raise InfrastructureError("private model path policy is absent")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InfrastructureError("private model path policy is invalid") from exc
    if not isinstance(policy, dict) or policy.get("schema_version") not in {
        _SCHEMA_VERSION,
        _EVAL_SCHEMA_VERSION,
    }:
        raise InfrastructureError("private model path policy schema is invalid")
    schema = policy["schema_version"]
    if schema == _SCHEMA_VERSION:
        allowed = set(_string_list(policy.get("allowed_paths"), "allowed_paths"))
        if not allowed:
            raise InfrastructureError("private model path policy has no allowed paths")
    else:
        if (
            policy.get("policy_mode") != "deny-sensitive-paths"
            or set(
                _string_list(policy.get("denied_basenames"), "denied_basenames")
            )
            != _EVAL_DENIED_BASENAMES
            or set(
                _string_list(policy.get("denied_components"), "denied_components")
            )
            != _EVAL_DENIED_COMPONENTS
            or policy.get("deny_test_name_patterns") is not True
        ):
            raise InfrastructureError("private evaluation path policy is incomplete")
        allowed = set()
    content = patch.read_bytes()
    _validate_patch_features(content)
    paths = _patch_paths(repo, patch)
    disallowed = (
        sorted(paths - allowed)
        if schema == _SCHEMA_VERSION
        else sorted(path for path in paths if _eval_path_is_denied(path))
    )
    if disallowed:
        raise RejectedPatch(f"model patch touches paths outside the private policy: {disallowed}")
    return paths


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: model_patch_policy.py REPO PATCH POLICY", file=sys.stderr)
        return 2
    try:
        paths = validate(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    except RejectedPatch as exc:
        print(f"model patch rejected: {exc}", file=sys.stderr)
        return 10
    except (InfrastructureError, OSError) as exc:
        print(f"model path-policy infrastructure error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"validated_model_paths": sorted(paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
