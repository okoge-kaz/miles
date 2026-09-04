"""Run the pinned SWE-Gym repository/version test command as UID 1000."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from swegym.harness.constants import MAP_REPO_VERSION_TO_SPECS, NON_TEST_EXTS


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value


def _test_directives(repo: str, test_patch: str) -> list[str]:
    if repo == "swe-bench/humaneval":
        return ["test.py"]
    directives = re.findall(r"^diff --git a/.* b/(.*)$", test_patch, re.MULTILINE)
    directives = [directive for directive in directives if not any(directive.endswith(extension) for extension in NON_TEST_EXTS)]
    if repo == "django/django":
        transformed = []
        for directive in directives:
            if directive.endswith(".py"):
                directive = directive[:-3]
            if directive.startswith("tests/"):
                directive = directive[len("tests/") :]
            transformed.append(directive.replace("/", "."))
        directives = transformed
    return directives


def _test_command(config: dict[str, Any], test_patch: str) -> tuple[dict[str, Any], str]:
    repo = str(config.get("repo") or "").lower()
    version = str(config.get("version") or "")
    try:
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    except KeyError as exc:
        raise ValueError(f"unsupported pinned SWE-Gym repo/version: {repo}@{version}") from exc
    command = specs.get("test_cmd")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"pinned SWE-Gym test command is absent for {repo}@{version}")
    if repo == "python/mypy":
        cases = re.findall(r"\[case ([^\]]+)\]", test_patch)
        if not cases:
            raise ValueError("SWE-Gym mypy patch contains no official case directives")
        target = " or ".join(cases)
        return specs, f"{command} {shlex.quote(target)}"
    directives = _test_directives(repo, test_patch)
    if not directives:
        raise ValueError("SWE-Gym hidden patch contains no official test directives")
    targets = " ".join(shlex.quote(directive) for directive in directives)
    return specs, f"{command} {targets}"


def _public_baseline_command(
    config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    repo = str(config.get("repo") or "").lower()
    version = str(config.get("version") or "")
    try:
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    except KeyError as exc:
        raise ValueError(
            f"unsupported pinned SWE-Gym repo/version: {repo}@{version}"
        ) from exc
    command = specs.get("test_cmd")
    pass_to_pass = _string_list(config.get("pass_to_pass"), "pass_to_pass")
    if not isinstance(command, str) or not command.strip() or not pass_to_pass:
        raise ValueError("pinned SWE-Gym public baseline is absent")
    if repo == "python/mypy":
        target = " or ".join(pass_to_pass)
        return specs, f"{command} {shlex.quote(target)}"
    targets = " ".join(shlex.quote(test_id) for test_id in pass_to_pass)
    return specs, f"{command} {targets}"


def _script_from_command(
    specs: dict[str, Any],
    test_command: str,
    repo: Path,
    *,
    include_install: bool,
) -> str:
    eval_commands = specs.get("eval_commands", [])
    if not isinstance(eval_commands, list) or any(not isinstance(command, str) or not command.strip() for command in eval_commands):
        raise ValueError("pinned SWE-Gym eval_commands are invalid")
    install = specs.get("install", "")
    if not isinstance(install, str):
        raise ValueError("pinned SWE-Gym install command is invalid")
    commands = [
        "set -uo pipefail",
        f"cd -- {shlex.quote(str(repo))}",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        *eval_commands,
    ]
    if include_install and install.strip():
        commands.append(install)
    commands.append(test_command)
    return "\n".join(commands)


def _script(config: dict[str, Any], test_patch: str, repo: Path) -> str:
    specs, test_command = _test_command(config, test_patch)
    return _script_from_command(
        specs,
        test_command,
        repo,
        include_install=True,
    )


def _public_baseline_script(config: dict[str, Any], repo: Path) -> str:
    specs, test_command = _public_baseline_command(config)
    return _script_from_command(
        specs,
        test_command,
        repo,
        include_install=False,
    )


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--public-baseline":
        config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("SWE-Gym public baseline config must be an object")
        print(_public_baseline_script(config, Path(sys.argv[3])))
        return 0
    if len(sys.argv) != 4:
        print(
            "usage: swe_gym_run.py CONFIG TEST_PATCH REPO | "
            "swe_gym_run.py --public-baseline CONFIG REPO",
            file=sys.stderr,
        )
        return 2
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SWE-Gym verifier config must be an object")
    test_patch = Path(sys.argv[2]).read_text(encoding="utf-8")
    print(_script(config, test_patch, Path(sys.argv[3])))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SWE-Gym test-runner infrastructure error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
