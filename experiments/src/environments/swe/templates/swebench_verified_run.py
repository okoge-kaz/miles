"""Render the pinned official SWE-bench v2.0.13 test command."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, NON_TEST_EXTS


def _test_directives(repo: str, test_patch: str) -> list[str]:
    if repo == "swe-bench/humaneval":
        return ["test.py"]
    directives = re.findall(r"^diff --git a/.* b/(.*)$", test_patch, re.MULTILINE)
    directives = [
        directive
        for directive in directives
        if not any(directive.endswith(extension) for extension in NON_TEST_EXTS)
    ]
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


def _test_command(
    config: dict[str, Any],
    test_patch: str,
) -> tuple[dict[str, Any], str]:
    repo = str(config.get("repo") or "").lower()
    version = str(config.get("version") or "")
    try:
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    except KeyError as exc:
        raise ValueError(
            f"unsupported pinned SWE-bench repo/version: {repo}@{version}"
        ) from exc
    command = specs.get("test_cmd")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"pinned SWE-bench test command is absent for {repo}@{version}")
    directives = _test_directives(repo, test_patch)
    if not directives:
        raise ValueError("Verified hidden patch contains no official test directives")
    targets = " ".join(shlex.quote(directive) for directive in directives)
    return specs, f"{command} {targets}"


def _script(config: dict[str, Any], test_patch: str, repo: Path) -> str:
    specs, test_command = _test_command(config, test_patch)
    eval_commands = specs.get("eval_commands", [])
    if not isinstance(eval_commands, list) or any(
        not isinstance(command, str) or not command.strip()
        for command in eval_commands
    ):
        raise ValueError("pinned SWE-bench eval_commands are invalid")
    install = specs.get("install", "")
    if not isinstance(install, str):
        raise ValueError("pinned SWE-bench install command is invalid")
    commands = [
        "set -uo pipefail",
        f"cd -- {shlex.quote(str(repo))}",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        *eval_commands,
    ]
    if install.strip():
        commands.append(install)
    commands.append(test_command)
    return "\n".join(commands)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: swebench_verified_run.py CONFIG TEST_PATCH REPO", file=sys.stderr)
        return 2
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SWE-bench verifier config must be an object")
    test_patch = Path(sys.argv[2]).read_text(encoding="utf-8")
    print(_script(config, test_patch, Path(sys.argv[3])))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SWE-bench test-runner infrastructure error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
