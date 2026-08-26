"""Load and execute Workplace resource modules without serving NeMo Gym."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

WORKPLACE_RESOURCE_COMMIT = "48d5b9c01e3fc59a49f19674d0034a6f06396074"
DEFAULT_RESOURCE_ROOT = Path(os.environ.get("WORKPLACE_RESOURCE_ROOT", "/data/nemo-gym-code"))
_TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]


@lru_cache(maxsize=1)
def _load_resource_functions() -> tuple[Any, Any]:
    root = DEFAULT_RESOURCE_ROOT
    expected = root / "resources_servers" / "workplace_assistant" / "utils.py"
    if not expected.is_file():
        raise RuntimeError(f"standalone Workplace resource modules are missing: {expected}")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if commit != WORKPLACE_RESOURCE_COMMIT:
        raise RuntimeError(
            f"Workplace resource commit is {commit}, expected {WORKPLACE_RESOURCE_COMMIT}"
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("resources_servers.workplace_assistant.utils")
    return module.get_tools, module.is_correct


def create_tool_environment() -> dict[str, Any]:
    """Create one isolated in-process environment from pinned resource modules."""

    get_tools, _ = _load_resource_functions()
    return get_tools(_TOOLKITS)


def execute_action(environment: dict[str, Any], name: str, arguments: dict[str, Any]) -> Any:
    """Execute one tool call and return an observation, including recoverable errors."""

    function = environment["functions"].get(name)
    if function is None:
        return f"Error executing tool {name!r}: unknown tool"
    try:
        return function(**arguments)
    except Exception as exc:  # noqa: BLE001 - tool errors are observations for self-correction
        return f"Error executing tool {name!r}: {exc}"
