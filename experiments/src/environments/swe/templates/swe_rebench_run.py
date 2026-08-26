"""Execute canonical SWE-rebench test commands sequentially."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any


def _commands(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("test_commands must be text or a list")
    commands = [item for item in value if isinstance(item, str) and item.strip()]
    if not commands:
        raise ValueError("test_commands must contain at least one command")
    return commands


def main() -> int:
    repo = Path(sys.argv[1])
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd -- {shlex.quote(str(repo))}",
    ]
    for command in _commands(config.get("test_commands")):
        lines.append(f"/bin/bash --noprofile --norc -c {shlex.quote(command)}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SWE-rebench test-runner infrastructure error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
