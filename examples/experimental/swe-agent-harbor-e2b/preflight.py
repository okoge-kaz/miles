#!/usr/bin/env python3
"""Offline compatibility check for the Harbor E2B agent server.

This imports SDK types but never creates a template or sandbox.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from miles.rollout.harbor.environment_config import build_harbor_environment_config

_REQUIRED_METHODS = (
    "start",
    "stop",
    "exec",
    "upload_file",
    "upload_dir",
    "download_file",
    "download_dir",
)


def _missing_methods(environment_class: type[Any]) -> tuple[str, ...]:
    return tuple(name for name in _REQUIRED_METHODS if not isinstance(getattr(environment_class, name, None), Callable))


def _environment_type_value(config: Any) -> str:
    raw_type = config.type
    return str(getattr(raw_type, "value", raw_type)).lower()


def main() -> None:
    try:
        from harbor.environments import e2b as e2b_module
        from harbor.environments.e2b import E2BEnvironment
    except ImportError as exc:
        raise SystemExit("Harbor E2B support is unavailable; install the checkout with `uv sync --extra e2b`") from exc

    if not getattr(e2b_module, "_HAS_E2B", False):
        raise SystemExit("The Harbor checkout is present but its optional E2B SDK is not installed")

    missing = _missing_methods(E2BEnvironment)
    if missing:
        raise SystemExit(f"Harbor E2BEnvironment lacks required methods: {', '.join(missing)}")

    config = build_harbor_environment_config()
    if _environment_type_value(config) != "e2b":
        raise SystemExit("Miles did not resolve Harbor's E2B environment")

    print("Harbor E2B preflight passed (no external API call was made).")


if __name__ == "__main__":
    main()
