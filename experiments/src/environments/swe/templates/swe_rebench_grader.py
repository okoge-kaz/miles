"""Strict SWE-rebench V2 grader using the pinned official parser module."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lib.agent import log_parsers, swe_constants

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]
_VALID_STATUS = {status.value for status in swe_constants.TestStatus}


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value


def _parser(name: Any) -> Callable[[str], dict[str, str]]:
    if not isinstance(name, str) or not name.startswith("parse_"):
        raise ValueError("log_parser is not an explicit official parser name")
    parser = log_parsers.NAME_TO_PARSER.get(name)
    if parser is None:
        parser = getattr(log_parsers, name, None)
    if not callable(parser):
        raise ValueError(f"unknown official SWE-rebench parser: {name!r}")
    return parser


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _actual(parser: Callable[[str], dict[str, str]], log: str) -> tuple[dict[str, str], bool]:
    try:
        parsed = parser(log)
    except Exception:  # Official parser consumes model-controlled output.
        return {}, True
    if not isinstance(parsed, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(status, str)
        or status not in _VALID_STATUS
        for name, status in parsed.items()
    ):
        return {}, True
    return parsed, False


def main() -> None:
    config = json.loads(Path("/tests/verifier_config.json").read_text(encoding="utf-8"))
    log = Path("/logs/verifier/test-output.log").read_text(encoding="utf-8", errors="replace")
    parser_name = config.get("log_parser")
    parsed, parser_error = _actual(_parser(parser_name), log)
    fail_to_pass = _string_list(config.get("fail_to_pass"), "fail_to_pass")
    pass_to_pass = _string_list(config.get("pass_to_pass"), "pass_to_pass")
    normalized = {_normalize_test_name(name): status for name, status in parsed.items()}
    expected_passed = sorted(
        _normalize_test_name(name) for name in pass_to_pass + fail_to_pass
    )
    actual_passed = sorted(name for name, status in normalized.items() if status == "PASSED")
    failed_actual = sorted(name for name, status in normalized.items() if status == "FAILED")
    reward = int(not parser_error and actual_passed == expected_passed)
    report = {
        "kind": "swe-rebench-v2",
        "parser": parser_name,
        "resolved": bool(reward),
        "expected_count": len(expected_passed),
        "actual_passed_count": len(actual_passed),
        "actual_failed_count": len(failed_actual),
        "missing_count": len(set(expected_passed) - set(actual_passed)),
        "unexpected_pass_count": len(set(actual_passed) - set(expected_passed)),
        "parser_error": parser_error,
        "reward": reward,
    }
    Path("/logs/verifier/report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("/logs/verifier/reward.txt").write_text(f"{reward}\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SWE-rebench verifier infrastructure error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
