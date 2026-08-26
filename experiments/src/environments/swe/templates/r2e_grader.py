"""Grade R2E-Gym's withheld pytest log against its exact expected map."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from r2e_execution_log_parser import decolor_dict_keys, parse_log_pytest

_VALID_STATUS = {"PASSED", "FAILED", "ERROR"}


def _expected(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("expected_output must be a non-empty object")
    normalized: dict[str, str] = {}
    for raw_name, raw_status in value.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("expected test names must be non-empty strings")
        status = str(raw_status).upper()
        if status not in _VALID_STATUS:
            raise ValueError(f"invalid expected status for {raw_name!r}: {raw_status!r}")
        normalized[raw_name] = status
    return dict(sorted(normalized.items()))


def _actual(log: str) -> tuple[dict[str, str], bool]:
    try:
        parsed = decolor_dict_keys(parse_log_pytest(log))
    except Exception:  # Official parser consumes model-controlled output.
        return {}, True
    if not isinstance(parsed, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(status, str)
        or status.upper() not in _VALID_STATUS
        for name, status in parsed.items()
    ):
        return {}, True
    return dict(sorted((name, status.upper()) for name, status in parsed.items())), False


def main() -> None:
    config = json.loads(Path("/tests/verifier_config.json").read_text(encoding="utf-8"))
    log = Path("/logs/verifier/test-output.log").read_text(encoding="utf-8", errors="replace")
    expected = _expected(config.get("expected_output"))
    actual, parser_error = _actual(log)
    reward = int(not parser_error and actual == expected)
    missing_count = len(set(expected) - set(actual))
    unexpected_count = len(set(actual) - set(expected))
    mismatched_count = sum(
        expected[name] != actual[name] for name in set(expected) & set(actual)
    )
    report = {
        "kind": "r2e-expected-pytest-map-v1",
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_count": missing_count,
        "unexpected_count": unexpected_count,
        "mismatched_count": mismatched_count,
        "no_tests_collected": not actual,
        "parser_error": parser_error,
        "reward": reward,
    }
    report["resolved"] = bool(reward)
    Path("/logs/verifier/report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("/logs/verifier/reward.txt").write_text(f"{reward}\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"R2E verifier infrastructure error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
