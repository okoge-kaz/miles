"""Grade hardened-local Verified output with a pinned official SWE-bench parser."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from swebench.harness.constants import (
    FAIL_TO_PASS,
    PASS_TO_PASS,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

_REPORT_KIND = "swebench-verified-hardened-local-v2.0.13"
_SCORE_SEMANTICS = "hardened-local-not-official-comparable-v1"
_VALID_STATUS = {status.value for status in TestStatus}


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value


def main() -> None:
    config = json.loads(
        Path("/tests/verifier_config.json").read_text(encoding="utf-8")
    )
    if config.get("score_semantics") != _SCORE_SEMANTICS:
        raise ValueError("SWE-bench Verified score semantics are not pinned")
    log = Path("/logs/verifier/test-output.log").read_text(
        encoding="utf-8",
        errors="replace",
    )
    repo = str(config.get("repo") or "").lower()
    parser = MAP_REPO_TO_PARSER.get(repo)
    if parser is None:
        raise ValueError(f"pinned official SWE-bench parser is absent for {repo!r}")
    fail_to_pass = _string_list(config.get("fail_to_pass"), "fail_to_pass")
    pass_to_pass = _string_list(config.get("pass_to_pass"), "pass_to_pass")
    try:
        status_map = parser(log)
    except Exception:  # Official parsers consume untrusted model/test output.
        status_map = {}
        parser_error = True
    else:
        parser_error = False
    if not isinstance(status_map, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(status, str)
        or status not in _VALID_STATUS
        for name, status in status_map.items()
    ):
        status_map = {}
        parser_error = True
    gold = {FAIL_TO_PASS: fail_to_pass, PASS_TO_PASS: pass_to_pass}
    report = get_eval_tests_report(status_map, gold)
    resolution = get_resolution_status(report)
    reward = int(not parser_error and resolution == ResolvedStatus.FULL.value)
    expected = sorted(fail_to_pass + pass_to_pass)
    passed = sorted(
        test
        for test in expected
        if test in status_map and status_map[test] == TestStatus.PASSED.value
    )
    result = {
        "kind": _REPORT_KIND,
        "score_semantics": _SCORE_SEMANTICS,
        "official_harness_commit": config.get("harness_commit"),
        "resolved": bool(reward),
        "resolution_status": resolution,
        "expected_count": len(expected),
        "passed_count": len(passed),
        "missing_or_failed_count": len(set(expected) - set(passed)),
        "parser_error": parser_error,
        "reward": reward,
    }
    Path("/logs/verifier/report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("/logs/verifier/reward.txt").write_text(
        f"{reward}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SWE-bench Verified verifier infrastructure error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
