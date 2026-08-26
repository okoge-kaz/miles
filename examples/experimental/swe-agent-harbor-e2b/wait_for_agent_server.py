#!/usr/bin/env python3
"""Wait until the private Harbor server is authenticated and rollout-ready."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from miles.rollout.harbor.auth import derive_harbor_health_bearer

_MAX_HEALTH_RESPONSE_BYTES = 4096
_MAX_SUMMARY_BYTES = 1024 * 1024
_MAX_WAIT_SECONDS = 24 * 60 * 60
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _expected_task_binding(summary_path: Path) -> tuple[str, str, str, int]:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(summary_path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_mode & 0o077
            or before.st_size > _MAX_SUMMARY_BYTES
        ):
            raise PermissionError(
                "SWE admission summary must be a bounded owner-only regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read(_MAX_SUMMARY_BYTES + 1)
        after = os.fstat(descriptor)
        if len(encoded) > _MAX_SUMMARY_BYTES or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
            before.st_mode,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
            after.st_mode,
        ):
            raise RuntimeError("SWE admission summary changed while reading")
    finally:
        os.close(descriptor)

    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("SWE admission summary must be a JSON object")
    schema = payload.get("schema_version")
    count_field = {
        "miles-swe-admitted-dataset-v1": "admitted_unique_tasks",
        "miles-swebench-verified-hardened-local-evaluation-v1": "unique_tasks",
    }.get(schema)
    if count_field is None:
        raise ValueError("SWE admission summary schema is unsupported")
    task_set_sha256 = payload.get("task_set_sha256")
    task_binding_sha256 = payload.get("task_binding_sha256")
    task_runtime_sha256 = payload.get("task_runtime_sha256")
    task_count = payload.get(count_field)
    if (
        not isinstance(task_set_sha256, str)
        or _DIGEST.fullmatch(task_set_sha256) is None
        or not isinstance(task_binding_sha256, str)
        or _DIGEST.fullmatch(task_binding_sha256) is None
        or not isinstance(task_runtime_sha256, str)
        or _DIGEST.fullmatch(task_runtime_sha256) is None
        or isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count <= 0
    ):
        raise ValueError("SWE admission summary has an invalid task-set binding")
    return (
        task_set_sha256,
        task_binding_sha256,
        task_runtime_sha256,
        task_count,
    )


def _health_url(raw_url: str) -> str:
    if not isinstance(raw_url, str) or not raw_url:
        raise ValueError("agent-server URL is missing")
    parsed = urlsplit(raw_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("agent-server URL must be a credential-free HTTP(S) origin")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("agent-server URL has an invalid port") from exc
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def wait_until_ready(
    server_url: str,
    master_secret: str,
    *,
    expected_task_set_sha256: str,
    expected_task_binding_sha256: str,
    expected_task_runtime_sha256: str,
    expected_task_count: int,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Poll authenticated health until ready or a bounded deadline expires."""

    if not 0 < timeout_seconds <= _MAX_WAIT_SECONDS:
        raise ValueError("readiness timeout must be in (0, 86400] seconds")
    if not 0 < poll_interval_seconds <= 60:
        raise ValueError("readiness poll interval must be in (0, 60] seconds")
    if (
        _DIGEST.fullmatch(expected_task_set_sha256) is None
        or _DIGEST.fullmatch(expected_task_binding_sha256) is None
        or _DIGEST.fullmatch(expected_task_runtime_sha256) is None
        or isinstance(expected_task_count, bool)
        or not isinstance(expected_task_count, int)
        or expected_task_count <= 0
    ):
        raise ValueError("expected task-set binding is invalid")
    health_url = _health_url(server_url)
    bearer = derive_harbor_health_bearer(master_secret)
    deadline = clock() + timeout_seconds
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(
                "Harbor agent server did not become authenticated-ready before timeout"
            )
        request = urllib.request.Request(
            health_url,
            headers={"Authorization": f"Bearer {bearer}"},
            method="GET",
        )
        try:
            with opener(request, timeout=min(5.0, remaining)) as response:
                payload = response.read(_MAX_HEALTH_RESPONSE_BYTES + 1)
                if (
                    response.status == 200
                    and len(payload) <= _MAX_HEALTH_RESPONSE_BYTES
                    and json.loads(payload)
                    == {
                        "status": "ok",
                        "ready": True,
                        "task_set_sha256": expected_task_set_sha256,
                        "task_binding_sha256": expected_task_binding_sha256,
                        "task_runtime_sha256": expected_task_runtime_sha256,
                        "task_count": expected_task_count,
                    }
                ):
                    return
        except (
            json.JSONDecodeError,
            OSError,
            TimeoutError,
            urllib.error.URLError,
        ):
            pass
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(
                "Harbor agent server did not become authenticated-ready before timeout"
            )
        sleeper(min(poll_interval_seconds, remaining))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--admission-summary", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--poll-interval-seconds", type=float, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    master_secret = os.environ.get("HARBOR_RUN_SECRET", "")
    (
        task_set_sha256,
        task_binding_sha256,
        task_runtime_sha256,
        task_count,
    ) = _expected_task_binding(args.admission_summary)
    wait_until_ready(
        args.server_url,
        master_secret,
        expected_task_set_sha256=task_set_sha256,
        expected_task_binding_sha256=task_binding_sha256,
        expected_task_runtime_sha256=task_runtime_sha256,
        expected_task_count=task_count,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    print("Harbor agent server is authenticated-ready.")


if __name__ == "__main__":
    main()
