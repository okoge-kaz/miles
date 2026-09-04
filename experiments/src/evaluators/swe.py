"""Run repository-level SWE evaluation through a Harbor agent server.

This client sends only task identity and local-policy connection information to
Harbor. Verifier payloads live in Harbor task directories and are never read
from the Miles prompt JSONL. The client derives a task-scoped bearer from the
process-only ``HARBOR_RUN_SECRET``; credentials are never accepted on the CLI
or copied into Harbor request bodies and request logs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from experiments.src.datasets.common.io import read_rows
from experiments.src.environments.swe.result import (
    HarborSWEOutcome,
    SWEEvaluationTrial,
    summarize_trials,
)
from experiments.src.environments.swe.timeouts import TRIAL_REQUEST_TIMEOUT_SEC
from miles.rollout.harbor.auth import (
    derive_harbor_cancel_bearer,
    derive_harbor_health_bearer,
    derive_harbor_run_bearer,
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_CLIENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MIN_SECRET_LENGTH = 32
_MAX_SECRET_LENGTH = 4096
_MAX_MODEL_LENGTH = 512
_SWEBENCH_VERIFIED_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
_SWEBENCH_HARNESS_COMMIT = "7336033d65d32ec62f9ce2419aa8f3a757b06ce2"
_SWEBENCH_HARNESS_FILES = {
    "constants.py": "6d189fcea0459897741eb241407b25e728467d3120f9feb8deaf2c61c030bc3e",
    "grading.py": "c953793204d52a7ac67b197e0479987efcdf13f381f87b685d711b2e156e3ce3",
    "log_parsers.py": "43ce7f06a562177ef82126f547e40fe19c51874148aae6027760d8b49b74bc89",
}


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    tasks, task_binding = _load_tasks_with_binding(
        args.input,
        limit=args.limit,
        input_summary=args.input_summary,
    )
    if task_binding is None:
        raise ValueError("SWE evaluation requires a task-set-bound input summary")
    args.server_url = _safe_service_url(args.server_url, name="--server-url")
    args.base_url = _safe_service_url(args.base_url, name="--base-url")
    args.model = _harbor_agent_model(args.model)
    run_secret = _required_process_secret("HARBOR_RUN_SECRET")
    client_id = _required_client_id()
    requests = [
        (instance_id, task_digest, trial_index)
        for instance_id, task_digest in tasks
        for trial_index in range(args.trials_per_task)
    ]
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[SWEEvaluationTrial] = []
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        await _require_server_ready(
            args.server_url,
            session=session,
            run_master_secret=run_secret,
            timeout_seconds=args.readiness_timeout,
            expected_task_binding=task_binding,
        )
        for start in range(0, len(requests), args.concurrency):
            request_batch = requests[start : start + args.concurrency]
            results.extend(
                await asyncio.gather(
                    *(
                        _run_trial(
                            args,
                            session,
                            semaphore,
                            instance_id=instance_id,
                            task_digest=task_digest,
                            trial_index=trial_index,
                            run_master_secret=run_secret,
                            client_id=client_id,
                        )
                        for instance_id, task_digest, trial_index in request_batch
                    )
                )
            )
    _write_results(args.output, results)
    summary = summarize_trials(results, task_count=len(tasks))
    summary.update(
        {
            "input": str(args.input),
            "input_summary": str(args.input_summary),
            "output": str(args.output),
            "model": args.model,
            "agent_name": args.agent_name,
            "harbor_server": args.server_url,
            "policy_base_url": args.base_url,
            "trials_per_task": args.trials_per_task,
            "external_judge": False,
            "score_semantics": "hardened-local-not-official-comparable-v1",
            "official_comparable": False,
            "official_harness_commit": _SWEBENCH_HARNESS_COMMIT,
            "maximum_infrastructure_failures": args.max_infrastructure_failures,
        }
    )
    summary["evaluation_valid"] = _evaluation_is_valid(
        summary,
        maximum_infrastructure_failures=args.max_infrastructure_failures,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if not summary["evaluation_valid"]:
        raise RuntimeError(
            "SWE evaluation is invalid: no graded trials or infrastructure "
            "failure threshold exceeded"
        )
    return summary


async def _require_server_ready(
    server_url: str,
    *,
    session: aiohttp.ClientSession,
    run_master_secret: str,
    timeout_seconds: int,
    expected_task_binding: tuple[str, str, str, int],
) -> None:
    bearer = derive_harbor_health_bearer(run_master_secret)

    async def request_health() -> None:
        async with session.get(
            f"{server_url.rstrip('/')}/health",
            headers={"Authorization": f"Bearer {bearer}"},
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Harbor health HTTP {response.status}")
            payload = await response.json()
        (
            expected_task_set,
            expected_binding,
            expected_runtime,
            expected_count,
        ) = expected_task_binding
        if payload != {
            "status": "ok",
            "ready": True,
            "task_set_sha256": expected_task_set,
            "task_binding_sha256": expected_binding,
            "task_runtime_sha256": expected_runtime,
            "task_count": expected_count,
        }:
            raise RuntimeError("Harbor health response is not ready")

    await asyncio.wait_for(request_health(), timeout=timeout_seconds)


def _evaluation_is_valid(
    summary: dict[str, Any],
    *,
    maximum_infrastructure_failures: int,
) -> bool:
    graded = summary.get("graded_trials")
    failures = summary.get("infrastructure_failures")
    return (
        isinstance(graded, int)
        and not isinstance(graded, bool)
        and graded > 0
        and isinstance(failures, int)
        and not isinstance(failures, bool)
        and 0 <= failures <= maximum_infrastructure_failures
    )


async def _run_trial(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    *,
    instance_id: str,
    task_digest: str,
    trial_index: int,
    run_master_secret: str,
    client_id: str,
) -> SWEEvaluationTrial:
    request_id = secrets.token_hex(16)
    body: dict[str, Any] = {
        "base_url": args.base_url,
        "model": _harbor_agent_model(args.model),
        "instance_id": instance_id,
        "task_digest": task_digest,
        "agent_name": args.agent_name,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_response_tokens,
        },
        "max_seq_len": args.max_sequence_length,
        "client_id": client_id,
        "request_id": request_id,
    }
    run_bearer = derive_harbor_run_bearer(
        run_master_secret,
        instance_id=instance_id,
        client_id=client_id,
        request_id=request_id,
    )
    started = monotonic()
    status_code = -1
    try:
        async with semaphore:
            async with session.post(
                f"{args.server_url.rstrip('/')}/run",
                json=body,
                headers={"Authorization": f"Bearer {run_bearer}"},
            ) as response:
                status_code = response.status
                text = await response.text()
        if status_code != 200:
            raise RuntimeError(f"Harbor HTTP {status_code}")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Harbor response must be a JSON object")
        outcome = HarborSWEOutcome.from_mapping(payload)
        return SWEEvaluationTrial(
            instance_id=instance_id,
            trial_index=trial_index,
            status_code=status_code,
            elapsed_seconds=monotonic() - started,
            reward=outcome.reward,
            exit_status=outcome.exit_status,
            eval_report=outcome.eval_report,
            agent_metrics=outcome.agent_metrics,
            error=None,
        )
    except asyncio.CancelledError:
        await _cancel_failed_trial(
            args,
            session,
            instance_id=instance_id,
            client_id=client_id,
            request_id=request_id,
            run_master_secret=run_master_secret,
        )
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        await _cancel_failed_trial(
            args,
            session,
            instance_id=instance_id,
            client_id=client_id,
            request_id=request_id,
            run_master_secret=run_master_secret,
        )
        return SWEEvaluationTrial(
            instance_id=instance_id,
            trial_index=trial_index,
            status_code=status_code,
            elapsed_seconds=monotonic() - started,
            reward=None,
            exit_status="ClientError",
            eval_report={},
            agent_metrics={},
            error=f"{type(exc).__name__}: {exc}",
        )


async def _cancel_failed_trial(
    args: argparse.Namespace,
    session: aiohttp.ClientSession,
    *,
    instance_id: str,
    client_id: str,
    request_id: str,
    run_master_secret: str,
) -> None:
    bearer = derive_harbor_cancel_bearer(
        run_master_secret,
        instance_id=instance_id,
        client_id=client_id,
        request_id=request_id,
    )

    async def cancel() -> None:
        async with session.post(
            f"{args.server_url.rstrip('/')}/cancel",
            json={
                "instance_id": instance_id,
                "client_id": client_id,
                "request_id": request_id,
            },
            headers={"Authorization": f"Bearer {bearer}"},
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Harbor cancel HTTP {response.status}")
            await response.read()

    try:
        await asyncio.wait_for(cancel(), timeout=30)
    except Exception:
        # The trial has its own bounded server-side wall timeout. Cancellation
        # is best effort when the client has already lost the transport.
        return


def _load_tasks(
    path: Path,
    *,
    limit: int | None,
    input_summary: Path | None = None,
) -> list[tuple[str, str]]:
    return _load_tasks_with_binding(
        path,
        limit=limit,
        input_summary=input_summary,
    )[0]


def _load_tasks_with_binding(
    path: Path,
    *,
    limit: int | None,
    input_summary: Path | None = None,
) -> tuple[list[tuple[str, str]], tuple[str, str, str, int] | None]:
    task_binding = None
    if input_summary is not None:
        rows, task_binding = _load_bound_evaluation_rows(
            input_summary,
            input_path=path,
        )
    else:
        rows = read_rows([path])
    tasks: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        instance_id = str(metadata.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError("every SWE evaluation row requires metadata.instance_id")
        swe_task = metadata.get("swe_task") or {}
        if not isinstance(swe_task, dict):
            raise ValueError("every SWE evaluation row requires metadata.swe_task")
        task_digest = str(swe_task.get("task_digest") or "").strip()
        if _DIGEST.fullmatch(task_digest) is None:
            raise ValueError(
                "every SWE evaluation row requires a lowercase SHA-256 task_digest"
            )
        previous_digest = seen.get(instance_id)
        if previous_digest is not None and previous_digest != task_digest:
            raise ValueError(
                f"duplicate SWE instance_id {instance_id!r} has conflicting task digests"
            )
        if previous_digest is not None:
            continue
        seen[instance_id] = task_digest
        tasks.append((instance_id, task_digest))
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise ValueError(f"no SWE evaluation tasks found in {path}")
    return tasks, task_binding


def _load_bound_evaluation_rows(
    summary_path: Path,
    *,
    input_path: Path,
) -> tuple[list[dict[str, Any]], tuple[str, str, str, int]]:
    summary_content = _read_owner_only_regular_file(
        summary_path,
        name="SWE evaluation input summary",
    )
    content = _read_owner_only_regular_file(
        input_path,
        name="SWE evaluation input",
    )
    try:
        summary = json.loads(summary_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SWE evaluation input summary is invalid JSON") from exc
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version")
        != "miles-swebench-verified-hardened-local-evaluation-v1"
        or summary.get("artifact_stage")
        != "hardened-local-environment-admitted-evaluation"
        or summary.get("environment_admitted") is not True
        or summary.get("evaluation_only") is not True
        or summary.get("official_comparable") is not False
        or summary.get("score_semantics")
        != "hardened-local-not-official-comparable-v1"
        or summary.get("source_dataset")
        != "princeton-nlp/SWE-bench_Verified"
        or summary.get("source_revision") != _SWEBENCH_VERIFIED_REVISION
        or summary.get("image_publisher_policy")
        != "miles-swe-image-publisher-policy-v1"
        or summary.get("harness_repository") != "SWE-bench/SWE-bench"
        or summary.get("harness_commit") != _SWEBENCH_HARNESS_COMMIT
        or summary.get("harness_version") != "2.0.13"
        or summary.get("harness_files_sha256") != _SWEBENCH_HARNESS_FILES
        or summary.get("model_path_policy_schema")
        != "miles-swe-model-path-policy-v2"
    ):
        raise ValueError("SWE evaluation input summary is not an admitted Verified artifact")
    expected_digest = summary.get("output_sha256")
    if _DIGEST.fullmatch(str(expected_digest or "")) is None:
        raise ValueError("SWE evaluation input summary has no valid output digest")
    actual_digest = hashlib.sha256(content).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("SWE evaluation input differs from its admitted summary")
    rows = _parse_bound_jsonl(content)
    if (
        isinstance(summary.get("rows"), bool)
        or summary.get("rows") != len(rows)
        or summary.get("unique_tasks") != len(rows)
        or not rows
    ):
        raise ValueError("SWE evaluation input row count differs from its summary")
    task_bindings = []
    for row in rows:
        metadata = row.get("metadata") or {}
        swe_task = metadata.get("swe_task") or {}
        task_bindings.append(
            [metadata.get("instance_id"), swe_task.get("task_digest")]
        )
    task_bindings.sort(key=lambda value: str(value[0]))
    task_ids = [value[0] for value in task_bindings]
    task_set_sha256 = _stable_digest(task_ids)
    task_binding_sha256 = _stable_digest(task_bindings)
    task_runtime_sha256 = summary.get("task_runtime_sha256")
    if (
        summary.get("task_set_sha256") != task_set_sha256
        or summary.get("task_binding_sha256") != task_binding_sha256
        or not isinstance(task_runtime_sha256, str)
        or _DIGEST.fullmatch(task_runtime_sha256) is None
    ):
        raise ValueError("SWE evaluation input task binding differs from its summary")
    return rows, (
        task_set_sha256,
        task_binding_sha256,
        task_runtime_sha256,
        len(task_ids),
    )


def _stable_digest(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _read_owner_only_regular_file(path: Path, *, name: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError(f"{name} is missing or unsafe: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise ValueError(f"{name} must be an owner-only regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _parse_bound_jsonl(content: bytes) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SWE evaluation input must be UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SWE evaluation input has invalid JSON on line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"SWE evaluation input row {line_number} is not an object"
            )
        rows.append(row)
    return rows


def _safe_service_url(value: str, *, name: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain credentials, a query, or a fragment")
    return value.rstrip("/")


def _harbor_agent_model(value: str) -> str:
    """Return the LiteLLM identifier used by the Harbor Terminus agent."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_MODEL_LENGTH
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ValueError("--model is missing or invalid")
    return value if value.startswith("openai/") else f"openai/{value}"


def _required_process_secret(setting: str) -> str:
    secret = os.getenv(setting)
    if (
        secret is None
        or not (_MIN_SECRET_LENGTH <= len(secret) <= _MAX_SECRET_LENGTH)
        or "\r" in secret
        or "\n" in secret
    ):
        raise ValueError(f"{setting} is missing or invalid")
    return secret


def _required_client_id() -> str:
    value = os.getenv("HARBOR_CLIENT_ID")
    if not isinstance(value, str) or _SAFE_CLIENT_ID.fullmatch(value) is None:
        raise ValueError("HARBOR_CLIENT_ID is missing or invalid")
    return value


def _write_results(path: Path, results: list[SWEEvaluationTrial]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for result in results:
                rendered = json.dumps(
                    asdict(result),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.write(rendered + "\n")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--server-url", required=True, help="running Harbor miles_agent_server URL")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible policy endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--agent-name", default="terminus-2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-infrastructure-failures",
        type=int,
        default=0,
        help="fail the evaluation when more ungraded trials are observed",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=int,
        default=30,
        help="fail-fast timeout for the authenticated Harbor readiness check",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=TRIAL_REQUEST_TIMEOUT_SEC,
        help="end-to-end trial timeout; must cover agent, verifier, and teardown",
    )
    args = parser.parse_args()
    for name in ("limit", "trials_per_task", "concurrency", "max_response_tokens", "max_sequence_length"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_infrastructure_failures < 0:
        parser.error("--max-infrastructure-failures must be non-negative")
    if args.readiness_timeout <= 0 or args.readiness_timeout > 60:
        parser.error("--readiness-timeout must be in [1, 60]")
    if args.request_timeout < TRIAL_REQUEST_TIMEOUT_SEC:
        parser.error(
            "--request-timeout must cover Harbor's bounded end-to-end trial "
            f"contract ({TRIAL_REQUEST_TIMEOUT_SEC}s)"
        )
    return args


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
