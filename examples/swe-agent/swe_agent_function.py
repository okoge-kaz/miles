"""
Custom agent function for ``agentic_tool_call.generate``.

Dispatches to a Harbor-based agent server and returns env metadata
as a plain dict. The generate layer merges this into sample.metadata so
downstream reward models (--custom-rm-path) can extract reward, eval
reports, etc.

Task-type agnostic — the server + Harbor task directory handle all
differentiation (environment, grading harness, agent selection).
"""

import asyncio
import logging
import os
import re
import secrets
import socket
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunparse

import httpx

from experiments.src.environments.swe.result import HarborSWEOutcome
from experiments.src.environments.swe.timeouts import TRIAL_REQUEST_TIMEOUT_SEC
from miles.rollout.generate_hub.agentic_types import AgentFunctionOutput
from miles.rollout.harbor.auth import (
    derive_harbor_cancel_bearer,
    derive_harbor_flush_bearer,
    derive_harbor_run_bearer,
)
from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

_agent_server_client: httpx.AsyncClient | None = None
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_SAFE_SESSION_BINDING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SAFE_CLIENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_REQUEST_TIMEOUT_SEC = TRIAL_REQUEST_TIMEOUT_SEC
_MIN_REQUEST_TIMEOUT_SEC = TRIAL_REQUEST_TIMEOUT_SEC
_MIN_SECRET_LENGTH = 32
_MAX_SECRET_LENGTH = 4096
_MAX_ABORT_INSTANCES = 256
_ABORT_CONCURRENCY = 16
_ABORT_REQUEST_TIMEOUT_SEC = 30


def _get_agent_server_client() -> httpx.AsyncClient:
    """Return a client whose long-running requests survive idle network paths."""
    global _agent_server_client
    if _agent_server_client is None:
        socket_options = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 60),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 30),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 5),
        ]
        transport = httpx.AsyncHTTPTransport(socket_options=socket_options)
        _agent_server_client = httpx.AsyncClient(
            transport=transport,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            timeout=None,
            trust_env=False,
        )
    return _agent_server_client


async def _post_agent_server(
    url: str,
    payload: dict[str, Any],
    *,
    bearer_token: str,
) -> dict[str, Any]:
    client = _get_agent_server_client()
    response = await client.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {bearer_token}"},
    )
    response.raise_for_status()
    return response.json()


def _validated_trial_metadata(response: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return verifier-scored metadata, or ``None`` for an ungraded trial.

    The Harbor server intentionally returns a JSON response for environment,
    agent, and verifier failures.  Those responses currently carry
    ``reward=0`` so the client must not confuse infrastructure failures with a
    model-produced incorrect solution.
    """
    try:
        outcome = HarborSWEOutcome.from_mapping(response)
    except (TypeError, ValueError):
        return None

    return {
        "reward": outcome.reward,
        "exit_status": outcome.exit_status,
        "eval_report": outcome.eval_report,
        "agent_metrics": outcome.agent_metrics,
    }


def _safe_http_endpoint(raw_url: str, *, setting: str) -> str:
    """Validate a non-secret HTTP endpoint before it can reach argv or logs."""
    if not raw_url or any(ord(character) < 0x20 for character in raw_url):
        raise ValueError(f"{setting} must be a non-empty HTTP(S) URL")
    parsed = urlsplit(raw_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{setting} must be HTTP(S) without credentials, query, or fragment"
        )
    return raw_url.rstrip("/")


def _request_timeout_seconds() -> int:
    raw_timeout = os.getenv(
        "SWE_TRIAL_REQUEST_TIMEOUT_SEC",
        str(_DEFAULT_REQUEST_TIMEOUT_SEC),
    )
    if not raw_timeout.isascii() or not raw_timeout.isdecimal():
        raise ValueError("SWE_TRIAL_REQUEST_TIMEOUT_SEC must be an integer")
    timeout = int(raw_timeout)
    if timeout < _MIN_REQUEST_TIMEOUT_SEC:
        raise ValueError(
            "SWE_TRIAL_REQUEST_TIMEOUT_SEC must cover Harbor's bounded "
            f"end-to-end trial contract ({_MIN_REQUEST_TIMEOUT_SEC}s)"
        )
    return timeout


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
    client_id = os.getenv("HARBOR_CLIENT_ID")
    if (
        not isinstance(client_id, str)
        or _SAFE_CLIENT_ID.fullmatch(client_id) is None
    ):
        raise ValueError("HARBOR_CLIENT_ID is missing or invalid")
    return client_id


def _required_task_binding(metadata: Mapping[str, Any]) -> tuple[str, str]:
    instance_id = metadata.get("instance_id")
    if not isinstance(instance_id, str) or _SAFE_TASK_ID.fullmatch(instance_id) is None:
        raise ValueError("metadata.instance_id is missing or unsafe")
    if metadata.get("agent_name") != "terminus-2":
        raise ValueError("metadata.agent_name must be exactly 'terminus-2'")
    swe_task = metadata.get("swe_task")
    task_digest = swe_task.get("task_digest") if isinstance(swe_task, Mapping) else None
    if not isinstance(task_digest, str) or _SHA256.fullmatch(task_digest) is None:
        raise ValueError("metadata.swe_task.task_digest is missing or invalid")
    return instance_id, task_digest


def _optional_binding(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or _SAFE_SESSION_BINDING.fullmatch(value) is None:
        raise ValueError(f"metadata.{key} is invalid")
    return value


def _abort_instance_ids(args: Any) -> tuple[str, ...]:
    """Return the validated driver-side session-server instance inventory."""

    instance_map = getattr(args, "session_server_instance_ids", None)
    if instance_map is not None:
        if not isinstance(instance_map, Mapping):
            raise ValueError("session_server_instance_ids must be a mapping")
        if len(instance_map) > _MAX_ABORT_INSTANCES:
            raise ValueError("too many session-server instances to flush")
        validated: list[tuple[int, str]] = []
        seen: set[str] = set()
        for port, instance_id in instance_map.items():
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError("session-server map contains an invalid port")
            if (
                not isinstance(instance_id, str)
                or _SAFE_SESSION_BINDING.fullmatch(instance_id) is None
                or instance_id in seen
            ):
                raise ValueError("session-server map contains an invalid instance ID")
            seen.add(instance_id)
            validated.append((port, instance_id))
        return tuple(instance_id for _, instance_id in sorted(validated))

    # Compatibility for older single-session launchers. Production Miles uses
    # the complete port-to-instance map above.
    instance_id = getattr(args, "session_server_instance_id", None)
    if instance_id is None:
        return ()
    if (
        not isinstance(instance_id, str)
        or _SAFE_SESSION_BINDING.fullmatch(instance_id) is None
    ):
        raise ValueError("session_server_instance_id is invalid")
    return (instance_id,)


async def _flush_abort_instance(
    *,
    agent_server_url: str,
    run_master_secret: str,
    instance_id: str,
    semaphore: asyncio.Semaphore,
) -> Any:
    """Flush one scoped Harbor trial inventory with a bounded request."""

    bearer = derive_harbor_flush_bearer(
        run_master_secret,
        session_server_instance_id=instance_id,
    )
    async with semaphore:
        return await asyncio.wait_for(
            post(
                f"{agent_server_url}/flush",
                {"session_server_instance_id": instance_id},
                max_retries=3,
                headers={"Authorization": f"Bearer {bearer}"},
            ),
            timeout=_ABORT_REQUEST_TIMEOUT_SEC,
        )


async def _cancel_failed_trial(
    *,
    agent_server_url: str,
    run_master_secret: str,
    client_id: str,
    request_id: str,
    instance_id: str,
    session_server_instance_id: str | None,
) -> None:
    """Best-effort cleanup after the long-running HTTP request is lost."""

    bearer = derive_harbor_cancel_bearer(
        run_master_secret,
        client_id=client_id,
        request_id=request_id,
        instance_id=instance_id,
        session_server_instance_id=session_server_instance_id,
    )
    payload: dict[str, Any] = {
        "client_id": client_id,
        "instance_id": instance_id,
        "request_id": request_id,
    }
    if session_server_instance_id is not None:
        payload["session_server_instance_id"] = session_server_instance_id
    try:
        await asyncio.wait_for(
            _post_agent_server(
                f"{agent_server_url}/cancel",
                payload,
                bearer_token=bearer,
            ),
            timeout=_ABORT_REQUEST_TIMEOUT_SEC,
        )
    except Exception:
        logger.warning("Failed to cancel a disconnected Harbor trial")


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> AgentFunctionOutput:
    """Run a single task instance via the Harbor agent server."""
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}

    try:
        agent_server_url = _safe_http_endpoint(
            os.getenv(
                "AGENT_SERVER_URL",
                os.getenv("SWE_AGENT_URL", "http://localhost:11000"),
            ),
            setting="AGENT_SERVER_URL",
        )
        model_name = os.getenv(
            "AGENT_MODEL_NAME",
            os.getenv("SWE_AGENT_MODEL_NAME", "model"),
        )
        if not model_name or any(character.isspace() for character in model_name):
            raise ValueError("AGENT_MODEL_NAME is invalid")
        instance_id, task_digest = _required_task_binding(metadata)
        request_timeout = _request_timeout_seconds()
        run_master_secret = _required_process_secret("HARBOR_RUN_SECRET")
        client_id = _required_client_id()
        request_id = secrets.token_hex(16)

        session_url = _safe_http_endpoint(f"{base_url.rstrip('/')}/v1", setting="base_url")
        external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
        if external_host:
            if "://" in external_host or any(
                character in external_host for character in "/?#@"
            ):
                raise ValueError("MILES_ROUTER_EXTERNAL_HOST must be a bare hostname")
            parsed = urlparse(session_url)
            port = parsed.port
            netloc = f"{external_host}:{port}" if port else external_host
            session_url = _safe_http_endpoint(
                urlunparse(parsed._replace(netloc=netloc)),
                setting="base_url",
            )

        request: dict[str, Any] = {
            "instance_id": instance_id,
            "agent_name": "terminus-2",
            "task_digest": task_digest,
            "base_url": session_url,
            "model": f"openai/{model_name}",
            "sampling_params": request_kwargs,
            "client_id": client_id,
            "request_id": request_id,
        }
        max_seq_len = metadata.get("max_seq_len")
        if max_seq_len is not None:
            if (
                isinstance(max_seq_len, bool)
                or not isinstance(max_seq_len, int)
                or max_seq_len <= 0
            ):
                raise ValueError("metadata.max_seq_len must be a positive integer")
            request["max_seq_len"] = max_seq_len

        session_server_id = _optional_binding(metadata, "session_server_id")
        if session_server_id is not None:
            if external_host:
                port = urlsplit(f"http://{session_server_id}").port
                if port is None:
                    raise ValueError("metadata.session_server_id must include a port")
                session_server_id = f"{external_host}:{port}"
            request["session_server_id"] = session_server_id

        session_server_instance_id = _optional_binding(
            metadata,
            "session_server_instance_id",
        )
        if session_server_instance_id is not None:
            request["session_server_instance_id"] = session_server_instance_id
        run_bearer = derive_harbor_run_bearer(
            run_master_secret,
            instance_id=instance_id,
            session_server_instance_id=session_server_instance_id,
            client_id=client_id,
            request_id=request_id,
        )
    except (TypeError, ValueError):
        logger.error("SWE trial request configuration or metadata is invalid")
        return AgentFunctionOutput.abort({"exit_status": "ClientConfigurationError"})

    try:
        response = await asyncio.wait_for(
            _post_agent_server(
                f"{agent_server_url}/run",
                request,
                bearer_token=run_bearer,
            ),
            timeout=request_timeout,
        )
    except asyncio.TimeoutError:
        await _cancel_failed_trial(
            agent_server_url=agent_server_url,
            run_master_secret=run_master_secret,
            client_id=client_id,
            request_id=request_id,
            instance_id=instance_id,
            session_server_instance_id=session_server_instance_id,
        )
        logger.error("Agent server call timed out after %ss", request_timeout)
        return AgentFunctionOutput.abort({"exit_status": "ClientTimeout"})
    except asyncio.CancelledError:
        await _cancel_failed_trial(
            agent_server_url=agent_server_url,
            run_master_secret=run_master_secret,
            client_id=client_id,
            request_id=request_id,
            instance_id=instance_id,
            session_server_instance_id=session_server_instance_id,
        )
        logger.warning("Agent server call cancelled (sibling task failure?)")
        return AgentFunctionOutput.abort({"exit_status": "Cancelled"})
    except Exception as e:
        await _cancel_failed_trial(
            agent_server_url=agent_server_url,
            run_master_secret=run_master_secret,
            client_id=client_id,
            request_id=request_id,
            instance_id=instance_id,
            session_server_instance_id=session_server_instance_id,
        )
        logger.error(f"Agent server call failed: {e}")
        return AgentFunctionOutput.abort({"exit_status": "ClientError"})

    trial_metadata = _validated_trial_metadata(response)
    if trial_metadata is None:
        logger.warning(
            "Harbor trial was not verifier-scored; marking the rollout aborted "
            "(exit_status=%r)",
            response.get("exit_status"),
        )
        return AgentFunctionOutput.abort(
            {"exit_status": str(response.get("exit_status") or "Ungraded")}
        )
    return AgentFunctionOutput(metadata=trial_metadata)


async def abort(args) -> None:
    """Teardown hook for oversampling abort (called by sglang_rollout.abort).

    When Miles has enough samples and aborts SGLang, the in-flight Harbor trials
    keep looping and hitting SGLang until they hit their own max_seq_len/timeout.
    Flush the agent server so it cancels those ``/run`` tasks and releases their
    containers. Production Miles provides a port-to-instance map because each
    session server has a separately scoped flush bearer.
    """
    raw_agent_server_url = os.getenv("AGENT_SERVER_URL", os.getenv("SWE_AGENT_URL"))
    if not raw_agent_server_url:
        return

    try:
        agent_server_url = _safe_http_endpoint(
            raw_agent_server_url,
            setting="AGENT_SERVER_URL",
        )
        run_master_secret = _required_process_secret("HARBOR_RUN_SECRET")
        instance_ids = _abort_instance_ids(args)
    except ValueError:
        logger.error("Refusing an invalid or unauthenticated SWE abort inventory")
        return
    if not instance_ids:
        return

    semaphore = asyncio.Semaphore(_ABORT_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _flush_abort_instance(
                agent_server_url=agent_server_url,
                run_master_secret=run_master_secret,
                instance_id=instance_id,
                semaphore=semaphore,
            )
            for instance_id in instance_ids
        ),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        logger.warning(
            "Failed to flush %s of %s agent-server trial inventories",
            len(failures),
            len(results),
        )
    logger.info(
        "Flushed %s of %s agent-server trial inventories",
        len(results) - len(failures),
        len(results),
    )
