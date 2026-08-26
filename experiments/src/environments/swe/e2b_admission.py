"""Backend-neutral lifecycle for fail-closed SWE admission on Harbor E2B.

This module intentionally never imports or invokes dotenv.  Harbor and the E2B
SDK receive ``E2B_API_KEY`` only from the admission process environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

_IMMUTABLE_IMAGE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,447}@sha256:[0-9a-f]{64}"
)
_E2B_ID = re.compile(r"[A-Za-z0-9_-]{6,128}")


@dataclass(frozen=True)
class RemoteResult:
    """Backend-neutral result of one remote command."""

    return_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SandboxSpec:
    """One source, agent, or verifier sandbox requested by admission."""

    role: Literal["source", "agent", "verifier"]
    name: str
    context_dir: Path
    task_dir: Path | None = None
    source_image: str | None = None
    expected_image: str | None = None


class AdmissionSandbox(Protocol):
    """Minimal remote lifecycle used by admission and fake fast tests."""

    @property
    def template_evidence(self) -> dict[str, str]: ...

    async def exec(
        self,
        command: str,
        *,
        user: int,
        timeout_sec: float,
    ) -> RemoteResult: ...

    async def upload_file(self, source: Path, destination: str) -> None: ...

    async def download_file(self, source: str, destination: Path) -> None: ...

    async def install_private_verifier(self, tests_dir: Path) -> None: ...

    async def close(self) -> None: ...


class AdmissionBackend(Protocol):
    """Factory abstraction shared by native E2B and deterministic fakes."""

    async def start(self, spec: SandboxSpec) -> AdmissionSandbox: ...


class RemoteCommandError(RuntimeError):
    """A checked remote command failed without exposing its private output."""

    def __init__(self, phase: str, return_code: int) -> None:
        super().__init__(f"{phase} failed with status {return_code}")
        self.phase = phase
        self.return_code = return_code


class LifecycleCleanupError(RuntimeError):
    """One or more cleanup failures chained behind the primary exception."""

    def __init__(self, message: str, failures: list[BaseException]) -> None:
        super().__init__(message)
        self.failures = tuple(failures)


@dataclass(frozen=True)
class _TemplatePin:
    template_id: str
    build_id: str
    alias_sha256: str
    template_identity_sha256: str


class _NativeSandbox:
    def __init__(
        self,
        environment: Any,
        temporary: tempfile.TemporaryDirectory[str],
        template_pin: _TemplatePin,
        sandbox_id: str,
    ) -> None:
        self._environment = environment
        self._temporary = temporary
        self._template_pin = template_pin
        self._sandbox_id = sandbox_id
        self._closed = False

    @property
    def template_evidence(self) -> dict[str, str]:
        return {
            "template_id": self._template_pin.template_id,
            "build_id": self._template_pin.build_id,
            "alias_sha256": self._template_pin.alias_sha256,
            "template_identity_sha256": (
                self._template_pin.template_identity_sha256
            ),
            "sandbox_id": self._sandbox_id,
        }

    async def exec(
        self,
        command: str,
        *,
        user: int,
        timeout_sec: float,
    ) -> RemoteResult:
        result = await self._environment.exec(
            command,
            user=user,
            timeout_sec=timeout_sec,
        )
        return RemoteResult(
            return_code=result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    async def upload_file(self, source: Path, destination: str) -> None:
        await self._environment.upload_file(source, destination)

    async def download_file(self, source: str, destination: Path) -> None:
        await self._environment.download_file(source, destination)

    async def install_private_verifier(self, tests_dir: Path) -> None:
        from harbor.trial.private_verifier_package import (
            upload_private_verifier_package,
        )

        await upload_private_verifier_package(self._environment, tests_dir)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failures: list[BaseException] = []
        try:
            if self._environment._sandbox is not None:
                await self._environment._stop_sandbox()
                self._environment._sandbox = None
        except BaseException as exc:
            failures.append(exc)
        try:
            self._temporary.cleanup()
        except BaseException as exc:
            failures.append(exc)
        if failures:
            if len(failures) == 1:
                raise failures[0]
            raise LifecycleCleanupError(
                "native E2B sandbox teardown failed",
                failures,
            ) from failures[0]


class NativeHarborE2BBackend:
    """Create native Harbor E2B sandboxes without forwarding credentials."""

    def __init__(self) -> None:
        if not os.environ.get("E2B_API_KEY", "").strip():
            raise ValueError(
                "E2B_API_KEY must be injected into the process environment"
            )
        if os.environ.get("HARBOR_ENV_TYPE", "").strip().lower() != "e2b":
            raise ValueError("HARBOR_ENV_TYPE=e2b is required for SWE admission")
        protected_users = {
            item.strip()
            for item in os.environ.get(
                "HARBOR_E2B_NO_NEW_PRIVS_USERS",
                "",
            ).split(",")
            if item.strip()
        }
        if "1000" not in protected_users:
            raise ValueError(
                "HARBOR_E2B_NO_NEW_PRIVS_USERS must include UID 1000"
            )
        try:
            from harbor.trial.private_verifier_package import (
                upload_private_verifier_package,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "pinned Harbor lacks the late private-verifier package overlay"
            ) from exc
        if not callable(upload_private_verifier_package):
            raise RuntimeError("Harbor private-verifier uploader is not callable")
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        self._template_pins: dict[str, _TemplatePin] = {}
        self._template_locks: dict[str, asyncio.Lock] = {}
        self._template_locks_guard = asyncio.Lock()

    async def start(self, spec: SandboxSpec) -> AdmissionSandbox:
        environment, temporary = self._create_environment(spec)
        failures: list[BaseException] = []
        try:
            if not hasattr(environment, "_with_no_new_privileges"):
                raise RuntimeError(
                    "pinned Harbor lacks the E2B no-new-privileges overlay"
                )
            template_pin = await self._fresh_template_pin(environment)
            environment._template_id = template_pin.template_id
            await environment._create_sandbox()
            if environment._sandbox is None:
                raise RuntimeError("E2B sandbox was absent immediately after creation")
            sandbox_id = getattr(environment._sandbox, "sandbox_id", None)
            if not isinstance(sandbox_id, str) or _E2B_ID.fullmatch(sandbox_id) is None:
                raise RuntimeError("E2B sandbox did not return a safe immutable ID")
            await environment.ensure_dirs(
                environment._mount_targets(writable_only=True)
            )
            late_tests = bool(getattr(environment, "_late_tests", False))
            if (spec.role == "verifier") != late_tests:
                raise RuntimeError("Harbor late-verifier classification is inconsistent")
            if not late_tests:
                await environment._upload_environment_dir_after_start()
            return _NativeSandbox(
                environment,
                temporary,
                template_pin,
                sandbox_id,
            )
        except BaseException as exc:
            failures.append(exc)
        try:
            if environment._sandbox is not None:
                await environment._stop_sandbox()
                environment._sandbox = None
        except BaseException as exc:
            failures.append(exc)
        try:
            temporary.cleanup()
        except BaseException as exc:
            failures.append(exc)
        primary = failures[0]
        cleanup_failures = failures[1:]
        if cleanup_failures:
            raise primary from LifecycleCleanupError(
                "native E2B sandbox startup cleanup failed",
                cleanup_failures,
            )
        raise primary

    async def _fresh_template_pin(self, environment: Any) -> _TemplatePin:
        identity = _template_build_identity(environment)
        async with self._template_locks_guard:
            lock = self._template_locks.setdefault(identity, asyncio.Lock())
        async with lock:
            cached = self._template_pins.get(identity)
            if cached is not None:
                return cached
            alias = f"miles-swe-admit-{secrets.token_hex(16)}"
            build_info = await environment._create_template(
                alias=alias,
                skip_cache=True,
            )
            template_id = getattr(build_info, "template_id", None)
            build_id = getattr(build_info, "build_id", None)
            if (
                not isinstance(template_id, str)
                or _E2B_ID.fullmatch(template_id) is None
                or not isinstance(build_id, str)
                or _E2B_ID.fullmatch(build_id) is None
            ):
                raise RuntimeError("fresh E2B build returned an invalid build identity")
            pin = _TemplatePin(
                template_id=template_id,
                build_id=build_id,
                alias_sha256=hashlib.sha256(alias.encode("utf-8")).hexdigest(),
                template_identity_sha256=hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest(),
            )
            self._template_pins[identity] = pin
            return pin

    def _create_environment(
        self,
        spec: SandboxSpec,
    ) -> tuple[Any, tempfile.TemporaryDirectory[str]]:
        from harbor.environments.e2b import E2BEnvironment
        from harbor.environments.factory import EnvironmentFactory
        from harbor.models.task.config import (
            EnvironmentConfig,
            NetworkMode,
            NetworkPolicy,
        )
        from harbor.models.task.task import Task
        from harbor.models.task.verifier_mode import (
            resolve_effective_verifier_env_config,
        )
        from harbor.models.trial.paths import TrialPaths
        from miles.rollout.harbor.environment_config import (
            build_harbor_environment_config,
        )

        expected_image = _validated_expected_image(spec)
        temporary = tempfile.TemporaryDirectory(prefix="miles-swe-admit-e2b-")
        try:
            trial_paths = TrialPaths(trial_dir=Path(temporary.name) / "trial")
            trial_paths.mkdir()
            if spec.role == "source":
                task_environment = EnvironmentConfig(
                    docker_image=expected_image,
                    cpus=4,
                    memory_mb=8192,
                    storage_mb=20480,
                    gpus=0,
                )
                network_policy = NetworkPolicy(
                    network_mode=NetworkMode.NO_NETWORK
                )
            else:
                if spec.task_dir is None:
                    raise ValueError(
                        f"{spec.role} sandbox requires a materialized task"
                    )
                task = Task(spec.task_dir)
                if spec.role == "agent":
                    dockerfile = spec.context_dir / "Dockerfile"
                    if (
                        not dockerfile.is_file()
                        or dockerfile.is_symlink()
                        or not dockerfile.read_text(encoding="utf-8").startswith(
                            f"FROM {expected_image}\n"
                        )
                    ):
                        raise ValueError(
                            "SWE agent Dockerfile differs from its admitted digest"
                        )
                    task_environment = task.config.environment.model_copy(deep=True)
                else:
                    task_environment = resolve_effective_verifier_env_config(
                        task.config,
                        step_cfg=None,
                    )
                    if task_environment is None:
                        raise ValueError(
                            "SWE task has no separate verifier environment"
                        )
                    if task_environment.docker_image != expected_image:
                        raise ValueError(
                            "SWE verifier image differs from its admitted digest"
                        )
                network_policy = task_environment.resolve_baseline()
            _require_no_network(network_policy)
            environment = EnvironmentFactory.create_environment_from_config(
                config=build_harbor_environment_config(),
                environment_dir=spec.context_dir,
                environment_name=spec.name,
                session_id=f"swe-admit-{uuid.uuid4().hex}",
                trial_paths=trial_paths,
                task_env_config=task_environment,
                network_policy=network_policy,
            )
            if not isinstance(environment, E2BEnvironment):
                raise TypeError(
                    "SWE admission requires Harbor's native E2B environment"
                )
            return environment, temporary
        except BaseException:
            temporary.cleanup()
            raise


def _validated_expected_image(spec: SandboxSpec) -> str:
    expected = spec.expected_image or spec.source_image
    if expected is None or _IMMUTABLE_IMAGE.fullmatch(expected) is None:
        raise ValueError("SWE admission requires an exact immutable image digest")
    if spec.role == "source" and spec.source_image != expected:
        raise ValueError("source sandbox image differs from its admitted digest")
    return expected


def _require_no_network(policy: Any) -> None:
    mode = getattr(policy, "network_mode", None)
    rendered = getattr(mode, "value", mode)
    if rendered != "no-network":
        raise ValueError("SWE admission sandbox baseline must be no-network")


def _template_build_identity(environment: Any) -> str:
    """Canonicalize exactly the inputs consumed by Harbor's E2B build."""

    task_environment = getattr(environment, "task_env_config", None)
    image = getattr(task_environment, "docker_image", None)
    if image is not None:
        if not isinstance(image, str) or _IMMUTABLE_IMAGE.fullmatch(image) is None:
            raise RuntimeError("E2B image template is not an immutable digest")
        source = {"kind": "docker-image", "docker_image": image}
    else:
        environment_id = getattr(environment, "environment_id", None)
        if not isinstance(environment_id, str) or not environment_id:
            raise RuntimeError("Harbor did not expose a Dockerfile content identity")
        source = {"kind": "dockerfile", "environment_id": environment_id}
    value = {
        **source,
        "effective_cpus": getattr(environment, "_effective_cpus", None),
        "effective_memory_mb": getattr(
            environment,
            "_effective_memory_mb",
            None,
        ),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@asynccontextmanager
async def sandbox(
    backend: AdmissionBackend,
    spec: SandboxSpec,
) -> AsyncIterator[AdmissionSandbox]:
    """Close every sandbox and preserve both body and teardown failures."""

    remote = await backend.start(spec)
    failure: BaseException | None = None
    try:
        yield remote
    except BaseException as exc:
        failure = exc
    close_failure: BaseException | None = None
    try:
        await remote.close()
    except BaseException as exc:
        close_failure = exc
    if failure is not None and close_failure is not None:
        raise failure from LifecycleCleanupError(
            "SWE admission sandbox teardown failed after body failure",
            [close_failure],
        )
    if failure is not None:
        raise failure
    if close_failure is not None:
        raise close_failure


async def require_ok(
    remote: AdmissionSandbox,
    command: str,
    *,
    user: int,
    timeout_sec: float,
    phase: str,
) -> RemoteResult:
    """Run a command without putting private stdout/stderr in exceptions."""

    result = await remote.exec(command, user=user, timeout_sec=timeout_sec)
    if result.return_code != 0:
        raise RemoteCommandError(phase, result.return_code)
    return result
