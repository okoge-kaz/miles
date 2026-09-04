"""Environment selection shared by Miles' Harbor agent-server integration.

The Harbor server is a separate process and owns sandbox execution.  This
module keeps its provider selection small, deterministic, and testable without
making Harbor or a cloud SDK a core Miles dependency.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

HarborEnvironmentType = Literal["docker", "daytona", "e2b"]

_TRUE_VALUES = frozenset({"1", "true", "t"})
_SUPPORTED_ENVIRONMENTS = frozenset({"docker", "daytona", "e2b"})


@dataclass(frozen=True)
class HarborEnvironmentSpec:
    """Backend-neutral inputs for Harbor's ``EnvironmentConfig``."""

    environment_type: HarborEnvironmentType
    delete: bool
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None
    extra_allowed_hosts: tuple[str, ...] = ()
    provider_kwargs: tuple[tuple[str, Any], ...] = ()

    def as_harbor_kwargs(self) -> dict[str, Any]:
        """Return constructor arguments without credentials or endpoint secrets."""
        kwargs: dict[str, Any] = {
            "type": self.environment_type,
            "delete": self.delete,
            "override_memory_mb": self.override_memory_mb,
        }
        if self.override_storage_mb is not None:
            kwargs["override_storage_mb"] = self.override_storage_mb
        if self.extra_allowed_hosts:
            kwargs["extra_allowed_hosts"] = list(self.extra_allowed_hosts)
        if self.provider_kwargs:
            kwargs["kwargs"] = dict(self.provider_kwargs)
        return kwargs


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _flag(environ: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _positive_int(environ: Mapping[str, str], name: str) -> int | None:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _allowed_hosts(environ: Mapping[str, str]) -> tuple[str, ...]:
    raw = environ.get("HARBOR_ENV_ALLOWED_HOSTS", "")
    return tuple(host for host in re.split(r"[,\s]+", raw) if host)


def _environment_type(environ: Mapping[str, str]) -> HarborEnvironmentType:
    raw = environ.get("HARBOR_ENV_TYPE", "docker").strip().lower()
    if raw not in _SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(sorted(_SUPPORTED_ENVIRONMENTS))
        raise ValueError(f"Unsupported HARBOR_ENV_TYPE {raw!r}; choose one of: {supported}")
    return cast(HarborEnvironmentType, raw)


def _docker_spec(environ: Mapping[str, str], memory_mb: int | None) -> HarborEnvironmentSpec:
    return HarborEnvironmentSpec(
        environment_type="docker",
        delete=_flag(environ, "HARBOR_DELETE_CONTAINERS"),
        override_memory_mb=memory_mb,
        extra_allowed_hosts=_allowed_hosts(environ),
    )


def _daytona_spec(environ: Mapping[str, str], memory_mb: int | None) -> HarborEnvironmentSpec:
    disk_gb = _positive_int(environ, "HARBOR_DAYTONA_DISK_GB") or 10
    auto_snapshot = _flag(environ, "HARBOR_DAYTONA_AUTO_SNAPSHOT", default=True)
    provider_kwargs: tuple[tuple[str, Any], ...] = (("auto_snapshot", True),) if auto_snapshot else ()
    return HarborEnvironmentSpec(
        environment_type="daytona",
        delete=not _flag(environ, "HARBOR_KEEP_SANDBOX"),
        override_memory_mb=memory_mb,
        override_storage_mb=disk_gb * 1024,
        provider_kwargs=provider_kwargs,
    )


def _e2b_spec(environ: Mapping[str, str], memory_mb: int | None) -> HarborEnvironmentSpec:
    if not environ.get("E2B_API_KEY", "").strip():
        raise ValueError("E2B requires E2B_API_KEY in the agent-server process environment")
    if _flag(environ, "HARBOR_KEEP_SANDBOX"):
        raise ValueError("HARBOR_KEEP_SANDBOX is unsupported for ephemeral E2B sandboxes")
    return HarborEnvironmentSpec(
        environment_type="e2b",
        delete=True,
        override_memory_mb=memory_mb,
        extra_allowed_hosts=_allowed_hosts(environ),
    )


def get_harbor_environment_spec(
    environ: Mapping[str, str] | None = None,
) -> HarborEnvironmentSpec:
    """Resolve Harbor provider settings from an explicit process environment."""
    source = _environment(environ)
    memory_mb = _positive_int(source, "HARBOR_OVERRIDE_MEMORY_MB")
    environment_type = _environment_type(source)
    if environment_type == "docker":
        return _docker_spec(source, memory_mb)
    if environment_type == "daytona":
        return _daytona_spec(source, memory_mb)
    return _e2b_spec(source, memory_mb)


def build_harbor_environment_config(
    environ: Mapping[str, str] | None = None,
    *,
    config_factory: Callable[..., Any] | None = None,
) -> Any:
    """Build Harbor's optional ``EnvironmentConfig`` from the resolved spec."""
    if config_factory is None:
        from harbor.models.trial.config import EnvironmentConfig

        config_factory = EnvironmentConfig
    spec = get_harbor_environment_spec(environ)
    return config_factory(**spec.as_harbor_kwargs())


def environment_uses_local_docker(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the selected provider owns local Docker resources."""
    return _environment_type(_environment(environ)) == "docker"
