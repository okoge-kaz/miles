#!/usr/bin/env python3
"""Prebuild and admit every E2B template used by selected Harbor tasks.

The Miles server forks one subprocess per trajectory.  Without this admission
step, several first-use trajectories can all observe a missing E2B alias and
race to build it.  This tool builds each distinct agent and separate-verifier
alias once, then verifies that every alias is visible before the server starts.
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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_IMMUTABLE_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_E2B_ID = re.compile(r"^[A-Za-z0-9_-]{6,128}$")
_LATE_TESTS_MARKER = ".harbor-e2b-late-tests"
_SEMANTIC_SCHEMAS = {
    "miles-r2e-admission-v1": "e2b_sandbox_evidence",
    "miles-swe-gym-admission-v1": "template_evidence",
    "miles-swe-rebench-admission-v1": "template_evidence",
    "miles-swebench-verified-hardened-local-admission-v1": (
        "template_evidence"
    ),
}
_MAX_ADMISSION_MANIFEST_BYTES = 512 * 1024 * 1024
_MAX_ADMISSION_LINE_BYTES = 1024 * 1024
_MAX_TASK_IDS_BYTES = 16 * 1024 * 1024
_MAX_TEMPLATE_PINS_BYTES = 64 * 1024 * 1024


def _task_set_sha256(task_ids: set[str]) -> str:
    rendered = json.dumps(
        sorted(task_ids),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _task_binding_sha256(tasks: dict[str, dict[str, str]]) -> str:
    rendered = json.dumps(
        [
            [instance_id, tasks[instance_id]["task_digest"]]
            for instance_id in sorted(tasks)
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _task_runtime_sha256(tasks: dict[str, dict[str, str]]) -> str:
    rendered = json.dumps(
        [
            [
                instance_id,
                tasks[instance_id]["task_digest"],
                tasks[instance_id]["task_tree_sha256"],
            ]
            for instance_id in sorted(tasks)
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TemplateSpec:
    task_id: str
    task_digest: str
    source_image: str
    task_tree_sha256: str
    role: str
    environment_dir: Path
    environment_name: str
    task_environment: Any


def _validate_private_task_tree(task_dir: Path) -> None:
    for path in [task_dir, *task_dir.rglob("*")]:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)) or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1):
            raise PermissionError("Harbor task tree contains an unsafe entry")
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise PermissionError("Harbor task tree must be owner-only")
        private_mode = stat.S_IMODE(metadata.st_mode)
        expected_modes = {0o500} if stat.S_ISDIR(metadata.st_mode) else {0o400, 0o500}
        if private_mode not in expected_modes:
            raise PermissionError("Selected Harbor task subtrees must be sealed read-only")


def _read_private_text_file(path: Path, *, max_bytes: int, name: str) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077 or metadata.st_size > max_bytes:
        raise PermissionError(f"{name} must be owner-only, single-link, and regular")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_uid != os.getuid() or opened.st_mode & 0o077 or opened.st_size > max_bytes or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise PermissionError(f"{name} changed before open")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        finished = os.fstat(descriptor)
        path_finished = path.lstat()
        stable = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
            opened.st_uid,
            stat.S_IMODE(opened.st_mode),
        )
        if (
            len(payload) > max_bytes
            or stable
            != (
                finished.st_dev,
                finished.st_ino,
                finished.st_size,
                finished.st_mtime_ns,
                finished.st_ctime_ns,
                finished.st_nlink,
                finished.st_uid,
                stat.S_IMODE(finished.st_mode),
            )
            or stable
            != (
                path_finished.st_dev,
                path_finished.st_ino,
                path_finished.st_size,
                path_finished.st_mtime_ns,
                path_finished.st_ctime_ns,
                path_finished.st_nlink,
                path_finished.st_uid,
                stat.S_IMODE(path_finished.st_mode),
            )
        ):
            raise RuntimeError(f"{name} changed while reading")
        return payload.decode("utf-8")
    finally:
        os.close(descriptor)


def _selected_task_dirs(tasks_dir: Path, task_ids_file: Path | None) -> list[Path]:
    tasks_absolute = tasks_dir.absolute()
    if tasks_dir.is_symlink() or not tasks_dir.is_dir() or tasks_absolute.resolve() != tasks_absolute or tasks_dir.stat().st_uid != os.getuid() or tasks_dir.stat().st_mode & 0o077:
        raise PermissionError("Harbor tasks directory must be owner-only and symlink-free")
    tasks_dir = tasks_absolute
    if task_ids_file is None:
        selected = sorted(path.parent for path in tasks_dir.glob("*/task.toml"))
    else:
        task_ids = []
        task_ids_text = _read_private_text_file(
            task_ids_file,
            max_bytes=_MAX_TASK_IDS_BYTES,
            name="E2B task-id file",
        )
        for raw_line in task_ids_text.splitlines():
            task_id = raw_line.strip()
            if not task_id or task_id.startswith("#"):
                continue
            if _SAFE_TASK_ID.fullmatch(task_id) is None:
                raise ValueError(f"Invalid task id in {task_ids_file}: {task_id!r}")
            task_ids.append(task_id)
        selected = [tasks_dir / task_id for task_id in task_ids]

    if not selected:
        raise ValueError("No Harbor task directories were selected for E2B admission")
    for task_dir in selected:
        if task_dir.is_symlink():
            raise ValueError(f"Harbor task directory must not be a symlink: {task_dir}")
        resolved = task_dir.resolve()
        if resolved.parent != tasks_dir or not (resolved / "task.toml").is_file():
            raise ValueError(f"Invalid Harbor task directory: {task_dir}")
        _validate_private_task_tree(resolved)
    return selected


def _validate_immutable_agent_image(task: Any, task_dir: Path) -> str:
    configured = task.config.environment.docker_image
    dockerfile = task.paths.environment_dir / "Dockerfile"
    if configured is not None:
        if _IMMUTABLE_IMAGE.fullmatch(configured) is None or dockerfile.exists():
            raise ValueError(f"Task {task_dir.name!r} agent image is not an unambiguous immutable image")
        return configured
    if dockerfile.is_symlink() or not dockerfile.is_file():
        raise ValueError(f"Task {task_dir.name!r} lacks a regular agent Dockerfile")
    from_lines = [line.strip() for line in dockerfile.read_text(encoding="utf-8").splitlines() if line.strip().upper().startswith("FROM ")]
    match = (
        re.fullmatch(
            r"FROM ([^\s]+@sha256:[0-9a-f]{64})",
            from_lines[0],
        )
        if len(from_lines) == 1
        else None
    )
    if len(from_lines) != 1 or match is None:
        raise ValueError(f"Task {task_dir.name!r} agent Dockerfile must have one immutable FROM")
    return match.group(1)


def _template_specs(task_dirs: list[Path]) -> list[TemplateSpec]:
    from experiments.src.environments.swe.materialize import _task_tree_sha256
    from harbor.models.task.config import NetworkMode
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_effective_verifier_env_config,
        resolve_task_verifier_mode,
    )
    from harbor.models.task.task import Task

    specs = []
    for task_dir in task_dirs:
        task = Task(task_dir)
        if task.has_steps:
            raise ValueError(f"Multi-step task {task_dir.name!r} needs explicit per-step template admission; this SWE launcher accepts single-step tasks only")
        task_digest = task.config.metadata.get("task_digest")
        if not isinstance(task_digest, str) or re.fullmatch(r"[0-9a-f]{64}", task_digest) is None:
            raise ValueError(f"Task {task_dir.name!r} lacks a valid task digest")
        if task.config.agent.user != 1000:
            raise ValueError(f"Task {task_dir.name!r} agent must run as UID 1000")
        if task.config.environment.resolve_baseline().network_mode != NetworkMode.NO_NETWORK or task.config.agent.network_mode not in (None, NetworkMode.NO_NETWORK):
            raise ValueError(f"Task {task_dir.name!r} agent must be no-network")
        source_image = _validate_immutable_agent_image(task, task_dir)
        task_tree_sha256 = _task_tree_sha256(task_dir)
        specs.append(
            TemplateSpec(
                task_id=task_dir.name,
                task_digest=task_digest,
                source_image=source_image,
                task_tree_sha256=task_tree_sha256,
                role="agent",
                environment_dir=task.paths.environment_dir,
                environment_name=task.short_name,
                task_environment=task.config.environment.model_copy(deep=True),
            )
        )
        mode = resolve_task_verifier_mode(task.config)
        if mode != VerifierEnvironmentMode.SEPARATE:
            raise ValueError(f"Task {task_dir.name!r} must use a separate verifier")
        verifier_environment = resolve_effective_verifier_env_config(
            task.config,
            step_cfg=None,
        )
        if verifier_environment is None:
            raise ValueError(f"Task {task_dir.name!r} declares a separate verifier without an effective verifier environment")
        if task.config.verifier.user != 0:
            raise ValueError(f"Task {task_dir.name!r} verifier must run as root")
        if verifier_environment.resolve_baseline().network_mode != NetworkMode.NO_NETWORK or task.config.verifier.network_mode not in (None, NetworkMode.NO_NETWORK):
            raise ValueError(f"Task {task_dir.name!r} verifier must be no-network")
        if not any(collect.required for collect in task.config.verifier.collect):
            raise ValueError(f"Task {task_dir.name!r} lacks a required collect hook")
        marker = task.paths.tests_dir / _LATE_TESTS_MARKER
        if marker.is_symlink() or not marker.is_file():
            raise ValueError(f"Task {task_dir.name!r} lacks the regular late-tests marker")
        if (task.paths.tests_dir / "Dockerfile").exists():
            raise ValueError(f"Task {task_dir.name!r} exposes private tests as a build context")
        if verifier_environment.docker_image is None or _IMMUTABLE_IMAGE.fullmatch(verifier_environment.docker_image) is None or verifier_environment.docker_image != source_image:
            raise ValueError(f"Task {task_dir.name!r} verifier must use the exact immutable agent source image")
        specs.append(
            TemplateSpec(
                task_id=task_dir.name,
                task_digest=task_digest,
                source_image=source_image,
                task_tree_sha256=task_tree_sha256,
                role="verifier",
                environment_dir=task.paths.tests_dir,
                environment_name=task.short_name,
                task_environment=verifier_environment.model_copy(deep=True),
            )
        )
    return specs


def _validate_template_evidence(value: Any) -> dict[str, str]:
    required = {
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
        "sandbox_id",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("semantic admission has invalid E2B template evidence")
    for key in ("template_id", "build_id", "sandbox_id"):
        if not isinstance(value[key], str) or _E2B_ID.fullmatch(value[key]) is None:
            raise ValueError("semantic admission has an invalid E2B identifier")
    for key in ("alias_sha256", "template_identity_sha256"):
        if not isinstance(value[key], str) or _DIGEST.fullmatch(value[key]) is None:
            raise ValueError("semantic admission has an invalid E2B identity digest")
    return value


def _validate_verified_admission(value: dict[str, Any]) -> None:
    """Apply the exact eval-only schema and pinned dependency contract."""

    from experiments.src.environments.swe import materialize

    pins = {
        "dataset_revision": materialize._SWEBENCH_VERIFIED_DATASET_REVISION,
        "harness_repository": materialize._SWEBENCH_HARNESS_REPOSITORY,
        "harness_commit": materialize._SWEBENCH_HARNESS_COMMIT,
        "harness_version": materialize._SWEBENCH_HARNESS_VERSION,
        "constants_sha256": materialize._SWEBENCH_CONSTANTS_SHA256,
        "log_parsers_sha256": materialize._SWEBENCH_LOG_PARSERS_SHA256,
        "grading_sha256": materialize._SWEBENCH_GRADING_SHA256,
        "score_semantics": materialize._SWEBENCH_HARDENED_SCORE_SEMANTICS,
        "harbor_adapter_commit": materialize._SWE_GYM_HARBOR_COMMIT,
        "harbor_adapter_sha256": materialize._SWE_GYM_HARBOR_ADAPTER_SHA256,
    }
    materialize._validate_repository_admission_record(
        value,
        schema="miles-swebench-verified-hardened-local-admission-v1",
        source_schema="swebench",
        pins=pins,
        name="SWE-bench Verified hardened-local",
    )


def _validate_semantic_admission(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("semantic admission record must be an object")
    evidence_key = _SEMANTIC_SCHEMAS.get(value.get("schema_version"))
    if evidence_key is None:
        raise ValueError("unsupported semantic admission schema")
    instance_id = value.get("instance_id")
    if not isinstance(instance_id, str) or _SAFE_TASK_ID.fullmatch(instance_id) is None:
        raise ValueError("semantic admission instance_id is invalid")
    for key in ("task_digest", "admitted_task_tree_sha256"):
        if not isinstance(value.get(key), str) or _DIGEST.fullmatch(value[key]) is None:
            raise ValueError(f"semantic admission {key} is invalid")
    source_image = value.get("source_image")
    if not isinstance(source_image, str) or _IMMUTABLE_IMAGE.fullmatch(source_image) is None:
        raise ValueError("semantic admission source_image is not immutable")
    evidence = value.get(evidence_key)
    roles = {"source", "agent", "empty_verifier", "oracle_verifier"}
    if not isinstance(evidence, dict) or set(evidence) != roles:
        raise ValueError("semantic admission E2B role set is invalid")
    validated = {role: _validate_template_evidence(evidence[role]) for role in roles}
    immutable_fields = (
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
    )
    if any(validated["empty_verifier"][key] != validated["oracle_verifier"][key] for key in immutable_fields):
        raise ValueError("semantic admission verifier template evidence conflicts")
    if any(validated["source"][key] != validated["empty_verifier"][key] for key in immutable_fields):
        raise ValueError("semantic admission source/verifier template evidence conflicts")
    checks = value.get("checks")
    required_checks: dict[str, bool | int] = {
        "empty_reward": 0,
        "oracle_reward": 1,
        "no_new_privileges": True,
        "effective_capabilities_zero": True,
        "fresh_separate_verifier": True,
    }
    if not isinstance(checks, dict) or any(checks.get(key) != expected for key, expected in required_checks.items()):
        raise ValueError("semantic admission lacks a required live security check")
    if evidence_key == "e2b_sandbox_evidence":
        if checks.get("late_verifier_tests_absent_before_upload") is not True:
            raise ValueError("R2E admission lacks late-verifier isolation evidence")
    elif checks.get("late_private_verifier_upload") is not True:
        raise ValueError("repository admission lacks late-verifier isolation evidence")
    if (
        value["schema_version"]
        == "miles-swebench-verified-hardened-local-admission-v1"
    ):
        _validate_verified_admission(value)
    return value


def _load_semantic_admissions(
    paths: list[Path],
    *,
    selected_task_ids: set[str],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in paths:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077 or metadata.st_size > _MAX_ADMISSION_MANIFEST_BYTES:
            raise PermissionError("semantic admission manifest must be owner-only, single-link, and regular")
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_uid != os.getuid() or opened.st_mode & 0o077 or opened.st_size > _MAX_ADMISSION_MANIFEST_BYTES or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise PermissionError("semantic admission manifest changed before open")
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
                while line := handle.readline(_MAX_ADMISSION_LINE_BYTES + 1):
                    if len(line.encode("utf-8")) > _MAX_ADMISSION_LINE_BYTES:
                        raise ValueError("semantic admission record is oversized")
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("semantic admission record must be an object")
                    instance_id = raw.get("instance_id")
                    if instance_id not in selected_task_ids:
                        continue
                    record = _validate_semantic_admission(raw)
                    if instance_id in selected:
                        raise ValueError("duplicate selected semantic admission")
                    selected[instance_id] = record
            finished = os.fstat(descriptor)
            path_finished = path.lstat()
            if (
                finished.st_dev,
                finished.st_ino,
                finished.st_size,
                finished.st_mtime_ns,
                finished.st_ctime_ns,
                finished.st_nlink,
                finished.st_uid,
                stat.S_IMODE(finished.st_mode),
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
                opened.st_uid,
                stat.S_IMODE(opened.st_mode),
            ):
                raise RuntimeError("semantic admission manifest changed while reading")
            if (
                path_finished.st_dev,
                path_finished.st_ino,
                path_finished.st_size,
                path_finished.st_mtime_ns,
                path_finished.st_ctime_ns,
                path_finished.st_nlink,
                path_finished.st_uid,
                stat.S_IMODE(path_finished.st_mode),
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
                opened.st_uid,
                stat.S_IMODE(opened.st_mode),
            ):
                raise RuntimeError("semantic admission path changed while reading")
        finally:
            os.close(descriptor)
    missing = selected_task_ids - set(selected)
    if missing:
        raise ValueError("selected task lacks an exact semantic admission")
    return selected


async def _prebuild(
    specs: list[TemplateSpec],
    *,
    concurrency: int,
    semantic_admissions: dict[str, dict[str, Any]] | None = None,
    allow_fresh_build: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from harbor.environments.e2b import E2BEnvironment
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.trial.paths import TrialPaths
    from miles.rollout.harbor.environment_config import build_harbor_environment_config

    if concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if semantic_admissions is None and not allow_fresh_build:
        raise ValueError("production template admission requires semantic admission manifests")
    if semantic_admissions is not None:
        semantic_admissions = {task_id: _validate_semantic_admission(record) for task_id, record in semantic_admissions.items()}

    runtime_config = build_harbor_environment_config()
    environments: dict[
        str,
        tuple[E2BEnvironment, set[tuple[str, str]], set[str]],
    ] = {}
    alias_identities: dict[str, str] = {}
    reuse_pins: dict[str, dict[str, str]] = {}
    identity_pins: dict[str, tuple[str, str]] = {}
    task_attestations: dict[str, dict[str, str]] = {}
    admitted_template_ids: set[str] = set()
    if semantic_admissions is not None:
        for admission in semantic_admissions.values():
            evidence_key = _SEMANTIC_SCHEMAS[admission["schema_version"]]
            admitted_template_ids.update(phase["template_id"] for phase in admission[evidence_key].values())
    with tempfile.TemporaryDirectory(prefix="miles-e2b-prebuild-") as raw_dir:
        root = Path(raw_dir)
        for index, spec in enumerate(specs):
            task_attestation = {
                "task_digest": spec.task_digest,
                "task_tree_sha256": spec.task_tree_sha256,
            }
            prior_attestation = task_attestations.setdefault(
                spec.task_id,
                task_attestation,
            )
            if prior_attestation != task_attestation:
                raise ValueError("one task has conflicting runtime attestations")
            trial_paths = TrialPaths(trial_dir=root / str(index))
            trial_paths.mkdir()
            environment = EnvironmentFactory.create_environment_from_config(
                config=runtime_config,
                environment_dir=spec.environment_dir,
                environment_name=spec.environment_name,
                session_id=f"prebuild-{index}",
                trial_paths=trial_paths,
                task_env_config=spec.task_environment,
                network_policy=spec.task_environment.resolve_baseline(),
            )
            if not isinstance(environment, E2BEnvironment):
                raise TypeError(f"Template admission requires HARBOR_ENV_TYPE=e2b; got {environment.type()}")
            alias = environment._template_name
            build_identity = environment._template_build_identity
            prior_identity = alias_identities.setdefault(alias, build_identity)
            if prior_identity != build_identity:
                raise RuntimeError("one E2B runtime alias resolves to multiple build identities")
            if build_identity in environments:
                environments[build_identity][1].add((spec.task_id, spec.role))
                environments[build_identity][2].add(alias)
            else:
                environments[build_identity] = (
                    environment,
                    {(spec.task_id, spec.role)},
                    {alias},
                )
            if semantic_admissions is not None:
                admission = semantic_admissions.get(spec.task_id)
                if admission is None:
                    raise ValueError("selected task lacks a semantic admission")
                if admission.get("task_digest") != spec.task_digest or admission.get("source_image") != spec.source_image or admission.get("admitted_task_tree_sha256") != spec.task_tree_sha256:
                    raise ValueError("semantic admission does not bind the materialized task")
                evidence_key = _SEMANTIC_SCHEMAS[admission["schema_version"]]
                evidence_role = "agent" if spec.role == "agent" else "empty_verifier"
                evidence = admission[evidence_key][evidence_role]
                if evidence["template_identity_sha256"] != build_identity:
                    raise ValueError("semantic admission template identity does not match runtime")
                pin = (evidence["template_id"], evidence["build_id"])
                prior_pin = identity_pins.setdefault(build_identity, pin)
                if prior_pin != pin:
                    raise ValueError("one E2B build identity has conflicting semantic pins")
                alias_pin = {
                    "template_id": pin[0],
                    "build_id": pin[1],
                }
                if alias in reuse_pins and reuse_pins[alias] != alias_pin:
                    raise ValueError("one E2B runtime alias has conflicting pins")
                reuse_pins[alias] = alias_pin

        semaphore = asyncio.Semaphore(concurrency)
        built = 0
        reused = 0
        run_nonce = secrets.token_hex(16)
        pins: dict[str, dict[str, str]] = {}

        async def admit(
            index: int,
            environment: E2BEnvironment,
            aliases: set[str],
        ) -> None:
            nonlocal built
            async with semaphore:
                build_alias = f"miles-swe-admit-{run_nonce}-{index}"
                build_info = await environment._create_template(
                    alias=build_alias,
                    skip_cache=True,
                )
                template_id = getattr(build_info, "template_id", None)
                build_id = getattr(build_info, "build_id", None)
                if not isinstance(template_id, str) or not template_id or not isinstance(build_id, str) or not build_id:
                    raise RuntimeError("E2B build did not return a pin-able identity")
                for alias in aliases:
                    pins[alias] = {
                        "template_id": template_id,
                        "build_id": build_id,
                    }
                built += 1

        if semantic_admissions is None:
            await asyncio.gather(*(admit(index, environment, aliases) for index, (_, (environment, _, aliases)) in enumerate(environments.items())))
        else:
            from e2b import AsyncTemplate

            semaphore = asyncio.Semaphore(concurrency)

            async def verify_template_id(template_id: str) -> None:
                async with semaphore:
                    await AsyncTemplate.get_tags(template_id)

            await asyncio.gather(*(verify_template_id(template_id) for template_id in sorted(admitted_template_ids)))
            pins.update(reuse_pins)
            reused = len(environments)

    selected_task_ids = set(task_attestations)
    task_set_sha256 = _task_set_sha256(selected_task_ids)
    task_binding_sha256 = _task_binding_sha256(task_attestations)
    task_runtime_sha256 = _task_runtime_sha256(task_attestations)
    report = {
        "schema_version": 4,
        "task_count": len(selected_task_ids),
        "task_set_sha256": task_set_sha256,
        "task_binding_sha256": task_binding_sha256,
        "task_runtime_sha256": task_runtime_sha256,
        "template_count": len(environments),
        "runtime_alias_count": len(alias_identities),
        "built_count": built,
        "reused_count": reused,
        "semantic_admission_reuse": semantic_admissions is not None,
        "template_id_access_checked": semantic_admissions is not None,
        "template_ids_pinned": True,
        "skip_cache": True,
        "late_tests_post_start": True,
        "templates": [
            {
                "consumer_count": len(consumers),
                "runtime_alias_count": len(aliases),
                "roles": sorted({role for _, role in consumers}),
            }
            for _, (_, consumers, aliases) in sorted(environments.items())
        ],
    }
    pin_payload = {
        "schema_version": 2,
        "pins": dict(sorted(pins.items())),
        "tasks": dict(sorted(task_attestations.items())),
        "task_count": len(selected_task_ids),
        "task_set_sha256": task_set_sha256,
        "task_binding_sha256": task_binding_sha256,
        "task_runtime_sha256": task_runtime_sha256,
    }
    return report, pin_payload


def _write_private_report(
    path: Path,
    report: dict[str, Any],
    *,
    max_bytes: int | None = None,
) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if max_bytes is not None and len(rendered.encode()) > max_bytes:
        raise ValueError("E2B template pin payload exceeds the explicit size cap")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_absolute = path.parent.absolute()
    parent = path.parent.lstat()
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent_absolute.resolve() != parent_absolute:
        raise PermissionError("E2B admission report parent is unsafe")
    path.parent.chmod(0o700)
    if path.exists() or path.is_symlink():
        target = path.lstat()
        if stat.S_ISLNK(target.st_mode) or not stat.S_ISREG(target.st_mode) or target.st_nlink != 1 or target.st_uid != os.getuid():
            raise PermissionError("E2B admission report target is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--task-ids-file", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument(
        "--semantic-admission-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--allow-fresh-build", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.report.resolve() == args.pins.resolve():
        raise ValueError("--report and --pins must be distinct private files")
    task_dirs = _selected_task_dirs(args.tasks_dir, args.task_ids_file)
    specs = _template_specs(task_dirs)
    selected_task_ids = {spec.task_id for spec in specs}
    semantic_admissions = (
        _load_semantic_admissions(
            args.semantic_admission_manifest,
            selected_task_ids=selected_task_ids,
        )
        if args.semantic_admission_manifest
        else None
    )
    report, pins = asyncio.run(
        _prebuild(
            specs,
            concurrency=args.concurrency,
            semantic_admissions=semantic_admissions,
            allow_fresh_build=args.allow_fresh_build,
        )
    )
    _write_private_report(args.report, report)
    _write_private_report(
        args.pins,
        pins,
        max_bytes=_MAX_TEMPLATE_PINS_BYTES,
    )
    print(f"E2B template admission passed: tasks={report['task_count']} templates={report['template_count']} built={report['built_count']}")


if __name__ == "__main__":
    main()
