"""Live-admit R2E-Gym tasks for native Harbor E2B training.

R2E-Gym images check out the answer commit's unique parent while retaining the
answer object in the source Git history. A task is admitted only after this
tool binds the dataset's private answer commit, verifies that image layout,
derives its canonical diff, and exercises independent agent and verifier
sandboxes. The output files are private inputs to Harbor; they must never be
copied into Miles prompt rows.

The command deliberately does not load dotenv files.  The E2B SDK obtains its
credential from ``E2B_API_KEY`` in the process environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock
from experiments.src.environments.swe.e2b_admission import (
    AdmissionBackend,
    AdmissionSandbox as AdmissionSandbox,
    NativeHarborE2BBackend,
    RemoteCommandError,
    RemoteResult as RemoteResult,
    SandboxSpec,
    require_ok as _require_ok,
    sandbox as _sandbox,
)

_ADMISSION_SCHEMA = "miles-r2e-admission-v1"
_CHECKPOINT_SCHEMA = "miles-r2e-admission-checkpoint-v2"
_LEGACY_CHECKPOINT_SCHEMA = "miles-r2e-admission-checkpoint-v1"
_QUARANTINE_SCHEMA = "miles-r2e-admission-quarantine-v2"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_E2B_ID = re.compile(r"[A-Za-z0-9_-]{6,128}")
_IMMUTABLE_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,447}@sha256:[0-9a-f]{64}")
_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,239}")
_MAX_ORACLE_PATCH_BYTES = 64 * 1024 * 1024
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_SMALL_EVIDENCE_BYTES = 64 * 1024
_QUARANTINE_REASONS = {
    "golden_outcome_mismatch",
    "materialization_unsupported",
    "oracle_patch_unsupported",
    "source_evidence_invalid",
    "source_image_unsupported",
    "unsafe_agent_image",
    "verifier_evidence_invalid",
    "verifier_incompatible",
}
_REQUIRED_CHECKS: dict[str, bool | int] = {
    "publisher_namespace_policy": True,
    "source_head_matches_base": True,
    "unique_parent_matches_base": True,
    "empty_reward": 0,
    "oracle_reward": 1,
    "runtime_smoke": True,
    "tool_smoke": True,
    "no_new_privileges": True,
    "effective_capabilities_zero": True,
    "suid_sgid_absent": True,
    "file_capabilities_absent": True,
    "fresh_separate_verifier": True,
    "source_network_denied": True,
    "agent_network_denied": True,
    "empty_verifier_network_denied": True,
    "oracle_verifier_network_denied": True,
    "late_verifier_tests_absent_before_upload": True,
    "gold_history_absent": True,
    "known_gold_artifacts_absent": True,
    "gold_blob_content_absent": True,
    "gold_commit_text_absent": True,
}

_SOURCE_INSPECTION_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/tmp/miles-r2e-root-home
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_COUNT=0
export GIT_ATTR_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1
export GIT_PAGER=cat
export PAGER=cat
export GIT_EXTERNAL_DIFF=
unset BASH_ENV CDPATH ENV GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR \
    GIT_DIR GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_WORK_TREE

for command in awk bash cat git python3 setpriv sha256sum stat timeout; do
    command -v "${command}" >/dev/null 2>&1 || exit 20
done
[[ -d /testbed ]] || exit 21
safe_git() {
    command git --no-pager \
        -c safe.directory=/testbed \
        -c core.fsmonitor=false \
        -c core.hooksPath=/dev/null \
        -c core.pager=cat \
        -c pager.diff=false \
        -c diff.external= \
        "$@"
}
repo="$(safe_git -C /testbed rev-parse --show-toplevel)"
[[ "${repo}" == /testbed ]] || exit 22
gold="$(cat /tmp/miles-r2e-expected-gold)"
published_base="$(cat /tmp/miles-r2e-published-base)"
head="$(safe_git -C "${repo}" rev-parse HEAD)"
safe_git -C "${repo}" cat-file -e "${gold}^{commit}" || exit 23
read -r -a parents <<<"$(safe_git -C "${repo}" show --no-ext-diff --no-textconv -s --format='%P' "${gold}")"
[[ "${#parents[@]}" == 1 ]] || exit 24
base="${parents[0]}"
[[ "${head}" == "${base}" ]] || exit 25
[[ -z "${published_base}" || "${published_base}" == "${base}" ]] || exit 26
safe_git -C "${repo}" cat-file -e "${base}^{commit}"
safe_git -C "${repo}" diff --no-ext-diff --no-textconv \
    --no-color --no-renames --no-indent-heuristic --diff-algorithm=myers \
    --src-prefix=a/ --dst-prefix=b/ --unified=3 --binary --full-index \
    "${base}..${gold}" -- \
    >/tmp/miles-r2e-oracle.patch
[[ -s /tmp/miles-r2e-oracle.patch ]] || exit 27
[[ -f /tmp/miles-r2e-oracle.patch && ! -L /tmp/miles-r2e-oracle.patch ]] || exit 28
[[ "$(stat -c %h /tmp/miles-r2e-oracle.patch)" == 1 ]] || exit 28
(( $(stat -c %s /tmp/miles-r2e-oracle.patch) <= 67108864 )) || exit 29
patch_sha="$(sha256sum /tmp/miles-r2e-oracle.patch | awk '{print $1}')"
printf '%s\n%s\n' "${base}" "${patch_sha}" >/tmp/miles-r2e-source-result
chmod 0600 /tmp/miles-r2e-source-result /tmp/miles-r2e-oracle.patch
[[ -f /tmp/miles-r2e-source-result && ! -L /tmp/miles-r2e-source-result ]] || exit 30
[[ "$(stat -c %h /tmp/miles-r2e-source-result)" == 1 ]] || exit 30
(( $(stat -c %s /tmp/miles-r2e-source-result) <= 65536 )) || exit 31
printf 'source-inspection-ok\n'
"""

_AGENT_ROOT_CHECK_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
repo="$(cat /opt/miles-swe/workdir)"
gitdir="$(cat /opt/miles-swe/gitdir)"
[[ -d "${repo}" && -d "${gitdir}" ]] || exit 30
gold="$(cat /tmp/miles-r2e-expected-gold)"
base="$(cat /tmp/miles-r2e-expected-base)"
for commit in "${gold}" "${base}"; do
    if git --git-dir="${gitdir}" --work-tree="${repo}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        exit 31
    fi
done
[[ "$(stat -c '%u:%g:%a' /opt/miles-swe/r2e-rootfs-attestation)" == 0:0:444 ]] || exit 32
[[ "$(cat /opt/miles-swe/r2e-rootfs-attestation)" == miles-r2e-visible-rootfs-attestation-v1 ]] || exit 32
for runtime_evidence in \
    /opt/miles-swe/r2e-runtime-imports \
    /opt/miles-swe/r2e-runtime-inventory \
    /opt/miles-swe/r2e-runtime-inventory.sha256 \
    /opt/miles-swe/r2e-runtime-links; do
    [[ "$(stat -c '%u:%g:%a' "${runtime_evidence}")" == 0:0:444 ]] || exit 32
    [[ -s "${runtime_evidence}" ]] || exit 32
done
expected_runtime_inventory_sha="$(cat /opt/miles-swe/r2e-runtime-inventory.sha256)"
[[ "${expected_runtime_inventory_sha}" =~ ^[0-9a-f]{64}$ ]] || exit 32
actual_runtime_inventory_sha="$(sha256sum /opt/miles-swe/r2e-runtime-inventory | awk '{print $1}')"
[[ "${actual_runtime_inventory_sha}" == "${expected_runtime_inventory_sha}" ]] || exit 32
[[ ! -e /r2e_tests && ! -e "${repo}/r2e_tests" && ! -e "${repo}/run_tests.sh" ]] || exit 33
if find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /run \) \
    -prune -o -type f \( \
    -name syn_issue.json -o -name expected_test_output.json -o \
    -name execution_result.json -o -name parsed_commit.json -o \
    -name modified_files.json -o -name modified_entities.json \
    \) -print -quit 2>/dev/null | grep -q .; then
    exit 34
fi
rm -f -- /tmp/miles-r2e-expected-gold /tmp/miles-r2e-expected-base
gold_text_leak=0
while IFS= read -r -d '' candidate; do
    if grep -a -F -q -- "${gold}" "${candidate}" 2>/dev/null; then
        gold_text_leak=1
        break
    fi
done < <(find / -xdev \( -path /proc -o -path /sys -o -path /dev -o \
    -path /run \) -prune -o -type f -readable -print0 2>/dev/null)
(( gold_text_leak == 0 )) || exit 35
if git --git-dir="${gitdir}" fsck --no-reflogs --unreachable --no-progress 2>&1 | grep -q .; then
    exit 36
fi
privilege_report="$(python3 /opt/miles-swe/strip_agent_privileges.py /)"
[[ "${privilege_report}" == *'modes=0 capabilities=0'* ]] || exit 37
printf 'agent-root-check-ok\n'
"""

_AGENT_USER_CHECK_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
[[ "$(id -u)" == 1000 ]] || exit 40
[[ ",$(id -G | tr ' ' ',')," != *,0,* ]] || exit 41
status=/proc/self/status
[[ "$(awk '/^NoNewPrivs:/ {print $2}' "${status}")" == 1 ]] || exit 42
cap_eff="$(awk '/^CapEff:/ {print $2}' "${status}")"
[[ "${cap_eff}" =~ ^0+$ ]] || exit 43
repo="$(cat /opt/miles-swe/workdir)"
probe="${repo}/.miles-r2e-tool-smoke"
printf 'tool-smoke\n' >"${probe}"
python3 -c 'from pathlib import Path; assert Path("'"${probe}"'").read_text() == "tool-smoke\n"'
rm -f -- "${probe}"
[[ -x "${repo}/.venv/bin/python" && -s /opt/miles-swe/r2e-runtime-imports ]] || exit 45
while IFS= read -r module; do
    [[ "${module}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || exit 45
    timeout 60 "${repo}/.venv/bin/python" -c \
        'import importlib, sys; importlib.import_module(sys.argv[1])' "${module}" || exit 45
done </opt/miles-swe/r2e-runtime-imports
if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    exit 44
fi
printf 'agent-user-check-ok\n'
"""

_NETWORK_DENIAL_SCRIPT = r"""python3 - <<'PY'
import socket

try:
    connection = socket.create_connection(("1.1.1.1", 443), timeout=2.0)
except OSError:
    raise SystemExit(0)
else:
    connection.close()
    raise SystemExit(71)
PY
"""


@dataclass(frozen=True)
class AdmissionConfig:
    """Private paths and optional selection for a resumable admission run."""

    private_manifest: Path
    admission_manifest: Path
    admitted_manifest: Path
    quarantine_manifest: Path
    work_root: Path
    r2e_execution_log_parser: Path
    instance_id: str | None = None
    limit: int | None = None
    concurrency: int = 4


@dataclass(frozen=True)
class SourceInspection:
    base_commit: str
    oracle_patch: str
    oracle_patch_sha256: str
    template_evidence: dict[str, str]


@dataclass(frozen=True)
class CheckpointOutcome:
    """One validated admitted or quarantined resumable checkpoint."""

    locked_digest: str
    disposition: Literal["admitted", "quarantined"]
    admission: dict[str, Any] | None = None
    admitted_task: dict[str, Any] | None = None
    quarantine: dict[str, Any] | None = None


class QuarantineTask(RuntimeError):
    """A task-local semantic mismatch that must not abort the whole batch."""

    def __init__(
        self,
        reason: str,
        detail: str,
        *,
        empty_reward: int | None = None,
        oracle_reward: int | None = None,
    ) -> None:
        if reason not in _QUARANTINE_REASONS:
            raise ValueError(f"invalid R2E quarantine reason: {reason}")
        if not detail or len(detail) > 512 or "\0" in detail:
            raise ValueError("invalid R2E quarantine detail")
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.empty_reward = empty_reward
        self.oracle_reward = oracle_reward


def _write_private(path: Path, content: str | bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    path.chmod(mode)


def _read_regular_bounded(path: Path, *, maximum_bytes: int) -> bytes:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise RuntimeError(f"private E2B readback is not a regular non-symlink: {path}")
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise RuntimeError(f"private E2B readback is not owner-only: {path}")
    if metadata.st_size > maximum_bytes:
        raise RuntimeError(f"private E2B readback exceeds its size limit: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or opened.st_mode & 0o077
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            != (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
        ):
            raise RuntimeError(f"private E2B readback changed before open: {path}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        try:
            path_after = os.lstat(path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"private E2B readback disappeared during read: {path}") from exc
        if (
            (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
        ):
            raise RuntimeError(f"private E2B readback changed during read: {path}")
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if len(content) > maximum_bytes:
        raise RuntimeError(f"private E2B readback exceeds its size limit: {path}")
    if len(content) != after.st_size:
        raise RuntimeError(f"private E2B readback changed during read: {path}")
    return content


def _remote_path(workspace: Path, name: str, content: str) -> Path:
    path = workspace / name
    _write_private(path, content)
    return path


def _remote_template_evidence(remote: AdmissionSandbox) -> dict[str, str]:
    evidence = dict(remote.template_evidence)
    _validate_template_evidence(evidence)
    return evidence


async def _inspect_source(
    row: Mapping[str, Any],
    backend: AdmissionBackend,
    workspace: Path,
) -> SourceInspection:
    verifier = _required_mapping(row, "verifier")
    gold_commit = _required_commit(verifier, "gold_commit")
    published_base = _optional_commit(row.get("base_commit"), "base_commit") or ""
    source_image = _source_image(row)
    source_context = workspace / "source"
    source_context.mkdir(mode=0o700)
    spec = SandboxSpec(
        role="source",
        name=f"r2e-source-{_image_digest(source_image)[:20]}",
        context_dir=source_context,
        source_image=source_image,
        expected_image=source_image,
    )
    script = _remote_path(workspace, "inspect-source.sh", _SOURCE_INSPECTION_SCRIPT)
    gold = _remote_path(workspace, "expected-gold", gold_commit + "\n")
    base = _remote_path(workspace, "published-base", published_base + "\n")
    result_path = workspace / "source-result"
    patch_path = workspace / "oracle.patch"
    _write_private(result_path, b"")
    _write_private(patch_path, b"")
    async with _sandbox(backend, spec) as remote:
        await _require_ok(
            remote,
            _NETWORK_DENIAL_SCRIPT,
            user=0,
            timeout_sec=10,
            phase="source network-denial check",
        )
        await remote.upload_file(script, "/tmp/miles-r2e-inspect.sh")
        await remote.upload_file(gold, "/tmp/miles-r2e-expected-gold")
        await remote.upload_file(base, "/tmp/miles-r2e-published-base")
        try:
            await _require_ok(
                remote,
                "chmod 0500 /tmp/miles-r2e-inspect.sh && chmod 0600 "
                "/tmp/miles-r2e-expected-gold /tmp/miles-r2e-published-base && "
                "/tmp/miles-r2e-inspect.sh",
                user=0,
                timeout_sec=300,
                phase="source inspection",
            )
        except RemoteCommandError as exc:
            raise QuarantineTask(
                "source_image_unsupported",
                f"source inspection exited with status {exc.return_code}",
            ) from exc
        await remote.download_file("/tmp/miles-r2e-source-result", result_path)
        await remote.download_file("/tmp/miles-r2e-oracle.patch", patch_path)
        template_evidence = _remote_template_evidence(remote)
    try:
        result_text = _read_regular_bounded(
            result_path,
            maximum_bytes=_MAX_SMALL_EVIDENCE_BYTES,
        ).decode("utf-8")
    except (RuntimeError, UnicodeDecodeError) as exc:
        raise QuarantineTask(
            "source_evidence_invalid",
            "source inspection returned invalid private evidence",
        ) from exc
    lines = result_text.splitlines()
    if len(lines) != 2 or _COMMIT.fullmatch(lines[0]) is None or _DIGEST.fullmatch(lines[1]) is None:
        raise QuarantineTask(
            "source_evidence_invalid",
            "source inspection returned an invalid commit/digest binding",
        )
    try:
        patch_bytes = _read_regular_bounded(
            patch_path,
            maximum_bytes=_MAX_ORACLE_PATCH_BYTES,
        )
    except RuntimeError as exc:
        raise QuarantineTask(
            "source_evidence_invalid",
            "source inspection returned an invalid oracle patch artifact",
        ) from exc
    if hashlib.sha256(patch_bytes).hexdigest() != lines[1]:
        raise QuarantineTask(
            "source_evidence_invalid",
            "source oracle patch digest does not match its evidence",
        )
    try:
        oracle_patch = patch_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QuarantineTask(
            "oracle_patch_unsupported",
            "source oracle diff is not UTF-8",
        ) from exc
    return SourceInspection(
        base_commit=lines[0],
        oracle_patch=oracle_patch,
        oracle_patch_sha256=lines[1],
        template_evidence=template_evidence,
    )


def _enrich_manifest(
    row: Mapping[str, Any],
    inspection: SourceInspection,
) -> dict[str, Any]:
    enriched = json.loads(json.dumps(row))
    enriched["base_commit"] = inspection.base_commit
    solution = enriched.get("solution")
    if not isinstance(solution, dict):
        raise ValueError("R2E private manifest solution must be an object")
    solution["oracle_patch"] = inspection.oracle_patch
    enriched["solution"] = solution
    enriched.pop("content_digest", None)
    enriched["content_digest"] = _stable_digest_without_bindings(enriched)
    return enriched


def _materialize_exact(
    manifest: dict[str, Any],
    *,
    workspace: Path,
    parser_path: Path,
) -> Path:
    private_manifest = workspace / "exact-task.private.jsonl"
    _write_private(private_manifest, json.dumps(manifest, sort_keys=True) + "\n")
    output = workspace / "harbor-tasks"
    arguments = argparse.Namespace(
        manifest=private_manifest,
        output=output,
        r2e_execution_log_parser=parser_path,
        r2e_admission_manifest=None,
        swe_rebench_log_parsers=None,
        swe_rebench_constants=None,
        swe_gym_harness_root=None,
        swe_gym_admission_manifest=None,
        allow_mutable_images=False,
        allow_unadmitted_r2e_dry_run=True,
        allow_unadmitted_swe_gym_dry_run=False,
        limit=None,
    )
    summary = materialize_module.materialize(arguments)
    if summary.get("tasks") != 1 or summary.get("schemas") != {"r2e-gym-v1": 1}:
        raise ValueError("exact R2E materialization produced an unexpected task set")
    task_dir = output / _required_text(manifest, "instance_id")
    if not (task_dir / "task.toml").is_file():
        raise ValueError("exact R2E Harbor task was not materialized")
    return task_dir


async def _check_agent_sandbox(
    manifest: Mapping[str, Any],
    task_dir: Path,
    backend: AdmissionBackend,
    workspace: Path,
) -> dict[str, str]:
    verifier = _required_mapping(manifest, "verifier")
    gold_commit = _required_commit(verifier, "gold_commit")
    base_commit = _required_commit(manifest, "base_commit")
    root_script = _remote_path(workspace, "agent-root-check.sh", _AGENT_ROOT_CHECK_SCRIPT)
    user_script = _remote_path(workspace, "agent-user-check.sh", _AGENT_USER_CHECK_SCRIPT)
    gold = _remote_path(workspace, "agent-expected-gold", gold_commit + "\n")
    base = _remote_path(workspace, "agent-expected-base", base_commit + "\n")
    spec = SandboxSpec(
        role="agent",
        name=f"r2e-agent-{_required_text(manifest, 'instance_id')}",
        context_dir=task_dir / "environment",
        task_dir=task_dir,
        expected_image=_source_image(manifest),
    )
    async with _sandbox(backend, spec) as remote:
        await _require_ok(
            remote,
            _NETWORK_DENIAL_SCRIPT,
            user=1000,
            timeout_sec=10,
            phase="agent network-denial check",
        )
        await remote.upload_file(root_script, "/tmp/miles-r2e-root-check.sh")
        await remote.upload_file(user_script, "/tmp/miles-r2e-user-check.sh")
        await remote.upload_file(gold, "/tmp/miles-r2e-expected-gold")
        await remote.upload_file(base, "/tmp/miles-r2e-expected-base")
        try:
            await _require_ok(
                remote,
                "chmod 0500 /tmp/miles-r2e-root-check.sh "
                "/tmp/miles-r2e-user-check.sh && "
                "chmod 0600 /tmp/miles-r2e-expected-gold "
                "/tmp/miles-r2e-expected-base && /tmp/miles-r2e-root-check.sh",
                user=0,
                timeout_sec=600,
                phase="agent root security check",
            )
            await _require_ok(
                remote,
                "/tmp/miles-r2e-user-check.sh",
                user=1000,
                timeout_sec=120,
                phase="agent tool/no-new-privileges check",
            )
        except RemoteCommandError as exc:
            raise QuarantineTask(
                "unsafe_agent_image",
                f"agent security check exited with status {exc.return_code}",
            ) from exc
        return _remote_template_evidence(remote)


async def _verify_patch(
    manifest: Mapping[str, Any],
    task_dir: Path,
    patch: bytes,
    backend: AdmissionBackend,
    workspace: Path,
    phase: Literal["empty", "oracle"],
) -> tuple[int, dict[str, str]]:
    local_patch = workspace / f"{phase}.patch"
    _write_private(local_patch, patch)
    reward_path = workspace / f"{phase}-reward.txt"
    report_path = workspace / f"{phase}-report.json"
    _write_private(reward_path, b"")
    _write_private(report_path, b"")
    spec = SandboxSpec(
        role="verifier",
        name=f"r2e-{phase}-verifier-{_required_text(manifest, 'instance_id')}",
        context_dir=task_dir / "tests",
        task_dir=task_dir,
        expected_image=_source_image(manifest),
    )
    async with _sandbox(backend, spec) as remote:
        await _require_ok(
            remote,
            _NETWORK_DENIAL_SCRIPT,
            user=0,
            timeout_sec=10,
            phase=f"{phase} verifier network-denial check",
        )
        await _require_ok(
            remote,
            "test ! -e /tests/.harbor-e2b-late-tests",
            user=0,
            timeout_sec=10,
            phase=f"{phase} verifier late-tests pre-upload check",
        )
        await remote.install_private_verifier(task_dir / "tests")
        await remote.upload_file(local_patch, "/tmp/miles-r2e-model.patch")
        patch_digest = hashlib.sha256(patch).hexdigest()
        try:
            await _require_ok(
                remote,
                "test \"$(sha256sum /tmp/miles-r2e-model.patch | "
                "awk '{print $1}')\" = "
                f"{patch_digest} && "
                "install -d -o root -g root -m 0700 /opt/miles-swe/collected && "
                "install -o root -g root -m 0600 /tmp/miles-r2e-model.patch "
                "/opt/miles-swe/collected/model.patch && /tests/test.sh && "
                "test -f /logs/verifier/reward.txt && "
                "test ! -L /logs/verifier/reward.txt && "
                "test \"$(stat -c %h /logs/verifier/reward.txt)\" = 1 && "
                "test \"$(stat -c %s /logs/verifier/reward.txt)\" -le 65536 && "
                "test -f /logs/verifier/report.json && "
                "test ! -L /logs/verifier/report.json && "
                "test \"$(stat -c %h /logs/verifier/report.json)\" = 1 && "
                "test \"$(stat -c %s /logs/verifier/report.json)\" -le 4194304",
                user=0,
                timeout_sec=1900,
                phase=f"{phase} verifier",
            )
        except RemoteCommandError as exc:
            raise QuarantineTask(
                "verifier_incompatible",
                f"{phase} verifier exited with status {exc.return_code}",
            ) from exc
        await remote.download_file("/logs/verifier/reward.txt", reward_path)
        await remote.download_file("/logs/verifier/report.json", report_path)
        template_evidence = _remote_template_evidence(remote)
    try:
        raw_reward = _read_regular_bounded(
            reward_path,
            maximum_bytes=_MAX_SMALL_EVIDENCE_BYTES,
        ).decode("utf-8").strip()
        report_text = _read_regular_bounded(
            report_path,
            maximum_bytes=_MAX_REPORT_BYTES,
        ).decode("utf-8")
    except (RuntimeError, UnicodeDecodeError) as exc:
        raise QuarantineTask(
            "verifier_evidence_invalid",
            f"{phase} verifier returned invalid private evidence",
        ) from exc
    if raw_reward not in {"0", "1"}:
        raise QuarantineTask(
            "verifier_evidence_invalid",
            f"{phase} verifier returned a non-binary reward",
        )
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as exc:
        raise QuarantineTask(
            "verifier_evidence_invalid",
            f"{phase} verifier returned invalid JSON",
        ) from exc
    reported_reward = report.get("reward") if isinstance(report, dict) else None
    if (
        type(reported_reward) is not int
        or reported_reward not in {0, 1}
        or reported_reward != int(raw_reward)
    ):
        raise QuarantineTask(
            "verifier_evidence_invalid",
            f"{phase} verifier report and reward disagree",
        )
    return int(raw_reward), template_evidence


async def _admit_one(
    row: dict[str, Any],
    *,
    original: dict[str, Any],
    backend: AdmissionBackend,
    config: AdmissionConfig,
    workspace: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inspection = await _inspect_source(row, backend, workspace)
    enriched = _enrich_manifest(row, inspection)
    try:
        task_dir = _materialize_exact(
            enriched,
            workspace=workspace,
            parser_path=config.r2e_execution_log_parser,
        )
    except ValueError as exc:
        raise QuarantineTask(
            "materialization_unsupported",
            "normalized task cannot be materialized under the safe R2E policy",
        ) from exc
    task_tree_sha256 = materialize_module._task_tree_sha256(task_dir)
    agent_evidence = await _check_agent_sandbox(enriched, task_dir, backend, workspace)
    empty_reward, empty_evidence = await _verify_patch(
        enriched,
        task_dir,
        b"",
        backend,
        workspace,
        "empty",
    )
    oracle_reward, oracle_evidence = await _verify_patch(
        enriched,
        task_dir,
        inspection.oracle_patch.encode("utf-8"),
        backend,
        workspace,
        "oracle",
    )
    if empty_reward != 0 or oracle_reward != 1:
        raise QuarantineTask(
            "golden_outcome_mismatch",
            "empty/oracle verifier outcomes do not match 0/1",
            empty_reward=empty_reward,
            oracle_reward=oracle_reward,
        )
    if materialize_module._task_tree_sha256(task_dir) != task_tree_sha256:
        raise RuntimeError("materialized R2E task changed during live admission")
    _validate_fresh_verifier_evidence(empty_evidence, oracle_evidence)
    _validate_source_verifier_template_evidence(
        inspection.template_evidence,
        empty_evidence,
    )
    admission = _admission_record(
        original=original,
        enriched=enriched,
        inspection=inspection,
        empty_reward=empty_reward,
        oracle_reward=oracle_reward,
        agent_evidence=agent_evidence,
        empty_evidence=empty_evidence,
        oracle_evidence=oracle_evidence,
        task_tree_sha256=task_tree_sha256,
    )
    return enriched, admission


def _admission_record(
    *,
    original: Mapping[str, Any],
    enriched: Mapping[str, Any],
    inspection: SourceInspection,
    empty_reward: int,
    oracle_reward: int,
    agent_evidence: Mapping[str, str],
    empty_evidence: Mapping[str, str],
    oracle_evidence: Mapping[str, str],
    task_tree_sha256: str,
) -> dict[str, Any]:
    requested_image, resolved_image, input_content_digest = _image_provenance(original)
    return {
        "schema_version": _ADMISSION_SCHEMA,
        "instance_id": _required_text(enriched, "instance_id"),
        "task_digest": _required_digest(enriched, "task_digest"),
        "input_content_digest": input_content_digest,
        "locked_content_digest": _required_digest(original, "content_digest"),
        "content_digest": _required_digest(enriched, "content_digest"),
        "source_image_requested": requested_image,
        "source_image_resolved": resolved_image,
        "source_image": resolved_image,
        "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
        "base_commit": inspection.base_commit,
        "oracle_patch_sha256": inspection.oracle_patch_sha256,
        "admitted_task_tree_sha256": task_tree_sha256,
        "e2b_sandbox_evidence": {
            "source": inspection.template_evidence,
            "agent": dict(agent_evidence),
            "empty_verifier": dict(empty_evidence),
            "oracle_verifier": dict(oracle_evidence),
        },
        "checks": {
            **_REQUIRED_CHECKS,
            "empty_reward": empty_reward,
            "oracle_reward": oracle_reward,
        },
    }


async def admit_r2e_tasks(
    config: AdmissionConfig,
    backend: AdmissionBackend,
) -> dict[str, int]:
    """Admit selected tasks with bounded E2B concurrency and private checkpoints."""
    if config.concurrency <= 0:
        raise ValueError("R2E admission concurrency must be positive")
    checkpoint_dir = _checkpoint_directory(config.admission_manifest)
    oci_image_lock._require_distinct_paths(
        config.private_manifest,
        config.admission_manifest,
        config.admitted_manifest,
        config.quarantine_manifest,
        checkpoint_dir,
        config.work_root,
        config.r2e_execution_log_parser,
    )
    _validate_private_input(config.private_manifest)
    _validate_private_dependency(config.r2e_execution_log_parser)
    input_fingerprint = oci_image_lock._capture_private_fingerprint(config.private_manifest)
    config.work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    work_metadata = os.lstat(config.work_root)
    if stat.S_ISLNK(work_metadata.st_mode) or not stat.S_ISDIR(work_metadata.st_mode):
        raise PermissionError("R2E admission work root must be a non-symlink directory")
    if work_metadata.st_uid != os.getuid():
        raise PermissionError("R2E admission work root is owned by another user")
    os.chmod(config.work_root, 0o700)
    oci_image_lock._ensure_private_directory(checkpoint_dir)
    checkpoints = _load_checkpoint_index(checkpoint_dir)
    selected, completed, quarantined, skipped = await _run_selected_admissions(
        config=config,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        checkpoints=checkpoints,
    )
    _compact_admission_outputs(
        config=config,
        checkpoints=checkpoints,
        input_fingerprint=input_fingerprint,
    )
    return {
        "selected": selected,
        "admitted": completed,
        "quarantined": quarantined,
        "resumed": skipped,
    }


def _validate_candidate(row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != "miles-swe-task-v1":
        raise ValueError("unsupported private R2E task schema")
    oci_image_lock.validate_task_image_policy(row)
    if row.get("source_schema") != "r2e-gym-v1":
        raise ValueError("R2E admission input contains a non-R2E task")
    instance_id = _required_text(row, "instance_id")
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise ValueError(f"invalid private R2E instance_id: {instance_id!r}")
    _required_digest(row, "task_digest")
    content_digest = _required_digest(row, "content_digest")
    if _stable_digest_without_bindings(row) != content_digest:
        raise ValueError(f"private R2E content digest mismatch for {instance_id}")
    source_image = _source_image_text(row)
    if _IMMUTABLE_IMAGE.fullmatch(source_image) is None:
        raise ValueError("R2E source image must use an immutable name@sha256 digest")
    _image_provenance(row)
    verifier = _required_mapping(row, "verifier")
    if verifier.get("kind") != "r2e-expected-pytest-map-v1":
        raise ValueError(f"private R2E verifier kind mismatch for {instance_id}")
    _required_commit(verifier, "gold_commit")
    _optional_commit(row.get("base_commit"), "base_commit")


async def _run_selected_admissions(
    *,
    config: AdmissionConfig,
    backend: AdmissionBackend,
    checkpoint_dir: Path,
    checkpoints: dict[str, Path],
) -> tuple[int, int, int, int]:
    running: set[
        asyncio.Task[tuple[str, Path, Literal["admitted", "quarantined"]]]
    ] = set()
    seen: set[str] = set()
    selected = 0
    completed = 0
    quarantined = 0
    resumed = 0
    try:
        for original in _read_jsonl(config.private_manifest):
            _validate_candidate(original)
            if config.instance_id is not None and original.get("instance_id") != config.instance_id:
                continue
            if config.limit is not None and selected >= config.limit:
                break
            selected += 1
            locked_digest = _required_digest(original, "content_digest")
            if locked_digest in seen:
                raise ValueError(f"duplicate private R2E task content digest: {locked_digest}")
            seen.add(locked_digest)
            checkpoint = checkpoints.get(locked_digest)
            if checkpoint is not None:
                _load_checkpoint(checkpoint, original=original)
                resumed += 1
                continue
            running.add(
                asyncio.create_task(
                    _admit_and_checkpoint(
                        original,
                        backend=backend,
                        config=config,
                        checkpoint_dir=checkpoint_dir,
                    )
                )
            )
            if len(running) >= config.concurrency:
                running, finished = await _wait_for_admissions(running)
                for digest, path, disposition in finished:
                    checkpoints[digest] = path
                    if disposition == "admitted":
                        completed += 1
                    else:
                        quarantined += 1
        while running:
            running, finished = await _wait_for_admissions(running)
            for digest, path, disposition in finished:
                checkpoints[digest] = path
                if disposition == "admitted":
                    completed += 1
                else:
                    quarantined += 1
    except BaseException:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise
    if selected == 0:
        raise ValueError("no R2E task matched the admission selection")
    return selected, completed, quarantined, resumed


async def _wait_for_admissions(
    running: set[
        asyncio.Task[tuple[str, Path, Literal["admitted", "quarantined"]]]
    ],
) -> tuple[
    set[asyncio.Task[tuple[str, Path, Literal["admitted", "quarantined"]]]],
    list[tuple[str, Path, Literal["admitted", "quarantined"]]],
]:
    done, pending = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
    return set(pending), [task.result() for task in done]


async def _admit_and_checkpoint(
    original: dict[str, Any],
    *,
    backend: AdmissionBackend,
    config: AdmissionConfig,
    checkpoint_dir: Path,
) -> tuple[str, Path, Literal["admitted", "quarantined"]]:
    workspace = Path(tempfile.mkdtemp(prefix=".r2e-admit-", dir=config.work_root))
    workspace.chmod(0o700)
    outcome: tuple[dict[str, Any], dict[str, Any]] | None = None
    quarantine: QuarantineTask | None = None
    admission_failure: BaseException | None = None
    try:
        outcome = await _admit_one(
            original,
            original=original,
            backend=backend,
            config=config,
            workspace=workspace,
        )
    except QuarantineTask as exc:
        quarantine = exc
    except BaseException as exc:
        admission_failure = exc
    cleanup_failure: BaseException | None = None
    try:
        _remove_admission_workspace(workspace)
    except BaseException as exc:
        cleanup_failure = exc
    if admission_failure is not None and cleanup_failure is not None:
        raise cleanup_failure from admission_failure
    if quarantine is not None and cleanup_failure is not None:
        raise cleanup_failure from quarantine
    if admission_failure is not None:
        raise admission_failure
    if cleanup_failure is not None:
        raise cleanup_failure
    locked_digest = _required_digest(original, "content_digest")
    checkpoint = checkpoint_dir / f"{locked_digest}.json"
    if quarantine is not None:
        record = _quarantine_record(original, quarantine)
        _write_quarantine_checkpoint(
            checkpoint,
            locked_digest=locked_digest,
            quarantine=record,
        )
        return locked_digest, checkpoint, "quarantined"
    if outcome is None:
        raise RuntimeError("R2E admission produced no outcome")
    admitted, admission = outcome
    _validate_checkpoint_binding(
        original=original,
        locked_digest=locked_digest,
        admission=admission,
        admitted=admitted,
    )
    _write_checkpoint(
        checkpoint,
        locked_digest=locked_digest,
        admission=admission,
        admitted=admitted,
    )
    return locked_digest, checkpoint, "admitted"


def _quarantine_record(
    original: Mapping[str, Any],
    failure: QuarantineTask,
) -> dict[str, Any]:
    _validate_candidate(original)
    return {
        "schema_version": _QUARANTINE_SCHEMA,
        "instance_id": _required_text(original, "instance_id"),
        "task_digest": _required_digest(original, "task_digest"),
        "locked_content_digest": _required_digest(original, "content_digest"),
        "reason": failure.reason,
        "reason_detail": failure.detail,
        "observed_empty_reward": failure.empty_reward,
        "observed_oracle_reward": failure.oracle_reward,
    }


def _remove_admission_workspace(workspace: Path) -> None:
    materialize_module._remove_private_tree(workspace)


def _checkpoint_directory(admission_manifest: Path) -> Path:
    return admission_manifest.with_name(admission_manifest.name + ".d")


def _load_checkpoint_index(directory: Path) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        outcome = _load_checkpoint(path)
        if path.stem != outcome.locked_digest:
            raise ValueError(f"R2E checkpoint filename does not bind its input: {path}")
        if outcome.locked_digest in checkpoints:
            raise ValueError(
                f"duplicate R2E admission checkpoint: {outcome.locked_digest}"
            )
        checkpoints[outcome.locked_digest] = path
    return checkpoints


def _load_checkpoint(
    path: Path,
    *,
    original: Mapping[str, Any] | None = None,
) -> CheckpointOutcome:
    _validate_private_input(path)
    values = list(_read_jsonl(path))
    if len(values) != 1:
        raise ValueError(f"invalid R2E admission checkpoint: {path}")
    value = values[0]
    schema = value.get("schema_version")
    if schema not in {_CHECKPOINT_SCHEMA, _LEGACY_CHECKPOINT_SCHEMA}:
        raise ValueError(f"invalid R2E admission checkpoint: {path}")
    locked_digest = _required_digest(value, "locked_content_digest")
    disposition = value.get("disposition", "admitted")
    if disposition == "admitted":
        expected = {
            "schema_version",
            "locked_content_digest",
            "admission",
            "admitted_task",
        }
        if schema == _CHECKPOINT_SCHEMA:
            expected.add("disposition")
        if set(value) != expected:
            raise ValueError("R2E admitted checkpoint field set is invalid")
        admission = dict(_required_mapping(value, "admission"))
        admitted = dict(_required_mapping(value, "admitted_task"))
        _validate_checkpoint_binding(
            original=original,
            locked_digest=locked_digest,
            admission=admission,
            admitted=admitted,
        )
        return CheckpointOutcome(
            locked_digest=locked_digest,
            disposition="admitted",
            admission=admission,
            admitted_task=admitted,
        )
    if disposition != "quarantined" or schema != _CHECKPOINT_SCHEMA:
        raise ValueError("R2E checkpoint disposition is invalid")
    if set(value) != {
        "schema_version",
        "locked_content_digest",
        "disposition",
        "quarantine",
    }:
        raise ValueError("R2E quarantine checkpoint field set is invalid")
    quarantine = dict(_required_mapping(value, "quarantine"))
    _validate_quarantine_record(
        quarantine,
        original=original,
        locked_digest=locked_digest,
    )
    return CheckpointOutcome(
        locked_digest=locked_digest,
        disposition="quarantined",
        quarantine=quarantine,
    )


def _write_checkpoint(
    path: Path,
    *,
    locked_digest: str,
    admission: Mapping[str, Any],
    admitted: Mapping[str, Any],
) -> None:
    value = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "locked_content_digest": locked_digest,
        "disposition": "admitted",
        "admission": dict(admission),
        "admitted_task": dict(admitted),
    }
    if path.exists() or path.is_symlink():
        existing = _load_checkpoint(path)
        if (
            existing.disposition != "admitted"
            or existing.admission != admission
            or existing.admitted_task != admitted
        ):
            raise ValueError(f"conflicting R2E admission checkpoint: {path}")
        return
    _atomic_write_jsonl(path, [value])


def _write_quarantine_checkpoint(
    path: Path,
    *,
    locked_digest: str,
    quarantine: Mapping[str, Any],
) -> None:
    _validate_quarantine_record(
        quarantine,
        original=None,
        locked_digest=locked_digest,
    )
    value = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "locked_content_digest": locked_digest,
        "disposition": "quarantined",
        "quarantine": dict(quarantine),
    }
    if path.exists() or path.is_symlink():
        existing = _load_checkpoint(path)
        if (
            existing.disposition != "quarantined"
            or existing.quarantine != quarantine
        ):
            raise ValueError(f"conflicting R2E admission checkpoint: {path}")
        return
    _atomic_write_jsonl(path, [value])


def _validate_quarantine_record(
    value: Mapping[str, Any],
    *,
    original: Mapping[str, Any] | None,
    locked_digest: str,
) -> None:
    if set(value) != {
        "schema_version",
        "instance_id",
        "task_digest",
        "locked_content_digest",
        "reason",
        "reason_detail",
        "observed_empty_reward",
        "observed_oracle_reward",
    }:
        raise ValueError("R2E quarantine record field set is invalid")
    if value.get("schema_version") != _QUARANTINE_SCHEMA:
        raise ValueError("R2E quarantine record schema is invalid")
    instance_id = _required_text(value, "instance_id")
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise ValueError("R2E quarantine instance_id is invalid")
    task_digest = _required_digest(value, "task_digest")
    if _required_digest(value, "locked_content_digest") != locked_digest:
        raise ValueError("R2E quarantine locked-content binding mismatch")
    reason = value.get("reason")
    if reason not in _QUARANTINE_REASONS:
        raise ValueError("R2E quarantine reason is invalid")
    detail = value.get("reason_detail")
    if not isinstance(detail, str) or not detail or len(detail) > 512 or "\0" in detail:
        raise ValueError("R2E quarantine detail is invalid")
    empty_reward = value.get("observed_empty_reward")
    oracle_reward = value.get("observed_oracle_reward")
    if reason == "golden_outcome_mismatch":
        if (
            type(empty_reward) is not int
            or empty_reward not in {0, 1}
            or type(oracle_reward) is not int
            or oracle_reward not in {0, 1}
            or (empty_reward, oracle_reward) == (0, 1)
        ):
            raise ValueError("R2E quarantine golden outcome is invalid")
    elif empty_reward is not None or oracle_reward is not None:
        raise ValueError("R2E non-verdict quarantine must not claim rewards")
    if original is not None:
        _validate_candidate(original)
        if (
            _required_digest(original, "content_digest") != locked_digest
            or _required_text(original, "instance_id") != instance_id
            or _required_digest(original, "task_digest") != task_digest
        ):
            raise ValueError("R2E quarantine resume binding mismatch")


def _validate_checkpoint_binding(
    *,
    original: Mapping[str, Any] | None,
    locked_digest: str,
    admission: Mapping[str, Any],
    admitted: Mapping[str, Any],
) -> None:
    _validate_admission_record(admission)
    _validate_candidate(admitted)
    if _required_digest(admission, "locked_content_digest") != locked_digest:
        raise ValueError("R2E checkpoint locked-content binding mismatch")
    shared = {
        "instance_id": _required_text(admitted, "instance_id"),
        "task_digest": _required_digest(admitted, "task_digest"),
        "content_digest": _required_digest(admitted, "content_digest"),
    }
    if any(admission.get(key) != value for key, value in shared.items()):
        raise ValueError("R2E checkpoint final task identity mismatch")
    requested, resolved, input_digest = _image_provenance(admitted)
    if (
        admission.get("source_image_requested") != requested
        or admission.get("source_image_resolved") != resolved
        or admission.get("source_image") != resolved
        or admission.get("input_content_digest") != input_digest
    ):
        raise ValueError("R2E checkpoint OCI provenance mismatch")
    base_commit = _required_commit(admitted, "base_commit")
    solution = _required_mapping(admitted, "solution")
    oracle_patch = _required_text(solution, "oracle_patch")
    if len(oracle_patch.encode("utf-8")) > _MAX_ORACLE_PATCH_BYTES:
        raise ValueError("R2E checkpoint oracle patch exceeds its size limit")
    if (
        admission.get("base_commit") != base_commit
        or admission.get("oracle_patch_sha256")
        != hashlib.sha256(oracle_patch.encode("utf-8")).hexdigest()
    ):
        raise ValueError("R2E checkpoint base/oracle binding mismatch")
    if original is not None:
        _validate_candidate(original)
        original_requested, original_resolved, original_input = _image_provenance(original)
        if (
            _required_digest(original, "content_digest") != locked_digest
            or _required_text(original, "instance_id") != shared["instance_id"]
            or _required_digest(original, "task_digest") != shared["task_digest"]
            or (original_requested, original_resolved, original_input)
            != (requested, resolved, input_digest)
        ):
            raise ValueError("R2E checkpoint resume binding mismatch")


def _validate_template_evidence(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
        "sandbox_id",
    }
    if set(value) != expected_keys:
        raise ValueError("R2E E2B template evidence field set is invalid")
    for key in ("template_id", "build_id", "sandbox_id"):
        item = value.get(key)
        if not isinstance(item, str) or _E2B_ID.fullmatch(item) is None:
            raise ValueError(f"R2E E2B {key} evidence is invalid")
    for key in ("alias_sha256", "template_identity_sha256"):
        item = value.get(key)
        if not isinstance(item, str) or _DIGEST.fullmatch(item) is None:
            raise ValueError(f"R2E E2B {key} evidence is invalid")


def _validate_fresh_verifier_evidence(
    empty: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> None:
    _validate_template_evidence(empty)
    _validate_template_evidence(oracle)
    template_keys = {
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
    }
    if any(empty.get(key) != oracle.get(key) for key in template_keys):
        raise ValueError("R2E empty/oracle verifiers did not share one exact template pin")
    if empty.get("sandbox_id") == oracle.get("sandbox_id"):
        raise ValueError("R2E empty/oracle verification did not use fresh sandboxes")


def _validate_source_verifier_template_evidence(
    source: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> None:
    _validate_template_evidence(source)
    _validate_template_evidence(verifier)
    template_keys = {
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
    }
    if any(source.get(key) != verifier.get(key) for key in template_keys):
        raise ValueError("R2E source/verifier phases did not share one exact image template")


def _validate_admission_record(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != _ADMISSION_SCHEMA:
        raise ValueError("unsupported R2E admission record schema")
    instance_id = _required_text(value, "instance_id")
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise ValueError("R2E admission instance_id is invalid")
    for key in (
        "task_digest",
        "input_content_digest",
        "locked_content_digest",
        "content_digest",
        "oracle_patch_sha256",
        "admitted_task_tree_sha256",
    ):
        _required_digest(value, key)
    _required_commit(value, "base_commit")
    requested = _required_text(value, "source_image_requested")
    oci_image_lock.parse_image_reference(requested)
    resolved = _required_text(value, "source_image_resolved")
    if _IMMUTABLE_IMAGE.fullmatch(resolved) is None or value.get("source_image") != resolved:
        raise ValueError("R2E admission resolved image binding is invalid")
    if value.get("image_publisher_policy") != oci_image_lock.IMAGE_PUBLISHER_POLICY:
        raise ValueError("R2E admission image publisher policy is invalid")
    checks = _required_mapping(value, "checks")
    if set(checks) != set(_REQUIRED_CHECKS):
        raise ValueError("R2E admission check set is invalid")
    for key, expected in _REQUIRED_CHECKS.items():
        actual = checks.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"R2E admission check {key} is invalid")
    evidence = _required_mapping(value, "e2b_sandbox_evidence")
    expected_roles = {"source", "agent", "empty_verifier", "oracle_verifier"}
    if set(evidence) != expected_roles:
        raise ValueError("R2E E2B sandbox evidence role set is invalid")
    records = {role: _required_mapping(evidence, role) for role in expected_roles}
    for record in records.values():
        _validate_template_evidence(record)
    _validate_fresh_verifier_evidence(
        records["empty_verifier"],
        records["oracle_verifier"],
    )
    _validate_source_verifier_template_evidence(
        records["source"],
        records["empty_verifier"],
    )
    sandbox_ids = {_required_text(record, "sandbox_id") for record in records.values()}
    if len(sandbox_ids) != len(records):
        raise ValueError("R2E admission reused an E2B sandbox across phases")


def _compact_admission_outputs(
    *,
    config: AdmissionConfig,
    checkpoints: Mapping[str, Path],
    input_fingerprint: Any,
) -> None:
    def unchanged() -> None:
        oci_image_lock._assert_private_unchanged(
            config.private_manifest,
            input_fingerprint,
        )

    _atomic_write_jsonl(
        config.admitted_manifest,
        _checkpointed_rows(
            config.private_manifest,
            checkpoints,
            disposition="admitted",
            field="admitted_task",
        ),
        before_replace=unchanged,
    )
    _atomic_write_jsonl(
        config.admission_manifest,
        _checkpointed_rows(
            config.private_manifest,
            checkpoints,
            disposition="admitted",
            field="admission",
        ),
        before_replace=unchanged,
    )
    _atomic_write_jsonl(
        config.quarantine_manifest,
        _checkpointed_rows(
            config.private_manifest,
            checkpoints,
            disposition="quarantined",
            field="quarantine",
        ),
        before_replace=unchanged,
    )


def _checkpointed_rows(
    private_manifest: Path,
    checkpoints: Mapping[str, Path],
    *,
    disposition: Literal["admitted", "quarantined"],
    field: Literal["admission", "admitted_task", "quarantine"],
) -> Iterable[Mapping[str, Any]]:
    if disposition == "admitted" and field == "quarantine":
        raise ValueError("R2E admitted output cannot select quarantine records")
    if disposition == "quarantined" and field != "quarantine":
        raise ValueError("R2E quarantine output cannot select admitted records")
    seen: set[str] = set()
    for original in _read_jsonl(private_manifest):
        _validate_candidate(original)
        locked_digest = _required_digest(original, "content_digest")
        if locked_digest in seen:
            raise ValueError(f"duplicate private R2E task content digest: {locked_digest}")
        seen.add(locked_digest)
        checkpoint = checkpoints.get(locked_digest)
        if checkpoint is None:
            continue
        outcome = _load_checkpoint(checkpoint, original=original)
        if outcome.disposition != disposition:
            continue
        selected = getattr(outcome, field)
        if selected is None:
            raise ValueError("R2E checkpoint output is missing its selected record")
        yield selected


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    yield from oci_image_lock._read_jsonl(path)


def _atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_metadata = os.lstat(path.parent)
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise PermissionError(f"private output parent must be a non-symlink directory: {path.parent}")
    if parent_metadata.st_uid != os.getuid():
        raise PermissionError(f"private output parent is owned by another user: {path.parent}")
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        target_metadata = os.lstat(path)
        if (
            stat.S_ISLNK(target_metadata.st_mode)
            or not stat.S_ISREG(target_metadata.st_mode)
            or target_metadata.st_nlink != 1
        ):
            raise PermissionError(f"private output must be a regular non-symlink: {path}")
        if target_metadata.st_uid != os.getuid() or target_metadata.st_mode & 0o077:
            raise PermissionError(f"private output must be owner-only: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_private_input(path: Path) -> None:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise PermissionError(f"private input must be a regular non-symlink: {path}")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"private input is not owned by the current user: {path}")
    if metadata.st_mode & 0o077:
        raise PermissionError(f"private input must deny group/other access: {path}")


def _validate_private_dependency(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"pinned R2E parser is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"pinned R2E parser is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = materialize_module._R2E_PARSER_SHA256
    if actual != expected:
        raise ValueError("pinned R2E parser checksum mismatch")


def _stable_digest_without_bindings(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_digest", None)
    payload.pop("task_digest", None)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be an object")
    return result


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _required_digest(value: Mapping[str, Any], key: str) -> str:
    result = _required_text(value, key)
    if _DIGEST.fullmatch(result) is None:
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return result


def _required_commit(value: Mapping[str, Any], key: str) -> str:
    result = _required_text(value, key)
    if _COMMIT.fullmatch(result) is None:
        raise ValueError(f"{key} must be a lowercase full Git commit")
    return result


def _optional_commit(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{key} must be null or a lowercase full Git commit")
    return value


def _source_image(value: Mapping[str, Any]) -> str:
    source_image = _source_image_text(value)
    if _IMMUTABLE_IMAGE.fullmatch(source_image) is None:
        raise ValueError("R2E source image must use an immutable name@sha256 digest")
    return source_image


def _source_image_text(value: Mapping[str, Any]) -> str:
    sandbox = _required_mapping(value, "sandbox")
    return _required_text(sandbox, "source_image")


def _image_provenance(value: Mapping[str, Any]) -> tuple[str, str, str]:
    sandbox = _required_mapping(value, "sandbox")
    resolved = _source_image(value)
    lock = sandbox.get("image_lock")
    if lock is None:
        raise ValueError("R2E semantic admission requires generic OCI image-lock provenance")
    if not isinstance(lock, dict) or lock.get("schema_version") != "miles-oci-image-lock-v1":
        raise ValueError("R2E task has invalid embedded OCI image-lock provenance")
    requested = _required_text(lock, "source_image_requested")
    _, display_registry, repository, reference = oci_image_lock.parse_image_reference(requested)
    if _required_text(lock, "source_image_resolved") != resolved:
        raise ValueError("R2E task OCI image lock does not bind the resolved image")
    input_digest = _required_digest(lock, "input_content_digest")
    child_digest = _required_text(lock, "child_manifest_digest")
    if not resolved.endswith("@" + child_digest):
        raise ValueError("R2E task OCI child digest does not bind the resolved image")
    if resolved != f"{display_registry}/{repository}@{child_digest}":
        raise ValueError("R2E task OCI lock does not use the canonical resolved image")
    if lock.get("platform") != {"os": "linux", "architecture": "amd64"}:
        raise ValueError("R2E task OCI image lock is not linux/amd64")
    index_digest = lock.get("index_digest")
    if index_digest is not None and (
        not isinstance(index_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", index_digest) is None
    ):
        raise ValueError("R2E task OCI index digest is invalid")
    if reference.startswith("sha256:") and (
        (index_digest is None and child_digest != reference)
        or (index_digest is not None and index_digest != reference)
    ):
        raise ValueError("R2E task OCI lock does not bind the requested digest")
    return requested, resolved, input_digest


def _image_digest(image: str) -> str:
    return image.rsplit("@sha256:", maxsplit=1)[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--admitted-manifest", type=Path, required=True)
    parser.add_argument("--quarantine-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--r2e-execution-log-parser", type=Path, required=True)
    parser.add_argument("--instance-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.instance_id is not None and _INSTANCE_ID.fullmatch(args.instance_id) is None:
        parser.error("--instance-id is invalid")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return args


def main() -> None:
    args = _parse_args()
    config = AdmissionConfig(
        private_manifest=args.private_manifest,
        admission_manifest=args.admission_manifest,
        admitted_manifest=args.admitted_manifest,
        quarantine_manifest=args.quarantine_manifest,
        work_root=args.work_root,
        r2e_execution_log_parser=args.r2e_execution_log_parser,
        instance_id=args.instance_id,
        limit=args.limit,
        concurrency=args.concurrency,
    )
    summary = asyncio.run(admit_r2e_tasks(config, NativeHarborE2BBackend()))
    print(
        "R2E native-E2B admission complete: "
        f"selected={summary['selected']} admitted={summary['admitted']} "
        f"quarantined={summary['quarantined']} resumed={summary['resumed']}"
    )


if __name__ == "__main__":
    main()
