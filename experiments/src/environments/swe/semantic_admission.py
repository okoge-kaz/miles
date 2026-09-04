"""Resumable semantic admission shared by repository-level SWE environments."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from experiments.src.environments.swe import e2b_admission
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock
from experiments.src.environments.swe import timeouts

_CHECKPOINT_SCHEMA = "miles-swe-semantic-admission-checkpoint-v1"
_QUARANTINE_SCHEMA = "miles-swe-semantic-quarantine-v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_IMAGE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,447}@sha256:[0-9a-f]{64}"
)
_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,239}")
_REASON = re.compile(r"[a-z][a-z0-9_]{2,95}")
_MAX_PATCH_BYTES = 64 * 1024 * 1024
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_SMALL_EVIDENCE_BYTES = 64 * 1024

_SOURCE_INSPECTION_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/tmp/miles-swe-source-home
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_ATTR_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1
unset BASH_ENV CDPATH ENV GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR \
    GIT_DIR GIT_EXTERNAL_DIFF GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
    GIT_SSH GIT_SSH_COMMAND GIT_WORK_TREE SSH_ASKPASS
for command in git sha256sum stat; do
    command -v "${command}" >/dev/null 2>&1 || exit 20
done
if [[ -d /testbed/.git && ! -L /testbed/.git ]]; then
    repo=/testbed
else
    repo="$(pwd -P)"
    while [[ "${repo}" != / && ! -d "${repo}/.git" ]]; do
        repo="${repo%/*}"
        [[ -n "${repo}" ]] || repo=/
    done
fi
[[ -d "${repo}/.git" && ! -L "${repo}/.git" ]] || exit 21
source_gitdir="$(cd "${repo}/.git" && pwd -P)"
[[ "${source_gitdir}" == "${repo}/.git" ]] || exit 21
base="$(cat /tmp/miles-swe-expected-base)"
[[ "${base}" =~ ^[0-9a-f]{40}$ ]] || exit 22
safe_git() {
    command git --no-pager -c safe.directory="${repo}" \
        -c core.fsmonitor=false -c core.hooksPath=/dev/null \
        -c core.pager=cat -c pager.diff=false -c diff.external= \
        -C "${repo}" "$@"
}
[[ "$(safe_git rev-parse HEAD)" == "${base}" ]] || exit 23
tree="$(safe_git rev-parse HEAD^{tree})"
[[ "${tree}" =~ ^[0-9a-f]{40}$ ]] || exit 24
for patch in /tmp/miles-swe-oracle.patch /tmp/miles-swe-hidden-tests.patch; do
    [[ -f "${patch}" && ! -L "${patch}" ]] || exit 25
    (( $(stat -c %s "${patch}") <= 67108864 )) || exit 26
done
safe_git reset --hard "${base}" >/dev/null
safe_git clean -ffdx >/dev/null
while IFS= read -r -d '' _untracked_path; do
    exit 30
done < <(safe_git ls-files --others -z)
safe_git apply --check --binary /tmp/miles-swe-hidden-tests.patch || exit 27
safe_git apply --check --binary /tmp/miles-swe-oracle.patch || exit 28
safe_git apply --binary /tmp/miles-swe-oracle.patch
safe_git apply --check --binary /tmp/miles-swe-hidden-tests.patch || exit 29
safe_git reset --hard "${base}" >/dev/null
printf '%s\n%s\n' "${base}" "${tree}" >/tmp/miles-swe-source-result
chmod 0600 /tmp/miles-swe-source-result
printf 'source-inspection-ok\n'
"""

_AGENT_ROOT_CHECK_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
repo="$(cat /opt/miles-swe/workdir)"
gitdir="$(cat /opt/miles-swe/gitdir)"
base="$(cat /tmp/miles-swe-expected-base)"
tree="$(cat /tmp/miles-swe-expected-tree)"
[[ -d "${repo}" && -d "${gitdir}" ]] || exit 30
[[ ! -e /tests/.harbor-e2b-late-tests ]] || exit 31
head="$(git --git-dir="${gitdir}" --work-tree="${repo}" rev-parse HEAD)"
[[ "$(git --git-dir="${gitdir}" --work-tree="${repo}" rev-parse HEAD^{tree})" == "${tree}" ]] || exit 32
[[ -z "$(git --git-dir="${gitdir}" --work-tree="${repo}" show -s --format='%P' "${head}")" ]] || exit 33
if git --git-dir="${gitdir}" --work-tree="${repo}" cat-file -e "${base}^{commit}" 2>/dev/null; then
    exit 34
fi
[[ -z "$(git --git-dir="${gitdir}" --work-tree="${repo}" fsck --no-reflogs --unreachable --no-progress 2>&1)" ]] || exit 35
privilege_report="$(python3 /opt/miles-swe/strip_agent_privileges.py /)"
[[ "${privilege_report}" == *'modes=0 capabilities=0'* ]] || exit 36
[[ "$(stat -c '%u:%a' /opt/miles-swe/workdir)" == 0:444 ]] || exit 37
[[ "$(stat -c '%u' "${gitdir}")" == 0 ]] || exit 38
[[ "$(stat -c '%u:%a' /opt/miles-swe/runtime-policy)" == 0:444 ]] || exit 39
runtime_policy="$(cat /opt/miles-swe/runtime-policy)"
case "${runtime_policy}" in
    none) ;;
    npm-node-modules-v2)
        [[ -L "${repo}/node_modules" ]] || exit 46
        [[ "$(readlink -f -- "${repo}/node_modules")" == \
            /opt/miles-swe/runtime/node_modules ]] || exit 47
        [[ "$(stat -c '%u' /opt/miles-swe/runtime/node_modules)" == 0 ]] || exit 48
        [[ -z "$(find /opt/miles-swe/runtime/node_modules -xdev \
            -perm /022 -print -quit)" ]] || exit 49
        [[ -f "${repo}/package-lock.json" ]] || exit 56
        [[ "$(stat -c '%u:%a' /opt/miles-swe/npm-repo-runtime-paths)" == \
            0:444 ]] || exit 74
        [[ "$(stat -c '%u:%a' /opt/miles-swe/npm-repo-runtime.inventory)" == \
            0:444 ]] || exit 75
        [[ "$(stat -c '%u:%a' \
            /opt/miles-swe/npm-repo-runtime.inventory.sha256)" == \
            0:444 ]] || exit 76
        expected_npm_inventory_digest="$(cat \
            /opt/miles-swe/npm-repo-runtime.inventory.sha256)"
        [[ "$(sha256sum /opt/miles-swe/npm-repo-runtime.inventory \
            | awk '{print $1}')" == "${expected_npm_inventory_digest}" ]] || exit 77
        recomputed_npm_inventory=/tmp/miles-swe-npm-repo-runtime.inventory
        : >"${recomputed_npm_inventory}"
        while IFS= read -r relative; do
            [[ "${relative}" == dist ]] || exit 78
            [[ -d "${repo}/${relative}" && ! -L "${repo}/${relative}" && \
                "$(stat -c '%u' "${repo}/${relative}")" == 0 ]] || exit 79
            [[ -z "$(find "${repo}/${relative}" -xdev \( \
                -type b -o -type c -o -type l -o -type p -o -type s -o \
                -perm /6222 \
            \) -print -quit)" ]] || exit 80
            while IFS= read -r -d '' runtime_file; do
                runtime_relative="${runtime_file#"${repo}/"}"
                runtime_size="$(stat -c %s "${runtime_file}")"
                runtime_digest="$(sha256sum "${runtime_file}" | awk '{print $1}')"
                printf '%s\0%s\0%s\0' "${runtime_relative}" \
                    "${runtime_size}" "${runtime_digest}" \
                    >>"${recomputed_npm_inventory}"
            done < <(find "${repo}/${relative}" -xdev -type f \
                -print0 | sort -z)
        done </opt/miles-swe/npm-repo-runtime-paths
        [[ "$(sha256sum "${recomputed_npm_inventory}" | awk '{print $1}')" == \
            "${expected_npm_inventory_digest}" ]] || exit 81
        rm -f -- "${recomputed_npm_inventory}"
        [[ "$(stat -c '%u:%a' /opt/miles-swe/playwright-browsers-path)" == \
            0:444 ]] || exit 65
        playwright_path="$(cat /opt/miles-swe/playwright-browsers-path)"
        if [[ "${playwright_path}" == none ]]; then
            [[ ! -e /opt/miles-swe/runtime/ms-playwright && \
                ! -e /opt/miles-swe/playwright-runtime.inventory && \
                ! -e /opt/miles-swe/playwright-runtime.inventory.sha256 ]] || exit 66
        else
            [[ "${playwright_path}" == /opt/miles-swe/runtime/ms-playwright && \
                -d "${playwright_path}" && ! -L "${playwright_path}" ]] || exit 67
            [[ "$(stat -c '%u:%a' /opt/miles-swe/playwright-runtime.inventory)" == \
                0:444 ]] || exit 68
            [[ "$(stat -c '%u:%a' \
                /opt/miles-swe/playwright-runtime.inventory.sha256)" == \
                0:444 ]] || exit 69
            expected_inventory_digest="$(cat \
                /opt/miles-swe/playwright-runtime.inventory.sha256)"
            [[ "$(sha256sum /opt/miles-swe/playwright-runtime.inventory \
                | awk '{print $1}')" == "${expected_inventory_digest}" ]] || exit 70
            [[ -z "$(find "${playwright_path}" -xdev \( \
                -type b -o -type c -o -type l -o -type p -o -type s -o \
                -perm /6222 \
            \) -print -quit)" ]] || exit 71
            recomputed_playwright_inventory=/tmp/miles-swe-playwright-runtime.inventory
            : >"${recomputed_playwright_inventory}"
            while IFS= read -r -d '' runtime_file; do
                runtime_relative="${runtime_file#"${playwright_path}/"}"
                runtime_size="$(stat -c %s "${runtime_file}")"
                runtime_digest="$(sha256sum "${runtime_file}" | awk '{print $1}')"
                printf '%s\0%s\0%s\0' "${runtime_relative}" \
                    "${runtime_size}" "${runtime_digest}" \
                    >>"${recomputed_playwright_inventory}"
            done < <(find "${playwright_path}" -xdev -type f \
                -print0 | sort -z)
            [[ "$(sha256sum "${recomputed_playwright_inventory}" \
                | awk '{print $1}')" == "${expected_inventory_digest}" ]] || exit 82
            rm -f -- "${recomputed_playwright_inventory}"
        fi
        ;;
    python-editable-metadata-v1)
        [[ -f /opt/miles-swe/runtime-links ]] || exit 59
        [[ "$(stat -c '%u:%a' /opt/miles-swe/runtime-links)" == 0:444 ]] || exit 60
        while IFS= read -r relative; do
            [[ -n "${relative}" && "${relative}" == *.egg-info && \
                "${relative}" != /* && "${relative}" != ../* && \
                "${relative}" != */../* ]] || exit 61
            [[ -L "${repo}/${relative}" ]] || exit 62
            [[ "$(readlink -f -- "${repo}/${relative}")" == \
                "/opt/miles-swe/runtime/python-editable/${relative}" ]] || exit 63
        done </opt/miles-swe/runtime-links
        [[ -z "$(find /opt/miles-swe/runtime/python-editable -xdev \
            -perm /022 -print -quit)" ]] || exit 64
        ;;
    *) exit 57 ;;
esac
[[ -z "$(git --git-dir="${gitdir}" --work-tree="${repo}" \
    status --porcelain --untracked-files=all)" ]] || exit 58
rm -f -- /tmp/miles-swe-expected-base /tmp/miles-swe-expected-tree
printf 'agent-root-check-ok\n'
"""

_AGENT_USER_CHECK_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH=/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
[[ "$(id -u)" == 1000 ]] || exit 40
[[ ",$(id -G | tr ' ' ',')," != *,0,* ]] || exit 41
[[ "$(awk '/^NoNewPrivs:/ {print $2}' /proc/self/status)" == 1 ]] || exit 42
cap_eff="$(awk '/^CapEff:/ {print $2}' /proc/self/status)"
[[ "${cap_eff}" =~ ^0+$ ]] || exit 43
repo="$(cat /opt/miles-swe/workdir)"
runtime_policy="$(cat /opt/miles-swe/runtime-policy)"
probe="${repo}/.miles-swe-tool-smoke"
printf 'tool-smoke\n' >"${probe}"
python3 -c 'from pathlib import Path; assert Path("'"${probe}"'").read_text() == "tool-smoke\\n"'
git -C "${repo}" diff --check -- "${probe}"
rm -f -- "${probe}"
if [[ "${runtime_policy}" == npm-node-modules-v2 ]]; then
    command -v node >/dev/null 2>&1 || exit 50
    command -v npm >/dev/null 2>&1 || exit 51
    cd "${repo}"
    node -e 'const fs=require("fs"); for (const p of process.argv.slice(1)) JSON.parse(fs.readFileSync(p,"utf8"));' \
        package.json package-lock.json || exit 52
    playwright_path="$(cat /opt/miles-swe/playwright-browsers-path)"
    if [[ "${playwright_path}" != none ]]; then
        browser="$(find "${playwright_path}" -xdev -type f -name chrome \
            -perm /111 -print -quit)"
        [[ -n "${browser}" ]] || exit 72
        PLAYWRIGHT_BROWSERS_PATH="${playwright_path}" \
            timeout --signal=TERM --kill-after=5s 30s \
            "${browser}" --version >/tmp/miles-swe-browser-version 2>&1 || exit 73
        rm -f -- /tmp/miles-swe-browser-version
    fi
elif [[ "${runtime_policy}" == python-editable-metadata-v1 ]]; then
    [[ -x /opt/miniconda3/bin/conda ]] || exit 53
    # The editable metadata remains linked to the exact base worktree while
    # the environment itself stays outside model-writable repository paths.
    source /opt/miniconda3/bin/activate
    conda activate testbed
    cd "${repo}"
    python -m pip check >/tmp/miles-swe-pip-check.log 2>&1 || exit 54
    rm -f -- /tmp/miles-swe-pip-check.log
fi
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

_VERIFIER_HISTORY_CHECK_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
repo="$(cat /opt/miles-swe/verifier-workdir)"
base="$(cat /tmp/miles-swe-expected-base)"
tree="$(cat /tmp/miles-swe-expected-tree)"
[[ -d "${repo}/.git" ]] || exit 50
head="$(git -C "${repo}" rev-parse HEAD)"
[[ "$(git -C "${repo}" rev-parse HEAD^{tree})" == "${tree}" ]] || exit 51
[[ -z "$(git -C "${repo}" show -s --format='%P' "${head}")" ]] || exit 52
if git -C "${repo}" cat-file -e "${base}^{commit}" 2>/dev/null; then
    exit 53
fi
[[ -z "$(git -C "${repo}" fsck --no-reflogs --unreachable --no-progress 2>&1)" ]] || exit 54
[[ "$(stat -c '%u' "${repo}/.git")" == 0 ]] || exit 55
printf 'verifier-history-check-ok\n'
"""


class QuarantineTask(Exception):
    """A deterministic unsafe or unsupported record must not be admitted."""

    def __init__(self, reason_code: str, detail: str) -> None:
        if _REASON.fullmatch(reason_code) is None:
            raise ValueError(f"invalid quarantine reason code: {reason_code}")
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail[:512]


class SystemicAdmissionError(RuntimeError):
    """A policy, auth, network, or admission infrastructure failure."""


class EnvironmentAdapter(Protocol):
    """Environment-specific validation and pinned materialization contract."""

    source_schema: str
    admission_schema: str
    checkpoint_label: str
    report_kind: str
    required_checks: Mapping[str, bool | int]

    def validate_dependencies(self) -> None: ...

    def validate_candidate(self, row: Mapping[str, Any]) -> None: ...

    def materialize_arguments(
        self,
        manifest: Path,
        output: Path,
    ) -> argparse.Namespace: ...

    def validate_materialized(
        self,
        task_dir: Path,
        row: Mapping[str, Any],
    ) -> None: ...

    def validate_report(
        self,
        row: Mapping[str, Any],
        report: Mapping[str, Any],
        reward: int,
    ) -> None: ...

    def admission_metadata(self, row: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class AdmissionConfig:
    """Private paths and bounded selection for one semantic admission run."""

    private_manifest: Path
    admitted_manifest: Path
    admission_manifest: Path
    quarantine_manifest: Path
    work_root: Path
    instance_id: str | None = None
    limit: int | None = None
    concurrency: int = 4


@dataclass(frozen=True)
class SourceInspection:
    base_commit: str
    base_tree: str
    template_evidence: Mapping[str, str]


def _write_private(
    path: Path,
    content: str | bytes,
    *,
    mode: int = 0o600,
) -> None:
    oci_image_lock._ensure_private_directory(path.parent)
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
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
        or metadata.st_size > maximum_bytes
    ):
        raise SystemicAdmissionError(f"unsafe private E2B readback: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or opened.st_mode & 0o077
            or opened.st_size > maximum_bytes
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise SystemicAdmissionError(f"private E2B readback changed: {path}")
        content = bytearray()
        while len(content) <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - len(content)),
            )
            if not chunk:
                break
            content.extend(chunk)
        finished = os.fstat(descriptor)
        try:
            path_after = os.lstat(path)
        except FileNotFoundError as exc:
            raise SystemicAdmissionError(
                f"private E2B readback disappeared: {path}"
            ) from exc
        if (
            (opened.st_dev, opened.st_ino, opened.st_size)
            != (finished.st_dev, finished.st_ino, finished.st_size)
            or (finished.st_dev, finished.st_ino)
            != (path_after.st_dev, path_after.st_ino)
            or opened.st_mtime_ns != finished.st_mtime_ns
            or opened.st_ctime_ns != finished.st_ctime_ns
            or len(content) != finished.st_size
        ):
            raise SystemicAdmissionError(f"private E2B readback changed: {path}")
    finally:
        os.close(descriptor)
    if len(content) > maximum_bytes:
        raise SystemicAdmissionError(f"private E2B readback is oversized: {path}")
    return bytes(content)


def _remote_file(workspace: Path, name: str, content: str | bytes) -> Path:
    path = workspace / name
    _write_private(path, content)
    return path


async def _inspect_source(
    row: Mapping[str, Any],
    backend: e2b_admission.AdmissionBackend,
    workspace: Path,
) -> SourceInspection:
    instance_id = _required_text(row, "instance_id")
    base_commit = _required_commit(row, "base_commit")
    oracle_patch, test_patch = _trusted_patches(row)
    context_dir = workspace / "source-context"
    oci_image_lock._ensure_private_directory(context_dir)
    script = _remote_file(workspace, "inspect-source.sh", _SOURCE_INSPECTION_SCRIPT)
    base = _remote_file(workspace, "expected-base", base_commit + "\n")
    oracle = _remote_file(workspace, "oracle.patch", oracle_patch)
    hidden = _remote_file(workspace, "hidden-tests.patch", test_patch)
    result_path = workspace / "source-result"
    image = _source_image(row)
    spec = e2b_admission.SandboxSpec(
        role="source",
        name=_sandbox_name("source", instance_id),
        context_dir=context_dir,
        source_image=image,
        expected_image=image,
    )
    try:
        async with e2b_admission.sandbox(backend, spec) as remote:
            await _require_network_denied(remote, role="source")
            await remote.upload_file(script, "/tmp/miles-swe-inspect-source.sh")
            await remote.upload_file(base, "/tmp/miles-swe-expected-base")
            await remote.upload_file(oracle, "/tmp/miles-swe-oracle.patch")
            await remote.upload_file(hidden, "/tmp/miles-swe-hidden-tests.patch")
            await e2b_admission.require_ok(
                remote,
                "chmod 0500 /tmp/miles-swe-inspect-source.sh && "
                "chmod 0600 /tmp/miles-swe-expected-base "
                "/tmp/miles-swe-oracle.patch /tmp/miles-swe-hidden-tests.patch && "
                "/tmp/miles-swe-inspect-source.sh",
                user=0,
                timeout_sec=600,
                phase="source integrity inspection",
            )
            await remote.download_file(
                "/tmp/miles-swe-source-result",
                result_path,
            )
    except e2b_admission.RemoteCommandError as exc:
        raise QuarantineTask(
            "source_image_unsupported",
            f"source integrity check status {exc.return_code}",
        ) from exc
    try:
        lines = _read_regular_bounded(
            result_path,
            maximum_bytes=_MAX_SMALL_EVIDENCE_BYTES,
        ).decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SystemicAdmissionError("source inspection returned non-UTF-8") from exc
    if (
        len(lines) != 2
        or lines[0] != base_commit
        or _COMMIT.fullmatch(lines[1]) is None
    ):
        raise SystemicAdmissionError("source inspection binding is invalid")
    return SourceInspection(
        base_commit=lines[0],
        base_tree=lines[1],
        template_evidence=dict(remote.template_evidence),
    )


def _materialize_exact(
    row: dict[str, Any],
    *,
    adapter: EnvironmentAdapter,
    workspace: Path,
) -> Path:
    manifest = workspace / "exact-task.private.jsonl"
    _write_private(manifest, json.dumps(row, sort_keys=True) + "\n")
    output = workspace / "harbor-tasks"
    try:
        summary = materialize_module.materialize(
            adapter.materialize_arguments(manifest, output)
        )
    except (ValueError, PermissionError) as exc:
        raise QuarantineTask(
            "unsupported_materialization",
            f"task rejected by pinned materializer: {type(exc).__name__}",
        ) from exc
    if summary.get("tasks") != 1 or summary.get("schemas") != {
        adapter.source_schema: 1
    }:
        raise SystemicAdmissionError("exact materialization returned an invalid summary")
    task_dir = output / _required_text(row, "instance_id")
    _validate_common_task_tree(task_dir, row)
    adapter.validate_materialized(task_dir, row)
    return task_dir


def _validate_common_task_tree(
    task_dir: Path,
    row: Mapping[str, Any],
) -> None:
    for path in [task_dir, *task_dir.rglob("*")]:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise QuarantineTask(
                "unsafe_materialized_tree",
                "materialized task is not an owner-only regular tree",
            )
        if not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode):
            raise QuarantineTask(
                "unsafe_materialized_tree",
                "materialized task contains a special file",
            )
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
            raise QuarantineTask(
                "unsafe_materialized_tree",
                "materialized task contains a hard-linked file",
            )
    config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    image = _source_image(row)
    if (
        config.get("metadata", {}).get("task_digest") != row.get("task_digest")
        or config.get("metadata", {}).get("source_schema")
        != row.get("source_schema")
        or config.get("agent", {}).get("user") != 1000
        or config.get("environment", {}).get("network_mode") != "no-network"
        or config.get("verifier", {}).get("environment_mode") != "separate"
        or config.get("verifier", {}).get("user") != 0
        or config.get("verifier", {})
        .get("environment", {})
        .get("network_mode")
        != "no-network"
        or config.get("verifier", {})
        .get("environment", {})
        .get("docker_image")
        != image
    ):
        raise QuarantineTask(
            "unsafe_materialized_config",
            "Harbor task security or identity binding mismatch",
        )
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    if not dockerfile.startswith(f"FROM {image}\n"):
        raise QuarantineTask(
            "image_binding_mismatch",
            "agent Dockerfile is not based on the admitted image digest",
        )
    tests_dir = task_dir / "tests"
    if (
        not (tests_dir / ".harbor-e2b-late-tests").is_file()
        or (tests_dir / "Dockerfile").exists()
    ):
        raise QuarantineTask(
            "unsafe_private_verifier",
            "private verifier is not a late-upload-only package",
        )


async def _check_agent(
    row: Mapping[str, Any],
    inspection: SourceInspection,
    task_dir: Path,
    backend: e2b_admission.AdmissionBackend,
    workspace: Path,
) -> dict[str, str]:
    root_script = _remote_file(workspace, "agent-root-check.sh", _AGENT_ROOT_CHECK_SCRIPT)
    user_script = _remote_file(workspace, "agent-user-check.sh", _AGENT_USER_CHECK_SCRIPT)
    baseline_script = _agent_baseline_script(row)
    local_baseline = (
        _remote_file(workspace, "agent-public-baseline.sh", baseline_script)
        if baseline_script is not None
        else None
    )
    base = _remote_file(workspace, "agent-base", inspection.base_commit + "\n")
    tree = _remote_file(workspace, "agent-tree", inspection.base_tree + "\n")
    image = _source_image(row)
    spec = e2b_admission.SandboxSpec(
        role="agent",
        name=_sandbox_name("agent", _required_text(row, "instance_id")),
        context_dir=task_dir / "environment",
        task_dir=task_dir,
        expected_image=image,
    )
    try:
        async with e2b_admission.sandbox(backend, spec) as remote:
            await _require_network_denied(remote, role="agent")
            await remote.upload_file(root_script, "/tmp/miles-swe-agent-root.sh")
            await remote.upload_file(user_script, "/tmp/miles-swe-agent-user.sh")
            if local_baseline is not None:
                await remote.upload_file(
                    local_baseline,
                    "/tmp/miles-swe-agent-public-baseline.sh",
                )
            await remote.upload_file(base, "/tmp/miles-swe-expected-base")
            await remote.upload_file(tree, "/tmp/miles-swe-expected-tree")
            await e2b_admission.require_ok(
                remote,
                "chmod 0500 /tmp/miles-swe-agent-root.sh "
                "/tmp/miles-swe-agent-user.sh && "
                "chmod 0600 /tmp/miles-swe-expected-base "
                "/tmp/miles-swe-expected-tree && "
                "/tmp/miles-swe-agent-root.sh",
                user=0,
                timeout_sec=600,
                phase="agent root security check",
            )
            await e2b_admission.require_ok(
                remote,
                "/tmp/miles-swe-agent-user.sh",
                user=1000,
                timeout_sec=120,
                phase="agent UID 1000 security check",
            )
            if local_baseline is not None:
                await e2b_admission.require_ok(
                    remote,
                    "chmod 0500 /tmp/miles-swe-agent-public-baseline.sh && "
                    "/tmp/miles-swe-agent-public-baseline.sh",
                    user=1000,
                    timeout_sec=2200,
                    phase="agent exact public baseline",
                )
            if row.get("source_schema") == "swe-gym":
                await _run_swe_gym_agent_baseline(
                    row,
                    task_dir=task_dir,
                    remote=remote,
                    workspace=workspace,
                )
            evidence = dict(remote.template_evidence)
    except e2b_admission.RemoteCommandError as exc:
        raise QuarantineTask(
            "unsafe_agent_image",
            f"agent security check status {exc.return_code}",
        ) from exc
    return evidence


def _agent_baseline_script(row: Mapping[str, Any]) -> str | None:
    """Build an unprivileged, no-network public-baseline check for ReBench."""

    if row.get("source_schema") != "swe-rebench-v2":
        return None
    verifier = row.get("verifier")
    if not isinstance(verifier, dict):
        raise ValueError("SWE-ReBench verifier metadata is missing")
    install_config = verifier.get("install_config")
    if not isinstance(install_config, dict):
        raise ValueError("SWE-ReBench install_config is missing")
    commands = materialize_module._test_commands(install_config.get("test_cmd"))
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "export PATH=/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "export HOME=/tmp/miles-swe-agent-baseline-home",
        "export PYTHON_DOTENV_DISABLED=1 WANDB_MODE=offline WANDB_DISABLED=true",
        "mkdir -m 0700 -p \"${HOME}\"",
        "repo=\"$(cat /opt/miles-swe/workdir)\"",
        "playwright_path=\"$(cat /opt/miles-swe/playwright-browsers-path)\"",
        "if [[ \"${playwright_path}\" != none ]]; then "
        "export PLAYWRIGHT_BROWSERS_PATH=\"${playwright_path}\"; fi",
        "cd \"${repo}\"",
    ]
    for index, command in enumerate(commands):
        log = f"/tmp/miles-swe-public-baseline-{index}.log"
        lines.extend(
            [
                "if ! timeout --signal=TERM --kill-after=10s "
                f"{timeouts.VERIFIER_EXECUTION_TIMEOUT_SEC}s "
                f"bash --noprofile --norc -c {shlex.quote(command)} "
                f">{shlex.quote(log)} 2>&1; then",
                f"    tail -c 65536 {shlex.quote(log)} >&2 || true",
                "    exit 70",
                "fi",
                f"rm -f -- {shlex.quote(log)}",
            ]
        )
    lines.append("printf 'agent-public-baseline-ok\\n'")
    return "\n".join(lines) + "\n"


async def _run_swe_gym_agent_baseline(
    row: Mapping[str, Any],
    *,
    task_dir: Path,
    remote: e2b_admission.AdmissionSandbox,
    workspace: Path,
) -> None:
    """Run one pinned PASS_TO_PASS baseline without uploading hidden tests."""

    verifier = row.get("verifier")
    source_metadata = row.get("source_metadata")
    if not isinstance(verifier, dict) or not isinstance(source_metadata, dict):
        raise ValueError("SWE-Gym public baseline metadata is missing")
    pass_to_pass = verifier.get("pass_to_pass")
    if not isinstance(pass_to_pass, list) or any(
        not isinstance(test_id, str) or not test_id for test_id in pass_to_pass
    ):
        raise ValueError("SWE-Gym PASS_TO_PASS inventory is invalid")
    public_config = {
        "repo": _required_text(row, "repo").lower(),
        "version": _required_text(source_metadata, "version"),
        "pass_to_pass": pass_to_pass[:1],
    }
    config_path = _remote_file(
        workspace,
        "swe-gym-public-baseline.json",
        json.dumps(public_config, sort_keys=True) + "\n",
    )
    tests_dir = task_dir / "tests"
    sources = {
        tests_dir / "swe_gym_run.py": "/tmp/miles-swe-public/swe_gym_run.py",
        tests_dir / "lib" / "swegym" / "__init__.py": (
            "/tmp/miles-swe-public/lib/swegym/__init__.py"
        ),
        tests_dir / "lib" / "swegym" / "harness" / "__init__.py": (
            "/tmp/miles-swe-public/lib/swegym/harness/__init__.py"
        ),
        tests_dir / "lib" / "swegym" / "harness" / "constants.py": (
            "/tmp/miles-swe-public/lib/swegym/harness/constants.py"
        ),
        tests_dir / "lib" / "swegym" / "harness" / "test_spec.py": (
            "/tmp/miles-swe-public/lib/swegym/harness/test_spec.py"
        ),
        config_path: "/tmp/miles-swe-public/config.json",
    }
    await e2b_admission.require_ok(
        remote,
        "install -d -m 0700 /tmp/miles-swe-public "
        "/tmp/miles-swe-public/lib/swegym/harness",
        user=1000,
        timeout_sec=30,
        phase="agent public-baseline staging",
    )
    for source, destination in sources.items():
        await remote.upload_file(source, destination)
    await e2b_admission.require_ok(
        remote,
        "export PYTHONPATH=/tmp/miles-swe-public/lib "
        "PYTHON_DOTENV_DISABLED=1 WANDB_MODE=offline WANDB_DISABLED=true; "
        "repo=\"$(cat /opt/miles-swe/workdir)\"; "
        "python3 /tmp/miles-swe-public/swe_gym_run.py --public-baseline "
        "/tmp/miles-swe-public/config.json \"${repo}\" "
        ">/tmp/miles-swe-public/run.sh && "
        "chmod 0500 /tmp/miles-swe-public/run.sh && "
        "if ! timeout --signal=TERM --kill-after=10s "
        f"{timeouts.VERIFIER_EXECUTION_TIMEOUT_SEC}s "
        "/bin/bash --noprofile --norc /tmp/miles-swe-public/run.sh "
        ">/tmp/miles-swe-public/output.log 2>&1; then "
        "tail -c 65536 /tmp/miles-swe-public/output.log >&2 || true; exit 70; "
        "fi; rm -rf -- /tmp/miles-swe-public",
        user=1000,
        timeout_sec=2200,
        phase="agent exact SWE-Gym public baseline",
    )


async def _require_network_denied(
    remote: e2b_admission.AdmissionSandbox,
    *,
    role: str,
) -> None:
    result = await remote.exec(
        _NETWORK_DENIAL_SCRIPT,
        user=1000 if role == "agent" else 0,
        timeout_sec=10,
    )
    if result.return_code != 0:
        raise SystemicAdmissionError(
            f"{role} sandbox public-network isolation failed"
        )


async def _verify_patch(
    row: Mapping[str, Any],
    inspection: SourceInspection,
    task_dir: Path,
    patch: bytes,
    backend: e2b_admission.AdmissionBackend,
    adapter: EnvironmentAdapter,
    workspace: Path,
    phase: str,
) -> tuple[int, dict[str, str]]:
    local_patch = workspace / f"{phase}.patch"
    _write_private(local_patch, patch)
    base = _remote_file(workspace, f"{phase}-base", inspection.base_commit + "\n")
    tree = _remote_file(workspace, f"{phase}-tree", inspection.base_tree + "\n")
    history_script = _remote_file(
        workspace,
        f"{phase}-verifier-history.sh",
        _VERIFIER_HISTORY_CHECK_SCRIPT,
    )
    reward_path = workspace / f"{phase}-reward.txt"
    report_path = workspace / f"{phase}-report.json"
    image = _source_image(row)
    spec = e2b_admission.SandboxSpec(
        role="verifier",
        name=_sandbox_name(f"{phase}-verifier", _required_text(row, "instance_id")),
        context_dir=task_dir / "tests",
        task_dir=task_dir,
        expected_image=image,
    )
    async with e2b_admission.sandbox(backend, spec) as remote:
        await _require_network_denied(remote, role="verifier")
        await e2b_admission.require_ok(
            remote,
            "test ! -e /tests/.harbor-e2b-late-tests",
            user=0,
            timeout_sec=30,
            phase=f"{phase} verifier pre-upload isolation",
        )
        await remote.install_private_verifier(task_dir / "tests")
        await e2b_admission.require_ok(
            remote,
            "test -f /tests/.harbor-e2b-late-tests && "
            "test -f /tests/test.sh && test -f /tests/model_path_policy.json && "
            "test ! -e /tests/Dockerfile && "
            "test \"$(stat -c '%u' /tests/test.sh)\" = 0 && "
            "test $(( $(stat -c '%a' /tests/test.sh) % 100 )) = 0",
            user=0,
            timeout_sec=30,
            phase=f"{phase} verifier late-package binding",
        )
        await remote.upload_file(local_patch, "/tmp/miles-swe-model.patch")
        await remote.upload_file(base, "/tmp/miles-swe-expected-base")
        await remote.upload_file(tree, "/tmp/miles-swe-expected-tree")
        await remote.upload_file(
            history_script,
            "/tmp/miles-swe-verifier-history.sh",
        )
        patch_digest = hashlib.sha256(patch).hexdigest()
        await e2b_admission.require_ok(
            remote,
            "test \"$(sha256sum /tmp/miles-swe-model.patch | awk '{print $1}')\" = "
            f"{patch_digest} && "
            "install -d -o root -g root -m 0700 /opt/miles-swe/collected && "
            "install -o root -g root -m 0600 /tmp/miles-swe-model.patch "
            "/opt/miles-swe/collected/model.patch && /tests/test.sh",
            user=0,
            timeout_sec=timeouts.VERIFIER_EXECUTION_TIMEOUT_SEC,
            phase=f"{phase} trusted verifier",
        )
        await e2b_admission.require_ok(
            remote,
            "chmod 0500 /tmp/miles-swe-verifier-history.sh && "
            "chmod 0600 /tmp/miles-swe-expected-base "
            "/tmp/miles-swe-expected-tree && "
            "/tmp/miles-swe-verifier-history.sh",
            user=0,
            timeout_sec=120,
            phase=f"{phase} verifier history check",
        )
        await remote.download_file("/logs/verifier/reward.txt", reward_path)
        await remote.download_file("/logs/verifier/report.json", report_path)
        template_evidence = dict(remote.template_evidence)
    try:
        raw_reward = _read_regular_bounded(
            reward_path,
            maximum_bytes=_MAX_SMALL_EVIDENCE_BYTES,
        ).decode("utf-8").strip()
        report = json.loads(
            _read_regular_bounded(
                report_path,
                maximum_bytes=_MAX_REPORT_BYTES,
            ).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemicAdmissionError(
            f"{phase} verifier returned malformed evidence"
        ) from exc
    if raw_reward not in {"0", "1"} or not isinstance(report, dict):
        raise SystemicAdmissionError(
            f"{phase} verifier returned non-binary evidence"
        )
    reward = int(raw_reward)
    if report.get("reward") != reward:
        raise SystemicAdmissionError(
            f"{phase} verifier report and reward disagree"
        )
    adapter.validate_report(row, report, reward)
    return reward, template_evidence


async def _admit_one(
    row: dict[str, Any],
    *,
    adapter: EnvironmentAdapter,
    backend: e2b_admission.AdmissionBackend,
    workspace: Path,
) -> dict[str, Any]:
    inspection = await _inspect_source(row, backend, workspace)
    task_dir = _materialize_exact(row, adapter=adapter, workspace=workspace)
    agent_template = await _check_agent(
        row,
        inspection,
        task_dir,
        backend,
        workspace,
    )
    oracle_patch, _ = _trusted_patches(row)
    empty_reward, empty_template = await _verify_patch(
        row,
        inspection,
        task_dir,
        b"",
        backend,
        adapter,
        workspace,
        "empty",
    )
    oracle_reward, oracle_template = await _verify_patch(
        row,
        inspection,
        task_dir,
        oracle_patch.encode("utf-8"),
        backend,
        adapter,
        workspace,
        "oracle",
    )
    if empty_reward != 0 or oracle_reward != 1:
        raise QuarantineTask(
            "golden_outcome_mismatch",
            f"expected empty=0/oracle=1, got {empty_reward}/{oracle_reward}",
        )
    _validate_template_reuse(
        inspection.template_evidence,
        agent_template,
        empty_template,
        oracle_template,
    )
    return _admission_record(
        row,
        inspection=inspection,
        adapter=adapter,
        task_dir=task_dir,
        empty_reward=empty_reward,
        oracle_reward=oracle_reward,
        template_evidence={
            "source": inspection.template_evidence,
            "agent": agent_template,
            "empty_verifier": empty_template,
            "oracle_verifier": oracle_template,
        },
    )


def _admission_record(
    row: Mapping[str, Any],
    *,
    inspection: SourceInspection,
    adapter: EnvironmentAdapter,
    task_dir: Path,
    empty_reward: int,
    oracle_reward: int,
    template_evidence: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    requested, resolved, input_digest = _image_provenance(row)
    oracle_patch, test_patch = _trusted_patches(row)
    policy = task_dir / "tests" / "model_path_policy.json"
    checks = dict(adapter.required_checks)
    checks["empty_reward"] = empty_reward
    checks["oracle_reward"] = oracle_reward
    return {
        "schema_version": adapter.admission_schema,
        "instance_id": _required_text(row, "instance_id"),
        "source_schema": adapter.source_schema,
        "task_digest": _required_digest(row, "task_digest"),
        "input_content_digest": input_digest,
        "locked_content_digest": _required_digest(row, "content_digest"),
        "content_digest": _required_digest(row, "content_digest"),
        "source_image_requested": requested,
        "source_image_resolved": resolved,
        "source_image": resolved,
        "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
        "base_commit": inspection.base_commit,
        "base_tree": inspection.base_tree,
        "oracle_patch_sha256": hashlib.sha256(
            oracle_patch.encode("utf-8")
        ).hexdigest(),
        "test_patch_sha256": hashlib.sha256(
            test_patch.encode("utf-8")
        ).hexdigest(),
        "model_path_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "admitted_task_tree_sha256": materialize_module._task_tree_sha256(
            task_dir
        ),
        **dict(adapter.admission_metadata(row)),
        "template_evidence": {
            role: dict(evidence)
            for role, evidence in template_evidence.items()
        },
        "checks": checks,
    }


def _validate_template_reuse(
    source: Mapping[str, str],
    agent: Mapping[str, str],
    empty: Mapping[str, str],
    oracle: Mapping[str, str],
) -> None:
    for evidence in (source, agent, empty, oracle):
        _validate_template_evidence(evidence)
    immutable_keys = (
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
    )
    if any(empty.get(key) != oracle.get(key) for key in immutable_keys):
        raise SystemicAdmissionError(
            "empty and oracle verifiers did not reuse one exact fresh template"
        )
    if empty.get("sandbox_id") == oracle.get("sandbox_id"):
        raise SystemicAdmissionError(
            "empty and oracle verification reused the same sandbox"
        )
    if any(source.get(key) != empty.get(key) for key in immutable_keys):
        raise SystemicAdmissionError(
            "source/verifier phases did not reuse one exact image template"
        )
    sandbox_ids = [
        source.get("sandbox_id"),
        agent.get("sandbox_id"),
        empty.get("sandbox_id"),
        oracle.get("sandbox_id"),
    ]
    if len(set(sandbox_ids)) != len(sandbox_ids):
        raise SystemicAdmissionError("semantic admission sandbox IDs are not unique")


async def admit_tasks(
    config: AdmissionConfig,
    adapter: EnvironmentAdapter,
    backend: e2b_admission.AdmissionBackend,
) -> dict[str, int]:
    """Admit or quarantine every selected record with resumable checkpoints."""

    if config.concurrency <= 0:
        raise ValueError("semantic admission concurrency must be positive")
    adapter.validate_dependencies()
    checkpoint_dir = _checkpoint_directory(config.admission_manifest)
    oci_image_lock._require_distinct_paths(
        config.private_manifest,
        config.admitted_manifest,
        config.admission_manifest,
        config.quarantine_manifest,
        checkpoint_dir,
        config.work_root,
    )
    oci_image_lock._validate_private(config.private_manifest)
    input_fingerprint = oci_image_lock._capture_private_fingerprint(
        config.private_manifest
    )
    oci_image_lock._ensure_private_directory(config.work_root)
    checkpoints = _load_checkpoint_index(checkpoint_dir, adapter=adapter)
    selected, admitted, quarantined, resumed = await _run_selected(
        config=config,
        adapter=adapter,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        checkpoints=checkpoints,
    )
    _compact_outputs(
        config=config,
        adapter=adapter,
        checkpoints=checkpoints,
        input_fingerprint=input_fingerprint,
    )
    return {
        "selected": selected,
        "admitted": admitted,
        "quarantined": quarantined,
        "resumed": resumed,
    }


async def _run_selected(
    *,
    config: AdmissionConfig,
    adapter: EnvironmentAdapter,
    backend: e2b_admission.AdmissionBackend,
    checkpoint_dir: Path,
    checkpoints: dict[str, Path],
) -> tuple[int, int, int, int]:
    running: set[asyncio.Task[tuple[str, Path, str]]] = set()
    seen: set[str] = set()
    selected = admitted = quarantined = resumed = 0
    try:
        for row in _selected_rows(config):
            _validate_candidate(row, adapter)
            selected += 1
            digest = _required_digest(row, "content_digest")
            if digest in seen:
                raise ValueError(f"duplicate semantic-admission digest: {digest}")
            seen.add(digest)
            checkpoint = checkpoints.get(digest)
            if checkpoint is not None:
                _load_checkpoint(checkpoint, adapter=adapter, original=row)
                resumed += 1
                continue
            running.add(
                asyncio.create_task(
                    _admit_and_checkpoint(
                        row,
                        config=config,
                        adapter=adapter,
                        backend=backend,
                        checkpoint_dir=checkpoint_dir,
                    )
                )
            )
            if len(running) >= config.concurrency:
                running, finished = await _wait(running)
                for result_digest, path, disposition in finished:
                    checkpoints[result_digest] = path
                    admitted += disposition == "admitted"
                    quarantined += disposition == "quarantined"
        while running:
            running, finished = await _wait(running)
            for result_digest, path, disposition in finished:
                checkpoints[result_digest] = path
                admitted += disposition == "admitted"
                quarantined += disposition == "quarantined"
    except BaseException:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise
    if selected == 0:
        raise ValueError("no task matched semantic admission selection")
    return selected, admitted, quarantined, resumed


async def _wait(
    running: set[asyncio.Task[tuple[str, Path, str]]],
) -> tuple[
    set[asyncio.Task[tuple[str, Path, str]]],
    list[tuple[str, Path, str]],
]:
    done, pending = await asyncio.wait(
        running,
        return_when=asyncio.FIRST_COMPLETED,
    )
    return set(pending), [task.result() for task in done]


async def _admit_and_checkpoint(
    row: dict[str, Any],
    *,
    config: AdmissionConfig,
    adapter: EnvironmentAdapter,
    backend: e2b_admission.AdmissionBackend,
    checkpoint_dir: Path,
) -> tuple[str, Path, str]:
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{adapter.checkpoint_label}-", dir=config.work_root)
    )
    workspace.chmod(0o700)
    digest = _required_digest(row, "content_digest")
    disposition: str | None = None
    record: dict[str, Any] | None = None
    admission_failure: BaseException | None = None
    try:
        try:
            _validate_trusted_inputs(row)
            adapter.validate_candidate(row)
            admission = await _admit_one(
                row,
                adapter=adapter,
                backend=backend,
                workspace=workspace,
            )
        except QuarantineTask as exc:
            disposition = "quarantined"
            record = _quarantine_record(row, exc)
        else:
            disposition = "admitted"
            record = admission
    except BaseException as exc:
        admission_failure = exc
    try:
        materialize_module._remove_private_tree(workspace)
    except BaseException as cleanup_failure:
        if isinstance(admission_failure, asyncio.CancelledError):
            raise admission_failure from cleanup_failure
        if admission_failure is not None:
            raise cleanup_failure from admission_failure
        raise
    if admission_failure is not None:
        raise admission_failure
    if disposition is None or record is None:
        raise AssertionError("semantic admission produced no disposition")
    checkpoint = checkpoint_dir / f"{digest}.json"
    _write_checkpoint(
        checkpoint,
        adapter=adapter,
        original=row,
        disposition=disposition,
        record=record,
    )
    return digest, checkpoint, disposition


def _quarantine_record(
    row: Mapping[str, Any],
    reason: QuarantineTask,
) -> dict[str, Any]:
    requested, resolved, input_digest = _image_provenance(row)
    return {
        "schema_version": _QUARANTINE_SCHEMA,
        "instance_id": _required_text(row, "instance_id"),
        "source_schema": _required_text(row, "source_schema"),
        "task_digest": _required_digest(row, "task_digest"),
        "input_content_digest": input_digest,
        "locked_content_digest": _required_digest(row, "content_digest"),
        "content_digest": _required_digest(row, "content_digest"),
        "source_image_requested": requested,
        "source_image": resolved,
        "reason_code": reason.reason_code,
        "reason_detail": reason.detail,
    }


def _checkpoint_directory(path: Path) -> Path:
    directory = path.with_name(path.name + ".d")
    oci_image_lock._ensure_private_directory(directory)
    return directory


def _load_checkpoint_index(
    directory: Path,
    *,
    adapter: EnvironmentAdapter,
) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        digest, _, _ = _load_checkpoint(path, adapter=adapter)
        if path.stem != digest or digest in checkpoints:
            raise ValueError(f"invalid or duplicate admission checkpoint: {path}")
        checkpoints[digest] = path
    return checkpoints


def _load_checkpoint(
    path: Path,
    *,
    adapter: EnvironmentAdapter,
    original: Mapping[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    oci_image_lock._validate_private(path)
    values = list(oci_image_lock._read_jsonl(path))
    if len(values) != 1 or values[0].get("schema_version") != _CHECKPOINT_SCHEMA:
        raise ValueError(f"invalid semantic admission checkpoint: {path}")
    value = values[0]
    if value.get("environment") != adapter.source_schema:
        raise ValueError(f"semantic checkpoint environment mismatch: {path}")
    digest = _required_digest(value, "locked_content_digest")
    disposition = _required_text(value, "disposition")
    record = _required_mapping(value, "record")
    _validate_checkpoint_binding(
        digest=digest,
        disposition=disposition,
        record=record,
        adapter=adapter,
        original=original,
    )
    return digest, disposition, dict(record)


def _write_checkpoint(
    path: Path,
    *,
    adapter: EnvironmentAdapter,
    original: Mapping[str, Any],
    disposition: str,
    record: Mapping[str, Any],
) -> None:
    digest = _required_digest(original, "content_digest")
    _validate_checkpoint_binding(
        digest=digest,
        disposition=disposition,
        record=record,
        adapter=adapter,
        original=original,
    )
    value = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "environment": adapter.source_schema,
        "locked_content_digest": digest,
        "disposition": disposition,
        "record": dict(record),
    }
    if path.exists() or path.is_symlink():
        _, existing_disposition, existing_record = _load_checkpoint(
            path,
            adapter=adapter,
            original=original,
        )
        if existing_disposition != disposition or existing_record != record:
            raise ValueError(f"conflicting semantic admission checkpoint: {path}")
        return
    oci_image_lock._atomic_write_jsonl(path, [value])


def _validate_checkpoint_binding(
    *,
    digest: str,
    disposition: str,
    record: Mapping[str, Any],
    adapter: EnvironmentAdapter,
    original: Mapping[str, Any] | None,
) -> None:
    if disposition == "admitted":
        _validate_admission_record(record, adapter)
    elif disposition == "quarantined":
        _validate_quarantine_record(record, adapter)
    else:
        raise ValueError("semantic checkpoint has an invalid disposition")
    if _required_digest(record, "locked_content_digest") != digest:
        raise ValueError("semantic checkpoint locked-content mismatch")
    if original is None:
        return
    _validate_candidate(original, adapter)
    requested, resolved, input_digest = _image_provenance(original)
    expected = {
        "instance_id": _required_text(original, "instance_id"),
        "task_digest": _required_digest(original, "task_digest"),
        "content_digest": digest,
        "locked_content_digest": digest,
        "input_content_digest": input_digest,
        "source_image_requested": requested,
        "source_image": resolved,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("semantic checkpoint resume binding mismatch")
    if disposition == "admitted":
        if (
            record.get("image_publisher_policy")
            != oci_image_lock.IMAGE_PUBLISHER_POLICY
        ):
            raise ValueError("semantic checkpoint publisher policy mismatch")
        expected_metadata = adapter.admission_metadata(original)
        if any(record.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("semantic checkpoint pinned metadata mismatch")


def _validate_admission_record(
    record: Mapping[str, Any],
    adapter: EnvironmentAdapter,
) -> None:
    if (
        record.get("schema_version") != adapter.admission_schema
        or record.get("source_schema") != adapter.source_schema
    ):
        raise ValueError("semantic admission schema mismatch")
    _validate_record_identity(record)
    if record.get("image_publisher_policy") != oci_image_lock.IMAGE_PUBLISHER_POLICY:
        raise ValueError("semantic admission publisher policy is invalid")
    _required_commit(record, "base_commit")
    _required_commit(record, "base_tree")
    for key in (
        "oracle_patch_sha256",
        "test_patch_sha256",
        "model_path_policy_sha256",
        "admitted_task_tree_sha256",
    ):
        _required_digest(record, key)
    template_evidence = _required_mapping(record, "template_evidence")
    required_roles = {
        "source",
        "agent",
        "empty_verifier",
        "oracle_verifier",
    }
    if set(template_evidence) != required_roles:
        raise ValueError("semantic admission template roles are incomplete")
    for role in sorted(required_roles):
        _validate_template_evidence(
            _required_mapping(template_evidence, role)
        )
    _validate_template_reuse(
        _required_mapping(template_evidence, "source"),
        _required_mapping(template_evidence, "agent"),
        _required_mapping(template_evidence, "empty_verifier"),
        _required_mapping(template_evidence, "oracle_verifier"),
    )
    checks = _required_mapping(record, "checks")
    if any(
        checks.get(name) != expected
        for name, expected in adapter.required_checks.items()
    ):
        raise ValueError("semantic admission is missing a required live check")


def _validate_quarantine_record(
    record: Mapping[str, Any],
    adapter: EnvironmentAdapter,
) -> None:
    if (
        record.get("schema_version") != _QUARANTINE_SCHEMA
        or record.get("source_schema") != adapter.source_schema
    ):
        raise ValueError("semantic quarantine schema mismatch")
    _validate_record_identity(record)
    if _REASON.fullmatch(_required_text(record, "reason_code")) is None:
        raise ValueError("semantic quarantine reason is invalid")
    detail = _required_text(record, "reason_detail")
    if len(detail) > 512:
        raise ValueError("semantic quarantine detail is oversized")


def _validate_record_identity(record: Mapping[str, Any]) -> None:
    instance_id = _required_text(record, "instance_id")
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise ValueError("semantic admission instance identity is invalid")
    for key in (
        "task_digest",
        "input_content_digest",
        "locked_content_digest",
        "content_digest",
    ):
        _required_digest(record, key)
    if _source_image_record(record) != _required_text(record, "source_image"):
        raise ValueError("semantic admission image is not immutable")


def _validate_template_evidence(evidence: Mapping[str, Any]) -> None:
    if set(evidence) != {
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
        "sandbox_id",
    }:
        raise ValueError("semantic admission template evidence fields are invalid")
    for key in ("alias_sha256", "template_identity_sha256"):
        _required_digest(evidence, key)
    for key in ("template_id", "build_id", "sandbox_id"):
        value = _required_text(evidence, key)
        if re.fullmatch(r"[A-Za-z0-9_-]{6,128}", value) is None:
            raise ValueError(f"semantic admission {key} is invalid")


def _compact_outputs(
    *,
    config: AdmissionConfig,
    adapter: EnvironmentAdapter,
    checkpoints: Mapping[str, Path],
    input_fingerprint: Any,
) -> None:
    def unchanged() -> None:
        oci_image_lock._assert_private_unchanged(
            config.private_manifest,
            input_fingerprint,
        )

    oci_image_lock._atomic_write_jsonl(
        config.admitted_manifest,
        _checkpointed_rows(
            config=config,
            adapter=adapter,
            checkpoints=checkpoints,
            disposition="admitted",
            original=True,
        ),
        before_replace=unchanged,
    )
    oci_image_lock._atomic_write_jsonl(
        config.admission_manifest,
        _checkpointed_rows(
            config=config,
            adapter=adapter,
            checkpoints=checkpoints,
            disposition="admitted",
            original=False,
        ),
        before_replace=unchanged,
    )
    oci_image_lock._atomic_write_jsonl(
        config.quarantine_manifest,
        _checkpointed_rows(
            config=config,
            adapter=adapter,
            checkpoints=checkpoints,
            disposition="quarantined",
            original=False,
        ),
        before_replace=unchanged,
    )


def _checkpointed_rows(
    *,
    config: AdmissionConfig,
    adapter: EnvironmentAdapter,
    checkpoints: Mapping[str, Path],
    disposition: str,
    original: bool,
) -> Iterable[Mapping[str, Any]]:
    for row in _selected_rows(config):
        _validate_candidate(row, adapter)
        digest = _required_digest(row, "content_digest")
        checkpoint = checkpoints.get(digest)
        if checkpoint is None:
            raise SystemicAdmissionError(f"missing completed checkpoint for {digest}")
        _, actual_disposition, record = _load_checkpoint(
            checkpoint,
            adapter=adapter,
            original=row,
        )
        if actual_disposition == disposition:
            yield row if original else record


def _selected_rows(config: AdmissionConfig) -> Iterable[dict[str, Any]]:
    selected = 0
    for row in oci_image_lock._read_jsonl(config.private_manifest):
        if config.instance_id is not None and row.get("instance_id") != config.instance_id:
            continue
        if config.limit is not None and selected >= config.limit:
            break
        selected += 1
        yield row


def _validate_candidate(
    row: Mapping[str, Any],
    adapter: EnvironmentAdapter,
) -> None:
    if row.get("schema_version") != "miles-swe-task-v1":
        raise ValueError("unsupported private SWE task schema")
    if row.get("source_schema") != adapter.source_schema:
        raise ValueError("semantic admission input contains another environment")
    oci_image_lock.validate_task_image_policy(row)
    instance_id = _required_text(row, "instance_id")
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise ValueError(f"invalid private SWE instance_id: {instance_id!r}")
    _required_digest(row, "task_digest")
    content_digest = _required_digest(row, "content_digest")
    if oci_image_lock._stable_digest_without_bindings(row) != content_digest:
        raise ValueError(f"private SWE content binding mismatch for {instance_id}")
    _required_commit(row, "base_commit")
    _image_provenance(row)


def _validate_trusted_inputs(row: Mapping[str, Any]) -> None:
    oracle_patch, test_patch = _trusted_patches(row)
    for name, patch in (("oracle", oracle_patch), ("test", test_patch)):
        if len(patch.encode("utf-8")) > _MAX_PATCH_BYTES:
            raise QuarantineTask(
                f"{name}_patch_oversized",
                f"{name} patch exceeds {_MAX_PATCH_BYTES} bytes",
            )


def _trusted_patches(row: Mapping[str, Any]) -> tuple[str, str]:
    solution = _required_mapping(row, "solution")
    verifier = _required_mapping(row, "verifier")
    oracle_patch = solution.get("oracle_patch")
    test_patch = verifier.get("test_patch")
    if not isinstance(oracle_patch, str) or not oracle_patch.strip():
        raise QuarantineTask("oracle_patch_missing", "trusted oracle patch is absent")
    if not isinstance(test_patch, str) or not test_patch.strip():
        raise QuarantineTask("test_patch_missing", "private test patch is absent")
    return oracle_patch, test_patch


def _image_provenance(row: Mapping[str, Any]) -> tuple[str, str, str]:
    sandbox = _required_mapping(row, "sandbox")
    resolved = _source_image(row)
    lock = _required_mapping(sandbox, "image_lock")
    if lock.get("schema_version") != oci_image_lock.LOCK_SCHEMA:
        raise ValueError("semantic admission requires OCI image-lock provenance")
    requested = _required_text(lock, "source_image_requested")
    if _required_text(lock, "source_image_resolved") != resolved:
        raise ValueError("OCI lock resolved-image binding mismatch")
    input_digest = _required_digest(lock, "input_content_digest")
    child = _required_text(lock, "child_manifest_digest")
    if (
        not resolved.endswith("@" + child)
        or lock.get("platform") != {"os": "linux", "architecture": "amd64"}
    ):
        raise ValueError("OCI child/platform binding mismatch")
    return requested, resolved, input_digest


def _source_image(row: Mapping[str, Any]) -> str:
    image = _required_text(_required_mapping(row, "sandbox"), "source_image")
    if _IMMUTABLE_IMAGE.fullmatch(image) is None:
        raise ValueError("semantic admission requires name@sha256 image digests")
    return image


def _source_image_record(record: Mapping[str, Any]) -> str:
    image = _required_text(record, "source_image")
    if _IMMUTABLE_IMAGE.fullmatch(image) is None:
        raise ValueError("semantic admission record image is mutable")
    return image


def _required_mapping(
    value: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be an object")
    return result


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result or "\0" in result:
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


def _sandbox_name(role: str, instance_id: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9-]+", "-", f"swe-{role}-{instance_id}")
    return rendered[:120].strip("-") or "swe-admission"
