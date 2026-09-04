"""Remove source-image file privilege escalation paths before agent runtime."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path

_CAPABILITY_XATTR = "security.capability"
_SKIP_ROOT_NAMES = {"dev", "proc", "sys"}
_RUNTIME_EXECUTABLES = (
    "/bin/bash",
    "/usr/bin/env",
    "/usr/bin/setpriv",
)
_RUNTIME_PATHS = (
    "/bin",
    "/usr/bin",
    "/usr/local/bin",
    "/lib",
    "/lib64",
    "/usr/lib",
    "/usr/lib64",
    "/opt/miniconda3/bin",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/ld.so.preload",
)


def _regular_files(root: Path) -> Iterator[Path]:
    root_device = root.stat(follow_symlinks=False).st_dev
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except FileNotFoundError:
            continue
        for entry in entries:
            if directory == root and entry.name in _SKIP_ROOT_NAMES:
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if metadata.st_dev != root_device or stat.S_ISLNK(metadata.st_mode):
                continue
            path = Path(entry.path)
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                yield path


def _has_capability(path: Path) -> bool:
    try:
        return _CAPABILITY_XATTR in os.listxattr(path, follow_symlinks=False)
    except FileNotFoundError:
        return False


def _rooted(root: Path, absolute: str) -> Path:
    return root / absolute.removeprefix("/")


def _validate_root_owned_path(root: Path, path: Path) -> None:
    """Reject runtime lookup or loader paths writable by the sandbox user."""
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"runtime path escapes image root: {path}") from exc
    current = root_resolved
    candidates = [current]
    for part in relative.parts:
        current /= part
        candidates.append(current)
    for candidate in candidates:
        metadata = candidate.stat(follow_symlinks=False)
        if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise RuntimeError(
                f"runtime path is not root-owned and non-writable: {candidate}"
            )


def validate_runtime_paths(root: Path) -> None:
    """Validate command and loader paths used after dropping to UID 1000."""
    for absolute in _RUNTIME_EXECUTABLES:
        path = _rooted(root, absolute)
        if not path.exists() or not os.access(path, os.X_OK):
            raise RuntimeError(f"required hardening executable is absent: {absolute}")
        _validate_root_owned_path(root, path)
    for absolute in _RUNTIME_PATHS:
        path = _rooted(root, absolute)
        if path.exists():
            _validate_root_owned_path(root, path)


def strip_and_verify(root: Path) -> tuple[int, int]:
    """Strip privileged bits/xattrs and fail if any remain on a second scan."""
    stripped_modes = 0
    stripped_capabilities = 0
    for path in _regular_files(root):
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
            path.chmod(stat.S_IMODE(metadata.st_mode) & ~(stat.S_ISUID | stat.S_ISGID))
            stripped_modes += 1
        if _has_capability(path):
            os.removexattr(path, _CAPABILITY_XATTR, follow_symlinks=False)
            stripped_capabilities += 1
    remaining = []
    for path in _regular_files(root):
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID) or _has_capability(path):
            remaining.append(str(path))
            if len(remaining) >= 20:
                break
    if remaining:
        raise RuntimeError(f"privileged source-image files remain: {remaining}")
    validate_runtime_paths(root)
    return stripped_modes, stripped_capabilities


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("/")
    if len(sys.argv) > 2:
        print("usage: strip_agent_privileges.py [ROOT]", file=sys.stderr)
        return 2
    try:
        stripped_modes, stripped_capabilities = strip_and_verify(root)
    except (OSError, RuntimeError) as exc:
        print(f"source-image privilege scrub failed: {exc}", file=sys.stderr)
        return 2
    print(
        "source-image privilege scrub complete: "
        f"modes={stripped_modes} capabilities={stripped_capabilities}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
