#!/bin/bash
# Opt-in overlay for the pinned OCI a8e5c63 image; never changes the base SQSH.
# Apply before importing SGLang or starting Ray, inside a writable container.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCH_FILE="${REPO_ROOT}/experiments/container/patches/sglang-a8e5c63-prefill.patch"
SGLANG_ROOT="$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("sglang").origin).resolve().parents[2])')"
test "$(git -C "${SGLANG_ROOT}" rev-parse HEAD)" = a8e5c632fe40555f720d4f2c69771ea8cf24f3c4
if git -C "${SGLANG_ROOT}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
    echo "OCI SGLang provenance overlay already applied"
else
    git -C "${SGLANG_ROOT}" apply --check "${PATCH_FILE}"
    git -C "${SGLANG_ROOT}" apply "${PATCH_FILE}"
fi
sha256sum "${PATCH_FILE}"
