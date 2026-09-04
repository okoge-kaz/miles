#!/bin/bash

# Exercise clean apply, idempotency, and malicious-untracked rejection without
# modifying the shared pinned checkout.

set -euo pipefail
umask 077

REPO_ROOT="${1:?usage: test_harbor_overlay_helper.sh REPO_ROOT HARBOR_ROOT}"
HARBOR_SOURCE="${2:?usage: test_harbor_overlay_helper.sh REPO_ROOT HARBOR_ROOT}"
HELPER="${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/apply_harbor_e2b_overlays.sh"
TEST_ROOT="$(mktemp -d /tmp/harbor-overlay-selftest.XXXXXX)"

cleanup() {
    case "${TEST_ROOT}" in
        /tmp/harbor-overlay-selftest.*) rm -rf -- "${TEST_ROOT}" ;;
        *) echo "refusing unsafe overlay self-test cleanup" >&2 ;;
    esac
}
trap cleanup EXIT

git clone --quiet --no-hardlinks "${HARBOR_SOURCE}" "${TEST_ROOT}/repo"
"${HELPER}" "${TEST_ROOT}/repo"
"${HELPER}" "${TEST_ROOT}/repo"

touch "${TEST_ROOT}/repo/sitecustomize.py"
if "${HELPER}" "${TEST_ROOT}/repo" >"${TEST_ROOT}/unexpected.log" 2>&1; then
    echo "overlay helper accepted an unrelated untracked Python file" >&2
    exit 1
fi
grep -q "unrelated untracked file" "${TEST_ROOT}/unexpected.log"
echo "Harbor overlay helper clean/idempotent/untracked checks passed."
