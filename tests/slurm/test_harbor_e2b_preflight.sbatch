#!/bin/bash
#SBATCH --job-name=test-harbor-e2b
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

# Offline-only validation of the native Harbor E2B provider and Miles overlay.
# Submit with --export=ALL,HARBOR_ROOT=/path/to/harbor-miles-v0.20.0.
# This script never reads .env and never calls the E2B API.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
: "${HARBOR_ROOT:?set HARBOR_ROOT to the official Harbor checkout}"

readonly EXPECTED_HARBOR_COMMIT=2ce5ba2af33a00c9fba0463f6403313996373f85
readonly PATCH_FILE="${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/harbor-miles-e2b.patch"
readonly HARBOR_PYTHON="${HARBOR_PYTHON:-${HARBOR_ROOT}/.venv/bin/python}"

[[ "$(git -C "${HARBOR_ROOT}" rev-parse HEAD)" == "${EXPECTED_HARBOR_COMMIT}" ]] || {
    echo "Harbor must be pinned to ${EXPECTED_HARBOR_COMMIT}" >&2
    exit 1
}
[[ -x "${HARBOR_PYTHON}" ]] || {
    echo "Missing Harbor Python; run 'uv sync --extra e2b' in ${HARBOR_ROOT}" >&2
    exit 1
}
git -C "${HARBOR_ROOT}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" >/dev/null || {
    echo "Miles E2B overlay is not applied to ${HARBOR_ROOT}" >&2
    exit 1
}

export HARBOR_ENV_TYPE=e2b
export E2B_API_KEY=offline-preflight-placeholder
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}:${HARBOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${HARBOR_PYTHON}" \
    "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/preflight.py"
"${HARBOR_PYTHON}" -m pytest -q \
    --confcutdir="${REPO_ROOT}/tests/fast/rollout/harbor" \
    "${REPO_ROOT}/tests/fast/rollout/harbor/test_environment_config.py" \
    "${REPO_ROOT}/tests/fast/rollout/harbor/test_worker.py" \
    "${REPO_ROOT}/tests/fast/rollout/harbor/test_native_e2b_contract.py"
"${HARBOR_PYTHON}" -m pytest -q \
    "${HARBOR_ROOT}/tests/unit/environments/test_e2b.py"

echo "Harbor E2B offline preflight passed; no external sandbox was created."
