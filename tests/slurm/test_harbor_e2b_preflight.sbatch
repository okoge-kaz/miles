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
#SBATCH --export=NIL

# Offline-only validation of the native Harbor E2B provider and Miles overlay.
# Submit with only HARBOR_ROOT and, when needed, HARBOR_PYTHON explicitly
# exported by name. Do not use --export=ALL.
# This script never reads .env and never calls the E2B API.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export HARBOR_TIMEOUT_MULTIPLIER=1
export HARBOR_TRIAL_WALL_TIMEOUT_SEC=12600
export AGENT_TIMEOUT=3600
export AGENT_SETUP_TIMEOUT=1800
export HARBOR_VERIFIER_TIMEOUT_SEC=2100
export HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER=1

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
: "${HARBOR_ROOT:?set HARBOR_ROOT to the official Harbor checkout}"

readonly EXPECTED_HARBOR_COMMIT=2ce5ba2af33a00c9fba0463f6403313996373f85
readonly PATCH_DIR="${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b"
readonly HARBOR_PYTHON="${HARBOR_PYTHON:-${HARBOR_ROOT}/.venv/bin/python}"

[[ "$(git -C "${HARBOR_ROOT}" rev-parse HEAD)" == "${EXPECTED_HARBOR_COMMIT}" ]] || {
    echo "Harbor must be pinned to ${EXPECTED_HARBOR_COMMIT}" >&2
    exit 1
}
[[ -x "${HARBOR_PYTHON}" ]] || {
    echo "Missing Harbor Python; run 'uv sync --extra e2b' in ${HARBOR_ROOT}" >&2
    exit 1
}
"${PATCH_DIR}/apply_harbor_e2b_overlays.sh" "${HARBOR_ROOT}"
bash "${REPO_ROOT}/tests/slurm/support/test_harbor_overlay_helper.sh" \
    "${REPO_ROOT}" "${HARBOR_ROOT}"

export HARBOR_ENV_TYPE=e2b
export HARBOR_E2B_NO_NEW_PRIVS_USERS=1000
export E2B_API_KEY=offline-preflight-placeholder
export PYTHONPATH="${REPO_ROOT}:${HARBOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${HARBOR_PYTHON}" \
    "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/preflight.py"
"${HARBOR_PYTHON}" -m pytest -q \
    --confcutdir="${REPO_ROOT}/tests/fast/rollout/harbor" \
    "${REPO_ROOT}/tests/fast/rollout/harbor/test_environment_config.py" \
    "${REPO_ROOT}/tests/fast/rollout/harbor/test_harbor_auth.py" \
    "${REPO_ROOT}/tests/fast/rollout/harbor/test_e2b_prebuild.py" \
    "${REPO_ROOT}/tests/fast/rollout/harbor/test_worker.py" \
    "${REPO_ROOT}/tests/fast/rollout/harbor/test_native_e2b_contract.py"
"${HARBOR_PYTHON}" -m pytest -q \
    "${HARBOR_ROOT}/tests/unit/environments/test_e2b.py" \
    "${HARBOR_ROOT}/tests/unit/test_single_step_trial.py" \
    "${HARBOR_ROOT}/tests/unit/models/test_artifact_validation.py" \
    "${HARBOR_ROOT}/tests/unit/test_trial_verifier_artifact_transfer.py" \
    "${HARBOR_ROOT}/tests/unit/test_miles_agent_server_results.py"

echo "Harbor E2B offline preflight passed; no external sandbox was created."
