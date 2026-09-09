#!/bin/bash

# Hiso runs this one script. The handoff manifest remains read-only in Kfujii's
# checkout, while the controller, evaluation runtime, cache, and results all use
# Hiso-owned paths.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_REPO_ROOT="$(realpath -- "${RUNTIME_REPO_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/kzk/miles}")"
RESULT_STUDY_BASE="${RESULT_STUDY_BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/evaluations/reasoning_eval/staleness-ratio-sweep}"
CONTROLLER_LOG_DIR="${CONTROLLER_LOG_DIR:-${RESULT_STUDY_BASE}/controller}"
ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
CONTROLLER_PARTITION="${CONTROLLER_PARTITION:-cpu_long}"
CONTROLLER_TIME="${CONTROLLER_TIME:-2-00:00:00}"

usage() {
    cat <<'EOF'
usage: start-hiso-handoff-400.sh [--dry-run]

Validate the exact 400-entry handoff and start a CPU controller that keeps
Hiso's batch evaluation queue filled within the per-user Slurm submit limit.
The default performs the submission; --dry-run only validates the handoff.
EOF
}

dry_run=0
case "${1:-}" in
    "") ;;
    --dry-run) dry_run=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
(( $# <= 1 )) || { usage >&2; exit 2; }

[[ "${USER:-}" == hiso ]] || {
    echo "refusing to submit the delegated queue as USER=${USER:-unset}; run this as hiso" >&2
    exit 3
}
[[ -x "${RUNTIME_REPO_ROOT}/experiments/scripts/reasoning_eval/run-evaluation.sbatch" ]] || {
    echo "evaluation runner is not executable: ${RUNTIME_REPO_ROOT}" >&2
    exit 4
}

export RUNTIME_REPO_ROOT RESULT_STUDY_BASE
export MILES_REPO="${RUNTIME_REPO_ROOT}"
export ROUTES="${ROUTES:-batch=04:00:00}"

echo "validating the exact 400-entry handoff with runtime ${RUNTIME_REPO_ROOT}"
PRINT_LIMIT=5 "${SCRIPT_DIR}/submit-hiso-handoff-400.sh"
if (( dry_run == 1 )); then
    exit 0
fi

mkdir -p "${CONTROLLER_LOG_DIR}"
cd "${RUNTIME_REPO_ROOT}"
raw_job_id="$(sbatch --parsable \
    -A "${ACCOUNT}" \
    --partition="${CONTROLLER_PARTITION}" \
    --time="${CONTROLLER_TIME}" \
    --output="${CONTROLLER_LOG_DIR}/reason-eval-refill-hiso400-%j.log" \
    --export="ALL,HANDOFF_SCRIPT_DIR=${SCRIPT_DIR},RUNTIME_REPO_ROOT=${RUNTIME_REPO_ROOT},MILES_REPO=${MILES_REPO},RESULT_STUDY_BASE=${RESULT_STUDY_BASE},ROUTES=${ROUTES}" \
    "${SCRIPT_DIR}/refill-hiso-handoff-400.sbatch")"
job_id="${raw_job_id%%;*}"
printf 'submitted Hiso handoff controller job=%s manifest_entries=400 routes=%s\n' \
    "${job_id}" "${ROUTES}"
printf 'controller log: %s/reason-eval-refill-hiso400-%s.log\n' \
    "${CONTROLLER_LOG_DIR}" "${job_id}"
