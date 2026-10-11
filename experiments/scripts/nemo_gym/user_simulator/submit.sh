#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export MILES_REPO="${REPO_ROOT}"
source "${REPO_ROOT}/experiments/common/pbs.sh"
source "${REPO_ROOT}/experiments/common/job_env.sh"

: "${TAU_SERVER_MODEL_PATH:?set the frozen simulator checkpoint path}"
: "${TAU_SERVER_MODEL_REVISION:?set its immutable checkpoint identity}"
export TAU_SERVER_RUN_NAME="${TAU_SERVER_RUN_NAME:-tau-user-$(date +%Y%m%d-%H%M%S)-p$$}"
[[ "${TAU_SERVER_RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
submit=0
options=()
for option in "$@"; do
    if [[ "${option}" == --submit ]]; then submit=1; else options+=("${option}"); fi
done

(( submit != 0 )) || options+=(--test-only)
log_dir="${REPO_ROOT}/experiments/outputs/multi_env/user_simulator"
mkdir -p "${log_dir}"

pbs_submit --profile=gpu --nodes=1 --job-kind=training --chdir="${REPO_ROOT}" \
    --job-name="${TAU_SERVER_RUN_NAME}" --output="${log_dir}/${TAU_SERVER_RUN_NAME}.ou" \
    --export="$(miles_job_export_spec)" "${options[@]}" \
    "${REPO_ROOT}/experiments/scripts/nemo_gym/user_simulator/run.sh"
