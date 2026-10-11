#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export MILES_REPO="${REPO_ROOT}"
source "${REPO_ROOT}/experiments/common/pbs.sh"
source "${REPO_ROOT}/experiments/common/job_env.sh"
submit=0
options=()
for option in "$@"; do
    if [[ "${option}" == --submit ]]; then submit=1; else options+=("${option}"); fi
done
(( submit != 0 )) || options+=(--test-only)
log_dir="${REPO_ROOT}/experiments/outputs/data"
mkdir -p "${log_dir}"

pbs_submit --profile=cpu --job-kind=prep --chdir="${REPO_ROOT}" \
    --output="${log_dir}/multi-env-$(date +%Y%m%d-%H%M%S)-p$$.ou" \
    --export="$(miles_job_export_spec)" "${options[@]}" \
    "${REPO_ROOT}/experiments/scripts/multi_env/prepare/run.sh"
