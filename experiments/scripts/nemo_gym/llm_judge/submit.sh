#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export MILES_REPO="${REPO_ROOT}"
source "${REPO_ROOT}/experiments/common/pbs.sh"
source "${REPO_ROOT}/experiments/common/job_env.sh"

: "${JUDGE_MODEL_PATH:?set the judge HF checkpoint path inside /ckpt/hf, or a pinned HF model id}"
: "${JUDGE_MODEL_REVISION:?set an immutable revision/checkpoint identifier}"
export JUDGE_RUN_NAME="${JUDGE_RUN_NAME:-judge-$(date +%Y%m%d-%H%M%S)-p$$}"
[[ "${JUDGE_RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid JUDGE_RUN_NAME" >&2; exit 2; }
submit=0
options=()
for option in "$@"; do
    if [[ "${option}" == --submit ]]; then submit=1; else options+=("${option}"); fi
done
(( submit != 0 )) || options+=(--test-only)
log_dir="${REPO_ROOT}/experiments/outputs/judge/sglang"
mkdir -p "${log_dir}"
pbs_submit --profile=gpu --nodes=1 --job-kind=training --chdir="${REPO_ROOT}" \
    --job-name="${JUDGE_RUN_NAME}" --output="${log_dir}/${JUDGE_RUN_NAME}.ou" \
    --export="$(miles_job_export_spec)" "${options[@]}" \
    "${REPO_ROOT}/experiments/scripts/nemo_gym/llm_judge/run.sh"
