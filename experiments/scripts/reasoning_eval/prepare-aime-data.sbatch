#!/bin/bash

#SBATCH --job-name=reason-eval-data
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/reasoning_eval/setup/%x-%j.log

set -euo pipefail

SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
else
    REPO_ROOT="$(cd -- "$(dirname -- "${SCRIPT_PATH}")/../../.." && pwd)"
fi
source "${REPO_ROOT}/experiments/env.sh"

CONTAINER_ROOT="${REASONING_EVAL_CONTAINER_ROOT:-${SHARED_WS}/containers}"
NEMO_SKILLS_IMAGE="${CONTAINER_ROOT}/nemo-evaluator-nemo-skills-26.03-ac1b048e.sqsh"
NEMO_SKILLS_DATA_ROOT="${REASONING_EVAL_DATA_ROOT:-${DATASET_DIR}/evaluation}/nemo-skills-26.03"
EVALUATION_CACHE_ROOT="${REASONING_EVAL_CACHE_ROOT:-${CACHE_DIR}/reasoning_eval}"
PREPARED_MARKER="${NEMO_SKILLS_DATA_ROOT}/_PREPARED"
BENCHMARKS=(aime24 aime25 aime26)
HOST_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "${REPO_ROOT}"
    mkdir -p experiments/outputs/reasoning_eval/setup
    sbatch --test-only --export=ALL "${SCRIPT_PATH}"
    exec sbatch --export=ALL "${SCRIPT_PATH}" "$@"
fi

[[ -f "${NEMO_SKILLS_IMAGE}" ]] || {
    echo "NeMo Skills image not found: ${NEMO_SKILLS_IMAGE}" >&2
    echo "Run experiments/scripts/reasoning_eval/import-evaluator-images.sbatch first." >&2
    exit 2
}

AUTH_CONTAINER_MOUNT=""
AUTH_CONTAINER_EXPORT=""
if [[ -z "${HF_TOKEN:-}" && -s "${HOST_HF_TOKEN_PATH}" ]]; then
    AUTH_CONTAINER_MOUNT=",${HOST_HF_TOKEN_PATH}:/evaluation-auth/token:ro"
    AUTH_CONTAINER_EXPORT=",HF_TOKEN_PATH=/evaluation-auth/token"
fi

mkdir -p "${NEMO_SKILLS_DATA_ROOT}" "${EVALUATION_CACHE_ROOT}/huggingface"
exec 9> "${NEMO_SKILLS_DATA_ROOT}/.prepare.lock"
flock --nonblock 9 || { echo "another AIME data preparation job is active" >&2; exit 3; }
if [[ -f "${PREPARED_MARKER}" ]]; then
    if grep -Fx "nemo_skills_image=${NEMO_SKILLS_IMAGE}" "${PREPARED_MARKER}" >/dev/null \
        && grep -Fx "benchmarks=${BENCHMARKS[*]}" "${PREPARED_MARKER}" >/dev/null; then
        echo "AIME24/25/26 data is already prepared: ${PREPARED_MARKER}"
        exit 0
    fi
    echo "existing preparation marker has different provenance: ${PREPARED_MARKER}" >&2
    echo "use a new REASONING_EVAL_DATA_ROOT instead of mixing evaluator versions" >&2
    exit 4
fi

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --container-image="${NEMO_SKILLS_IMAGE}" \
    --container-mounts="${NEMO_SKILLS_DATA_ROOT}:/evaluation-data,${EVALUATION_CACHE_ROOT}:/evaluation-cache${AUTH_CONTAINER_MOUNT}" \
    --container-workdir=/nemo_run/code \
    --no-container-mount-home \
    --export="ALL,HF_HOME=/evaluation-cache/huggingface,NEMO_SKILLS_DATA_DIR=/evaluation-data,XDG_CACHE_HOME=/evaluation-cache/xdg,PYTHONUNBUFFERED=1${AUTH_CONTAINER_EXPORT}" \
    bash -lc '
        set -euo pipefail
        for benchmark in "$@"; do
            ns prepare_data "${benchmark}" --cluster=local --data_dir=/evaluation-data
        done
    ' _ "${BENCHMARKS[@]}"

MARKER_PARTIAL="${PREPARED_MARKER}.partial-${SLURM_JOB_ID}"
{
    printf 'nemo_skills_image=%s\n' "${NEMO_SKILLS_IMAGE}"
    printf 'benchmarks=%s\n' "${BENCHMARKS[*]}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'prepared_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${MARKER_PARTIAL}"
mv -- "${MARKER_PARTIAL}" "${PREPARED_MARKER}"
echo "AIME24/25/26 data preparation completed: ${PREPARED_MARKER}"
