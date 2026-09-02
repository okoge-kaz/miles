#!/bin/bash

#PBS -N reason-eval-data
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -j oe

set -euo pipefail

SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
if [[ -n "${MILES_JOB_ID:-}" ]]; then
    REPO_ROOT="$(realpath -- "${PBS_O_WORKDIR:?submit from the Miles repository root}")"
else
    REPO_ROOT="$(cd -- "$(dirname -- "${SCRIPT_PATH}")/../../.." && pwd)"
fi
source "${REPO_ROOT}/experiments/env.sh"

CONTAINER_ROOT="${REASONING_EVAL_CONTAINER_ROOT:-${CONTAINER_DIR}/reasoning-eval}"
NEMO_SKILLS_IMAGE="${CONTAINER_ROOT}/nemo-evaluator-nemo-skills-26.03-ac1b048e.sif"
NEMO_SKILLS_DATA_ROOT="${REASONING_EVAL_DATA_ROOT:-${DATASET_DIR}/evaluation}/nemo-skills-26.03"
EVALUATION_CACHE_ROOT="${REASONING_EVAL_CACHE_ROOT:-${CACHE_DIR}/reasoning_eval}"
PREPARED_MARKER="${NEMO_SKILLS_DATA_ROOT}/_PREPARED"
BENCHMARKS=(aime24 aime25 aime26)
PREPARED_DATA_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/prepared_data.py"
MILES_EXPORT_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/export_miles_aime.py"
MILES_AIME_DATA_ROOT="${MILES_AIME_DATA_ROOT:-${DATASET_DIR}}"
HOST_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
PREPARED_DATA_ARGS=(
    --data-root "${NEMO_SKILLS_DATA_ROOT}"
    --image "${NEMO_SKILLS_IMAGE}"
)
for benchmark in "${BENCHMARKS[@]}"; do
    PREPARED_DATA_ARGS+=(--benchmark "${benchmark}")
done

export_miles_aime_data() {
    python3 "${MILES_EXPORT_TOOL}" \
        --source-root "${NEMO_SKILLS_DATA_ROOT}" \
        --output-root "${MILES_AIME_DATA_ROOT}"
}

if [[ -z "${MILES_JOB_ID:-}" ]]; then
    cd "${REPO_ROOT}"
    mkdir -p experiments/outputs/reasoning_eval/setup
    pbs_submit --test-only --profile cpu --export=ALL "${SCRIPT_PATH}"
    exec pbs_submit --profile cpu --export=ALL "${SCRIPT_PATH}" "$@"
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
    if python3 "${PREPARED_DATA_TOOL}" \
        "${PREPARED_DATA_ARGS[@]}" --marker "${PREPARED_MARKER}"; then
        export_miles_aime_data
        echo "AIME24/25/26 data is already prepared: ${PREPARED_MARKER}"
        exit 0
    fi
    echo "existing preparation marker has different provenance: ${PREPARED_MARKER}" >&2
    echo "use a new REASONING_EVAL_DATA_ROOT instead of mixing evaluator versions" >&2
    exit 4
fi

miles_srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --image="${NEMO_SKILLS_IMAGE}" \
    --bind="${NEMO_SKILLS_DATA_ROOT}:/evaluation-data,${EVALUATION_CACHE_ROOT}:/evaluation-cache${AUTH_CONTAINER_MOUNT}" \
    --cwd=/nemo_run/code \
    --no-home \
    --export="ALL,HF_HOME=/evaluation-cache/huggingface,NEMO_SKILLS_DATA_DIR=/evaluation-data,XDG_CACHE_HOME=/evaluation-cache/xdg,PYTHONUNBUFFERED=1${AUTH_CONTAINER_EXPORT}" \
    bash -lc '
        set -euo pipefail
        for benchmark in "$@"; do
            ns prepare_data "${benchmark}" --cluster=local --data_dir=/evaluation-data
        done
    ' _ "${BENCHMARKS[@]}"

DATASET_PROVENANCE="$(
    python3 "${PREPARED_DATA_TOOL}" "${PREPARED_DATA_ARGS[@]}" --print-env
)"
MARKER_PARTIAL="${PREPARED_MARKER}.partial-${MILES_JOB_ID}"
{
    printf 'nemo_skills_image=%s\n' "${NEMO_SKILLS_IMAGE}"
    printf 'benchmarks=%s\n' "${BENCHMARKS[*]}"
    printf '%s\n' "${DATASET_PROVENANCE}"
    printf 'pbs_job_id=%s\n' "${MILES_JOB_ID}"
    printf 'prepared_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${MARKER_PARTIAL}"
mv -- "${MARKER_PARTIAL}" "${PREPARED_MARKER}"
export_miles_aime_data
echo "AIME24/25/26 data preparation completed: ${PREPARED_MARKER}"
