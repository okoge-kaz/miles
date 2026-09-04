#!/bin/bash

#SBATCH --job-name=reason-eval-delta
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/reasoning_eval/setup/%x-%A_%a.log

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"

: "${ANALYSIS_ROOT:?set ANALYSIS_ROOT to the analysis snapshot directory}"
: "${STUDY_ROOT:?set STUDY_ROOT to the checkpoint study root}"
: "${RUN_NAMESPACE:?set RUN_NAMESPACE to the training namespace}"
RESULT_STUDY_ROOT="${RESULT_STUDY_ROOT:-${ANALYSIS_ROOT%/analysis/*}}"
if [[ -f "${RESULT_STUDY_ROOT}/grid.env" ]]; then
    source "${RESULT_STUDY_ROOT}/grid.env"
fi
STALENESS_LEVELS="${STALENESS_LEVELS:-1 2 4 8}"
RATIOS="${RATIOS:-1:7 2:6 3:5 4:4}"
INCLUDE_COLOCATED="${INCLUDE_COLOCATED:-1}"
TRAINING_BUFFER_QUEUE_SIZE="${TRAINING_BUFFER_QUEUE_SIZE:-1000}"
ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-}"
STUDY_IDENTITY="$(realpath -- "${STUDY_ROOT}")"

MERGE_ONLY="${MERGE_ONLY:-0}"
SEED_PARTS_ROOT="${SEED_PARTS_ROOT:-}"
PARTS_ROOT="${ANALYSIS_ROOT}/staleness/checkpoint-displacements-parts"
OUTPUT_CSV="${ANALYSIS_ROOT}/staleness/checkpoint-displacements.csv"
SUCCESS_MARKER="${ANALYSIS_ROOT}/staleness/checkpoint-displacements._SUCCESS"
arms=()
for staleness in ${STALENESS_LEVELS}; do
    for ratio in ${RATIOS}; do
        arms+=("s${staleness}-t${ratio%%:*}r${ratio#*:}")
    done
done
if (( INCLUDE_COLOCATED == 1 )); then
    arms+=(s0-colocated)
fi
checkpoint_identity_args=(
    --training-buffer-queue-size "${TRAINING_BUFFER_QUEUE_SIZE}"
)
if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES}" ]]; then
    checkpoint_identity_args+=(
        --async-max-concurrent-samples "${ASYNC_MAX_CONCURRENT_SAMPLES}"
    )
fi

[[ "${MERGE_ONLY}" =~ ^[01]$ ]] || {
    echo "MERGE_ONLY must be 0 or 1" >&2
    exit 2
}
[[ -f "${ANALYSIS_ROOT}/aggregate-results.csv" ]] || {
    echo "aggregate results not found: ${ANALYSIS_ROOT}/aggregate-results.csv" >&2
    exit 3
}

if (( MERGE_ONLY == 1 )); then
    for arm in "${arms[@]}"; do
        part_path="${PARTS_ROOT}/${arm}.csv"
        [[ -s "${part_path}" ]] || {
            echo "checkpoint displacement part is missing: ${part_path}" >&2
            exit 4
        }
    done
    srun \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
        --container-image="${SQSH_IMAGE}" \
        --container-mounts="${REPO_ROOT}:/workspace/miles,${ANALYSIS_ROOT}:/analysis,${STUDY_ROOT}:/study:ro" \
        --container-workdir=/workspace/miles \
        --no-container-mount-home \
        python experiments/tools/reasoning_eval/compute_checkpoint_displacements.py \
            --aggregate-results-csv /analysis/aggregate-results.csv \
            --study-root /study \
            --study-identity "${STUDY_IDENTITY}" \
            --namespace "${RUN_NAMESPACE}" \
            "${checkpoint_identity_args[@]}" \
            --output-csv /analysis/staleness/checkpoint-displacements.csv \
            --merge-parts-root /analysis/staleness/checkpoint-displacements-parts \
            --torch-threads "${SLURM_CPUS_PER_TASK}"
    marker_temporary="${SUCCESS_MARKER}.partial-${SLURM_JOB_ID}"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${marker_temporary}"
    mv -- "${marker_temporary}" "${SUCCESS_MARKER}"
    echo "Merged checkpoint displacements: ${OUTPUT_CSV}"
    exit 0
fi

array_index="${SLURM_ARRAY_TASK_ID:?run compute mode as a zero-based per-arm job array}"
(( array_index >= 0 && array_index < ${#arms[@]} )) || {
    echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#arms[@]} - 1))" >&2
    exit 2
}
arm="${arms[${array_index}]}"
mkdir -p "${PARTS_ROOT}"
rm -f -- "${SUCCESS_MARKER}"
part_path="${PARTS_ROOT}/${arm}.csv"
if [[ -n "${SEED_PARTS_ROOT}" && ! -s "${part_path}" \
    && -s "${SEED_PARTS_ROOT}/${arm}.csv" ]]; then
    cp -- "${SEED_PARTS_ROOT}/${arm}.csv" "${part_path}"
    echo "Seeded ${arm} from ${SEED_PARTS_ROOT}/${arm}.csv"
fi

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${REPO_ROOT}:/workspace/miles,${ANALYSIS_ROOT}:/analysis,${STUDY_ROOT}:/study:ro" \
    --container-workdir=/workspace/miles \
    --no-container-mount-home \
    python experiments/tools/reasoning_eval/compute_checkpoint_displacements.py \
        --aggregate-results-csv /analysis/aggregate-results.csv \
        --study-root /study \
        --study-identity "${STUDY_IDENTITY}" \
        --namespace "${RUN_NAMESPACE}" \
        "${checkpoint_identity_args[@]}" \
        --output-csv "/analysis/staleness/checkpoint-displacements-parts/${arm}.csv" \
        --arm "${arm}" \
        --torch-threads "${SLURM_CPUS_PER_TASK}"
