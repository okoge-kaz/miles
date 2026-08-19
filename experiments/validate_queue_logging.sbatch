#!/bin/bash
#SBATCH --job-name=queue-logging-ab
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/queue-logging-ab-%j.log

set -euo pipefail

CANDIDATE_REPO="${CANDIDATE_REPO:-${SLURM_SUBMIT_DIR}}"
BASELINE_REPO="${BASELINE_REPO:-${CANDIDATE_REPO}-baseline}"
VALIDATION_DIR="${CANDIDATE_REPO}/experiments/outputs/queue_logging/${SLURM_JOB_ID}"
RECIPE="experiments/math_async/dapo-math-p10-90/qwen3-4b/run.sbatch"

mkdir -p "${VALIDATION_DIR}"

run_one() {
    local repo="$1"
    local label="$2"
    local output="${VALIDATION_DIR}/${label}.log"

    echo "starting ${label} from ${repo}; log: ${output}"
    (
        export MILES_REPO="${repo}"
        export SLURM_SUBMIT_DIR="${repo}"
        export CONFIG_TAG="queue-logging-${label}-${SLURM_JOB_ID}"
        export NUM_ROLLOUT=300
        export DEBUG_EXIT_AFTER_ROLLOUT=4
        export ROLLOUT_BATCH_SIZE=8
        export N_SAMPLES_PER_PROMPT=8
        export GLOBAL_BATCH_SIZE=64
        export MAX_RESPONSE_LEN=2048
        export MAX_WEIGHT_STALENESS=2
        export STALENESS_REFERENCE=prefill
        export ACTOR_NUM_NODES=1
        export ACTOR_GPUS_PER_NODE=8
        export ROLLOUT_NUM_GPUS=8
        export SAVE_INTERVAL=1000
        export SAVE_RETAIN_INTERVAL=1000
        export SAVE_HF=0
        export EVAL_INTERVAL=0
        export DUMP_TRAIN_DATA=0
        cd "${repo}"
        bash "${RECIPE}"
    ) >"${output}" 2>&1
    echo "finished ${label}"
}

run_one "${BASELINE_REPO}" baseline_before
run_one "${CANDIDATE_REPO}" instrumented
run_one "${BASELINE_REPO}" baseline_after

echo "all queue logging validation runs finished: ${VALIDATION_DIR}"
