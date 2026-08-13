#!/bin/bash
#SBATCH --job-name=queue-policy-smoke
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/queue-policy-smoke-%j.log

set -euo pipefail

CANDIDATE_REPO="${CANDIDATE_REPO:-${SLURM_SUBMIT_DIR}}"
BASELINE_REPO="${BASELINE_REPO:-${CANDIDATE_REPO}-baseline}"
VALIDATION_DIR="${CANDIDATE_REPO}/experiments/outputs/queue_policies/${SLURM_JOB_ID}"
RECIPE="experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch"

mkdir -p "${VALIDATION_DIR}"

run_one() {
    local repo="$1"
    local label="$2"
    local policy="$3"
    local output="${VALIDATION_DIR}/${label}.log"

    echo "starting ${label} from ${repo}; policy=${policy}; log=${output}"
    (
        export MILES_REPO="${repo}"
        export SLURM_SUBMIT_DIR="${repo}"
        export CONFIG_TAG="queue-policy-smoke-${label}-${SLURM_JOB_ID}"
        export NUM_ROLLOUT=300
        export DEBUG_EXIT_AFTER_ROLLOUT=4
        export ROLLOUT_BATCH_SIZE=8
        export N_SAMPLES_PER_PROMPT=8
        export GLOBAL_BATCH_SIZE=64
        export MAX_RESPONSE_LEN=2048
        export STALENESS_REFERENCE=prefill
        export ASYNC_MAX_CONCURRENT_SAMPLES=128
        export ACTOR_NUM_NODES=1
        export ACTOR_GPUS_PER_NODE=8
        export ROLLOUT_NUM_GPUS=8
        export SAVE_INTERVAL=1000
        export SAVE_RETAIN_INTERVAL=1000
        export SAVE_HF=0
        export EVAL_INTERVAL=0
        export DUMP_TRAIN_DATA=0

        if [[ "${policy}" == baseline ]]; then
            unset QUEUE_POLICY QUEUE_FACTOR
            export MAX_WEIGHT_STALENESS=2
        elif [[ "${policy}" == queue-drop ]]; then
            export QUEUE_POLICY=queue-drop
            export QUEUE_FACTOR=1
            unset MAX_WEIGHT_STALENESS
        else
            export QUEUE_POLICY="${policy}"
            export QUEUE_FACTOR=1
            if [[ "${policy}" == queue-max ]]; then
                export MAX_WEIGHT_STALENESS=0
            else
                export MAX_WEIGHT_STALENESS=2
            fi
        fi

        cd "${repo}"
        bash "${RECIPE}"
    ) >"${output}" 2>&1
    echo "finished ${label}"
}

run_one "${BASELINE_REPO}" baseline baseline
run_one "${CANDIDATE_REPO}" queue_recycle queue-recycle
run_one "${CANDIDATE_REPO}" queue_max queue-max
run_one "${CANDIDATE_REPO}" queue_drop queue-drop

echo "all queue-policy smoke runs finished: ${VALIDATION_DIR}"
