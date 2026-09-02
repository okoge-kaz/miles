#!/bin/bash
# Submit a CPU readiness gate and make the four-node SWE job depend on it.

set -euo pipefail

umask 077

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd -P)}"
USER="${USER:-$(id -un)}"
export USER
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
source "${REPO_ROOT}/experiments/env.sh"
: "${AGENT_SERVER_URL:?set the private Harbor agent-server origin}"
: "${HARBOR_RUN_SECRET:?export the job-scoped Harbor run secret}"
(( ${#HARBOR_RUN_SECRET} >= 32 && ${#HARBOR_RUN_SECRET} <= 4096 )) && \
    [[ "${HARBOR_RUN_SECRET}" != *$'\r'* && \
        "${HARBOR_RUN_SECRET}" != *$'\n'* ]] || {
    echo "HARBOR_RUN_SECRET must be 32-4096 characters without CR/LF" >&2
    exit 2
}

: "${SWE_DATASET:=swe-rebench-v2}"
case "${SWE_DATASET}" in
    r2e-gym-v1 | nemotron3-super-swe2-r2e | \
        nemotron3-super-swe2-swe-gym | swe-gym | swe-rebench-v2 | \
        swe-rebench-v2-filtered-verified | nemotron3-ultra-swe-rebench-v2 | \
        nemotron3-ultra-swe-gym)
        dataset_tag="${SWE_DATASET}"
        ;;
    *)
        echo "unknown SWE_DATASET=${SWE_DATASET}" >&2
        exit 2
        ;;
esac
SWE_HOST_ADMISSION_SUMMARY="${DATASET_DIR}/miles-swe/admitted/${dataset_tag}-train.summary.json"
export SWE_HOST_ADMISSION_SUMMARY SWE_DATASET

readonly GATE_JOB="${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/wait_agent_server.sbatch"
readonly TRAINING_JOB="${REPO_ROOT}/experiments/scripts/swe/async/swe-rebench-v2-swe-gym/qwen3-4b/run.sbatch"
MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly GATE_EXPORTS="AGENT_SERVER_URL,HARBOR_READINESS_TIMEOUT_SEC,HARBOR_RUN_SECRET,MILES_SWE_FIXED_EXPORTS,SWE_HOST_ADMISSION_SUMMARY"
readonly TRAINING_EXPORTS="AGENT_SERVER_URL,ASYNC_MAX_CONCURRENT_SAMPLES,DEBUG_EXIT_AFTER_ROLLOUT,HARBOR_RUN_SECRET,MILES_SWE_FIXED_EXPORTS,NUM_ROLLOUT,ROLLOUT_BATCH_SIZE,SWE_AGENT_NAME,SWE_AGENT_SERVER_READY_TIMEOUT_SEC,SWE_DATASET,SWE_TRIAL_REQUEST_TIMEOUT_SEC"

gate_submission="$(
    pbs_submit \
        --parsable \
        --profile cpu \
        --chdir="${REPO_ROOT}" \
        --export="${GATE_EXPORTS}" \
        "${GATE_JOB}"
)"
gate_job_id="${gate_submission%%;*}"
[[ "${gate_job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || {
    echo "PBS returned an invalid readiness job ID" >&2
    exit 2
}
training_submission="$(
    pbs_submit \
        --parsable \
        --profile gpu \
        --chdir="${REPO_ROOT}" \
        --dependency="afterok:${gate_job_id}" \
        --export="${TRAINING_EXPORTS}" \
        "${TRAINING_JOB}"
)"
training_job_id="${training_submission%%;*}"
[[ "${training_job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || {
    echo "PBS returned an invalid training job ID" >&2
    exit 2
}

printf 'readiness_job_id=%s\n' "${gate_job_id}"
printf 'training_job_id=%s\n' "${training_job_id}"
