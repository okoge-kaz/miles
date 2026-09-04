#!/bin/bash
#SBATCH --job-name=tau2-40k-staleness-chain
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=experiments/outputs/training/tau_bench/areal-tau2/qwen3-4b-agentic-sft-953/%x-%j.log

# Internal worker for one four-hour segment of the Tau2 40K staleness study.
# Invoke it through submit_staleness_truncation_sweep.sh. Fifteen minutes before
# the wall limit, request a transactionally matched model/replay checkpoint and
# enqueue exactly one afterany successor. A normally completed training process
# receives no signal and therefore ends its chain.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." &>/dev/null && pwd)"
SEGMENT_SCRIPT="${REPO_ROOT}/experiments/scripts/tau_bench/async/areal-tau2/internal/run_staleness_truncation_segment.sbatch"
BASE_RECIPE="${REPO_ROOT}/experiments/scripts/tau_bench/async/areal-tau2/qwen3-4b-agentic-sft-953/run.sbatch"
readonly IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Tau2 multi-turn RL waits for external user-simulator turns and long policy trajectories"}}'

: "${RUN_NAME:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_INDEX:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_MAX_SEGMENTS:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_ACCOUNT:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_PARTITION:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_QOS:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_LOG_DIR:?submit with submit_staleness_truncation_sweep.sh}"
: "${SAVE_TRIGGER_SENTINEL:?submit with submit_staleness_truncation_sweep.sh}"

[[ "${CHAIN_INDEX}" =~ ^[1-9][0-9]*$ ]] || {
    echo "CHAIN_INDEX must be positive" >&2
    exit 1
}
[[ "${CHAIN_MAX_SEGMENTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "CHAIN_MAX_SEGMENTS must be positive" >&2
    exit 1
}
[[ "${SAVE_TRIGGER_SENTINEL}" == /root/miles/experiments/outputs/.save-trigger/* ]] || {
    echo "unexpected SAVE_TRIGGER_SENTINEL: ${SAVE_TRIGGER_SENTINEL}" >&2
    exit 1
}

HOST_SAVE_TRIGGER="${REPO_ROOT}/${SAVE_TRIGGER_SENTINEL#/root/miles/}"
mkdir -p "$(dirname -- "${HOST_SAVE_TRIGGER}")" "${CHAIN_LOG_DIR}"

SUCCESSOR_SUBMITTED=0
SUCCESSOR_JOB_ID=
request_checkpoint_and_successor() {
    local next_index raw_job_id job_id

    touch -- "${HOST_SAVE_TRIGGER}"
    printf 'chain segment %s received the wall-time signal; requested checkpoint via %s\n' \
        "${CHAIN_INDEX}" "${SAVE_TRIGGER_SENTINEL}"

    ((SUCCESSOR_SUBMITTED == 0)) || return 0
    if ((CHAIN_INDEX >= CHAIN_MAX_SEGMENTS)); then
        echo "CHAIN_MAX_SEGMENTS=${CHAIN_MAX_SEGMENTS} reached; no successor submitted" >&2
        return 0
    fi

    next_index=$((CHAIN_INDEX + 1))
    if raw_job_id="$(sbatch --parsable \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --account="${CHAIN_ACCOUNT}" \
        --partition="${CHAIN_PARTITION}" \
        --qos="${CHAIN_QOS}" \
        --nodes=8 \
        --time=04:00:00 \
        --job-name="${RUN_NAME}" \
        --comment="${IDLE_EXEMPTION}" \
        --output="${CHAIN_LOG_DIR}/${RUN_NAME}-%j.log" \
        --export="ALL,CHAIN_INDEX=${next_index}" \
        "${SEGMENT_SCRIPT}")"; then
        job_id="${raw_job_id%%;*}"
        SUCCESSOR_SUBMITTED=1
        SUCCESSOR_JOB_ID="${job_id}"
        printf 'submitted chain successor %s/%s job=%s afterany:%s\n' \
            "${next_index}" "${CHAIN_MAX_SEGMENTS}" "${job_id}" "${SLURM_JOB_ID}"
    else
        echo "failed to submit a successor; the current checkpoint request remains active" >&2
    fi
}
trap request_checkpoint_and_successor USR1

printf 'starting Tau2 chain segment %s/%s for %s (job %s)\n' \
    "${CHAIN_INDEX}" "${CHAIN_MAX_SEGMENTS}" "${RUN_NAME}" "${SLURM_JOB_ID}"

SLURM_SUBMIT_DIR="${REPO_ROOT}" bash "${BASE_RECIPE}" &
TRAIN_PID=$!
train_status=0
set +e
while true; do
    wait "${TRAIN_PID}"
    train_status=$?
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
        # USR1 interrupted wait while the training child kept running.
        continue
    fi
    break
done
set -e

if ((train_status == 0)); then
    # Do not leave a late wall-signal request for a future manual resume.
    rm -f -- "${HOST_SAVE_TRIGGER}"
    if [[ -n "${SUCCESSOR_JOB_ID}" ]]; then
        scancel "${SUCCESSOR_JOB_ID}" || \
            echo "could not cancel completed run's pre-submitted successor ${SUCCESSOR_JOB_ID}" >&2
    fi
fi
exit "${train_status}"
