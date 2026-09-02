#!/bin/bash
#PBS -N tau2-40k-staleness-chain
#PBS -q R9920261300
#PBS -l select=8:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=24:00:00
#PBS -j oe

# Internal worker for one Tau2 40K staleness segment. A local timer requests a
# transactionally matched model/replay checkpoint 15 minutes before the PBS
# walltime and enqueues exactly one afterany successor when training is ongoing.

set -euo pipefail

REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." &>/dev/null && pwd)}}"
source "${REPO_ROOT}/experiments/env.sh"
SEGMENT_SCRIPT="${REPO_ROOT}/experiments/scripts/tau_bench/async/areal-tau2/internal/run_staleness_truncation_segment.sbatch"
BASE_RECIPE="${REPO_ROOT}/experiments/scripts/tau_bench/async/areal-tau2/qwen3-4b-agentic-sft-953/run.sbatch"

: "${RUN_NAME:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_INDEX:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_MAX_SEGMENTS:?submit with submit_staleness_truncation_sweep.sh}"
: "${CHAIN_WALLTIME:=${PBS_DEFAULT_WALLTIME}}"
: "${CHAIN_SIGNAL_LEAD_SECONDS:=900}"
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
[[ "${CHAIN_WALLTIME}" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]] || {
    echo "CHAIN_WALLTIME must use HH:MM:SS: ${CHAIN_WALLTIME}" >&2
    exit 1
}
segment_seconds=$((
    10#${BASH_REMATCH[1]} * 3600 +
    10#${BASH_REMATCH[2]} * 60 +
    10#${BASH_REMATCH[3]}
))
[[ "${CHAIN_SIGNAL_LEAD_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
    && (( CHAIN_SIGNAL_LEAD_SECONDS < segment_seconds )) || {
    echo "CHAIN_SIGNAL_LEAD_SECONDS must be positive and shorter than ${CHAIN_WALLTIME}" >&2
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
    if raw_job_id="$(pbs_submit --parsable --profile gpu \
        --dependency="afterany:${MILES_JOB_ID}" \
        --nodes=8 \
        --time="${CHAIN_WALLTIME}" \
        --job-name="${RUN_NAME}" \
        --output="${CHAIN_LOG_DIR}/${RUN_NAME}-%j.log" \
        --export="ALL,CHAIN_INDEX=${next_index}" \
        "${SEGMENT_SCRIPT}")"; then
        job_id="${raw_job_id}"
        SUCCESSOR_SUBMITTED=1
        SUCCESSOR_JOB_ID="${job_id}"
        printf 'submitted chain successor %s/%s job=%s afterany:%s\n' \
            "${next_index}" "${CHAIN_MAX_SEGMENTS}" "${job_id}" "${MILES_JOB_ID}"
    else
        echo "failed to submit a successor; the current checkpoint request remains active" >&2
    fi
}
trap request_checkpoint_and_successor USR1

printf 'starting Tau2 chain segment %s/%s for %s (job %s)\n' \
    "${CHAIN_INDEX}" "${CHAIN_MAX_SEGMENTS}" "${RUN_NAME}" "${MILES_JOB_ID}"

(
    sleep "$(( segment_seconds - CHAIN_SIGNAL_LEAD_SECONDS ))"
    kill -USR1 "$$"
) &
WALL_TIMER_PID=$!

MILES_SUBMIT_DIR="${REPO_ROOT}" bash "${BASE_RECIPE}" &
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

kill "${WALL_TIMER_PID}" 2>/dev/null || true
wait "${WALL_TIMER_PID}" 2>/dev/null || true

if ((train_status == 0)); then
    # Do not leave a late wall-signal request for a future manual resume.
    rm -f -- "${HOST_SAVE_TRIGGER}"
    if [[ -n "${SUCCESSOR_JOB_ID}" ]]; then
        qdel "${SUCCESSOR_JOB_ID}" || \
            echo "could not cancel completed run's pre-submitted successor ${SUCCESSOR_JOB_ID}" >&2
    fi
fi
exit "${train_status}"
