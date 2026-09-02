#!/bin/bash
#PBS -N reason-eval-refill
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=24:00:00
#PBS -j oe

# Keep a fixed number of snapshot evaluations in the single PBS GPU queue.

set -euo pipefail

REPO_ROOT="$(realpath -- "${PBS_O_WORKDIR:?submit from the Miles repository root}")"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/experiments/scripts/reasoning_eval/submit-staleness-sweep.sh}"
source "${REPO_ROOT}/experiments/env.sh"

: "${TRAINING_ROOT:?set TRAINING_ROOT to the checkpoint root}"
: "${RUN_NAMESPACE:?set RUN_NAMESPACE to the training namespace}"
: "${SNAPSHOT_ARM_MAX_STEPS:?set the arm=max_step snapshot}"

EXPECTED_CHECKPOINTS="${EXPECTED_CHECKPOINTS:-130}"
POLL_SECONDS="${POLL_SECONDS:-30}"
INFLIGHT_TARGET="${INFLIGHT_TARGET:-4}"
EVAL_WALL="${EVAL_WALL:-04:00:00}"
QSTAT_TIMEOUT_SECONDS="${QSTAT_TIMEOUT_SECONDS:-10}"
TRUST_PINNED_SNAPSHOT="${TRUST_PINNED_SNAPSHOT:-1}"
REFILL_ONCE="${REFILL_ONCE:-0}"

for value in \
    "${EXPECTED_CHECKPOINTS}" "${POLL_SECONDS}" "${INFLIGHT_TARGET}" \
    "${QSTAT_TIMEOUT_SECONDS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "controller integer controls must be positive: ${value}" >&2
        exit 2
    }
done
for value in "${TRUST_PINNED_SNAPSHOT}" "${REFILL_ONCE}"; do
    [[ "${value}" =~ ^[01]$ ]] || {
        echo "controller boolean controls must be 0 or 1: ${value}" >&2
        exit 2
    }
done
[[ -x "${LAUNCHER}" ]] || { echo "launcher is not executable: ${LAUNCHER}" >&2; exit 3; }

run_launcher() {
    TRAINING_ROOT="${TRAINING_ROOT}" \
    RUN_NAMESPACE="${RUN_NAMESPACE}" \
    SNAPSHOT_ARM_MAX_STEPS="${SNAPSHOT_ARM_MAX_STEPS}" \
    TRUST_PINNED_SNAPSHOT="${TRUST_PINNED_SNAPSHOT}" \
    WALL="${EVAL_WALL}" \
    PRINT_LIMIT=0 \
    QSTAT_TIMEOUT_SECONDS="${QSTAT_TIMEOUT_SECONDS}" \
        "${LAUNCHER}" "$@"
}

extract_status_count() {
    local status_output="$1"
    local label="$2"
    sed -n "s/^status:.*${label}=\([0-9][0-9]*\).*/\1/p" <<< "${status_output}"
}

active_evaluation_count() {
    local query_output
    local query_status
    local -a job_ids=()

    set +e
    query_output="$(
        timeout "${QSTAT_TIMEOUT_SECONDS}" qselect -u "${USER:?USER is not set}" 2>/dev/null
    )"
    query_status=$?
    set -e
    (( query_status == 0 || query_status == 162 )) || return 1
    mapfile -t job_ids <<< "${query_output}"
    if (( ${#job_ids[@]} == 0 )) || [[ -z "${job_ids[0]}" ]]; then
        echo 0
        return
    fi
    timeout "${QSTAT_TIMEOUT_SECONDS}" qstat -f -F json "${job_ids[@]}" 2>/dev/null \
        | jq '[
            (.Jobs // {} | to_entries[])
            | select((.value.Job_Name // "") | startswith("q3e-"))
            | select((.value.job_state // "") | IN("Q", "H", "W", "R", "E", "B"))
        ] | length'
}

while true; do
    scan_output="$(run_launcher)"
    printf '%s\n' "${scan_output}"
    available_count="$(extract_status_count "${scan_output}" available)"
    complete_count="$(extract_status_count "${scan_output}" complete)"
    pending_count="$(extract_status_count "${scan_output}" pending)"
    [[ -n "${available_count}" && -n "${complete_count}" && -n "${pending_count}" ]] || {
        echo "could not parse launcher status" >&2
        exit 4
    }
    (( available_count == EXPECTED_CHECKPOINTS )) || {
        echo "snapshot availability changed: expected=${EXPECTED_CHECKPOINTS} actual=${available_count}" >&2
        exit 5
    }
    if (( complete_count == EXPECTED_CHECKPOINTS )); then
        echo "all ${EXPECTED_CHECKPOINTS} checkpoint evaluations are complete"
        exit 0
    fi

    if ! active_count="$(active_evaluation_count)"; then
        echo "could not query active PBS evaluation jobs; retrying" >&2
        sleep "${POLL_SECONDS}"
        continue
    fi
    slots=$((INFLIGHT_TARGET - active_count))
    if (( slots > pending_count )); then
        slots="${pending_count}"
    fi
    if (( slots > 0 )); then
        echo "submitting ${slots} snapshot evaluation job(s); active=${active_count} target=${INFLIGHT_TARGET}"
        if ! run_launcher --submit --max-submissions "${slots}"; then
            echo "PBS submission was not fully accepted; retrying after the poll interval" >&2
        fi
    else
        echo "no refill needed: active=${active_count} pending=${pending_count} target=${INFLIGHT_TARGET}"
    fi

    if (( REFILL_ONCE == 1 )); then
        echo "completed one refill controller iteration"
        exit 0
    fi
    sleep "${POLL_SECONDS}"
done
