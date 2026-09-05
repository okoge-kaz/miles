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

SNAPSHOT_ARM_MAX_STEPS="${SNAPSHOT_ARM_MAX_STEPS:-}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${WS}/evaluations/reasoning_eval}"
RESULT_STUDY_ROOT="${EVALUATION_ROOT}/staleness-ratio-sweep/${RUN_NAMESPACE}"
if [[ -f "${RESULT_STUDY_ROOT}/grid.env" ]]; then
    source "${RESULT_STUDY_ROOT}/grid.env"
fi
STALENESS_LEVELS="${STALENESS_LEVELS:-1 2 4 8}"
RATIOS="${RATIOS:-1:7 2:6 3:5 4:4}"
INCLUDE_COLOCATED="${INCLUDE_COLOCATED:-1}"
START_STEP="${START_STEP:-10}"
END_STEP="${END_STEP:-300}"
STEP_INTERVAL="${STEP_INTERVAL:-10}"
staleness_count=$(wc -w <<< "${STALENESS_LEVELS}")
ratio_count=$(wc -w <<< "${RATIOS}")
arm_count=$(( staleness_count * ratio_count + INCLUDE_COLOCATED ))
default_expected_checkpoints=$(( arm_count * ((END_STEP - START_STEP) / STEP_INTERVAL + 1) ))
if [[ -n "${SNAPSHOT_ARM_MAX_STEPS}" ]]; then
    default_expected_checkpoints=0
    IFS=',' read -r -a snapshot_entries <<< "${SNAPSHOT_ARM_MAX_STEPS}"
    for snapshot_entry in "${snapshot_entries[@]}"; do
        max_step="${snapshot_entry#*=}"
        default_expected_checkpoints=$((
            default_expected_checkpoints + (max_step - START_STEP) / STEP_INTERVAL + 1
        ))
    done
fi
EXPECTED_CHECKPOINTS="${EXPECTED_CHECKPOINTS:-${default_expected_checkpoints}}"
POLL_SECONDS="${POLL_SECONDS:-30}"
INFLIGHT_TARGET="${INFLIGHT_TARGET:-4}"
EVAL_WALL="${EVAL_WALL:-04:00:00}"
QSTAT_TIMEOUT_SECONDS="${QSTAT_TIMEOUT_SECONDS:-10}"
DYNAMIC_QUIESCENT_SCANS="${DYNAMIC_QUIESCENT_SCANS:-12}"
CONTROLLER_DRY_RUN="${CONTROLLER_DRY_RUN:-0}"
default_trust_pinned_snapshot=0
[[ -z "${SNAPSHOT_ARM_MAX_STEPS}" ]] || default_trust_pinned_snapshot=1
TRUST_PINNED_SNAPSHOT="${TRUST_PINNED_SNAPSHOT:-${default_trust_pinned_snapshot}}"
REFILL_ONCE="${REFILL_ONCE:-0}"

for value in \
    "${EXPECTED_CHECKPOINTS}" "${POLL_SECONDS}" "${INFLIGHT_TARGET}" \
    "${QSTAT_TIMEOUT_SECONDS}" "${DYNAMIC_QUIESCENT_SCANS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "controller integer controls must be positive: ${value}" >&2
        exit 2
    }
done
for value in "${TRUST_PINNED_SNAPSHOT}" "${REFILL_ONCE}" "${CONTROLLER_DRY_RUN}"; do
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
    awk -v label="${label}" '
        /^status:/ {
            for (field_index = 1; field_index <= NF; field_index++) {
                split($field_index, field_parts, "=")
                if (field_parts[1] == label && field_parts[2] ~ /^[0-9]+$/) {
                    print field_parts[2]
                }
            }
        }
    ' <<< "${status_output}"
}

extract_active_job_ids() {
    awk '
        /^active job ids:/ {
            sub(/^active job ids:[[:space:]]*/, "")
            print
            found = 1
            exit
        }
        END { if (!found) print "unavailable" }
    ' <<< "$1"
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

last_available_count=-1
quiescent_scan_count=0
while true; do
    scan_output="$(run_launcher)"
    printf '%s\n' "${scan_output}"
    available_count="$(extract_status_count "${scan_output}" available)"
    complete_count="$(extract_status_count "${scan_output}" complete)"
    active_count="$(extract_status_count "${scan_output}" active)"
    pending_count="$(extract_status_count "${scan_output}" pending)"
    incomplete_count="$(extract_status_count "${scan_output}" incomplete)"
    unreadable_count="$(extract_status_count "${scan_output}" unreadable)"
    active_job_ids="$(extract_active_job_ids "${scan_output}")"
    [[ -n "${available_count}" && -n "${complete_count}" \
        && -n "${active_count}" && -n "${pending_count}" \
        && -n "${incomplete_count}" && -n "${unreadable_count}" ]] || {
        echo "could not parse launcher status" >&2
        exit 4
    }
    if [[ "${active_job_ids}" == unavailable ]]; then
        echo "could not query tracked evaluation jobs; retrying" >&2
        sleep "${POLL_SECONDS}"
        continue
    fi
    if [[ -n "${SNAPSHOT_ARM_MAX_STEPS}" ]]; then
        if (( available_count != EXPECTED_CHECKPOINTS )); then
            echo "snapshot availability changed: expected=${EXPECTED_CHECKPOINTS} actual=${available_count}" >&2
            exit 5
        fi
        if (( complete_count == EXPECTED_CHECKPOINTS )); then
            echo "all ${EXPECTED_CHECKPOINTS} pinned checkpoint evaluations are complete"
            exit 0
        fi
    else
        if (( available_count > 0 && complete_count == available_count \
            && active_count == 0 && pending_count == 0 \
            && incomplete_count == 0 && unreadable_count == 0 )); then
            if (( available_count == last_available_count )); then
                ((quiescent_scan_count += 1))
            else
                quiescent_scan_count=1
            fi
            echo "all ${available_count} currently available evaluations are complete; " \
                "quiescent scan ${quiescent_scan_count}/${DYNAMIC_QUIESCENT_SCANS}"
            if (( quiescent_scan_count >= DYNAMIC_QUIESCENT_SCANS )); then
                echo "available checkpoint set remained complete and unchanged; controller is done"
                exit 0
            fi
        else
            if (( available_count > 0 && complete_count == available_count \
                && active_count == 0 && pending_count == 0 \
                && (incomplete_count > 0 || unreadable_count > 0) )); then
                echo "all readable evaluations are complete; waiting for " \
                    "incomplete=${incomplete_count} unreadable=${unreadable_count} checkpoint(s)"
            fi
            quiescent_scan_count=0
        fi
        last_available_count="${available_count}"
    fi

    if (( CONTROLLER_DRY_RUN == 1 )); then
        echo "controller dry run; stopping before queue reassignment or submission"
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
