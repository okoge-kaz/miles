#!/bin/bash

#SBATCH --job-name=reason-eval-refill
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=experiments/outputs/reasoning_eval/setup/%x-%j.log

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
LAUNCHER="${REPO_ROOT}/experiments/scripts/reasoning_eval/submit-staleness-sweep.sh"

: "${TRAINING_ROOT:?set TRAINING_ROOT to the checkpoint root}"
: "${RUN_NAMESPACE:?set RUN_NAMESPACE to the training namespace}"

SNAPSHOT_ARM_MAX_STEPS="${SNAPSHOT_ARM_MAX_STEPS:-}"
EXPECTED_CHECKPOINTS="${EXPECTED_CHECKPOINTS:-510}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DYNAMIC_QUIESCENT_SCANS="${DYNAMIC_QUIESCENT_SCANS:-12}"
CONTROLLER_DRY_RUN="${CONTROLLER_DRY_RUN:-0}"
EVAL_QOS="${EVAL_QOS:-normal}"
BATCH_PARTITION="${BATCH_PARTITION:-batch_short}"
BATCH_WALL="${BATCH_WALL:-02:00:00}"
BATCH_INFLIGHT_TARGET="${BATCH_INFLIGHT_TARGET:-4}"
REGULAR_PARTITION="${REGULAR_PARTITION:-batch}"
REGULAR_WALL="${REGULAR_WALL:-02:30:00}"
REGULAR_INFLIGHT_TARGET="${REGULAR_INFLIGHT_TARGET:-130}"
INTERACTIVE_PARTITION="${INTERACTIVE_PARTITION:-interactive}"
INTERACTIVE_WALL="${INTERACTIVE_WALL:-04:00:00}"
INTERACTIVE_INFLIGHT_TARGET="${INTERACTIVE_INFLIGHT_TARGET:-4}"
SQUEUE_TIMEOUT_SECONDS="${SQUEUE_TIMEOUT_SECONDS:-10}"
PROMOTION_WAIT_ATTEMPTS="${PROMOTION_WAIT_ATTEMPTS:-15}"
PROMOTION_WAIT_SECONDS="${PROMOTION_WAIT_SECONDS:-2}"
if [[ -n "${SNAPSHOT_ARM_MAX_STEPS}" ]]; then
    default_trust_pinned_snapshot=1
else
    default_trust_pinned_snapshot=0
fi
TRUST_PINNED_SNAPSHOT="${TRUST_PINNED_SNAPSHOT:-${default_trust_pinned_snapshot}}"

for integer_value in \
    "${EXPECTED_CHECKPOINTS}" "${POLL_SECONDS}" "${DYNAMIC_QUIESCENT_SCANS}" \
    "${SQUEUE_TIMEOUT_SECONDS}" \
    "${PROMOTION_WAIT_ATTEMPTS}" "${PROMOTION_WAIT_SECONDS}"; do
    [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "controller integer controls must be positive: ${integer_value}" >&2
        exit 2
    }
done
for integer_value in \
    "${BATCH_INFLIGHT_TARGET}" "${REGULAR_INFLIGHT_TARGET}" \
    "${INTERACTIVE_INFLIGHT_TARGET}" "${TRUST_PINNED_SNAPSHOT}" \
    "${CONTROLLER_DRY_RUN}"; do
    [[ "${integer_value}" =~ ^[0-9]+$ ]] || {
        echo "controller queue targets must be nonnegative: ${integer_value}" >&2
        exit 2
    }
done
(( TRUST_PINNED_SNAPSHOT <= 1 )) || {
    echo "TRUST_PINNED_SNAPSHOT must be 0 or 1" >&2
    exit 2
}
(( CONTROLLER_DRY_RUN <= 1 )) || {
    echo "CONTROLLER_DRY_RUN must be 0 or 1" >&2
    exit 2
}
if (( TRUST_PINNED_SNAPSHOT == 1 )) && [[ -z "${SNAPSHOT_ARM_MAX_STEPS}" ]]; then
    echo "TRUST_PINNED_SNAPSHOT=1 requires SNAPSHOT_ARM_MAX_STEPS" >&2
    exit 2
fi
[[ "${INTERACTIVE_PARTITION}" != "${BATCH_PARTITION}" \
    && "${INTERACTIVE_PARTITION}" != "${REGULAR_PARTITION}" \
    && "${BATCH_PARTITION}" != "${REGULAR_PARTITION}" ]] || {
    echo "interactive, short-batch, and regular-batch partitions must be distinct" >&2
    exit 2
}
[[ -x "${LAUNCHER}" ]] || { echo "launcher is not executable: ${LAUNCHER}" >&2; exit 3; }

requeue_controller() {
    echo "controller wall time is nearly exhausted; requeueing Slurm job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_controller USR1

run_launcher() {
    local partition="$1"
    local wall="$2"
    shift 2
    TRAINING_ROOT="${TRAINING_ROOT}" \
    RUN_NAMESPACE="${RUN_NAMESPACE}" \
    SNAPSHOT_ARM_MAX_STEPS="${SNAPSHOT_ARM_MAX_STEPS}" \
    TRUST_PINNED_SNAPSHOT="${TRUST_PINNED_SNAPSHOT}" \
    PARTITION="${partition}" \
    QOS="${EVAL_QOS}" \
    WALL="${wall}" \
    PRINT_LIMIT=0 \
    SQUEUE_TIMEOUT_SECONDS="${SQUEUE_TIMEOUT_SECONDS}" \
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

cancel_pending_job_for_reassignment() {
    local job_id="$1"
    local destination="$2"
    local attempt state

    # The state filter prevents a race from canceling a job that started after
    # the queue snapshot. Running evaluations are never migrated.
    scancel --state=PENDING "${job_id}"
    for ((attempt = 1; attempt <= PROMOTION_WAIT_ATTEMPTS; attempt++)); do
        state="$(
            timeout "${SQUEUE_TIMEOUT_SECONDS}" \
                squeue --noheader --jobs="${job_id}" --format='%T'
        )" || return 1
        if [[ -z "${state}" ]]; then
            echo "released pending job ${job_id} for ${destination}"
            return 0
        fi
        if [[ "${state}" != PENDING ]]; then
            echo "job ${job_id} became ${state}; leaving it untouched" >&2
            return 1
        fi
        sleep "${PROMOTION_WAIT_SECONDS}"
    done
    echo "pending job ${job_id} did not leave the queue after cancellation" >&2
    return 1
}

last_available_count=-1
quiescent_scan_count=0
while true; do
    scan_output="$(run_launcher "${BATCH_PARTITION}" "${BATCH_WALL}")"
    printf '%s\n' "${scan_output}"
    available_count="$(extract_status_count "${scan_output}" available)"
    complete_count="$(extract_status_count "${scan_output}" complete)"
    active_count="$(extract_status_count "${scan_output}" active)"
    pending_count="$(extract_status_count "${scan_output}" pending)"
    [[ -n "${available_count}" && -n "${complete_count}" \
        && -n "${active_count}" && -n "${pending_count}" ]] || {
        echo "could not parse launcher status" >&2
        exit 4
    }
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
            && active_count == 0 && pending_count == 0 )); then
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
            quiescent_scan_count=0
        fi
        last_available_count="${available_count}"
    fi

    if (( CONTROLLER_DRY_RUN == 1 )); then
        echo "controller dry run; stopping before queue reassignment or submission"
        exit 0
    fi

    if ! queue_snapshot="$(
        timeout "${SQUEUE_TIMEOUT_SECONDS}" \
            squeue --noheader --user="${SLURM_JOB_USER}" --format='%i|%P|%T|%j'
    )"; then
        echo "could not query current partition job counts; retrying" >&2
        sleep "${POLL_SECONDS}"
        continue
    fi
    read -r interactive_job_count batch_job_count regular_job_count < <(
        awk -F '|' \
            -v interactive="${INTERACTIVE_PARTITION}" \
            -v batch="${BATCH_PARTITION}" \
            -v regular="${REGULAR_PARTITION}" '
            $2 == interactive && $4 ~ /^q3e-/ { interactive_count += 1 }
            $2 == batch && $4 ~ /^q3e-/ { batch_count += 1 }
            $2 == regular && $4 ~ /^q3e-/ { regular_count += 1 }
            END { print interactive_count + 0, batch_count + 0, regular_count + 0 }
        ' <<< "${queue_snapshot}"
    )

    batch_excess=$((batch_job_count - BATCH_INFLIGHT_TARGET))
    if (( batch_excess > 0 )); then
        mapfile -t short_batch_pending_job_ids < <(
            awk -F '|' \
                -v batch="${BATCH_PARTITION}" '
                $2 == batch && $3 == "PENDING" && $4 ~ /^q3e-/ { print $1 }
            ' <<< "${queue_snapshot}"
        )
        reassigned_count=0
        for short_batch_job_id in "${short_batch_pending_job_ids[@]}"; do
            (( reassigned_count < batch_excess )) || break
            if cancel_pending_job_for_reassignment \
                "${short_batch_job_id}" "${REGULAR_PARTITION} overflow"; then
                ((reassigned_count += 1))
            fi
        done
        if (( reassigned_count > 0 )); then
            batch_job_count=$((batch_job_count - reassigned_count))
            pending_count=$((pending_count + reassigned_count))
            echo "released ${reassigned_count} excess ${BATCH_PARTITION} job(s)"
            if ! queue_snapshot="$(
                timeout "${SQUEUE_TIMEOUT_SECONDS}" \
                    squeue --noheader --user="${SLURM_JOB_USER}" --format='%i|%P|%T|%j'
            )"; then
                echo "could not refresh the queue after reassignment; retrying" >&2
                sleep "${POLL_SECONDS}"
                continue
            fi
        fi
    fi

    interactive_needed=$((INTERACTIVE_INFLIGHT_TARGET - interactive_job_count))
    promotion_needed=$((interactive_needed - pending_count))
    if (( promotion_needed > 0 )); then
        # Prefer regular batch overflow, then batch_short. Preserve work that
        # has already started, and let the launcher choose the released task.
        mapfile -t promotion_job_ids < <(
            awk -F '|' \
                -v regular="${REGULAR_PARTITION}" '
                $2 == regular && $3 == "PENDING" && $4 ~ /^q3e-/ { print $1 }
            ' <<< "${queue_snapshot}"
            awk -F '|' \
                -v batch="${BATCH_PARTITION}" '
                $2 == batch && $3 == "PENDING" && $4 ~ /^q3e-/ { print $1 }
            ' <<< "${queue_snapshot}"
        )
        promoted_count=0
        for promotion_job_id in "${promotion_job_ids[@]}"; do
            (( promoted_count < promotion_needed )) || break
            if cancel_pending_job_for_reassignment \
                "${promotion_job_id}" "${INTERACTIVE_PARTITION} promotion"; then
                ((promoted_count += 1))
            fi
        done
        if (( promoted_count > 0 )); then
            pending_count=$((pending_count + promoted_count))
            echo "made ${promoted_count} checkpoint(s) available for interactive submission"
        elif (( interactive_needed > 0 )); then
            echo "no queued batch evaluation can currently be promoted to interactive" >&2
        fi
    fi
    if (( pending_count > 0 && interactive_needed > 0 )); then
        interactive_limit="${interactive_needed}"
        (( interactive_limit <= pending_count )) || interactive_limit="${pending_count}"
        echo "submitting up to ${interactive_limit} snapshot job(s) to ${INTERACTIVE_PARTITION}/${INTERACTIVE_WALL}"
        if run_launcher \
            "${INTERACTIVE_PARTITION}" "${INTERACTIVE_WALL}" \
            --submit --max-submissions "${interactive_limit}"; then
            pending_count=$((pending_count - interactive_limit))
        else
            echo "interactive submission was not fully accepted; retrying after the poll interval" >&2
            sleep "${POLL_SECONDS}"
            continue
        fi
    fi

    batch_needed=$((BATCH_INFLIGHT_TARGET - batch_job_count))
    if (( pending_count > 0 && batch_needed > 0 )); then
        submission_limit="${batch_needed}"
        (( submission_limit <= pending_count )) || submission_limit="${pending_count}"
        echo "submitting up to ${submission_limit} snapshot job(s) to ${BATCH_PARTITION}/${BATCH_WALL}"
        if run_launcher \
            "${BATCH_PARTITION}" "${BATCH_WALL}" \
            --submit --max-submissions "${submission_limit}"; then
            pending_count=$((pending_count - submission_limit))
        else
            echo "submission attempt was not fully accepted; retrying after the poll interval" >&2
            sleep "${POLL_SECONDS}"
            continue
        fi
    fi

    regular_needed=$((REGULAR_INFLIGHT_TARGET - regular_job_count))
    if (( pending_count > 0 && regular_needed > 0 )); then
        regular_limit="${regular_needed}"
        (( regular_limit <= pending_count )) || regular_limit="${pending_count}"
        echo "submitting up to ${regular_limit} snapshot job(s) to ${REGULAR_PARTITION}/${REGULAR_WALL}"
        if ! run_launcher \
            "${REGULAR_PARTITION}" "${REGULAR_WALL}" \
            --submit --max-submissions "${regular_limit}"; then
            echo "regular batch submission was not fully accepted; retrying after the poll interval" >&2
        fi
    fi
    sleep "${POLL_SECONDS}"
done
