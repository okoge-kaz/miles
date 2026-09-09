#!/bin/bash

#SBATCH --job-name=reason-eval-refill-local75
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

set -uo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
LAUNCHER="${REPO_ROOT}/experiments/scripts/reasoning_eval/submit-local-evaluation-remainder-75.sh"
POLL_SECONDS="${POLL_SECONDS:-60}"

[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || { echo "POLL_SECONDS must be positive" >&2; exit 2; }
[[ -x "${LAUNCHER}" ]] || { echo "launcher is not executable: ${LAUNCHER}" >&2; exit 3; }

requeue_controller() {
    echo "controller wall time is nearly exhausted; requeueing Slurm job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_controller USR1

while true; do
    if ! scan_output="$(PRINT_LIMIT=0 "${LAUNCHER}" 2>&1)"; then
        echo "manifest scan failed; retrying" >&2
        printf '%s\n' "${scan_output}" | tail -20 >&2
        sleep "${POLL_SECONDS}"
        continue
    fi
    printf '%s\n' "${scan_output}" | awk '/^entries:/ || /^routes:/ || /^dry run/'
    status_line="$(awk '/^entries:/ {print; exit}' <<< "${scan_output}")"
    complete_count="$(sed -n 's/^entries:.* complete=\([0-9][0-9]*\).*$/\1/p' <<< "${status_line}")"
    active_count="$(sed -n 's/^entries:.* active=\([0-9][0-9]*\).*$/\1/p' <<< "${status_line}")"
    pending_count="$(sed -n 's/^entries:.* pending=\([0-9][0-9]*\).*$/\1/p' <<< "${status_line}")"
    if [[ -z "${complete_count}" || -z "${active_count}" || -z "${pending_count}" ]]; then
        echo "could not parse manifest status; retrying" >&2
        sleep "${POLL_SECONDS}"
        continue
    fi
    if (( complete_count == 75 )); then
        echo "all 75 local evaluation jobs are complete"
        exit 0
    fi
    if (( pending_count > 0 )); then
        echo "trying one refill submission: complete=${complete_count} active=${active_count} pending=${pending_count}"
        if submit_output="$(PRINT_LIMIT=0 "${LAUNCHER}" --submit --max-submissions 1 2>&1)"; then
            printf '%s\n' "${submit_output}" | awk '/^submitted source=/ || /^submitted [0-9]+ checkpoint/'
        else
            printf '%s\n' "${submit_output}" | tail -8 >&2
            echo "refill was not accepted (usually the per-user submit limit); retrying" >&2
        fi
    fi
    sleep "${POLL_SECONDS}"
done
