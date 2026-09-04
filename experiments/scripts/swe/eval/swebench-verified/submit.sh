#!/bin/bash

# Submit hardened-local SWE evaluation with a fixed-name environment.

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:-$(realpath "${SCRIPT_DIR}/../../../../..")}"
: "${HARBOR_SWE_SERVER_URL:?set the private Harbor server URL}"
: "${POLICY_BASE_URL:?set the policy service URL}"
: "${SERVED_MODEL:?set the served model name}"
: "${HARBOR_RUN_SECRET:?export the job-scoped Harbor run secret}"

MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="HARBOR_RUN_SECRET,HARBOR_SWE_SERVER_URL,MILES_SWE_FIXED_EXPORTS,POLICY_BASE_URL,SERVED_MODEL,SWE_EVAL_CONCURRENCY,SWE_EVAL_LIMIT,SWE_EVAL_OUTPUT_ROOT,SWE_EVAL_TASK_ROWS,SWE_EVAL_TASK_SUMMARY"

submission="$(
    sbatch \
        --parsable \
        --chdir="${REPO_ROOT}" \
        --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/run.sbatch"
)"
job_id="${submission%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Slurm returned an invalid SWE evaluation job ID" >&2
    exit 2
}
printf 'swe_evaluation_job_id=%s\n' "${job_id}"
