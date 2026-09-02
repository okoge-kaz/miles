#!/bin/bash

# Submit R2E semantic admission with only fixed approved environment names.
# E2B secrets remain in the process environment and never appear in argv.

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(realpath "${SCRIPT_DIR}/../../..")}}"
source "${REPO_ROOT}/experiments/env.sh"
: "${E2B_API_KEY:?export E2B_API_KEY before submission}"
: "${HARBOR_ROOT:?set HARBOR_ROOT}"

MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="DOCKERHUB_TOKEN,DOCKERHUB_USERNAME,E2B_ACCESS_TOKEN,E2B_API_KEY,E2B_API_URL,E2B_DOMAIN,E2B_SANDBOX_URL,HARBOR_PYTHON,HARBOR_ROOT,MILES_SWE_FIXED_EXPORTS,R2E_ADMISSION_CONCURRENCY,R2E_ADMISSION_INSTANCE_ID,R2E_ADMISSION_LIMIT,R2E_ADMISSION_MANIFEST,R2E_ADMISSION_WORK_ROOT,R2E_ADMITTED_MANIFEST,R2E_EXECUTION_LOG_PARSER,R2E_IMAGE_LOCK_CHECKPOINT_BATCH_SIZE,R2E_IMAGE_LOCK_CONCURRENCY,R2E_IMAGE_LOCK_MANIFEST,R2E_LOCKED_TASK_MANIFEST,R2E_QUARANTINE_MANIFEST,R2E_REFRESH_MISSING_IMAGE_LOCKS,R2E_RESOLVE_MISSING_IMAGE_LOCKS,SWE_R2E_PRIVATE_MANIFEST,SWE_R2E_PRIVATE_ROOT,SWE_TASKSET"

submission="$(
    pbs_submit --profile cpu \
        --parsable \
        --chdir="${REPO_ROOT}" \
        --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/admit_r2e_e2b.sbatch"
)"
job_id="${submission}"
[[ "${job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || {
    echo "PBS returned an invalid R2E-admission job ID" >&2
    exit 2
}
printf 'r2e_admission_job_id=%s\n' "${job_id}"
