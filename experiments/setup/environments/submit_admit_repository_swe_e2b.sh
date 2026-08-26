#!/bin/bash

# Submit repository semantic admission with a fixed-name environment. Secret
# values remain out of sbatch argv and unrelated submission credentials are not
# copied into the allocation.

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:-$(realpath "${SCRIPT_DIR}/../../..")}"
: "${E2B_API_KEY:?export E2B_API_KEY before submission}"
: "${HARBOR_ROOT:?set HARBOR_ROOT}"
: "${SWE_TASKSET:?set SWE_TASKSET}"

MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="DOCKERHUB_TOKEN,DOCKERHUB_USERNAME,E2B_ACCESS_TOKEN,E2B_API_KEY,E2B_API_URL,E2B_DOMAIN,E2B_SANDBOX_URL,HARBOR_PYTHON,HARBOR_ROOT,MILES_SWE_FIXED_EXPORTS,SWE_ADMISSION_CONCURRENCY,SWE_ADMISSION_INSTANCE_ID,SWE_ADMISSION_LIMIT,SWE_ADMISSION_MANIFEST,SWE_ADMISSION_WORK_ROOT,SWE_ADMITTED_MANIFEST,SWE_DATA_ROOT,SWE_IMAGE_LOCK_CONCURRENCY,SWE_IMAGE_LOCK_MANIFEST,SWE_LOCKED_MANIFEST,SWE_PRIVATE_MANIFEST,SWE_QUARANTINE_MANIFEST,SWE_REFRESH_MISSING_IMAGE_LOCKS,SWE_RESOLVE_MISSING_IMAGE_LOCKS,SWE_TASKSET"

submission="$(
    sbatch \
        --parsable \
        --chdir="${REPO_ROOT}" \
        --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/admit_repository_swe_e2b.sbatch"
)"
job_id="${submission%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Slurm returned an invalid repository-admission job ID" >&2
    exit 2
}
printf 'repository_admission_job_id=%s\n' "${job_id}"
