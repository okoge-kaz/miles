#!/bin/bash

# Submit OCI locking without copying unrelated scheduler-shell credentials.

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(realpath "${SCRIPT_DIR}/../../..")}}" # submission root
source "${REPO_ROOT}/experiments/env.sh"
: "${SWE_PRIVATE_MANIFEST:?set SWE_PRIVATE_MANIFEST}"
: "${SWE_LOCKED_MANIFEST:?set SWE_LOCKED_MANIFEST}"
: "${SWE_IMAGE_LOCK_MANIFEST:?set SWE_IMAGE_LOCK_MANIFEST}"
MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="DOCKERHUB_TOKEN,DOCKERHUB_USERNAME,MILES_SWE_FIXED_EXPORTS,SWE_IMAGE_LOCK_CHECKPOINT_BATCH_SIZE,SWE_IMAGE_LOCK_CONCURRENCY,SWE_IMAGE_LOCK_INSTANCE_ID,SWE_IMAGE_LOCK_LIMIT,SWE_IMAGE_LOCK_MANIFEST,SWE_LOCKED_MANIFEST,SWE_PRIVATE_MANIFEST,SWE_REFRESH_MISSING_IMAGE_LOCKS,SWE_RESOLVE_MISSING_IMAGE_LOCKS"

submission="$(
    pbs_submit --parsable --profile cpu --chdir="${REPO_ROOT}" --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/lock_swe_oci_images.sbatch"
)"
job_id="${submission}"
[[ "${job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || exit 2
printf 'swe_oci_lock_job_id=%s\n' "${job_id}"
