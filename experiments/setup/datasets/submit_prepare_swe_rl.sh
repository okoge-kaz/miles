#!/bin/bash

# Submit SWE normalization with only the public source selector in scope.

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:-$(realpath "${SCRIPT_DIR}/../../..")}" # repository checkout
: "${SWE_SOURCE:?set SWE_SOURCE to a supported prepared-source selector}"

MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="MILES_SWE_FIXED_EXPORTS,SWE_SOURCE"

submission="$(
    sbatch \
        --parsable \
        --chdir="${REPO_ROOT}" \
        --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/prepare_swe_rl.sbatch"
)"
job_id="${submission%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Slurm returned an invalid SWE preparation job ID" >&2
    exit 2
}
printf 'swe_prepare_job_id=%s\n' "${job_id}"
