#!/bin/bash

# Submit the explicit live provider lifecycle probe with a fixed environment
# allowlist. Provider credentials stay in the process environment and never
# appear as values in argv or logs.

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(realpath "${SCRIPT_DIR}/../..")}}"
source "${REPO_ROOT}/experiments/env.sh"
umask 077
: "${E2B_API_KEY:?export E2B_API_KEY; .env is never read}"
: "${E2B_LIVE_SOURCE_IMAGE:?set an admitted linux/amd64 image@sha256 digest}"
: "${HARBOR_ROOT:?set HARBOR_ROOT to the pinned Harbor checkout}"

MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="E2B_ACCESS_TOKEN,E2B_API_KEY,E2B_API_URL,E2B_DOMAIN,E2B_LIVE_SOURCE_IMAGE,E2B_SANDBOX_URL,HARBOR_PYTHON,HARBOR_ROOT,MILES_SWE_FIXED_EXPORTS"

submission="$(
    pbs_submit \
        --parsable \
        --profile=cpu \
        --chdir="${REPO_ROOT}" \
        --output="${REPO_ROOT}/experiments/outputs/validation/%x-%j.log" \
        --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/test_harbor_e2b_live.sbatch"
)"
job_id="${submission##*$'\n'}"
[[ "${job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || {
    echo "PBS returned an invalid live E2B job ID" >&2
    exit 2
}
printf 'harbor_e2b_live_job_id=%s\n' "${job_id}"
