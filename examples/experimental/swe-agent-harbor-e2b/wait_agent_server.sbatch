#!/bin/bash
#PBS -N wait-miles-swe-server
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -j oe

# CPU-only dependency gate for the production GPU recipe. It never reads a
# dotenv file and receives only the agent-server URL and run-secret names.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)}}"
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_when_ready.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || {
    echo "MILES_SWE_FIXED_EXPORTS must be 1" >&2
    exit 2
}
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached the readiness allocation" >&2
    exit 2
}
: "${AGENT_SERVER_URL:?set the private Harbor agent-server origin}"
: "${HARBOR_RUN_SECRET:?export the job-scoped Harbor run secret}"
: "${SWE_HOST_ADMISSION_SUMMARY:?set the host-side admitted dataset summary path}"
HARBOR_READINESS_TIMEOUT_SEC="${HARBOR_READINESS_TIMEOUT_SEC:-13800}"
[[ "${HARBOR_READINESS_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] && \
    (( HARBOR_READINESS_TIMEOUT_SEC <= 13800 )) || {
    echo "HARBOR_READINESS_TIMEOUT_SEC must be in [1, 13800]" >&2
    exit 2
}

export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
unset BASH_ENV ENV
PYTHONPATH="${REPO_ROOT}" \
    python3 \
    "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/wait_for_agent_server.py" \
    --server-url "${AGENT_SERVER_URL}" \
    --admission-summary "${SWE_HOST_ADMISSION_SUMMARY}" \
    --timeout-seconds "${HARBOR_READINESS_TIMEOUT_SEC}"
