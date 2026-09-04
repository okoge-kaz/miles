#!/bin/bash
#SBATCH --job-name=wait-miles-swe-server
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --export=NIL
#SBATCH --output=experiments/outputs/validation/%x-%j.log

# CPU-only dependency gate for the production GPU recipe. It never reads a
# dotenv file and receives only the agent-server URL and run-secret names.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
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
