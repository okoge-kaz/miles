#!/bin/bash
#SBATCH --job-name=miles-swe-e2b-server
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log
#SBATCH --export=NIL

# Durable one-day Harbor/E2B agent-server allocation. This leaves room for
# full-shard admission before a four-hour training allocation starts. The
# 32 CPUs provide one host core per two maximum concurrent async trajectories
# at the recipe default of 64; live throughput remains an explicit scale gate.
# Credentials come only from the process environment inherited by Slurm; this
# script never reads a dotenv file or writes credentials to argv/log/metadata.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_agent_server.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || {
    echo "MILES_SWE_FIXED_EXPORTS must be 1" >&2
    exit 2
}
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached the server allocation" >&2
    exit 2
}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
: "${E2B_API_KEY:?export E2B_API_KEY before sbatch; .env is never read}"
: "${HARBOR_ROOT:?set HARBOR_ROOT to the pinned Harbor checkout}"
: "${HARBOR_TASKS_DIR:?set HARBOR_TASKS_DIR to production-materialized tasks}"
: "${HARBOR_E2B_PREBUILD_TASK_IDS_FILE:?set an owner-only admitted task-id file}"
: "${HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS:?set colon-separated owner-only semantic admission JSONL files}"
: "${HARBOR_RUN_SECRET:?export a dedicated /run bearer secret before sbatch}"
: "${HARBOR_ADMIN_SECRET:?export a distinct admin bearer secret before sbatch}"
: "${MAX_CONCURRENT:?set MAX_CONCURRENT equal to the training ASYNC_MAX_CONCURRENT_SAMPLES}"

[[ "$HARBOR_RUN_SECRET" != "$HARBOR_ADMIN_SECRET" ]] || {
    echo "Harbor run and admin secrets must be distinct" >&2
    exit 2
}
export HARBOR_RUN_SECRET HARBOR_ADMIN_SECRET

export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export MILES_ROOT="${REPO_ROOT}"
export PORT="${PORT:-$((12000 + SLURM_JOB_ID % 20000))}"
export TRIALS_DIR="${TRIALS_DIR:-${REPO_ROOT}/outputs/validation/harbor-e2b-server-${SLURM_JOB_ID}}"

if [[ -e "${TRIALS_DIR}" ]]; then
    [[ -d "${TRIALS_DIR}" && ! -L "${TRIALS_DIR}" && \
        "$(stat -c '%u' "${TRIALS_DIR}")" == "$(id -u)" ]] || {
        echo "TRIALS_DIR is unsafe" >&2
        exit 2
    }
else
    mkdir -p -m 0700 "${TRIALS_DIR}"
fi
chmod 0700 "${TRIALS_DIR}"
slurm_log="${REPO_ROOT}/experiments/outputs/validation/miles-swe-e2b-server-${SLURM_JOB_ID}.log"
if [[ -e "${slurm_log}" ]]; then
    [[ -f "${slurm_log}" && ! -L "${slurm_log}" && \
        "$(stat -c '%h' "${slurm_log}")" == 1 && \
        "$(stat -c '%u' "${slurm_log}")" == "$(id -u)" ]] || {
        echo "Slurm output path is unsafe" >&2
        exit 2
    }
    chmod 0600 "${slurm_log}"
fi
server_host="$(hostname -f)"
printf 'AGENT_SERVER_URL=http://%s:%s\n' "${server_host}" "${PORT}"
printf 'TRIALS_DIR=%s\n' "${TRIALS_DIR}"

export MILES_ROOT="${REPO_ROOT}"
exec "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/prune_agent_server_environment.sh" \
    /bin/bash \
    "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/launch_agent_server.sh"
