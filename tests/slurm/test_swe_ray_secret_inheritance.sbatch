#!/bin/bash
#SBATCH --job-name=test-swe-ray-secret
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log
#SBATCH --export=NIL

# Production-container contract for fixed-name Slurm -> Ray environment
# inheritance. The sentinel is generated inside the allocation, never supplied
# on argv or printed, and no dotenv file is read.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit this test from the repository root}"
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
source "${REPO_ROOT}/experiments/env.sh"

DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || {
    echo "tracked dotenv mask is missing or unsafe" >&2
    exit 2
}
SWE_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"

HARBOR_RUN_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export HARBOR_RUN_SECRET
HARBOR_CLIENT_ID="ray-client-${SLURM_JOB_ID}"
export HARBOR_CLIENT_ID
export RAY_DASHBOARD_HOST=127.0.0.1
export RAY_TEST_ROOT="/tmp/miles-ray-secret-${SLURM_JOB_ID}"

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${SWE_CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=HARBOR_CLIENT_ID,HARBOR_RUN_SECRET,PYTHON_DOTENV_DISABLED,RAY_DASHBOARD_HOST,RAY_TEST_ROOT,WANDB_DISABLED,WANDB_MODE \
    env -u BASH_ENV -u ENV bash --noprofile --norc \
    /root/miles/tests/slurm/support/check_swe_ray_secret_inheritance.sh
