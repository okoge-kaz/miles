#!/bin/bash
#PBS -N test-swe-ray-secret
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

# Production-container contract for fixed-name PBS -> Ray environment
# inheritance. The sentinel is generated inside the allocation, never supplied
# on argv or printed, and no dotenv file is read.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

REPO_ROOT="${PBS_O_WORKDIR:?submit this test from the repository root}"
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
source "${REPO_ROOT}/experiments/env.sh"
umask 077

DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || {
    echo "tracked dotenv mask is missing or unsafe" >&2
    exit 2
}
SWE_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"

HARBOR_RUN_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export HARBOR_RUN_SECRET
HARBOR_CLIENT_ID="ray-client-${MILES_JOB_ID}"
export HARBOR_CLIENT_ID
export RAY_DASHBOARD_HOST=127.0.0.1
export RAY_TEST_ROOT="/tmp/miles-ray-secret-${MILES_JOB_ID}"

miles_srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --image="${CONTAINER_IMAGE}" \
    --bind="${SWE_CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=HARBOR_CLIENT_ID,HARBOR_RUN_SECRET,PYTHON_DOTENV_DISABLED,RAY_DASHBOARD_HOST,RAY_TEST_ROOT,WANDB_DISABLED,WANDB_MODE \
    env -u BASH_ENV -u ENV bash --noprofile --norc \
    /root/miles/tests/slurm/support/check_swe_ray_secret_inheritance.sh
