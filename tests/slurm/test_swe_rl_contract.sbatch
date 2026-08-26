#!/bin/bash
#SBATCH --job-name=test-swe-rl-contract
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log
#SBATCH --export=NIL

# Run the SWE rollout/result boundary in the production Miles SQSH, where
# torch and the complete agentic rollout dependency set are installed. This
# job is offline-only and never reads a dotenv file.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
source "${REPO_ROOT}/experiments/env.sh"
umask 077

DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || {
    echo "tracked dotenv mask is missing or unsafe" >&2
    exit 2
}
SWE_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"

[[ -r "${SQSH_IMAGE}" ]] || {
    echo "missing production Miles SQSH_IMAGE" >&2
    exit 1
}

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${SWE_CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=PYTHON_DOTENV_DISABLED,PYTHONDONTWRITEBYTECODE,WANDB_DISABLED,WANDB_MODE \
    bash --noprofile --norc -c '
set -euo pipefail
umask 077
unset BASH_ENV ENV
cd /root/miles
export PYTHONPATH=/root/miles
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1

python3 - <<"PY"
import torch

print("production torch available", torch.__version__)
PY

python3 -m pytest -q \
    tests/fast/rollout/test_swe_agent_function.py \
    tests/fast/rollout/generate_hub/test_multi_turn.py \
    tests/fast/router/test_session_samples_op.py
'

echo "CPU-safe SWE RL client/result and agentic rollout contract passed in production SQSH."
