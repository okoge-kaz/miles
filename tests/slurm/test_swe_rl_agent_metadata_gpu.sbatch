#!/bin/bash
#SBATCH --job-name=test-swe-agent-meta
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log
#SBATCH --export=NIL

# The production SQSH imports Transformer Engine for this integration test, so
# a CUDA-capable node is required even though the SGLang server is mocked.
# This job is offline-only and never reads a dotenv file.

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

assert torch.cuda.is_available(), "CUDA must be visible for this integration contract"
print("production CUDA available", torch.version.cuda)
PY

python3 -m pytest -q \
    tests/fast/rollout/inference_rollout/integration/test_agent_metadata.py
'

echo "GPU-dependent SWE agent metadata integration passed in production SQSH."
