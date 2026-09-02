#!/bin/bash
#PBS -N test-swe-agent-meta
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=00:30:00
#PBS -j oe

# The production SIF imports Transformer Engine for this integration test, so
# a CUDA-capable node is required even though the SGLang server is mocked.
# This job is offline-only and never reads a dotenv file.

set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
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

[[ -r "${CONTAINER_IMAGE}" ]] || {
    echo "missing production Miles CONTAINER_IMAGE" >&2
    exit 1
}

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${SWE_CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
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

echo "GPU-dependent SWE agent metadata integration passed in production SIF."
