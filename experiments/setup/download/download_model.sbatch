#!/bin/bash
#SBATCH --job-name=miles-download-model
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log
#SBATCH --export=NIL

# Download one HuggingFace model into $HF_CKPT_DIR/$MODEL_NAME.
#
#   sbatch -A coreai_horizon_dilations \
#     --export=USER,HF_REPO=Qwen/Qwen3-8B,MODEL_NAME=Qwen3-8B,HF_TOKEN \
#     experiments/setup/download/download_model.sbatch
#
# Idempotent: hf download verifies existing files instead of refetching them.

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="${USER:-$(id -un)}"
WANDB_MODE=disabled
export PATH USER WANDB_MODE

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${HF_REPO:?set HF_REPO, e.g. Qwen/Qwen3-8B}"
: "${MODEL_NAME:?set MODEL_NAME, e.g. Qwen3-8B}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
echo "downloading ${HF_REPO} -> /ckpt/hf/${MODEL_NAME}"
time hf download "${HF_REPO}" --local-dir "/ckpt/hf/${MODEL_NAME}"
# Marker written only on success: a directory that merely exists may be a
# partial (cancelled) download, which then fails deep inside the converter.
date -u +%FT%TZ > "/ckpt/hf/${MODEL_NAME}/.download_complete"
du -sh "/ckpt/hf/${MODEL_NAME}"
'
