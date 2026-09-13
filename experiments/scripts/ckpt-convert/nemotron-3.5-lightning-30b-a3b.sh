#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=convert-lightning
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/lightning-setup/%x-%j.log

set -euo pipefail
REPO_ROOT="${MILES_REPO:-${SLURM_SUBMIT_DIR}}"
export HF_CKPT_DIR="${HF_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/${USER}/checkpoints/hf}"
source "${REPO_ROOT}/experiments/env.sh"
: "${MODEL_NAME:=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
: "${HF_MODEL_NAME:=${MODEL_NAME}-policy}"
export MODEL_NAME HF_MODEL_NAME
[[ -f "${HF_CKPT_DIR}/${HF_MODEL_NAME}/.download_complete" ]]
TRACKER="${MEGATRON_CKPT_DIR}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt"
if [[ -f "${TRACKER}" ]]; then
    [[ "$(cat "${TRACKER}")" == release ]]
    echo "Already converted: ${TRACKER}"
    exit 0
fi
# One GB200 holds the BF16 model without optimizer state. Save a release
# checkpoint that the training backend can reshard to its TP/EP layout.
srun --nodes=1 --ntasks=1 --container-image="${SQSH_IMAGE}" \
    --container-name="lightning-convert-${SLURM_JOB_ID}" \
    --container-mounts="${CONTAINER_MOUNTS}" --container-writable \
    --no-container-mount-home --export=ALL bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles:/root/Megatron-LM
export HF_HOME=/root/.cache/huggingface
export CUDA_DEVICE_MAX_CONNECTIONS=1
read -r -a MODEL_ARGS <<< "$(python -m miles.utils.external_utils.model_args_utils nemotron-3-nano-30b-a3b)"
torchrun --standalone --nproc-per-node=1 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" --megatron-to-hf-mode bridge --moe-token-dispatcher-type alltoall \
    --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}" \
    --save "/ckpt/megatron/${MODEL_NAME}_torch_dist"
'
