#!/bin/bash
#SBATCH --job-name=miles-convert-ckpt
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/convert/%x-%j.log
#SBATCH --export=NIL

# Convert the HuggingFace checkpoint into the torch_dist format Megatron loads.
#
#   sbatch -A coreai_horizon_dilations experiments/setup/models/convert_checkpoint.sbatch
#
#   $HF_CKPT_DIR/$HF_MODEL_NAME  ->  $MEGATRON_CKPT_DIR/${MODEL_NAME}_torch_dist
#
# The HF directory is kept: SGLang still serves from it (--hf-checkpoint), only
# the trainer reads torch_dist (--ref-load / --load).

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="${USER:-$(id -un)}"
WANDB_MODE=disabled
export PATH USER WANDB_MODE

# sbatch copies this script into Slurm's spool directory, so BASH_SOURCE does
# not point at the repo. Resolve through the submit directory (submit from the
# repo root), falling back to the path for a direct `bash` invocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
HF_MODEL_NAME="${HF_MODEL_NAME:-${MODEL_NAME}}"
MEGATRON_MODEL_TYPE="${MEGATRON_MODEL_TYPE:-qwen3-4B}"   # scripts/models/<type>.sh
# Per-model overrides from models.txt, e.g. parallelism for large MoE or a
# fusion flag a given architecture cannot use.
CONVERT_EXTRA_ARGS="${CONVERT_EXTRA_ARGS:-}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
MASTER_PORT="${MASTER_PORT:-29517}"
echo "converting ${HF_MODEL_NAME} -> ${MODEL_NAME}_torch_dist (${MEGATRON_MODEL_TYPE}) on ${NODES} node(s); extra: ${CONVERT_EXTRA_ARGS:-none}"

[[ -f "${HF_CKPT_DIR}/${HF_MODEL_NAME}/config.json" ]] || {
    echo "missing HF checkpoint: ${HF_CKPT_DIR}/${HF_MODEL_NAME}/config.json" >&2
    exit 1
}

TRACKER="${MEGATRON_CKPT_DIR}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt"
if [[ -f "${TRACKER}" && "$(cat "${TRACKER}")" == "release" ]]; then
    echo "skip ${MODEL_NAME}: already converted (${TRACKER} == release)"
    exit 0
fi

# One task per node; torchrun then forks 8 ranks locally. Multi-node uses the
# static rendezvous so no extra service is needed.
srun --nodes="${NODES}" --ntasks-per-node=1 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc "
set -euo pipefail
cd /root/miles

# MODEL_ARGS: the Megatron-side description of the architecture.
source scripts/models/${MEGATRON_MODEL_TYPE}.sh

# /root/miles is on the path too: Qwen3.5/3.6 MODEL_ARGS reference specs that
# live in miles_plugins (--spec miles_plugins.models.qwen3_5 ...).
PYTHONPATH=/root/Megatron-LM:/root/miles torchrun \
    --nnodes ${NODES} --node-rank \${SLURM_NODEID:-0} \
    --master-addr ${MASTER_NODE} --master-port ${MASTER_PORT} \
    --nproc-per-node ${GPUS_PER_NODE} \
    tools/convert_hf_to_torch_dist.py \
    \${MODEL_ARGS[@]} \
    --hf-checkpoint /ckpt/hf/${HF_MODEL_NAME} \
    --save /ckpt/megatron/${MODEL_NAME}_torch_dist \
    ${CONVERT_EXTRA_ARGS}

cat /ckpt/megatron/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt
"
