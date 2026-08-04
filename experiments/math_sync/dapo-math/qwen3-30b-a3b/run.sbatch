#!/bin/bash
#SBATCH --job-name=miles-dapo-math-qwen3-30b-a3b
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${WANDB_API_KEY:?set WANDB_API_KEY, or put an api.wandb.ai entry in ~/.netrc}"

MODEL_NAME=Qwen3-30B-A3B
ROLLOUT_MODE=async-off
DATASET_TAG=dapo-math
: "${PROMPT_DATA:=/data/dapo-math-17k/dapo-math-17k.jsonl}"

: "${LR:=1e-6}"
: "${MAX_RESPONSE_LEN:=24576}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"

: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_GPUS_PER_NODE:=8}"
: "${ROLLOUT_NUM_GPUS:=0}"

: "${TENSOR_PARALLEL_SIZE:=4}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=8}"
: "${MAX_TOKENS_PER_GPU:=32768}"

: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=8}"
: "${SGLANG_MEM_FRACTION:=0.7}"

: "${ROLLOUT_BATCH_SIZE:=32}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${GLOBAL_BATCH_SIZE:=256}"
: "${NUM_ROLLOUT:=3000}"
: "${OVER_SAMPLING_BATCH_SIZE:=$(( ROLLOUT_BATCH_SIZE * 2 ))}"
: "${RM_TYPE:=deepscaler}"

: "${DUMP_TRAIN_DATA:=0}"

export MODEL_NAME ROLLOUT_MODE DATASET_TAG PROMPT_DATA LR MAX_RESPONSE_LEN NUM_STEPS_PER_ROLLOUT
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
export TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE MAX_TOKENS_PER_GPU
export ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_ROLLOUT
export OVER_SAMPLING_BATCH_SIZE RM_TYPE
export DUMP_TRAIN_DATA

source "${REPO_ROOT}/experiments/common/run_identity.sh"
export RUN_NAME CONFIG_TAG CKPT_PATH

source "${REPO_ROOT}/experiments/common/placement.sh"

export RAY_HEAD_IP="$(hostname -I | awk '{print $1}')"
export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"

echo "run ${RUN_NAME}"
echo "checkpoints ${CKPT_PATH}"

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/math_sync/dapo-math/qwen3-30b-a3b/train.sh
