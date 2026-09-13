#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=lightning-dapo-math
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/lightning-setup/%x-%j.log

set -euo pipefail
REPO_ROOT="${MILES_REPO:-${SLURM_SUBMIT_DIR}}"
# The Step4000 HF root in .env is specific to Qwen; Lightning uses the HF parent.
export HF_CKPT_DIR="${HF_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/${USER}/checkpoints/hf}"
source "${REPO_ROOT}/experiments/env.sh"

: "${MODEL_NAME:=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
: "${HF_MODEL_NAME:=${MODEL_NAME}-policy}"
: "${ROLLOUT_BATCH_SIZE:=32}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"
: "${GLOBAL_BATCH_SIZE:=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))}"
# The initial unfiltered DAPO smoke kept 4 of 22 completed groups (~18%).
# Start with enough candidates for that acceptance rate; the sampler replenishes.
: "${OVER_SAMPLING_BATCH_SIZE:=$((6 * ROLLOUT_BATCH_SIZE))}"
: "${DYNAMIC_SAMPLING_FILTER_PATH:=miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter}"
: "${NUM_ROLLOUT:=100}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_CONTEXT_LEN:=18432}"
: "${LR:=1e-6}"
: "${ACTOR_NUM_NODES:=2}"
: "${ACTOR_GPUS_PER_NODE:=${GPUS_PER_NODE}}"
: "${ROLLOUT_NUM_GPUS:=0}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=4}"
: "${MAX_TOKENS_PER_GPU:=18432}"
: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=1}"
: "${SGLANG_MEM_FRACTION:=0.60}"
: "${SGLANG_MAX_RUNNING_REQUESTS:=32}"
: "${SGLANG_CUDA_GRAPH_MAX_BS:=32}"
: "${SGLANG_MOE_RUNNER_BACKEND:=triton}"
: "${SAVE_INTERVAL:=5}"
: "${SAVE_RETAIN_INTERVAL:=5}"
: "${SAVE_HF:=1}"
: "${HF_SAVE_INTERVAL:=5}"
: "${EVAL_INTERVAL:=5}"
: "${EVAL_MAX_RESPONSE_LEN:=${MAX_RESPONSE_LEN}}"
: "${N_SAMPLES_PER_EVAL_PROMPT:=1}"
: "${SKIP_EVAL_BEFORE_TRAIN:=0}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"
: "${CONFIG_TAG:=grpo-lr${LR}-b${ROLLOUT_BATCH_SIZE}n${N_SAMPLES_PER_PROMPT}-o${OVER_SAMPLING_BATCH_SIZE}-df-tp${TENSOR_PARALLEL_SIZE}ep${EXPERT_PARALLEL_SIZE}-${SLURM_JOB_ID}}"
: "${RUN_NAME:=lightning-${CONFIG_TAG}}"
: "${CKPT_PATH:=/ckpt/training/math/dapo-math-17k/${MODEL_NAME}/${CONFIG_TAG}}"
: "${PROMPT_DATA:=/data/dapo-math-17k/dapo-math-17k.jsonl}"
: "${EVAL_PROMPT_DATA:=/data/aime-2024/aime-2024.jsonl}"

(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT == GLOBAL_BATCH_SIZE * NUM_STEPS_PER_ROLLOUT ))
(( N_SAMPLES_PER_PROMPT > 1 && OVER_SAMPLING_BATCH_SIZE >= ROLLOUT_BATCH_SIZE ))
(( MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE >= MAX_CONTEXT_LEN ))
(( MAX_CONTEXT_LEN > MAX_RESPONSE_LEN ))
(( SAVE_INTERVAL > 0 && SAVE_RETAIN_INTERVAL > 0 && SAVE_HF == 1 && HF_SAVE_INTERVAL > 0 ))
[[ -f "${HF_CKPT_DIR}/${HF_MODEL_NAME}/.download_complete" ]]
[[ -f "${MEGATRON_CKPT_DIR}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt" ]]
PLACEMENT=colocated
source "${REPO_ROOT}/experiments/common/placement.sh"

export MODEL_NAME HF_MODEL_NAME ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_STEPS_PER_ROLLOUT
export OVER_SAMPLING_BATCH_SIZE DYNAMIC_SAMPLING_FILTER_PATH NUM_ROLLOUT MAX_RESPONSE_LEN MAX_CONTEXT_LEN LR
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE
export MAX_TOKENS_PER_GPU ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION SGLANG_MAX_RUNNING_REQUESTS SGLANG_CUDA_GRAPH_MAX_BS
export SGLANG_MOE_RUNNER_BACKEND
export SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL EVAL_INTERVAL EVAL_MAX_RESPONSE_LEN
export N_SAMPLES_PER_EVAL_PROMPT SKIP_EVAL_BEFORE_TRAIN TRAIN_SEED ROLLOUT_SEED CONFIG_TAG RUN_NAME CKPT_PATH PROMPT_DATA EVAL_PROMPT_DATA

RAY_HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
RAY_HEAD_IP=$(getent ahostsv4 "${RAY_HEAD_NODE}" | awk 'NR==1 {print $1}')
[[ -n "${RAY_HEAD_IP}" && "${RAY_HEAD_IP}" != 169.254.* ]]
RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
export RAY_HEAD_IP RAY_DONE_FLAG
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
echo "run ${RUN_NAME}; checkpoints: ${CKPT_PATH}"
echo "dynamic filter: ${DYNAMIC_SAMPLING_FILTER_PATH}; target groups=${ROLLOUT_BATCH_SIZE}; sampling groups=${OVER_SAMPLING_BATCH_SIZE}"
echo "checkpoint cadence: torch_dist=${SAVE_INTERVAL}, retain=${SAVE_RETAIN_INTERVAL}; HF=${HF_SAVE_INTERVAL}"
srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
    --container-image="${SQSH_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    --container-name="lightning-train-${SLURM_JOB_ID}" \
    --container-writable --no-container-mount-home --export=ALL \
    bash /root/miles/experiments/scripts/math/sync/dapo-math-17k/nemotron-3.5-lightning/train.sh
