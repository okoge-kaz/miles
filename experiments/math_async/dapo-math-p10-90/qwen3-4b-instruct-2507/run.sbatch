#!/bin/bash
#SBATCH --job-name=miles-dapo-math-p10-90-async-qwen3-4b-instruct-2507
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/math/dapo-math-p10-90/qwen3-4b-instruct-2507/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${WANDB_API_KEY:?set WANDB_API_KEY, or put an api.wandb.ai entry in ~/.netrc}"

# run identity
MODEL_NAME=Qwen3-4B-Instruct-2507
PLACEMENT=async
DATASET_TAG=dapo-math-p10-90
PROMPT_DATA=/data/dapo-math-p10-90/dapo-math-p10-90.jsonl

# rl algorithm: these derive RL_ALGORITHM in common/run_identity.sh, and every
# one of them is turned into a flag by train.sh
: "${ADVANTAGE_ESTIMATOR:=grpo}"
: "${ENTROPY_COEF:=0.00}"
: "${KL_LOSS_COEF:=0.00}"
: "${EPS_CLIP:=0.2}"
: "${EPS_CLIP_HIGH:=0.28}"
: "${EPS_CLIP_C:=}"                 # empty disables Dual-clip PPO
: "${RATIO_DENOMINATOR:=actor}"     # actor | rollout-logprobs | old-actor
: "${IS_CORRECTION:=tis}"           # none | tis | icepop | mis
: "${TIS_CLIP:=2.0}"
: "${TIS_CLIP_LOW:=0}"
: "${MIS_PROFILE:=}"                # experiments/configs/mis/<profile>.yaml
: "${M2PO_BUDGET:=0.04}"
: "${USE_OPSM:=0}"
: "${OPSM_DELTA:=1e-4}"

# training hyperparameters
: "${LR:=1e-6}"
: "${MAX_RESPONSE_LEN:=32768}"
: "${NUM_ROLLOUT:=300}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"

# batch shape: miles asserts rollout_batch * n_samples == global_batch * num_steps
: "${ROLLOUT_BATCH_SIZE:=192}"
: "${N_SAMPLES_PER_PROMPT:=16}"
: "${GLOBAL_BATCH_SIZE:=3072}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"

# queue selection and what happens to in-flight generation at a weight update
: "${QUEUE_POLICY:=queue-recycle}" # queue-recycle | queue-max | queue-drop
: "${QUEUE_FACTOR:=1}"                       # queue-drop capacity in training batches
if [[ "${QUEUE_POLICY}" == queue-drop ]]; then
   [[ -z "${MAX_WEIGHT_STALENESS:-}" ]] || {
      echo "QUEUE_POLICY=queue-drop does not use MAX_WEIGHT_STALENESS" >&2
      exit 1
   }
   MAX_WEIGHT_STALENESS=
else
   : "${MAX_WEIGHT_STALENESS:=2}"
fi
: "${STALENESS_REFERENCE:=prefill}"   # completion | submission | prefill
: "${PAUSE_GENERATION_MODE:=in_place}"
: "${ASYNC_MAX_CONCURRENT_SAMPLES:=}"
: "${FULLY_ASYNC_ROLLOUT_CHECKPOINT:=0}"  # opt-in; older checkpoints have no replay sidecar

# distributed setting
: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_GPUS_PER_NODE:=8}"
: "${ROLLOUT_NUM_GPUS:=8}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=1}"

# throughput / memory
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${RECOMPUTE_GRANULARITY:=full}"
: "${OVERLAP_COMM:=0}"
: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=1}"
: "${SGLANG_MEM_FRACTION:=0.70}"

# reward
: "${RM_TYPE:=math}"
: "${CUSTOM_RM_PATH:=}"

# evaluation during training; the reported numbers come from src/offline_eval
: "${EVAL_INTERVAL:=0}"       # 0 disables in-run eval; score HF exports offline
: "${N_SAMPLES_PER_EVAL_PROMPT:=8}"
: "${EVAL_MAX_RESPONSE_LEN:=32768}"
: "${SKIP_EVAL_BEFORE_TRAIN:=0}"

# checkpointing: torch_dist for resume, hf for offline eval, on separate cadences
: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=100}"
: "${SAVE_HF:=1}"
: "${HF_SAVE_INTERVAL:=5}"

# diagnostics
: "${DUMP_TRAIN_DATA:=0}"
: "${DUMP_POLICY_LOSS_DEBUG:=0}"
: "${FUSE_ONE_STEP_ACTOR_LOGPROBS:=1}"
: "${VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS:=0}"  # debug only; retains legacy forward
: "${LOG_STALENESS_GRADIENT_METRICS:=0}"
: "${LOG_STALENESS_GRADIENT_RATIO_HISTOGRAM:=0}"  # high-cardinality, debug only
: "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS:=0}"  # exact token boundaries, opt-in
: "${DEBUG_EXIT_AFTER_ROLLOUT:=}"               # validation only; empty runs the full schedule

export MODEL_NAME PLACEMENT DATASET_TAG PROMPT_DATA
export ADVANTAGE_ESTIMATOR ENTROPY_COEF KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
export RATIO_DENOMINATOR IS_CORRECTION TIS_CLIP TIS_CLIP_LOW MIS_PROFILE M2PO_BUDGET USE_OPSM OPSM_DELTA
export LR MAX_RESPONSE_LEN NUM_ROLLOUT TRAIN_SEED ROLLOUT_SEED
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_STEPS_PER_ROLLOUT
export QUEUE_POLICY QUEUE_FACTOR MAX_WEIGHT_STALENESS STALENESS_REFERENCE
export PAUSE_GENERATION_MODE ASYNC_MAX_CONCURRENT_SAMPLES
export FULLY_ASYNC_ROLLOUT_CHECKPOINT
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
export TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE
export MAX_TOKENS_PER_GPU RECOMPUTE_GRANULARITY OVERLAP_COMM
export ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION
export RM_TYPE CUSTOM_RM_PATH
export EVAL_INTERVAL N_SAMPLES_PER_EVAL_PROMPT EVAL_MAX_RESPONSE_LEN SKIP_EVAL_BEFORE_TRAIN
export SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL
export DUMP_TRAIN_DATA DUMP_POLICY_LOSS_DEBUG FUSE_ONE_STEP_ACTOR_LOGPROBS VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS
export LOG_STALENESS_GRADIENT_METRICS LOG_STALENESS_GRADIENT_RATIO_HISTOGRAM
export SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS
export DEBUG_EXIT_AFTER_ROLLOUT

source "${REPO_ROOT}/experiments/common/run_identity.sh"
export RUN_NAME CONFIG_TAG CKPT_PATH

source "${REPO_ROOT}/experiments/common/placement.sh"

RAY_HEAD_IP="$(getent hosts "$(hostname)" | awk '{print $1; exit}')"
if [[ -z "${RAY_HEAD_IP}" || "${RAY_HEAD_IP}" == 169.254.* ]]; then
    RAY_HEAD_IP="$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')"
fi
if [[ -z "${RAY_HEAD_IP}" || "${RAY_HEAD_IP}" == 169.254.* ]]; then
    echo "no routable address for the ray head on $(hostname): got '${RAY_HEAD_IP}'" >&2
    exit 1
fi
export RAY_HEAD_IP
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
    bash /root/miles/experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/train.sh
