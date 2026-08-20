#!/bin/bash
#SBATCH --job-name=miles-dapo-math-p10-90-qwen3-4b
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/math/dapo-math-p10-90/qwen3-4b/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." &>/dev/null && pwd)}"
export HF_CKPT_DIR="${QWEN3_4B_BASE_HF_ROOT:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000}"
source "${REPO_ROOT}/experiments/env.sh"

: "${WANDB_API_KEY:?set WANDB_API_KEY, or put an api.wandb.ai entry in ~/.netrc}"

# run identity
MODEL_NAME=Qwen3-4B-Base-LR2e-5-Step4000
HF_MODEL_NAME=iter_0004000
PLACEMENT=colocated
DATASET_TAG=dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000
PROMPT_DATA=/data/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl

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
: "${MAX_RESPONSE_LEN:=16384}"
: "${NUM_ROLLOUT:=300}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"

# batch shape: miles asserts rollout_batch * n_samples == global_batch * num_steps
: "${ROLLOUT_BATCH_SIZE:=192}"
: "${N_SAMPLES_PER_PROMPT:=16}"
: "${GLOBAL_BATCH_SIZE:=3072}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"

# distributed setting
: "${ACTOR_NUM_NODES:=2}"
: "${ACTOR_GPUS_PER_NODE:=8}"
: "${ROLLOUT_NUM_GPUS:=0}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=1}"

# throughput / memory
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${LOG_PROBS_CHUNK_SIZE:=-1}"    # set 8192 when training-entropy observation is enabled
: "${RECOMPUTE_GRANULARITY:=full}"
: "${OVERLAP_COMM:=0}"
: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=2}"
: "${SGLANG_MEM_FRACTION:=0.65}"   # colocated shares the GPU with training

# reward
: "${RM_TYPE:=deepscaler}"
: "${ZERO_REWARD_ON_TRUNCATED:=1}"

# evaluation during training; the reported numbers come from src/offline_eval
: "${EVAL_INTERVAL:=0}"       # 0 disables in-run eval; score HF exports offline
: "${N_SAMPLES_PER_EVAL_PROMPT:=8}"
: "${EVAL_MAX_RESPONSE_LEN:=16384}"
: "${SKIP_EVAL_BEFORE_TRAIN:=0}"

# checkpointing: torch_dist for resume, hf for offline eval, on separate cadences
: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=100}"
: "${SAVE_HF:=1}"
: "${HF_SAVE_INTERVAL:=10}"

# diagnostics
: "${DUMP_TRAIN_DATA:=0}"
: "${OBSERVE_TRAINING_ENTROPY:=0}" # full-vocab entropy is diagnostic-only when ENTROPY_COEF=0

export MODEL_NAME HF_MODEL_NAME PLACEMENT DATASET_TAG PROMPT_DATA
export ADVANTAGE_ESTIMATOR ENTROPY_COEF KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
export RATIO_DENOMINATOR IS_CORRECTION TIS_CLIP TIS_CLIP_LOW MIS_PROFILE M2PO_BUDGET USE_OPSM OPSM_DELTA
export LR MAX_RESPONSE_LEN NUM_ROLLOUT TRAIN_SEED ROLLOUT_SEED
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_STEPS_PER_ROLLOUT
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
export TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE
export MAX_TOKENS_PER_GPU LOG_PROBS_CHUNK_SIZE RECOMPUTE_GRANULARITY OVERLAP_COMM
export ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION
export RM_TYPE ZERO_REWARD_ON_TRUNCATED
export EVAL_INTERVAL N_SAMPLES_PER_EVAL_PROMPT EVAL_MAX_RESPONSE_LEN SKIP_EVAL_BEFORE_TRAIN
export SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL
export DUMP_TRAIN_DATA OBSERVE_TRAINING_ENTROPY

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

source "${REPO_ROOT}/experiments/common/clean_checkpoint.sh"

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
    bash /root/miles/experiments/scripts/math/sync/dapo-math-p10-90/qwen3-4b/train.sh
