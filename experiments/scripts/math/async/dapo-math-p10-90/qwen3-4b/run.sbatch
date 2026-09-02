#!/bin/bash
#PBS -N miles-dapo-math-p10-90-async-qwen3-4b
#PBS -q R9920261300
#PBS -l select=2:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." &>/dev/null && pwd)}}"
source "${REPO_ROOT}/experiments/env.sh"

if [[ "${WANDB_MODE:-online}" != offline && "${WANDB_MODE:-online}" != disabled ]]; then
    : "${WANDB_API_KEY:?set WANDB_API_KEY, use WANDB_MODE=offline, or disable wandb}"
fi
: "${WANDB_TEAM:=okoge}"
: "${WANDB_PROJECT:=miles-async-rl-math}"
export WANDB_TEAM WANDB_PROJECT

# run identity
: "${MODEL_NAME:=Qwen3-4B-Base-LR2e-5-Step4000}"
: "${HF_MODEL_NAME:=${QWEN3_4B_BASE_HF_MODEL}}"
: "${MODEL_PROFILE:=qwen3-4B}"
PLACEMENT=async
: "${DATASET_TAG:=dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000}"
: "${PROMPT_DATA:=/data/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl}"

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

# queue selection and what happens to in-flight generation at a weight update
: "${QUEUE_TYPE:=queue-recycle}" # queue-recycle | queue-max | queue-drop
: "${QUEUE_FACTOR:=1}"                       # queue-drop capacity in training batches
if [[ "${QUEUE_TYPE}" == queue-drop ]]; then
   [[ -z "${MAX_WEIGHT_STALENESS:-}" ]] || {
      echo "QUEUE_TYPE=queue-drop does not use MAX_WEIGHT_STALENESS" >&2
      exit 1
   }
   unset MAX_WEIGHT_STALENESS
else
   : "${MAX_WEIGHT_STALENESS:=8}"
fi
: "${STALENESS_REFERENCE:=prefill}"   # completion | submission | prefill
: "${PAUSE_GENERATION_MODE:=in_place}"
: "${ASYNC_MAX_CONCURRENT_SAMPLES:=}"
: "${USE_REPLAY_BUFFER:=1}"
: "${REPLAY_BUFFER_TYPE:=inflight}" # rollout | inflight
: "${REPLAY_BUFFER_IDENTITY_TAG:=1}" # keep buffer types in separate checkpoint namespaces
[[ "${USE_REPLAY_BUFFER}" =~ ^[01]$ ]] || {
   echo "USE_REPLAY_BUFFER must be 0 or 1" >&2
   exit 1
}
[[ "${REPLAY_BUFFER_TYPE}" =~ ^(rollout|inflight)$ ]] || {
   echo "REPLAY_BUFFER_TYPE must be rollout or inflight" >&2
   exit 1
}

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
: "${DUMP_POLICY_LOSS_DEBUG:=0}"
: "${FUSE_ONE_STEP_ACTOR_LOGPROBS:=1}"
: "${VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS:=0}"  # debug only; retains legacy forward
: "${LOG_SAMPLE_STALENESS_METRICS:=1}"
: "${LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM:=0}"  # high-cardinality, debug only
: "${LOG_UPDATE_DIAGNOSTICS:=1}"                 # no parameter/update tensor scan
# The managed SIF pins and tests the SGLang response-version extension. Keep
# exact token/version segments on unless an explicit ablation disables them.
: "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS:=1}"
: "${DEBUG_EXIT_AFTER_ROLLOUT:=}"               # validation only; empty runs the full schedule

: "${MILES_NCCL_TRANSPORT:=system}"
: "${ASYNC_CONTAINER_IMAGE_OVERRIDE:=${CONTAINER_IMAGE}}"
case "${MILES_NCCL_TRANSPORT}" in
   tcp)
      NCCL_IB_DISABLE=1
      NCCL_NET=Socket
      unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER
      ;;
   system)
      NCCL_IB_DISABLE=0
      unset NCCL_NET NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER
      ;;
   *)
      echo "unknown MILES_NCCL_TRANSPORT=${MILES_NCCL_TRANSPORT}; expected tcp or system" >&2
      exit 1
      ;;
esac

export MODEL_NAME HF_MODEL_NAME MODEL_PROFILE PLACEMENT DATASET_TAG PROMPT_DATA
export ADVANTAGE_ESTIMATOR ENTROPY_COEF KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
export RATIO_DENOMINATOR IS_CORRECTION TIS_CLIP TIS_CLIP_LOW MIS_PROFILE M2PO_BUDGET USE_OPSM OPSM_DELTA
export LR MAX_RESPONSE_LEN NUM_ROLLOUT TRAIN_SEED ROLLOUT_SEED
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_STEPS_PER_ROLLOUT
export QUEUE_TYPE QUEUE_FACTOR MAX_WEIGHT_STALENESS STALENESS_REFERENCE
export PAUSE_GENERATION_MODE ASYNC_MAX_CONCURRENT_SAMPLES
export USE_REPLAY_BUFFER REPLAY_BUFFER_TYPE REPLAY_BUFFER_IDENTITY_TAG
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
export TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE
export MAX_TOKENS_PER_GPU RECOMPUTE_GRANULARITY OVERLAP_COMM
export ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION
export RM_TYPE ZERO_REWARD_ON_TRUNCATED
export EVAL_INTERVAL N_SAMPLES_PER_EVAL_PROMPT EVAL_MAX_RESPONSE_LEN SKIP_EVAL_BEFORE_TRAIN
export SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL
export DUMP_TRAIN_DATA DUMP_POLICY_LOSS_DEBUG FUSE_ONE_STEP_ACTOR_LOGPROBS VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS
export LOG_SAMPLE_STALENESS_METRICS LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM LOG_UPDATE_DIAGNOSTICS
export SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS
export DEBUG_EXIT_AFTER_ROLLOUT
export MILES_NCCL_TRANSPORT ASYNC_CONTAINER_IMAGE_OVERRIDE NCCL_IB_DISABLE LD_LIBRARY_PATH
export NCCL_NET

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
export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${MILES_JOB_ID}.done"
export RAY_HEAD_IP_FILE="${RAY_DONE_FLAG}.head-ip"
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f \
   "${OUTPUT_DIR}/.ray/${RUN_NAME}-${MILES_JOB_ID}.done" \
   "${OUTPUT_DIR}/.ray/${RUN_NAME}-${MILES_JOB_ID}.done.head-ip"

source "${REPO_ROOT}/experiments/common/clean_checkpoint.sh"

echo "run ${RUN_NAME}"
echo "checkpoints ${CKPT_PATH}"

# The validation jobs are submitted by a small CPU finalizer. Use the GPU
# profile's per-node CPU count rather than an inherited allocation-wide value.
STEP_CPUS_PER_NODE="${PBS_GPU_CPUS_PER_NODE}"
[[ "${STEP_CPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid PBS_NCPUS=${PBS_NCPUS:-unset}" >&2
    exit 1
}
MOUNTS="${CONTAINER_MOUNTS},/usr/bin/bwrap:/usr/local/bin/bwrap"

container_command=(bash /root/miles/experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/train.sh)

miles_srun \
    --nodes="${MILES_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${STEP_CPUS_PER_NODE}" \
    --mem=0 \
    --image="${CONTAINER_IMAGE}" \
    --bind="${MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    "${container_command[@]}"
