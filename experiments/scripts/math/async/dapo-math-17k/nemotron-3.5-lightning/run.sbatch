#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=lightning-node-ratio
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/nemotron-3.5-lightning/math/train-rollout-ratio/%x-%j.log

set -euo pipefail
# sbatch spools this file outside the checkout; submission always sets MILES_REPO.
REPO_ROOT="${MILES_REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)}}"

# --print-config resolves the same defaults without env.sh, Slurm, or asset writes.

# Run identity and training data.
: "${MODEL_NAME:=Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000}"
: "${HF_MODEL_NAME:=${MODEL_NAME}-policy}"
: "${PROMPT_DATA:=/data/dapo-math-17k/dapo-math-17k.jsonl}"

# RL algorithm: actor denominator with truncated importance sampling.
: "${ADVANTAGE_ESTIMATOR:=grpo}"
: "${ADVANTAGE_CLIP_LOW:=-20}"
: "${ADVANTAGE_CLIP_HIGH:=20}"
: "${FUSE_ONE_STEP_ACTOR_LOGPROBS:=1}"
: "${EPS_CLIP:=0.2}"
: "${EPS_CLIP_HIGH:=0.28}"
: "${TIS_CLIP:=5.0}"
: "${TIS_CLIP_LOW:=0.2}"
: "${USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER:=0}"
: "${TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER_THRESHOLD:=2.0}"

# A fresh run stops after 300 updates (one per rollout), or the four-hour wall limit.
: "${LR:=4e-6}"
: "${ROLLOUT_TOP_P:=1.0}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_CONTEXT_LEN:=18432}"
: "${NUM_ROLLOUT:=300}"
# Optional per-launch step boundary; leave the scheduler's total horizon intact.
: "${DEBUG_EXIT_AFTER_ROLLOUT:=}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"

# Batch shape: rollout_batch * n_samples = global_batch * num_steps.
# LCM of DP={8,12,16,20,24,28,32} is 3360 at TP=2, CP=1.
: "${ROLLOUT_BATCH_SIZE:=420}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"
: "${GLOBAL_BATCH_SIZE:=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))}"
: "${OVER_SAMPLING_BATCH_SIZE:=$((6 * ROLLOUT_BATCH_SIZE))}"
: "${DYNAMIC_SAMPLING_FILTER_PATH:=miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter}"

# Queue admission and in-flight generation at each weight update.
# The completed-group capacity is shared by s8 and s1 to isolate the bound.
: "${QUEUE_TYPE:=queue-recycle}"
: "${MAX_WEIGHT_STALENESS:=8}"
: "${TRAINING_BUFFER_QUEUE_SIZE:=6000}"
: "${STALENESS_REFERENCE:=prefill}"
: "${PAUSE_GENERATION_MODE:=in_place}"
: "${UPDATE_WEIGHTS_INTERVAL:=1}"
: "${USE_REPLAY_BUFFER:=1}"
: "${REPLAY_BUFFER_TYPE:=inflight}"

# Distributed placement: separate trainer and rollout nodes, four GB200 GPUs per node.
: "${GPUS_PER_NODE:=4}"
: "${ACTOR_NUM_NODES:=4}"
: "${ACTOR_GPUS_PER_NODE:=4}"
: "${ROLLOUT_NUM_GPUS:=$(( (32 - ACTOR_NUM_NODES) * GPUS_PER_NODE ))}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=4}"

# Throughput and memory.
: "${MAX_TOKENS_PER_GPU:=18432}"
: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=1}"
: "${SGLANG_MEM_FRACTION:=0.70}"
: "${SGLANG_MAX_RUNNING_REQUESTS:=32}"
: "${SGLANG_CUDA_GRAPH_MAX_BS:=32}"
: "${SGLANG_MOE_RUNNER_BACKEND:=triton}"

# Fully async admits this many samples, independently of OVER_SAMPLING_BATCH_SIZE.
# Fix the largest fleet's 112 engines * requests within each serving profile.
# 32 requests -> 3584, 64 requests -> 7168; an explicit override fixes both.
: "${ASYNC_MAX_CONCURRENT_SAMPLES:=$((112 * SGLANG_MAX_RUNNING_REQUESTS))}"

# Evaluation: use HF exports offline; shared-engine eval pauses producer submissions.
: "${EVAL_INTERVAL:=0}"

# Checkpointing: resumable Megatron and retained HF exports on explicit cadences.
: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=50}"
: "${SAVE_HF:=1}"
: "${HF_SAVE_INTERVAL:=10}"

# Runtime overlays and local/W&B telemetry.
: "${OCI_SGLANG_PROVENANCE_PATCH:=1}"
: "${MILES_EXPERIMENTAL_ROLLOUT_REFACTOR:=1}"
: "${WANDB_PROJECT:=async-rl-nemotron-lightning}"

# Host asset mounts and unique run paths. Environment values override these defaults.
: "${WS:=/lustre/fsw/portfolios/coreai/users/${USER}}"
: "${TRAIN_CKPT_DIR:=${WS}/checkpoints/training}"
: "${HF_CKPT_DIR:=${WS}/checkpoints/hf}"
: "${MEGATRON_CKPT_DIR:=${WS}/checkpoints/megatron}"
: "${DATASET_DIR:=${WS}/datasets}"
: "${SQSH_IMAGE:=${WS}/container/miles-oci-aarch64-7035384.sqsh}"
: "${CONFIG_TAG:=lightning-ratio-s${MAX_WEIGHT_STALENESS}-t${ACTOR_NUM_NODES}r$((ROLLOUT_NUM_GPUS / GPUS_PER_NODE))-${SLURM_JOB_ID:-preview}}"
: "${RUN_NAME:=${CONFIG_TAG}}"
: "${CKPT_PATH:=/ckpt/training/math/dapo-math-17k/${MODEL_NAME}/${CONFIG_TAG}}"

# One inventory for exported job settings and the review JSON.
CONFIG_VARS=(
    GPUS_PER_NODE ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
    TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE
    ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT NUM_STEPS_PER_ROLLOUT GLOBAL_BATCH_SIZE
    OVER_SAMPLING_BATCH_SIZE DYNAMIC_SAMPLING_FILTER_PATH ASYNC_MAX_CONCURRENT_SAMPLES
    MAX_WEIGHT_STALENESS TRAINING_BUFFER_QUEUE_SIZE QUEUE_TYPE STALENESS_REFERENCE PAUSE_GENERATION_MODE
    UPDATE_WEIGHTS_INTERVAL USE_REPLAY_BUFFER REPLAY_BUFFER_TYPE OCI_SGLANG_PROVENANCE_PATCH
    MILES_EXPERIMENTAL_ROLLOUT_REFACTOR NUM_ROLLOUT DEBUG_EXIT_AFTER_ROLLOUT MAX_RESPONSE_LEN MAX_CONTEXT_LEN MAX_TOKENS_PER_GPU
    LR ROLLOUT_TOP_P ADVANTAGE_ESTIMATOR ADVANTAGE_CLIP_LOW ADVANTAGE_CLIP_HIGH FUSE_ONE_STEP_ACTOR_LOGPROBS
    EPS_CLIP EPS_CLIP_HIGH TIS_CLIP TIS_CLIP_LOW
    USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER_THRESHOLD
    ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION SGLANG_MAX_RUNNING_REQUESTS
    SGLANG_CUDA_GRAPH_MAX_BS SGLANG_MOE_RUNNER_BACKEND
    SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL EVAL_INTERVAL TRAIN_SEED ROLLOUT_SEED
    MODEL_NAME HF_MODEL_NAME PROMPT_DATA CONFIG_TAG RUN_NAME CKPT_PATH WS TRAIN_CKPT_DIR
    HF_CKPT_DIR MEGATRON_CKPT_DIR DATASET_DIR SQSH_IMAGE WANDB_PROJECT
)
export "${CONFIG_VARS[@]}"

[[ -z "${DEBUG_EXIT_AFTER_ROLLOUT}" || "${DEBUG_EXIT_AFTER_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]
(( GPUS_PER_NODE == 4 && ACTOR_GPUS_PER_NODE == 4 && ACTOR_NUM_NODES > 0 && ROLLOUT_NUM_GPUS > 0 ))
(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT == GLOBAL_BATCH_SIZE * NUM_STEPS_PER_ROLLOUT ))
(( NUM_STEPS_PER_ROLLOUT == 1 && N_SAMPLES_PER_PROMPT > 1 && OVER_SAMPLING_BATCH_SIZE >= ROLLOUT_BATCH_SIZE ))
(( FUSE_ONE_STEP_ACTOR_LOGPROBS == 1 ))
[[ "${USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER}" =~ ^[01]$ ]]
(( MAX_CONTEXT_LEN > MAX_RESPONSE_LEN && MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE >= MAX_CONTEXT_LEN ))
(( ASYNC_MAX_CONCURRENT_SAMPLES >= N_SAMPLES_PER_PROMPT && ASYNC_MAX_CONCURRENT_SAMPLES % N_SAMPLES_PER_PROMPT == 0 ))
(( SAVE_INTERVAL > 0 && SAVE_RETAIN_INTERVAL > 0 && SAVE_HF == 1 && HF_SAVE_INTERVAL > 0 ))
(( UPDATE_WEIGHTS_INTERVAL == 1 && EVAL_INTERVAL == 0 && USE_REPLAY_BUFFER == 1 ))
(( OCI_SGLANG_PROVENANCE_PATCH == 1 && MILES_EXPERIMENTAL_ROLLOUT_REFACTOR == 1 ))
[[ "${QUEUE_TYPE}" == queue-recycle && "${STALENESS_REFERENCE}" == prefill && "${REPLAY_BUFFER_TYPE}" == inflight ]]
[[ "${PAUSE_GENERATION_MODE}" == in_place && -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]]
(( ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE % EXPERT_PARALLEL_SIZE == 0 ))
source "${REPO_ROOT}/experiments/common/staleness_buffer_policy.sh"
require_staleness_training_buffer_queue_size "${MAX_WEIGHT_STALENESS}" "${TRAINING_BUFFER_QUEUE_SIZE}" \
    "${GLOBAL_BATCH_SIZE}" "${N_SAMPLES_PER_PROMPT}"

if [[ "${1:-}" == --print-config ]]; then
    SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-32}"
    # Skip the runtime-only async image override while checking the placement.
    PLACEMENT=preview
    source "${REPO_ROOT}/experiments/common/placement.sh" >&2
    python3 -c 'import json, os, sys; print(json.dumps({k: os.environ[k] for k in sys.argv[1:]}, sort_keys=True))' "${CONFIG_VARS[@]}"
    exit 0
fi
[[ $# == 0 ]]
: "${SLURM_JOB_ID:?submit through Slurm; use --print-config for local inspection}"
export MILES_REPO="${REPO_ROOT}"
# Keep the reviewed image even if .env names a different async experiment image.
export ASYNC_SQSH_IMAGE_OVERRIDE="${SQSH_IMAGE}"
source "${REPO_ROOT}/experiments/env.sh"
: "${WANDB_API_KEY:?W&B is required; set WANDB_API_KEY or configure api.wandb.ai in ~/.netrc}"
PLACEMENT=async
source "${REPO_ROOT}/experiments/common/placement.sh"
[[ -f "${HF_CKPT_DIR}/${HF_MODEL_NAME}/.download_complete" ]]
[[ -f "${MEGATRON_CKPT_DIR}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt" ]]
[[ -f "${DATASET_DIR}/dapo-math-17k/dapo-math-17k.jsonl" && -r "${SQSH_IMAGE}" ]]

RAY_HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
RAY_HEAD_IP=$(getent ahostsv4 "${RAY_HEAD_NODE}" | awk 'NR==1 {print $1}')
[[ -n "${RAY_HEAD_IP}" && "${RAY_HEAD_IP}" != 169.254.* ]]
RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
export RAY_HEAD_IP RAY_DONE_FLAG
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
echo "run ${RUN_NAME}; checkpoints: ${CKPT_PATH}; torch_dist=${SAVE_INTERVAL}, retain=${SAVE_RETAIN_INTERVAL}, HF=${HF_SAVE_INTERVAL}"
echo "fully async: staleness=${MAX_WEIGHT_STALENESS}, queue=${TRAINING_BUFFER_QUEUE_SIZE} groups, concurrent=${ASYNC_MAX_CONCURRENT_SAMPLES} samples"
echo "telemetry: ${TRAIN_CKPT_DIR}/${CKPT_PATH#/ckpt/training/}/dump/dashboard"
python3 -c 'import json, os, sys; print("resolved_config=" + json.dumps({k: os.environ[k] for k in sys.argv[1:]}, sort_keys=True))' "${CONFIG_VARS[@]}"
srun --nodes="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --mem=0 \
    --container-image="${SQSH_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable --no-container-mount-home --export=ALL \
    bash /root/miles/experiments/scripts/math/async/dapo-math-17k/nemotron-3.5-lightning/train.sh
