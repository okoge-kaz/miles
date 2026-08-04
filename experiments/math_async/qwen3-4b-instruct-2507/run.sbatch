#!/bin/bash
#SBATCH --job-name=miles-math-async-qwen3-4b-instruct-2507
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/%x-%j.log

# Fully-async math RL on Qwen3-4B-Instruct-2507: rollout and training run concurrently.
#
#   sbatch -A coreai_horizon_dilations experiments/math_async/qwen3-4b-instruct-2507/run.sbatch
#   sbatch -A coreai_horizon_dilations -N 4 -p batch experiments/math_async/qwen3-4b-instruct-2507/run.sbatch
#
# Env vars carried in by --export=ALL:
#     placement:  ACTOR_NUM_NODES, ACTOR_GPUS_PER_NODE, ROLLOUT_NUM_GPUS
#     batch:      ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, GLOBAL_BATCH_SIZE,
#                 NUM_STEPS_PER_ROLLOUT, NUM_ROLLOUT, MAX_RESPONSE_LEN
#     parallel:   TENSOR_PARALLEL_SIZE, CONTEXT_PARALLEL_SIZE,
#                 EXPERT_PARALLEL_SIZE, MAX_TOKENS_PER_GPU
#     rollout:    ROLLOUT_NUM_GPUS_PER_ENGINE, SGLANG_MEM_FRACTION,
#                 SGLANG_MAX_RUNNING_REQUESTS, SGLANG_CUDA_GRAPH_MAX_BS
#     sampling:   DYNAMIC_SAMPLING=0, PARTIAL_ROLLOUT=0, MASK_OFFPOLICY=1, MAX_WEIGHT_STALENESS,
#                 ASYNC_MAX_CONCURRENT_SAMPLES
#     telemetry:  DUMP_DETAILS=0, USE_DASHBOARD=0, ROLLOUT_ENTROPY=0, DUMP_DIR
#     data:       PROMPT_DATA, RM_TYPE, SAVE_INTERVAL

set -euo pipefail

# sbatch copies this script into Slurm's spool directory, so BASH_SOURCE does
# not point at the repo. Resolve through the submit directory (submit from the
# repo root), falling back to the path for a direct `bash` invocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

export RUN_NAME="${RUN_NAME:-math-grpo-async-qwen3-4b-instruct-2507-${SLURM_JOB_ID}}"

# The batch script runs on the first node of the allocation, which is therefore
# the Ray head; workers dial this address. The done-flag lives on the shared
# filesystem so the workers can see it, and is cleared first in case a previous
# job with the same RUN_NAME left one behind — a stale flag would make every
# worker exit immediately and the run would fail with a one-node cluster.
export RAY_HEAD_IP="$(hostname -I | awk '{print $1}')"
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}.done"
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f "${OUTPUT_DIR}/.ray/${RUN_NAME}.done"

echo "nodes=${SLURM_JOB_NUM_NODES} head=${RAY_HEAD_IP} run=${RUN_NAME}"

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/math_async/qwen3-4b-instruct-2507/train.sh
