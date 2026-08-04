#!/bin/bash
#SBATCH --job-name=miles-math-qwen3-4b-instruct-2507
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/%x-%j.log

# Single-turn math RL (GRPO) on Qwen3-4B-Instruct-2507 — colocated.
#
#   sbatch -A coreai_horizon_dilations experiments/math_sync/qwen3-4b-instruct-2507/run.sbatch
#
# Default partition is interactive (4h, <=2 nodes) — the fast lane. Longer runs:
# -p batch_long (8h) or -p batch_large_long (14d), and raise --time to match.
#
# Knobs are env vars carried in by --export=ALL:
#     NUM_ROLLOUT, ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, GLOBAL_BATCH_SIZE,
#     NUM_STEPS_PER_ROLLOUT, MAX_RESPONSE_LEN, RM_TYPE, PROMPT_DATA,
#     DYNAMIC_SAMPLING=0 / PARTIAL_ROLLOUT=0 / MASK_OFFPOLICY=0 to disable those,
#     TENSOR_PARALLEL_SIZE, CONTEXT_PARALLEL_SIZE, MAX_TOKENS_PER_GPU
#
#   sbatch --export=ALL,NUM_ROLLOUT=5,ROLLOUT_BATCH_SIZE=8,GLOBAL_BATCH_SIZE=64 \
#          -A coreai_horizon_dilations experiments/math_sync/qwen3-4b-instruct-2507/run.sbatch

set -euo pipefail

# sbatch copies this script into Slurm's spool directory, so BASH_SOURCE does
# not point at the repo. Resolve through the submit directory (submit from the
# repo root), falling back to the path for a direct `bash` invocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

export RUN_NAME="${RUN_NAME:-math-grpo-qwen3-4b-instruct-2507-${SLURM_JOB_ID}}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/math_sync/qwen3-4b-instruct-2507/train.sh
