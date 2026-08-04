#!/bin/bash
#SBATCH --job-name=miles-math-grpo-async
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/%x-%j.log

# Fully-async math RL on Qwen3-4B: rollout and training run concurrently on
# disjoint GPUs (4 + 4 by default).
#
#   sbatch -A coreai_horizon_dilations experiments/math_async/qwen3-4b/run.sbatch
#
# Knobs: ACTOR_GPUS, ROLLOUT_GPUS (must sum to 8),
#        ASYNC_MAX_CONCURRENT_SAMPLES, MAX_WEIGHT_STALENESS,
#        NUM_ROLLOUT, ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, GLOBAL_BATCH_SIZE

set -euo pipefail

# sbatch copies this script into Slurm's spool directory, so BASH_SOURCE does
# not point at the repo. Resolve through the submit directory (submit from the
# repo root), falling back to the path for a direct `bash` invocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

export RUN_NAME="${RUN_NAME:-math-grpo-async-qwen3-4b-${SLURM_JOB_ID}}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/math_async/qwen3-4b/train.sh
