#!/bin/bash
#SBATCH --job-name=miles-tool-multiturn
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/%x-%j.log

# Multi-turn tool-calling RL (ReTool v2 style) on Qwen3-4B.
#
#   sbatch -A coreai_horizon_dilations experiments/tool_multiturn/qwen3-4b/run.sbatch
#
# Same assets as math_sync/qwen3-4b; the difference is the multi-turn generate function
# plus a local Python-interpreter tool. Knobs:
#   GENERATE_MAX_TURNS, NUM_ROLLOUT, ROLLOUT_BATCH_SIZE, GLOBAL_BATCH_SIZE

set -euo pipefail

# sbatch copies this script into Slurm's spool directory, so BASH_SOURCE does
# not point at the repo. Resolve through the submit directory (submit from the
# repo root), falling back to the path for a direct `bash` invocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

export RUN_NAME="${RUN_NAME:-tool-multiturn-qwen3-4b-${SLURM_JOB_ID}}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/tool_multiturn/qwen3-4b/train.sh
