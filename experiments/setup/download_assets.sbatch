#!/bin/bash
#SBATCH --job-name=miles-download-assets
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Download the model and the datasets used by the math / tool-multiturn recipes.
#
#   sbatch -A coreai_horizon_dilations experiments/setup/download_assets.sbatch
#
# Lands in:
#   $HF_CKPT_DIR/Qwen3-4B              (HuggingFace weights, also used by SGLang)
#   $DATASET_DIR/dapo-math-17k         (training prompts)
#   $DATASET_DIR/aime-2024             (eval prompts)
#
# CPU nodes have egress to huggingface.co (verified). Set HF_TOKEN in your
# environment first if you ever point this at a gated repo.

set -euo pipefail

# sbatch copies this script into Slurm's spool directory, so BASH_SOURCE does
# not point at the repo. Resolve through the submit directory (submit from the
# repo root), falling back to the path for a direct `bash` invocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface

hf download Qwen/Qwen3-4B                       --local-dir /ckpt/hf/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /data/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /data/aime-2024

echo "--- downloaded ---"
ls -la /ckpt/hf/Qwen3-4B | head
ls -la /data/dapo-math-17k /data/aime-2024
'
