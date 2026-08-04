#!/bin/bash
#SBATCH --job-name=miles-download-dataset
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Download one HuggingFace dataset into $DATASET_DIR/$LOCAL_NAME.
#
#   sbatch -A coreai_horizon_dilations \
#     --export=ALL,HF_REPO=SWE-Gym/SWE-Gym,LOCAL_NAME=swe-gym \
#     experiments/setup/download_dataset.sbatch
#
# Idempotent: hf download verifies existing files instead of refetching them.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${HF_REPO:?set HF_REPO, e.g. SWE-Gym/SWE-Gym}"
: "${LOCAL_NAME:?set LOCAL_NAME, e.g. swe-gym}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
echo "downloading dataset ${HF_REPO} -> /data/${LOCAL_NAME}"
time hf download --repo-type dataset "${HF_REPO}" --local-dir "/data/${LOCAL_NAME}"
du -sh "/data/${LOCAL_NAME}"
find "/data/${LOCAL_NAME}" -maxdepth 2 -type f | head -20
'
