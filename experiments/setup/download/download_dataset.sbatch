#!/bin/bash
#SBATCH --job-name=miles-download-dataset
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log
#SBATCH --export=NIL

# Download one HuggingFace dataset into $DATASET_DIR/$LOCAL_NAME.
#
#   sbatch -A coreai_horizon_dilations \
#     --export=USER,HF_REPO=SWE-Gym/SWE-Gym,LOCAL_NAME=swe-gym,HF_TOKEN \
#     experiments/setup/download/download_dataset.sbatch
#
# Idempotent: hf download verifies existing files instead of refetching them.
#
# Gated repos (Idavidrein/gpqa) need an exported HF_TOKEN and the terms accepted
# by that token's account. Repository-local `.env` files are never read. Without
# a usable token, `hf download` can fetch only the public README, so the checks
# below inspect what actually landed rather than trusting the exit status.

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="${USER:-$(id -un)}"
WANDB_MODE=disabled
export PATH USER WANDB_MODE

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
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

# A gated repo without an accepted licence yields README.md and nothing else,
# with a zero exit status. Left alone it looks staged and only fails much later,
# in a job that expected data. Fail here instead.
_payload=$(find "/data/${LOCAL_NAME}" -maxdepth 2 -type f \
    ! -name "README.md" ! -name ".gitattributes" ! -name "*.md" ! -path "*/.cache/*" | wc -l)
if [[ "${_payload}" -eq 0 ]]; then
    echo "ERROR: ${HF_REPO} produced no data files, only documentation."
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "  HF_TOKEN is unset. Export it in the submitting process; repository .env files are not read."
    else
        echo "  HF_TOKEN is set, so the licence is probably not accepted by that"
        echo "  account yet: open https://huggingface.co/datasets/${HF_REPO} and agree."
    fi
    exit 1
fi
'
