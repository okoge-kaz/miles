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
#
# Gated repos (Idavidrein/gpqa) need HF_TOKEN in .env *and* the terms accepted by
# that token's account. Without one, `hf download` exits 0 having fetched only the
# public README -- so the checks below look at what actually landed rather than at
# the exit status.

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

# A gated repo without an accepted licence yields README.md and nothing else,
# with a zero exit status. Left alone it looks staged and only fails much later,
# in a job that expected data. Fail here instead.
_payload=$(find "/data/${LOCAL_NAME}" -maxdepth 2 -type f \
    ! -name "README.md" ! -name ".gitattributes" ! -name "*.md" ! -path "*/.cache/*" | wc -l)
if [[ "${_payload}" -eq 0 ]]; then
    echo "ERROR: ${HF_REPO} produced no data files, only documentation."
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "  HF_TOKEN is unset. Gated repos need it in .env (see .env.example)."
    else
        echo "  HF_TOKEN is set, so the licence is probably not accepted by that"
        echo "  account yet: open https://huggingface.co/datasets/${HF_REPO} and agree."
    fi
    exit 1
fi
'
