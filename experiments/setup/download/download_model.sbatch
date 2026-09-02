#!/bin/bash
#PBS -N miles-download-model
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=24:00:00
#PBS -j oe

# Download one HuggingFace model into $HF_CKPT_DIR/$MODEL_NAME.
#
#   experiments/setup/download/stage_model.sh Qwen3-8B
#
# Idempotent: hf download verifies existing files instead of refetching them.

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="${USER:-$(id -un)}"
WANDB_MODE=disabled
export PATH USER WANDB_MODE

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/singularity.sh"

: "${HF_REPO:?set HF_REPO, e.g. Qwen/Qwen3-8B}"
: "${MODEL_NAME:?set MODEL_NAME, e.g. Qwen3-8B}"
HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-2}"
HF_DOWNLOAD_ATTEMPTS="${HF_DOWNLOAD_ATTEMPTS:-5}"
HF_DOWNLOAD_RETRY_DELAY_SECONDS="${HF_DOWNLOAD_RETRY_DELAY_SECONDS:-60}"
[[ "${HF_DOWNLOAD_MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_DOWNLOAD_MAX_WORKERS must be a positive integer" >&2
    exit 2
}
[[ "${HF_DOWNLOAD_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_DOWNLOAD_ATTEMPTS must be a positive integer" >&2
    exit 2
}
[[ "${HF_DOWNLOAD_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]] || {
    echo "HF_DOWNLOAD_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
}
export HF_DOWNLOAD_MAX_WORKERS HF_DOWNLOAD_ATTEMPTS HF_DOWNLOAD_RETRY_DELAY_SECONDS
[[ -r "${CONTAINER_IMAGE}" ]] || {
    echo "missing CONTAINER_IMAGE=${CONTAINER_IMAGE}; run experiments/setup.sh container --submit" >&2
    exit 1
}

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
echo "downloading ${HF_REPO} -> /ckpt/hf/${MODEL_NAME}"
for ((attempt = 1; attempt <= HF_DOWNLOAD_ATTEMPTS; attempt++)); do
    echo "hf download attempt ${attempt}/${HF_DOWNLOAD_ATTEMPTS} (max-workers=${HF_DOWNLOAD_MAX_WORKERS})"
    if time hf download "${HF_REPO}" \
        --max-workers "${HF_DOWNLOAD_MAX_WORKERS}" \
        --local-dir "/ckpt/hf/${MODEL_NAME}"; then
        break
    fi
    if (( attempt == HF_DOWNLOAD_ATTEMPTS )); then
        echo "ERROR: model download failed after ${HF_DOWNLOAD_ATTEMPTS} attempts" >&2
        exit 1
    fi
    sleep "${HF_DOWNLOAD_RETRY_DELAY_SECONDS}"
done
# Publish a nonempty marker atomically. A killed job may leave only its uniquely
# named partial marker, which the staging wrappers never accept as complete.
_completion_marker="/ckpt/hf/${MODEL_NAME}/.download_complete"
_completion_partial="${_completion_marker}.partial-${PBS_JOBID:-$$}"
cleanup_completion_marker() {
    rm -f -- "${_completion_partial}"
}
trap cleanup_completion_marker EXIT
date -u +%FT%TZ > "${_completion_partial}"
[[ -s "${_completion_partial}" ]]
mv -f -- "${_completion_partial}" "${_completion_marker}"
du -sh "/ckpt/hf/${MODEL_NAME}"
'
