#!/bin/bash
#PBS -N miles-download-dataset
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=24:00:00
#PBS -j oe

# Download one HuggingFace dataset into $DATASET_DIR/$LOCAL_NAME.
#
#   experiments/setup.sh datasets --submit
#
# Set HF_REVISION to a commit when the result will feed training. Idempotent:
# hf download verifies existing files instead of refetching them.
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

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/singularity.sh"

: "${HF_REPO:?set HF_REPO, e.g. SWE-Gym/SWE-Gym}"
: "${LOCAL_NAME:?set LOCAL_NAME, e.g. swe-gym}"
HF_REVISION="${HF_REVISION:-main}"
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
export HF_REVISION HF_DOWNLOAD_MAX_WORKERS HF_DOWNLOAD_ATTEMPTS
export HF_DOWNLOAD_RETRY_DELAY_SECONDS
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
echo "downloading dataset ${HF_REPO} -> /data/${LOCAL_NAME}"
_attempt=1
while :; do
    echo "hf download attempt ${_attempt}/${HF_DOWNLOAD_ATTEMPTS} (max-workers=${HF_DOWNLOAD_MAX_WORKERS})"
    if time hf download \
        --repo-type dataset \
        --revision "${HF_REVISION}" \
        --max-workers "${HF_DOWNLOAD_MAX_WORKERS}" \
        "${HF_REPO}" \
        --local-dir "/data/${LOCAL_NAME}"; then
        break
    fi
    if (( _attempt >= HF_DOWNLOAD_ATTEMPTS )); then
        echo "ERROR: hf download failed after ${HF_DOWNLOAD_ATTEMPTS} attempts: ${HF_REPO}" >&2
        exit 1
    fi
    echo "hf download failed; resuming in ${HF_DOWNLOAD_RETRY_DELAY_SECONDS}s" >&2
    sleep "${HF_DOWNLOAD_RETRY_DELAY_SECONDS}"
    (( _attempt += 1 ))
done
du -sh "/data/${LOCAL_NAME}"
find "/data/${LOCAL_NAME}" -type f ! -path "*/.cache/*" -print | sed -n "1,20p"

# A gated repo without an accepted licence yields README.md and nothing else,
# with a zero exit status. Left alone it looks staged and only fails much later,
# in a job that expected data. Fail here instead.
_payload=$(find "/data/${LOCAL_NAME}" -type f \
    ! -path "*/.cache/*" \
    ! -name "README.md" \
    ! -name ".gitattributes" \
    ! -name "*.md" \
    ! -name "MILES_SOURCE_PROVENANCE*" \
    -printf . | wc -c)
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
_provenance="/data/${LOCAL_NAME}/MILES_SOURCE_PROVENANCE"
_provenance_partial="${_provenance}.partial-${MILES_JOB_ID_SHORT:-$$}"
printf "repo=%s\nrevision=%s\n" "${HF_REPO}" "${HF_REVISION}" \
    >"${_provenance_partial}"
mv -f -- "${_provenance_partial}" "${_provenance}"
'
