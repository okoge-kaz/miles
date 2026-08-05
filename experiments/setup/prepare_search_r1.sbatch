#!/bin/bash
#SBATCH --job-name=prep-search-r1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Assemble the Search-R1 retrieval assets into the shapes the retriever wants.
#
#   sbatch -A coreai_horizon_dilations experiments/setup/prepare_search_r1.sbatch
#
# The three downloads land in forms nothing can read directly:
#   wiki18-e5-index   split into part_aa (43 GB) + part_ab (22 GB); FAISS needs
#                     one file, so they are concatenated back
#   wiki18-corpus     wiki-18.jsonl.gz; the retriever mmaps the plain jsonl
#   e5-base-v2        the query encoder, not shipped with either of the above
#
# ~65 GB of sequential I/O on lustre, hence `cpu` (1 day) rather than the
# 4 h interactive lane. Everything is skipped if already present, so a rerun
# after a timeout costs a stat() per artifact.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

SEARCH_DIR="${DATASET_DIR}/search-r1"
mkdir -p "${SEARCH_DIR}"

INDEX="${SEARCH_DIR}/e5_Flat.index"
CORPUS="${SEARCH_DIR}/wiki-18.jsonl"

# --- 1. index: concatenate the split parts -----------------------------------
if [[ -s "${INDEX}" ]]; then
    echo "skip index: ${INDEX} exists ($(du -h "${INDEX}" | cut -f1))"
else
    echo "concatenating e5 index parts -> ${INDEX}"
    # Write to .partial first: a truncated index looks valid to FAISS and fails
    # much later, inside a GPU job.
    time cat "${DATASET_DIR}"/wiki18-e5-index/part_a* > "${INDEX}.partial"
    mv "${INDEX}.partial" "${INDEX}"
    du -h "${INDEX}"
fi

# --- 2. corpus: decompress ---------------------------------------------------
if [[ -s "${CORPUS}" ]]; then
    echo "skip corpus: ${CORPUS} exists ($(du -h "${CORPUS}" | cut -f1))"
else
    echo "decompressing wiki-18 corpus -> ${CORPUS}"
    time gunzip -c "${DATASET_DIR}/wiki18-corpus/wiki-18.jsonl.gz" > "${CORPUS}.partial"
    mv "${CORPUS}.partial" "${CORPUS}"
    du -h "${CORPUS}"
    echo "corpus lines: $(wc -l < "${CORPUS}")"
fi

# --- 3. query encoder --------------------------------------------------------
E5="${HF_CKPT_DIR}/e5-base-v2"
if [[ -f "${E5}/.download_complete" ]]; then
    echo "skip e5-base-v2: already downloaded"
else
    srun --container-image="${SQSH_IMAGE}" \
         --container-mounts="${CONTAINER_MOUNTS}" \
         --container-writable --no-container-mount-home --export=ALL \
         bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
hf download intfloat/e5-base-v2 --local-dir /ckpt/hf/e5-base-v2
date -u +%FT%TZ > /ckpt/hf/e5-base-v2/.download_complete
'
fi

echo
echo "search-r1 assets ready:"
ls -la "${SEARCH_DIR}"
echo "  encoder: ${E5}"
