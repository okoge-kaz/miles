#!/bin/bash
#PBS -N prep-search-r1
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=08:00:00
#PBS -j oe

# Assemble the Search-R1 retrieval assets into the shapes the retriever wants.
#
#   pbs_submit --profile cpu experiments/setup/environments/prepare_search_r1.sbatch
#
# The three downloads land in forms nothing can read directly:
#   wiki18-e5-index   split into part_aa (43 GB) + part_ab (22 GB); FAISS needs
#                     one file, so they are concatenated back
#   wiki18-corpus     wiki-18.jsonl.gz; the retriever mmaps the plain jsonl
#   e5-base-v2        the query encoder, not shipped with either of the above
#
# ~65 GB of sequential I/O on lustre belongs on the dedicated datamover lane.
# Everything is skipped if already present, so a rerun after a timeout costs a
# stat() per artifact.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
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
# Despite the name, wiki-18.jsonl.gz is a gzipped *tar* (file(1): "POSIX tar
# archive"), so `gunzip -c` yields tar framing rather than jsonl and every read
# dies on a stray 0x80. Detect the container format instead of trusting the
# extension.
if [[ -s "${CORPUS}" ]] && head -c 1 "${CORPUS}" | grep -q '{'; then
    echo "skip corpus: ${CORPUS} exists ($(du -h "${CORPUS}" | cut -f1))"
else
    rm -f "${CORPUS}"
    SRC="${DATASET_DIR}/wiki18-corpus/wiki-18.jsonl.gz"
    echo "decompressing wiki-18 corpus -> ${CORPUS}"
    # Ask tar directly. `file -` on a 512-byte fragment does not see the ustar
    # magic reliably (it sits at offset 257 and needs the whole header block),
    # which is how the first attempt silently took the plain-gunzip branch.
    if tar -tzf "${SRC}" >/dev/null 2>&1; then
        echo "  (gzipped tar; extracting the jsonl member)"
        mkdir -p "${SEARCH_DIR}/_corpus_tmp"
        time tar -xzf "${SRC}" -C "${SEARCH_DIR}/_corpus_tmp"
        MEMBER=$(find "${SEARCH_DIR}/_corpus_tmp" -type f -name "*.jsonl" | head -1)
        [[ -n "${MEMBER}" ]] || { echo "no .jsonl inside the archive"; exit 1; }
        mv "${MEMBER}" "${CORPUS}"
        rm -rf "${SEARCH_DIR}/_corpus_tmp"
    else
        time gunzip -c "${SRC}" > "${CORPUS}.partial"
        mv "${CORPUS}.partial" "${CORPUS}"
    fi
    du -h "${CORPUS}"
    # Remove the bad artifact rather than leaving it to look prepared next run.
    head -c 1 "${CORPUS}" | grep -q '{' || {
        echo "corpus does not start with JSON; removing ${CORPUS}"
        rm -f "${CORPUS}"
        exit 1
    }
    echo "corpus lines: $(wc -l < "${CORPUS}")"
fi

# --- 3. query encoder --------------------------------------------------------
E5="${HF_CKPT_DIR}/e5-base-v2"
if [[ -f "${E5}/.download_complete" ]]; then
    echo "skip e5-base-v2: already downloaded"
else
    miles_srun --image="${CONTAINER_IMAGE}" \
         --bind="${CONTAINER_MOUNTS}" \
         --writable-tmpfs --no-home --export=ALL \
         bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
hf download intfloat/e5-base-v2 --local-dir /ckpt/hf/e5-base-v2
date -u +%FT%TZ > /ckpt/hf/e5-base-v2/.download_complete
'
fi

# --- 4. train/eval task data -------------------------------------------------
# stage_all.sh downloads these two Hugging Face datasets. Convert FlashRAG's
# bare question/answer rows to the same prompt/label schema as the training
# parquet, so the rollout and reward paths are identical during evaluation.
TRAIN_DATA="${DATASET_DIR}/searchr1-nq-hotpotqa/train.parquet"
FLASHRAG_DIR="${DATASET_DIR}/flashrag-datasets"
EVAL_DIR="${SEARCH_DIR}/eval"

[[ -s "${TRAIN_DATA}" ]] || {
    echo "missing ${TRAIN_DATA}; stage searchr1-nq-hotpotqa from experiments/setup/manifests/datasets.txt"
    exit 1
}
[[ -d "${FLASHRAG_DIR}" ]] || {
    echo "missing ${FLASHRAG_DIR}; stage flashrag-datasets from experiments/setup/manifests/datasets.txt"
    exit 1
}

python3 "${REPO_ROOT}/experiments/src/datasets/search_r1/build_eval.py" \
    --flashrag-dir "${FLASHRAG_DIR}" \
    --out-dir "${EVAL_DIR}"
python3 "${REPO_ROOT}/experiments/src/datasets/search_r1/build_eval.py" \
    --out-dir "${EVAL_DIR}" \
    --validate-only

echo
echo "search-r1 assets ready:"
ls -la "${SEARCH_DIR}"
echo "  encoder: ${E5}"
echo "  train:   ${TRAIN_DATA}"
echo "  eval:    ${EVAL_DIR}"
echo "next: run experiments/tools/difficulty_filter/run_measure_search_r1.sbatch to build the fixed p10-90 train set"
