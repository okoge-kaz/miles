#!/bin/bash
#SBATCH --job-name=offline-search-r1
#SBATCH --partition=batch,batch_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/offline_eval/%x-%j.log

# Offline Search-R1 evaluation for one HF checkpoint.  The inner script starts
# one 8-way data-parallel SGLang server and one CPU wiki-18 retriever, then runs
# all seven published QA benchmarks through the training rollout function.
#
# Required:
#   CKPT        in-container HF checkpoint path
#
# Common knobs:
#   UNPAD_VOCAB=1 for HF snapshots exported by Megatron
#   N_SAMPLES=1, MAX_NEW_TOKENS=512, MAX_TURNS=3, SEARCH_TOPK=3
#   LIMIT=<n> for a smoke run
#   OUT_DIR=/data/offline_eval/search_r1/<tag>
#   EXTRA_MOUNTS=<host>:<container> for checkpoints outside TRAIN_CKPT_DIR

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${CKPT:?set CKPT to an in-container HF checkpoint, for example /ckpt/training/.../hf/19}"

export BENCHMARKS="${BENCHMARKS:-nq:/data/search-r1/eval/nq-miles.jsonl+hotpotqa:/data/search-r1/eval/hotpotqa-miles.jsonl+triviaqa:/data/search-r1/eval/triviaqa-miles.jsonl+popqa:/data/search-r1/eval/popqa-miles.jsonl+2wikimultihopqa:/data/search-r1/eval/2wikimultihopqa-miles.jsonl+musique:/data/search-r1/eval/musique-miles.jsonl+bamboogle:/data/search-r1/eval/bamboogle-miles.jsonl}"
export N_SAMPLES="${N_SAMPLES:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export MAX_TURNS="${MAX_TURNS:-3}"
export SEARCH_TOPK="${SEARCH_TOPK:-3}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:--1}"
export CONCURRENCY_PER_BENCHMARK="${CONCURRENCY_PER_BENCHMARK:-4}"
export SEARCH_TIMEOUT="${SEARCH_TIMEOUT:-60}"
export SEARCH_MAX_ATTEMPTS="${SEARCH_MAX_ATTEMPTS:-3}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
export RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
export SERVER_PORT="${SERVER_PORT:-30000}"
export RETRIEVER_BATCH_MAX_REQUESTS="${RETRIEVER_BATCH_MAX_REQUESTS:-64}"
export RETRIEVER_BATCH_WAIT_MS="${RETRIEVER_BATCH_WAIT_MS:-50}"
export RETRIEVER_FAISS_THREADS="${RETRIEVER_FAISS_THREADS:-32}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export UNPAD_VOCAB="${UNPAD_VOCAB:-0}"
export FORMAT_SCORE="${FORMAT_SCORE:-0.0}"
export LIMIT="${LIMIT:-}"

export TAG="${TAG:-$(echo "${CKPT}" | sed 's|^/ckpt/training/||; s|/|_|g')}"
export OUT_DIR="${OUT_DIR:-/data/offline_eval/search_r1/${TAG}}"
export CKPT

for asset in "${DATASET_DIR}/search-r1/e5_Flat.index" \
             "${DATASET_DIR}/search-r1/wiki-18.jsonl" \
             "${HF_CKPT_DIR}/e5-base-v2/.download_complete"; do
    [[ -e "${asset}" ]] || {
        echo "missing ${asset}; run experiments/setup/prepare_search_r1.sbatch first" >&2
        exit 1
    }
done
for name in nq hotpotqa triviaqa popqa 2wikimultihopqa musique bamboogle; do
    [[ -s "${DATASET_DIR}/search-r1/eval/${name}-miles.jsonl" ]] || {
        echo "missing Search-R1 eval set ${name}; run experiments/setup/prepare_search_r1.sbatch first" >&2
        exit 1
    }
done

mkdir -p "${OUTPUT_DIR}/offline_eval" "${DATASET_DIR}/offline_eval/search_r1"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/src/offline_eval/search_r1_eval_job.sh
