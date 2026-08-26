#!/bin/bash
#SBATCH --job-name=search-r1-passrate
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/difficulty/%x-%j.log

# Measure the initial policy's Search-R1 pass rate on every training prompt,
# including the same multi-turn E5 retrieval environment used during training.
# The output is resumable; resubmit the same command after a time limit.
#
# Smoke (measures 64 prompts and does not write the production filtered set):
#
#   sbatch -A coreai_horizon_dilations --export=ALL,LIMIT=64 \
#     experiments/tools/difficulty_filter/run_measure_search_r1.sbatch
#
# Full measurement + fixed p10-90 dataset:
#
#   sbatch -A coreai_horizon_dilations \
#     experiments/tools/difficulty_filter/run_measure_search_r1.sbatch
#
# Defaults intentionally match the training recipe's step-4000 SFT policy,
# n=8, temperature/top-p 1, 512 generated tokens per action, three turns, E5
# top-k 3, outcome-only EM. Pass-rate 0.1--0.9 is inclusive, so with n=8 it
# keeps prompts with 1--7 correct samples and drops only unanimous groups.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
SEARCH_R1_IMAGE_WAS_SET="${SQSH_IMAGE+x}"
source "${REPO_ROOT}/experiments/env.sh"
if [[ -z "${SEARCH_R1_IMAGE_WAS_SET}" ]]; then
    SQSH_IMAGE="${CONTAINER_DIR}/miles-search-r1-b300-faiss-1.12.0-20260826.sqsh"
    export SQSH_IMAGE
fi
unset SEARCH_R1_IMAGE_WAS_SET
[[ -r "${SQSH_IMAGE}" ]] || { echo "missing Search-R1 image: ${SQSH_IMAGE}" >&2; exit 1; }

export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Base-LR2e-5-Step4000}"
export HF_MODEL_NAME="${HF_MODEL_NAME:-Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000/iter_0004000}"
export MODEL_PATH="${MODEL_PATH:-/ckpt/hf/${HF_MODEL_NAME}}"
export PROMPT_DATA="${PROMPT_DATA:-/data/searchr1-nq-hotpotqa/train.parquet}"
export FILTERED_OUTPUT="${FILTERED_OUTPUT:-/data/searchr1-nq-hotpotqa/searchr1-nq-hotpotqa-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl}"
export LIMIT="${LIMIT:-}"
if [[ -n "${LIMIT}" ]]; then
    DEFAULT_OUTPUT="/data/difficulty/smoke/searchr1-nq-hotpotqa.${MODEL_NAME}.limit-${LIMIT}.passrate.jsonl"
    DEFAULT_DUMP_RESPONSES="/data/difficulty/smoke/searchr1-nq-hotpotqa.${MODEL_NAME}.limit-${LIMIT}.audit.jsonl"
else
    DEFAULT_OUTPUT="/data/difficulty/searchr1-nq-hotpotqa.${MODEL_NAME}.passrate.jsonl"
    DEFAULT_DUMP_RESPONSES="/data/difficulty/searchr1-nq-hotpotqa.${MODEL_NAME}.passrate.audit.jsonl"
fi
export OUTPUT="${OUTPUT:-${DEFAULT_OUTPUT}}"
export N_SAMPLES="${N_SAMPLES:-8}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:--1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export MAX_TURNS="${MAX_TURNS:-3}"
export SEARCH_TOPK="${SEARCH_TOPK:-3}"
export FORMAT_SCORE="${FORMAT_SCORE:-0.0}"
export PASS_RATE_MIN="${PASS_RATE_MIN:-0.1}"
export PASS_RATE_MAX="${PASS_RATE_MAX:-0.9}"
export CONCURRENCY="${CONCURRENCY:-64}"
export SEARCH_CONCURRENCY="${SEARCH_CONCURRENCY:-16}"
export SEARCH_TIMEOUT="${SEARCH_TIMEOUT:-60}"
export SEARCH_MAX_ATTEMPTS="${SEARCH_MAX_ATTEMPTS:-3}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
export RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
export SERVER_PORT="${SERVER_PORT:-30000}"
export RETRIEVER_BATCH_MAX_REQUESTS="${RETRIEVER_BATCH_MAX_REQUESTS:-64}"
export RETRIEVER_BATCH_WAIT_MS="${RETRIEVER_BATCH_WAIT_MS:-50}"
export RETRIEVER_FAISS_THREADS="${RETRIEVER_FAISS_THREADS:-32}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export DUMP_LIMIT="${DUMP_LIMIT:-8}"
export DUMP_RESPONSES="${DUMP_RESPONSES:-${DEFAULT_DUMP_RESPONSES}}"
if [[ -z "${APPLY_FILTER:-}" ]]; then
    if [[ -z "${LIMIT}" ]]; then
        APPLY_FILTER=1
    else
        APPLY_FILTER=0
    fi
fi
export APPLY_FILTER
if [[ -n "${LIMIT}" && "${APPLY_FILTER}" == "1" ]]; then
    echo "refusing to materialize a training filter from a LIMIT=${LIMIT} measurement" >&2
    exit 1
fi
unset DEFAULT_OUTPUT DEFAULT_DUMP_RESPONSES

for asset in "${DATASET_DIR}/searchr1-nq-hotpotqa/train.parquet" \
             "${DATASET_DIR}/search-r1/e5_Flat.index" \
             "${DATASET_DIR}/search-r1/wiki-18.jsonl" \
             "${HF_CKPT_DIR}/e5-base-v2/.download_complete" \
             "${HF_CKPT_DIR}/${HF_MODEL_NAME}/config.json"; do
    [[ -e "${asset}" ]] || {
        echo "missing ${asset}; run experiments/setup/environments/prepare_search_r1.sbatch first" >&2
        exit 1
    }
done

mkdir -p "${OUTPUT_DIR}/difficulty" "${DATASET_DIR}/difficulty"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/search_r1/common/run_measurement.sh
