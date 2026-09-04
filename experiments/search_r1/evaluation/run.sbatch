#!/bin/bash
#SBATCH --job-name=search-r1-eval
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/%x-%j.log

# Held-out Search-R1 evaluation. This starts one local E5/FAISS retriever and
# one data-parallel SGLang server, then evaluates immutable benchmark JSONL
# files with the exact multi-turn generator and reward used for training.
#
# Smoke (two prompts from NQ and HotpotQA):
#   sbatch -A coreai_horizon_dilations -p batch --qos=interactive \
#     --export=ALL,WANDB_MODE=offline,EVAL_MODE=smoke \
#     experiments/search_r1/evaluation/run.sbatch
#
# Full held-out evaluation of an exported checkpoint:
#   sbatch -A coreai_horizon_dilations -p batch --qos=interactive \
#     --export=ALL,WANDB_MODE=offline,EVAL_MODE=full,MODEL_PATH=/ckpt/training/<run>/hf/<step> \
#     experiments/search_r1/evaluation/run.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
SEARCH_R1_IMAGE_WAS_SET="${SQSH_IMAGE+x}"
source "${REPO_ROOT}/experiments/env.sh"
if [[ -z "${SEARCH_R1_IMAGE_WAS_SET}" ]]; then
    SQSH_IMAGE="${CONTAINER_DIR}/miles-search-r1-b300-faiss-1.12.0-20260826.sqsh"
    export SQSH_IMAGE
fi
unset SEARCH_R1_IMAGE_WAS_SET
[[ -r "${SQSH_IMAGE}" ]] || { echo "missing Search-R1 image: ${SQSH_IMAGE}" >&2; exit 3; }

export WANDB_MODE="${WANDB_MODE:-offline}"
[[ "${WANDB_MODE}" == offline ]] || {
    echo "Search-R1 validation requires WANDB_MODE=offline" >&2
    exit 1
}

export MEASUREMENT_MODE=evaluation
export EVAL_MODE="${EVAL_MODE:-smoke}"
case "${EVAL_MODE}" in
    smoke)
        export BENCHMARKS="${BENCHMARKS:-nq hotpotqa}"
        export LIMIT="${LIMIT:-2}"
        ;;
    full)
        export BENCHMARKS="${BENCHMARKS:-nq hotpotqa triviaqa popqa 2wikimultihopqa musique bamboogle}"
        export LIMIT="${LIMIT:-}"
        ;;
    *)
        echo "EVAL_MODE must be smoke or full, got ${EVAL_MODE}" >&2
        exit 2
        ;;
esac

export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Base-LR2e-5-Step4000}"
export MODEL_PATH="${MODEL_PATH:-${HF_CKPT_DIR}/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000/iter_0004000}"
export POLICY_NAME="${POLICY_NAME:-$(basename -- "${MODEL_PATH}")}"
RESULT_ROOT_HOST="$(realpath -m -- "${RESULT_ROOT:-${OUTPUT_DIR}/evaluation/search_r1/${POLICY_NAME}/${EVAL_MODE}}")"
export N_SAMPLES="${N_SAMPLES:-1}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:--1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export MAX_TURNS="${MAX_TURNS:-3}"
export SEARCH_TOPK="${SEARCH_TOPK:-3}"
export FORMAT_SCORE="${FORMAT_SCORE:-0.0}"
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

VALIDATE_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/validate_checkpoint.py"
UNPAD_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/unpad_vocab.py"
MODEL_SOURCE="$(realpath -- "${MODEL_PATH}")"
CHECKPOINT_MANIFEST_SHA256="$(
    python3 "${VALIDATE_TOOL}" --quiet --print-manifest "${MODEL_SOURCE}"
)"
MODEL_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/search-r1-model-${SLURM_JOB_ID}"
python3 "${UNPAD_TOOL}" "${MODEL_SOURCE}" "${MODEL_RUNTIME_DIR}"
if [[ -f "${MODEL_RUNTIME_DIR}/config.json" ]]; then
    MODEL_MOUNTS="${MODEL_RUNTIME_DIR}:/search-eval-model,${MODEL_SOURCE}:${MODEL_SOURCE}:ro"
else
    MODEL_MOUNTS="${MODEL_SOURCE}:/search-eval-model:ro"
fi

for asset in "${DATASET_DIR}/search-r1/e5_Flat.index" \
             "${DATASET_DIR}/search-r1/wiki-18.jsonl" \
             "${HF_CKPT_DIR}/e5-base-v2/.download_complete"; do
    [[ -e "${asset}" ]] || {
        echo "missing ${asset}; run experiments/setup/environments/prepare_search_r1.sbatch first" >&2
        exit 3
    }
done
for benchmark in ${BENCHMARKS}; do
    [[ -s "${DATASET_DIR}/search-r1/eval/${benchmark}-miles.jsonl" ]] || {
        echo "missing benchmark data: ${DATASET_DIR}/search-r1/eval/${benchmark}-miles.jsonl" >&2
        exit 4
    }
done
python3 "${REPO_ROOT}/experiments/src/datasets/search_r1/build_eval.py" \
    --out-dir "${DATASET_DIR}/search-r1/eval" \
    --validate-only

declare -A EXPECTED_REVISIONS=(
    ["${DATASET_DIR}/wiki18-e5-index/.cache/huggingface/trees/a4d31160a035f30764604f4827cd8f1d0315eb86.json"]=1
    ["${DATASET_DIR}/wiki18-corpus/.cache/huggingface/trees/69c1c00ffe7c5554c68d8548355cb22e46aabc51.json"]=1
    ["${DATASET_DIR}/flashrag-datasets/.cache/huggingface/trees/bcafb8dd07d453be3cbeeeb3f78be1841bddf92c.json"]=1
    ["${HF_CKPT_DIR}/e5-base-v2/.cache/huggingface/trees/f52bf8ec8c7124536f0efb74aca902b2995e5bcd.json"]=1
)
for revision_marker in "${!EXPECTED_REVISIONS[@]}"; do
    [[ -s "${revision_marker}" ]] || {
        echo "missing pinned Search-R1 source revision: ${revision_marker}" >&2
        exit 5
    }
done

mkdir -p "${RESULT_ROOT_HOST}"
desired_contract="${SLURM_TMPDIR:-/tmp}/search-r1-eval-contract-${SLURM_JOB_ID}.env"
cat > "${desired_contract}" <<EOF
checkpoint=${MODEL_SOURCE}
checkpoint_manifest_sha256=${CHECKPOINT_MANIFEST_SHA256}
policy=${POLICY_NAME}
eval_mode=${EVAL_MODE}
benchmarks=${BENCHMARKS}
limit=${LIMIT}
n_samples=${N_SAMPLES}
temperature=${TEMPERATURE}
top_p=${TOP_P}
top_k=${TOP_K}
max_new_tokens_per_turn=${MAX_NEW_TOKENS}
max_turns=${MAX_TURNS}
search_topk=${SEARCH_TOPK}
format_score=${FORMAT_SCORE}
index_revision=a4d31160a035f30764604f4827cd8f1d0315eb86
corpus_revision=69c1c00ffe7c5554c68d8548355cb22e46aabc51
flashrag_revision=bcafb8dd07d453be3cbeeeb3f78be1841bddf92c
e5_revision=f52bf8ec8c7124536f0efb74aca902b2995e5bcd
container_image=${SQSH_IMAGE}
EOF
for benchmark in ${BENCHMARKS}; do
    benchmark_sha256="$(sha256sum "${DATASET_DIR}/search-r1/eval/${benchmark}-miles.jsonl")"
    printf 'dataset_%s_sha256=%s\n' "${benchmark}" "${benchmark_sha256%% *}" \
        >> "${desired_contract}"
done
if [[ -f "${RESULT_ROOT_HOST}/evaluation-contract.env" ]]; then
    cmp --silent "${desired_contract}" "${RESULT_ROOT_HOST}/evaluation-contract.env" || {
        diff --unified "${RESULT_ROOT_HOST}/evaluation-contract.env" "${desired_contract}" || true
        echo "refusing to reuse RESULT_ROOT with a changed Search-R1 protocol" >&2
        exit 6
    }
else
    if find "${RESULT_ROOT_HOST}" -mindepth 2 -type f \
        \( -name records.jsonl -o -name _SUCCESS \) -print -quit | grep -q .; then
        echo "legacy Search-R1 result contains artifacts without a contract" >&2
        exit 6
    fi
    contract_partial="${RESULT_ROOT_HOST}/.evaluation-contract.partial-${SLURM_JOB_ID}"
    cp -- "${desired_contract}" "${contract_partial}"
    mv -- "${contract_partial}" "${RESULT_ROOT_HOST}/evaluation-contract.env"
fi

export MODEL_PATH=/search-eval-model
export RESULT_ROOT=/results

mkdir -p "${OUTPUT_DIR}/evaluation/search_r1"
echo "Search-R1 ${EVAL_MODE} evaluation: ${BENCHMARKS}"
echo "checkpoint: ${MODEL_SOURCE}"
echo "results: ${RESULT_ROOT_HOST}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${MODEL_MOUNTS},${RESULT_ROOT_HOST}:/results" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/search_r1/common/run_measurement.sh
