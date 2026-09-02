#!/bin/bash
#PBS -N q3-4b-tool-call-pivot-eval
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=04:00:00
#PBS -j oe

# Held-out exact next-tool-action evaluation. No external model or sandbox API
# is used. TRAINING_HF_ROOT selects the newest complete periodic HF export.

set -euo pipefail

REPO_ROOT="$(realpath -- "${PBS_O_WORKDIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"
: "${RESULT_ROOT:?set RESULT_ROOT}"

VALIDATE_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/validate_checkpoint.py"
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    : "${TRAINING_HF_ROOT:?set CHECKPOINT_PATH or TRAINING_HF_ROOT}"
    TRAINING_HF_ROOT="$(realpath -- "${TRAINING_HF_ROOT}")"
    mapfile -t CANDIDATES < <(
        find "${TRAINING_HF_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\n' \
            | awk -F '\t' '$1 ~ /^[0-9]+$/ { print }' \
            | sort -t $'\t' -k1,1nr \
            | cut -f2-
    )
    for candidate in "${CANDIDATES[@]}"; do
        if python3 "${VALIDATE_TOOL}" --quiet "${candidate}"; then
            CHECKPOINT_PATH="${candidate}"
            break
        fi
    done
fi
: "${CHECKPOINT_PATH:?no complete checkpoint found}"
CHECKPOINT_PATH="$(realpath -- "${CHECKPOINT_PATH}")"
python3 "${VALIDATE_TOOL}" --quiet "${CHECKPOINT_PATH}"

RESULT_ROOT="$(realpath -m -- "${RESULT_ROOT}")"
mkdir -p "${RESULT_ROOT}" "${CACHE_DIR}/reasoning_eval/huggingface"
printf '%s\n' "${CHECKPOINT_PATH}" > "${RESULT_ROOT}/selected-checkpoint.txt"

VLLM_IMAGE="${REASONING_EVAL_CONTAINER_ROOT:-${CONTAINER_DIR}/reasoning-eval}/vllm-openai-v0.20.2-70a098d9.sif"
UNPAD_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/unpad_vocab.py"
MODEL_RUNTIME_DIR="${MILES_JOB_TMPDIR:-/tmp}/tool-call-eval-model-${MILES_JOB_ID}"
TOKENIZER_RUNTIME_DIR="${MILES_JOB_TMPDIR:-/tmp}/tool-call-eval-tokenizer-${MILES_JOB_ID}"
python3 "${UNPAD_TOOL}" "${CHECKPOINT_PATH}" "${MODEL_RUNTIME_DIR}"
if [[ -f "${MODEL_RUNTIME_DIR}/config.json" ]]; then
    MODEL_MOUNTS="${MODEL_RUNTIME_DIR}:/checkpoint,${CHECKPOINT_PATH}:${CHECKPOINT_PATH}:ro"
else
    MODEL_MOUNTS="${CHECKPOINT_PATH}:/checkpoint:ro"
fi
mkdir -p "${TOKENIZER_RUNTIME_DIR}"
cp -- "${CHECKPOINT_PATH}/chat_template.jinja" "${CHECKPOINT_PATH}/config.json" \
    "${CHECKPOINT_PATH}/tokenizer.json" "${TOKENIZER_RUNTIME_DIR}/"
extra_type="$(jq -r 'if has("extra_special_tokens") then (.extra_special_tokens | type) else "absent" end' "${CHECKPOINT_PATH}/tokenizer_config.json")"
if [[ "${extra_type}" == array ]]; then
    jq 'del(.extra_special_tokens)' "${CHECKPOINT_PATH}/tokenizer_config.json" \
        > "${TOKENIZER_RUNTIME_DIR}/tokenizer_config.json"
else
    cp -- "${CHECKPOINT_PATH}/tokenizer_config.json" "${TOKENIZER_RUNTIME_DIR}/"
fi

VLLM_PORT="${VLLM_PORT:-8000}"
TOOL_CALL_EVAL_LIMIT="${TOOL_CALL_EVAL_LIMIT:-}"
TOOL_CALL_EVAL_INPUT="${TOOL_CALL_EVAL_INPUT:-/data/nemotron-agentic-conv-tooluse-pivot/nemotron-agentic-conv-tooluse-pivot-heldout.jsonl}"
[[ -z "${TOOL_CALL_EVAL_LIMIT}" || "${TOOL_CALL_EVAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TOOL_CALL_EVAL_LIMIT must be a positive integer" >&2
    exit 1
}
SERVED_MODEL=tool-call-eval-qwen3-4b
VLLM_LOG="${RESULT_ROOT}/vllm-${MILES_JOB_ID}.log"
miles_srun --overlap --exact --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=96 \
    --image="${VLLM_IMAGE}" \
    --bind="${MODEL_MOUNTS},${TOKENIZER_RUNTIME_DIR}:/runtime-tokenizer:ro,${CACHE_DIR}/reasoning_eval:/evaluation-cache" \
    --cwd=/ --no-home \
    --export="ALL,HF_HOME=/evaluation-cache/huggingface,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,NCCL_NVLS_ENABLE=0,VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    bash -lc '
exec vllm serve /checkpoint --tokenizer /runtime-tokenizer --host 0.0.0.0 --port "$1" \
    --served-model-name "$2" --tensor-parallel-size 1 --data-parallel-size 8 --dtype bfloat16 \
    --max-model-len 49152 --generation-config /checkpoint --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --seed 1234 --reasoning-parser qwen3 --tool-call-parser hermes \
    --enable-auto-tool-choice --enable-prefix-caching
' _ "${VLLM_PORT}" "${SERVED_MODEL}" > "${VLLM_LOG}" 2>&1 &
SERVER_PID=$!
stop_server() {
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap stop_server EXIT INT TERM
for _ in $(seq 1 180); do
    curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 120 "${VLLM_LOG}" >&2; exit 1; }
    sleep 5
done
curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null

miles_srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},${RESULT_ROOT}:/results" \
    --writable-tmpfs --no-home --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
limit_args=()
[[ -z "$4" ]] || limit_args=(--limit "$4")
python3 -m experiments.src.evaluators.tool_call_pivot \
    --input "$5" \
    --output /results/results.jsonl \
    --summary /results/summary.json \
    --endpoint "http://127.0.0.1:$1/v1/chat/completions" \
    --model "$2" \
    --max-response-tokens 16384 \
    --concurrency "$3" \
    "${limit_args[@]}"
' _ "${VLLM_PORT}" "${SERVED_MODEL}" "${TOOL_CALL_EVAL_CONCURRENCY:-32}" \
    "${TOOL_CALL_EVAL_LIMIT}" "${TOOL_CALL_EVAL_INPUT}"
touch "${RESULT_ROOT}/_SUCCESS"

stop_server
trap - EXIT INT TERM
echo "tool-call downstream evaluation completed under ${RESULT_ROOT}"
