#!/bin/bash
#PBS -N q3-4b-tau3-eval
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=04:00:00
#PBS -j oe

# Held-out Tau v3 v1.0.1 evaluation for checkpoints trained on the external
# AReaL Tau2 RL split. Official Tau v3 train/base tasks are not accepted by
# this runner.

set -euo pipefail

REPO_ROOT="$(realpath -- "${PBS_O_WORKDIR:?submit from the Miles repository root}")"
export WANDB_MODE="${WANDB_MODE:-offline}"
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/load_job_env.sh"
load_job_env \
    "${REPO_ROOT}/.env" \
    NVIDIA_INFERENCE_API_KEY \
    NVIDIA_INFERENCE_BASE_URL \
    TAU_USER_MODEL \
    TAU_USER_PROVIDER \
    GEMINI_API_KEY
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -r "${DOTENV_MASK}" ]] || { echo "dotenv mask not found: ${DOTENV_MASK}" >&2; exit 1; }
: "${RESULT_ROOT:?set RESULT_ROOT}"

: "${TAU_USER_PROVIDER:=nvidia}"
: "${TAU_USER_REQUEST_TIMEOUT:=120}"
: "${TAU_USER_MAX_RETRIES:=4}"
: "${TAU_AGENT_EMPTY_RESPONSE_RETRIES:=2}"
: "${TAU_AGENT_ENABLE_THINKING:=true}"
readonly VERIFIED_GEMINI_USER_MODEL=gemini-2.5-flash-lite
readonly VERIFIED_NVIDIA_USER_MODEL=us/gcp/google/eccn-gemini-2.5-flash-lite
case "${TAU_USER_PROVIDER}" in
    gemini)
        : "${TAU_USER_MODEL:=${VERIFIED_GEMINI_USER_MODEL}}"
        [[ "${TAU_USER_MODEL}" == "${VERIFIED_GEMINI_USER_MODEL}" ]] || {
            echo "Tau Gemini evaluation requires ${VERIFIED_GEMINI_USER_MODEL}" >&2
            exit 1
        }
        : "${GEMINI_API_KEY:?set GEMINI_API_KEY in the repository .env or job environment}"
        TAU_SECRET_NAME=GEMINI_API_KEY
        ;;
    nvidia)
        : "${TAU_USER_MODEL:=${VERIFIED_NVIDIA_USER_MODEL}}"
        [[ "${TAU_USER_MODEL}" == "${VERIFIED_NVIDIA_USER_MODEL}" ]] || {
            echo "Tau NVIDIA evaluation requires ${VERIFIED_NVIDIA_USER_MODEL}" >&2
            exit 1
        }
        : "${NVIDIA_INFERENCE_API_KEY:?set NVIDIA_INFERENCE_API_KEY in the repository .env or job environment}"
        TAU_SECRET_NAME=NVIDIA_INFERENCE_API_KEY
        ;;
    *)
        echo "unknown TAU_USER_PROVIDER=${TAU_USER_PROVIDER}; expected gemini or nvidia" >&2
        exit 1
        ;;
esac
[[ "${TAU_AGENT_ENABLE_THINKING}" == true || "${TAU_AGENT_ENABLE_THINKING}" == false ]] || {
    echo "TAU_AGENT_ENABLE_THINKING must be true or false" >&2
    exit 1
}

TAU3_CONTAINER_IMAGE="${TAU3_CONTAINER_IMAGE:-${CONTAINER_IMAGE}}"
[[ -r "${TAU3_CONTAINER_IMAGE}" ]] || { echo "Tau three image not found: ${TAU3_CONTAINER_IMAGE}" >&2; exit 1; }

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
MODEL_RUNTIME_DIR="${MILES_JOB_TMPDIR:-/tmp}/tau3-eval-model-${MILES_JOB_ID}"
TOKENIZER_RUNTIME_DIR="${MILES_JOB_TMPDIR:-/tmp}/tau3-eval-tokenizer-${MILES_JOB_ID}"
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
SERVED_MODEL=tau3-eval-qwen3-4b
VLLM_LOG="${RESULT_ROOT}/vllm-${MILES_JOB_ID}.log"
miles_srun --overlap --exact --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=96 \
    --image="${VLLM_IMAGE}" \
    --bind="${MODEL_MOUNTS},${TOKENIZER_RUNTIME_DIR}:/runtime-tokenizer:ro,${CACHE_DIR}/reasoning_eval:/evaluation-cache" \
    --cwd=/ --no-home \
    --export="USER,HF_HOME=/evaluation-cache/huggingface,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,NCCL_NVLS_ENABLE=0,VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    bash -lc '
exec vllm serve /checkpoint --tokenizer /runtime-tokenizer --host 0.0.0.0 --port "$1" \
    --served-model-name "$2" --tensor-parallel-size 1 --data-parallel-size 8 --dtype bfloat16 \
    --max-model-len 32768 --generation-config /checkpoint --gpu-memory-utilization 0.90 \
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

TASK_SPLITS="${TASK_SPLITS:-test}"
[[ -z "${TAU_EVAL_LIMIT:-}" || "${TAU_EVAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TAU_EVAL_LIMIT must be a positive integer" >&2
    exit 1
}
for task_split in ${TASK_SPLITS}; do
    case "${task_split}" in
        test) input=/data/tau-bench/tau3-test-miles.jsonl ;;
        retail_test) input=/data/tau-bench/tau3-retail-test-miles.jsonl ;;
        airline_test) input=/data/tau-bench/tau3-airline-test-miles.jsonl ;;
        telecom_test) input=/data/tau-bench/tau3-telecom-test-miles.jsonl ;;
        *) echo "unsupported Tau three split ${task_split}" >&2; exit 1 ;;
    esac
    mkdir -p "${RESULT_ROOT}/${task_split}"
    limit_args=()
    [[ -z "${TAU_EVAL_LIMIT:-}" ]] || limit_args=(--limit "${TAU_EVAL_LIMIT}")
    thinking_arg=--no-agent-enable-thinking
    [[ "${TAU_AGENT_ENABLE_THINKING}" == true ]] && thinking_arg=--agent-enable-thinking
    export TAU_USER_PROVIDER TAU_USER_MODEL TAU_USER_REQUEST_TIMEOUT TAU_USER_MAX_RETRIES
    export NVIDIA_INFERENCE_BASE_URL
    TAU_EVAL_EXPORTS="USER,TAU_USER_PROVIDER,TAU_USER_MODEL,TAU_USER_REQUEST_TIMEOUT,TAU_USER_MAX_RETRIES,NVIDIA_INFERENCE_BASE_URL,${TAU_SECRET_NAME}"
    miles_srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=32 \
        --image="${TAU3_CONTAINER_IMAGE}" \
        --bind="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro,${RESULT_ROOT}:/results" \
        --cwd=/root/miles --writable-tmpfs --no-home \
        --export="${TAU_EVAL_EXPORTS}" \
        /opt/tau3/.venv/bin/python -m experiments.src.evaluators.tau_bench \
            --input "${input}" \
            --output "/results/${task_split}/results.jsonl" \
            --summary "/results/${task_split}/summary.json" \
            --endpoint "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
            --model "${SERVED_MODEL}" \
            --repeats "${TAU_EVAL_REPEATS:-1}" \
            --concurrency "${TAU_EVAL_CONCURRENCY:-32}" \
            --max-response-tokens 16384 \
            --agent-empty-response-retries "${TAU_AGENT_EMPTY_RESPONSE_RETRIES}" \
            "${thinking_arg}" \
            --user-provider "${TAU_USER_PROVIDER}" \
            --user-model "${TAU_USER_MODEL}" \
            --user-request-timeout "${TAU_USER_REQUEST_TIMEOUT}" \
            --user-max-retries "${TAU_USER_MAX_RETRIES}" \
            "${limit_args[@]}"
    touch "${RESULT_ROOT}/${task_split}/_SUCCESS"
done

stop_server
trap - EXIT INT TERM
echo "Tau three downstream evaluation completed under ${RESULT_ROOT}"
