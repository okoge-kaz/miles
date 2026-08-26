#!/bin/bash
#SBATCH --job-name=q3-4b-tau-downstream-eval
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/domain_eval/%x-%j.log

# Held-out Tau v1 retail/airline evaluation. The agent runs on the selected
# local checkpoint. TAU_USER_BACKEND selects either that same local checkpoint
# or Gemini 2.5 Flash-Lite as the user simulator.

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
export WANDB_MODE="${WANDB_MODE:-offline}"
source "${REPO_ROOT}/experiments/env.sh"
: "${RESULT_ROOT:?set RESULT_ROOT}"

: "${TAU_USER_BACKEND:=local-policy}"
: "${TAU_USER_MODEL:=gemini-2.5-flash-lite}"
: "${TAU_USER_MODEL_PROVIDER:=gemini}"
: "${TAU_USER_REQUEST_TIMEOUT:=120}"
: "${TAU_USER_MAX_RETRIES:=4}"
: "${TAU_USER_RETRY_BACKOFF:=1}"
case "${TAU_USER_BACKEND}" in
    local-policy) ;;
    gemini)
        [[ "${TAU_USER_MODEL_PROVIDER}" == gemini ]] || {
            echo "TAU_USER_MODEL_PROVIDER must be gemini for TAU_USER_BACKEND=gemini" >&2
            exit 1
        }
        [[ "${TAU_USER_MODEL}" =~ ^[A-Za-z0-9._-]+$ ]] || {
            echo "invalid TAU_USER_MODEL=${TAU_USER_MODEL}" >&2
            exit 1
        }
        : "${GEMINI_API_KEY:?export GEMINI_API_KEY in the job environment; .env files are not read}"
        export GEMINI_API_KEY
        ;;
    *)
        echo "unknown TAU_USER_BACKEND=${TAU_USER_BACKEND}; expected local-policy or gemini" >&2
        exit 1
        ;;
esac
export TAU_USER_BACKEND TAU_USER_MODEL TAU_USER_MODEL_PROVIDER
export TAU_USER_REQUEST_TIMEOUT TAU_USER_MAX_RETRIES TAU_USER_RETRY_BACKOFF WANDB_MODE

TAU_COMMIT=09c26a85efd1d65168cfb57865ca2ca278c8153d
TAU_SOURCE_DIR="${TAU_SOURCE_DIR:-${WS}/src/tau-bench}"
[[ "$(git -C "${TAU_SOURCE_DIR}" rev-parse HEAD)" == "${TAU_COMMIT}" ]] || {
    echo "Tau source is not pinned to ${TAU_COMMIT}" >&2
    exit 1
}

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

VLLM_IMAGE="${SHARED_WS}/containers/vllm-openai-v0.20.2-70a098d9.sqsh"
UNPAD_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/unpad_vocab.py"
MODEL_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/tau-eval-model-${SLURM_JOB_ID}"
TOKENIZER_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/tau-eval-tokenizer-${SLURM_JOB_ID}"
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
SERVED_MODEL=tau-eval-qwen3-4b
VLLM_LOG="${RESULT_ROOT}/vllm-${SLURM_JOB_ID}.log"
srun --overlap --exact --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=96 \
    --container-image="${VLLM_IMAGE}" \
    --container-mounts="${MODEL_MOUNTS},${TOKENIZER_RUNTIME_DIR}:/runtime-tokenizer:ro,${CACHE_DIR}/reasoning_eval:/evaluation-cache" \
    --container-workdir=/ --no-container-mount-home \
    --export="ALL,HF_HOME=/evaluation-cache/huggingface,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,NCCL_NVLS_ENABLE=0,VLLM_WORKER_MULTIPROC_METHOD=spawn" \
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

TASK_SPLITS="${TASK_SPLITS:-retail_test airline_test}"
TAU_EVAL_LIMIT="${TAU_EVAL_LIMIT:-}"
[[ -z "${TAU_EVAL_LIMIT}" || "${TAU_EVAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TAU_EVAL_LIMIT must be a positive integer" >&2
    exit 1
}
for task_split in ${TASK_SPLITS}; do
    case "${task_split}" in
        retail_test) input=/data/tau-bench/tau1-retail-test-miles.jsonl ;;
        airline_test) input=/data/tau-bench/tau1-airline-test-miles.jsonl ;;
        *) echo "unsupported Tau split ${task_split}" >&2; exit 1 ;;
    esac
    mkdir -p "${RESULT_ROOT}/${task_split}"
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=32 \
        --container-image="${SQSH_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS},${TAU_SOURCE_DIR}:/root/tau-bench,${RESULT_ROOT}:/results" \
        --container-writable --no-container-mount-home --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/tau-bench:/root/miles
limit_args=()
[[ -z "$7" ]] || limit_args=(--limit "$7")
python3 -m experiments.src.evaluators.tau_bench \
    --input "$1" --output "/results/$2/results.jsonl" --summary "/results/$2/summary.json" \
    --endpoint "http://127.0.0.1:$3/v1/chat/completions" --model "$4" \
    --repeats "$5" --max-response-tokens 16384 --concurrency "$6" \
    --user-backend "${TAU_USER_BACKEND}" --user-model "${TAU_USER_MODEL}" \
    --user-request-timeout "${TAU_USER_REQUEST_TIMEOUT}" \
    --user-max-retries "${TAU_USER_MAX_RETRIES}" \
    --user-retry-backoff "${TAU_USER_RETRY_BACKOFF}" "${limit_args[@]}"
' _ "${input}" "${task_split}" "${VLLM_PORT}" "${SERVED_MODEL}" \
            "${TAU_EVAL_REPEATS:-1}" "${TAU_EVAL_CONCURRENCY:-32}" "${TAU_EVAL_LIMIT}"
    touch "${RESULT_ROOT}/${task_split}/_SUCCESS"
done

stop_server
trap - EXIT INT TERM
echo "Tau downstream evaluation completed under ${RESULT_ROOT}"
