#!/bin/bash

#SBATCH --job-name=q3-4b-rl-eval
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=experiments/outputs/reasoning_eval/%x-%j.log

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"

: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to a Qwen3-4B Hugging Face checkpoint}"
: "${RESULT_ROOT:?set RESULT_ROOT to the checkpoint evaluation result directory}"

TASKS_TEXT="${TASKS:-aime24 aime25 aime26}"
EVAL_MODE="${EVAL_MODE:-full}"
PROTOCOL_NAME="${PROTOCOL_NAME:-eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-thinking-t0.6-p0.95-k20-aime64-v1}"
ARM_NAME="${ARM_NAME:-unknown}"
TRAINING_STEP="${TRAINING_STEP:-unknown}"
RUN_NAMESPACE="${RUN_NAMESPACE:-unknown}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
PROMPT_TOKEN_RESERVE="${PROMPT_TOKEN_RESERVE:-4096}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
AIME_REPEATS="${AIME_REPEATS:-64}"
AIME_PARALLELISM="${AIME_PARALLELISM:-1024}"
SMOKE_LIMIT_SAMPLES="${SMOKE_LIMIT_SAMPLES:-2}"
MAX_RETRIES="${MAX_RETRIES:-10}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-3600}"
VLLM_SEED="${VLLM_SEED:-1234}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_STARTUP_ATTEMPTS="${VLLM_STARTUP_ATTEMPTS:-3}"
EVALUATOR_ATTEMPTS="${EVALUATOR_ATTEMPTS:-3}"
VLLM_ENGINE_READY_TIMEOUT_SECONDS="${VLLM_ENGINE_READY_TIMEOUT_SECONDS:-2400}"
SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-540}"
SERVER_READY_INTERVAL_SECONDS="${SERVER_READY_INTERVAL_SECONDS:-5}"
VLLM_PREFLIGHT_MAX_TOKENS="${VLLM_PREFLIGHT_MAX_TOKENS:-512}"
VLLM_TRITON_PTXAS_PATH="${VLLM_TRITON_PTXAS_PATH:-/usr/local/cuda-13.0/bin/ptxas}"
CONTAINER_ROOT="${REASONING_EVAL_CONTAINER_ROOT:-${SHARED_WS}/containers}"
VLLM_IMAGE="${CONTAINER_ROOT}/vllm-openai-v0.20.2-70a098d9.sqsh"
NEMO_SKILLS_IMAGE="${CONTAINER_ROOT}/nemo-evaluator-nemo-skills-26.03-ac1b048e.sqsh"
VLLM_OCI_DIGEST="sha256:70a098d90dbab428a001d9e852fc0fc8d67da5beb03e7851a22247653bf35923"
NEMO_SKILLS_OCI_DIGEST="sha256:ac1b048e13fe7f2a59751b528fc23f5f471452197ad9ae40b715a77cda0a9612"
NEMO_SKILLS_DATA_ROOT="${REASONING_EVAL_DATA_ROOT:-${DATASET_DIR}/evaluation}/nemo-skills-26.03"
EVALUATION_CACHE_ROOT="${REASONING_EVAL_CACHE_ROOT:-${CACHE_DIR}/reasoning_eval}"
UNPAD_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/unpad_vocab.py"
VALIDATE_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/validate_checkpoint.py"
EXPORT_TOOL_DIR="${REPO_ROOT}/experiments/tools/reasoning_eval"
VLLM_RUNTIME_HOOKS="${EXPORT_TOOL_DIR}/vllm_runtime_hooks"
API_BASE_URL="http://127.0.0.1:${VLLM_PORT}"
API_ENDPOINT="${API_BASE_URL}/v1/chat/completions"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "submit this script with sbatch" >&2; exit 2; }
[[ "${EVAL_MODE}" == smoke || "${EVAL_MODE}" == full ]] || {
    echo "EVAL_MODE must be smoke or full" >&2
    exit 3
}
[[ "${PROTOCOL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "PROTOCOL_NAME contains unsupported characters" >&2
    exit 4
}
[[ "${ARM_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ARM_NAME contains unsupported characters" >&2; exit 5; }
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "RUN_NAMESPACE contains unsupported characters" >&2
    exit 5
}
[[ "${TRAINING_STEP}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAINING_STEP must be a positive integer" >&2
    exit 5
}
for integer_value in \
    "${CONTEXT_LENGTH}" "${PROMPT_TOKEN_RESERVE}" "${TOP_K}" "${AIME_REPEATS}" \
    "${AIME_PARALLELISM}" "${SMOKE_LIMIT_SAMPLES}" "${MAX_RETRIES}" \
    "${REQUEST_TIMEOUT_SECONDS}" "${VLLM_SEED}" "${VLLM_PORT}" \
    "${VLLM_MAX_NUM_SEQS}" "${VLLM_STARTUP_ATTEMPTS}" "${EVALUATOR_ATTEMPTS}" \
    "${VLLM_ENGINE_READY_TIMEOUT_SECONDS}" "${SERVER_READY_ATTEMPTS}" \
    "${SERVER_READY_INTERVAL_SECONDS}" "${VLLM_PREFLIGHT_MAX_TOKENS}"; do
    [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "integer evaluation controls must be positive: ${integer_value}" >&2
        exit 5
    }
done
for decimal_value in "${TEMPERATURE}" "${TOP_P}" "${VLLM_GPU_MEMORY_UTILIZATION}"; do
    [[ "${decimal_value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
        echo "floating-point evaluation controls must be nonnegative decimals" >&2
        exit 6
    }
done
awk -v value="${TOP_P}" 'BEGIN { exit !(value > 0 && value <= 1) }' || {
    echo "TOP_P must be in (0, 1]" >&2
    exit 6
}
awk -v value="${VLLM_GPU_MEMORY_UTILIZATION}" 'BEGIN { exit !(value > 0 && value < 1) }' || {
    echo "VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1)" >&2
    exit 6
}
(( PROMPT_TOKEN_RESERVE < CONTEXT_LENGTH )) || {
    echo "PROMPT_TOKEN_RESERVE must be smaller than CONTEXT_LENGTH" >&2
    exit 7
}
(( VLLM_PORT <= 65535 )) || { echo "VLLM_PORT must be at most 65535" >&2; exit 8; }
MAX_NEW_TOKENS=$((CONTEXT_LENGTH - PROMPT_TOKEN_RESERVE))

CHECKPOINT_SOURCE="$(realpath -- "${CHECKPOINT_PATH}")"
RESULT_ROOT="$(realpath -m -- "${RESULT_ROOT}")"
for path_value in "${CHECKPOINT_SOURCE}" "${RESULT_ROOT}" "${EVALUATION_CACHE_ROOT}"; do
    [[ "${path_value}" != *","* && "${path_value}" != *$'\n'* ]] || {
        echo "container-mounted paths may not contain commas or newlines: ${path_value}" >&2
        exit 9
    }
done

[[ -f "${VLLM_IMAGE}" ]] || { echo "vLLM image not found: ${VLLM_IMAGE}" >&2; exit 10; }
[[ -f "${NEMO_SKILLS_IMAGE}" ]] || {
    echo "NeMo Skills image not found: ${NEMO_SKILLS_IMAGE}" >&2
    exit 11
}
[[ -f "${VLLM_IMAGE}.oci-digest" && "$(< "${VLLM_IMAGE}.oci-digest")" == "${VLLM_OCI_DIGEST}" ]] || {
    echo "vLLM image OCI provenance is missing or mismatched" >&2
    exit 11
}
[[ -f "${NEMO_SKILLS_IMAGE}.oci-digest" \
    && "$(< "${NEMO_SKILLS_IMAGE}.oci-digest")" == "${NEMO_SKILLS_OCI_DIGEST}" ]] || {
    echo "NeMo Skills image OCI provenance is missing or mismatched" >&2
    exit 11
}
[[ -f "${NEMO_SKILLS_DATA_ROOT}/_PREPARED" ]] || {
    echo "prepared AIME data not found: ${NEMO_SKILLS_DATA_ROOT}/_PREPARED" >&2
    exit 12
}
[[ -r "${VLLM_RUNTIME_HOOKS}/sitecustomize.py" ]] || {
    echo "vLLM runtime hook is missing: ${VLLM_RUNTIME_HOOKS}/sitecustomize.py" >&2
    exit 13
}
grep -Fx "nemo_skills_image=${NEMO_SKILLS_IMAGE}" "${NEMO_SKILLS_DATA_ROOT}/_PREPARED" >/dev/null || {
    echo "AIME data was prepared with a different NeMo Skills image" >&2
    exit 13
}
for checkpoint_file in \
    config.json tokenizer.json tokenizer_config.json generation_config.json \
    chat_template.jinja model.safetensors.index.json; do
    [[ -f "${CHECKPOINT_SOURCE}/${checkpoint_file}" ]] || {
        echo "checkpoint file not found: ${CHECKPOINT_SOURCE}/${checkpoint_file}" >&2
        exit 14
    }
done
[[ "$(jq -r '.model_type // empty' "${CHECKPOINT_SOURCE}/config.json")" == qwen3 ]] || {
    echo "checkpoint is not Qwen3: ${CHECKPOINT_SOURCE}" >&2
    exit 15
}
CHECKPOINT_CONTEXT_LENGTH="$(jq -r '.max_position_embeddings // empty' "${CHECKPOINT_SOURCE}/config.json")"
[[ "${CHECKPOINT_CONTEXT_LENGTH}" =~ ^[1-9][0-9]*$ ]] || {
    echo "checkpoint max_position_embeddings is missing" >&2
    exit 16
}
(( CONTEXT_LENGTH <= CHECKPOINT_CONTEXT_LENGTH )) || {
    echo "CONTEXT_LENGTH exceeds checkpoint max_position_embeddings" >&2
    exit 17
}
mapfile -t CHECKPOINT_SHARDS < <(
    jq -r '.weight_map[]' "${CHECKPOINT_SOURCE}/model.safetensors.index.json" | sort -u
)
for checkpoint_shard in "${CHECKPOINT_SHARDS[@]}"; do
    [[ -s "${CHECKPOINT_SOURCE}/${checkpoint_shard}" ]] || {
        echo "checkpoint shard not found: ${CHECKPOINT_SOURCE}/${checkpoint_shard}" >&2
        exit 18
    }
done
python3 "${VALIDATE_TOOL}" --quiet "${CHECKPOINT_SOURCE}"

read -r -a REQUESTED_TASKS <<< "${TASKS_TEXT}"
(( ${#REQUESTED_TASKS[@]} > 0 )) || { echo "TASKS is empty" >&2; exit 19; }
declare -A SEEN_TASKS=()
for task in "${REQUESTED_TASKS[@]}"; do
    [[ "${task}" =~ ^aime(24|25|26)$ ]] || { echo "unsupported task: ${task}" >&2; exit 20; }
    [[ -z "${SEEN_TASKS[${task}]:-}" ]] || { echo "duplicate task: ${task}" >&2; exit 21; }
    SEEN_TASKS["${task}"]=1
done

mkdir -p "${RESULT_ROOT}" "${EVALUATION_CACHE_ROOT}/huggingface"
exec 9> "${RESULT_ROOT}/.checkpoint.lock"
if ! flock --nonblock 9; then
    echo "another evaluation job is active for ${RESULT_ROOT}; nothing to do"
    exit 0
fi
printf '%s\n' "${SLURM_JOB_ID}" > "${RESULT_ROOT}/.active-job"
trap 'rm -f -- "${RESULT_ROOT}/.active-job"' EXIT

declare -a PENDING_TASKS=()
for task in "${REQUESTED_TASKS[@]}"; do
    final_result="${RESULT_ROOT}/${task}"
    partial_result="${final_result}.partial"
    if [[ -f "${final_result}/_SUCCESS" ]]; then
        echo "Already complete: ${final_result}"
        continue
    fi
    [[ ! -e "${final_result}" ]] || {
        echo "final result exists without _SUCCESS: ${final_result}" >&2
        exit 22
    }
    if [[ -f "${partial_result}/_SUCCESS" \
        && -s "${partial_result}/model-outputs.jsonl" \
        && -s "${partial_result}/evaluator/eval-results/${task}/metrics.json" ]]; then
        mv -- "${partial_result}" "${final_result}"
        echo "Finalized completed partial result: ${final_result}"
        continue
    fi
    PENDING_TASKS+=("${task}")
done
if (( ${#PENDING_TASKS[@]} == 0 )); then
    rm -f -- "${RESULT_ROOT}/.active-job"
    echo "All requested tasks are already complete"
    exit 0
fi

JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp}"
MODEL_RUNTIME_DIR="${JOB_TMP_ROOT}/reasoning-eval-model-${SLURM_JOB_ID}"
TOKENIZER_RUNTIME_DIR="${JOB_TMP_ROOT}/reasoning-eval-tokenizer-${SLURM_JOB_ID}"
VLLM_IMAGE_CACHE_DIR="${JOB_TMP_ROOT}/reasoning-eval-images"
VLLM_RUNTIME_IMAGE="${VLLM_IMAGE_CACHE_DIR}/$(basename -- "${VLLM_IMAGE}")"
stage_node_local_image() {
    local source="$1"
    local destination="$2"
    local source_size temporary

    mkdir -p "$(dirname -- "${destination}")"
    source_size="$(stat -c '%s' "${source}")"
    (
        flock --exclusive 8
        if [[ -s "${destination}" \
            && "$(stat -c '%s' "${destination}")" == "${source_size}" ]]; then
            exit 0
        fi
        temporary="${destination}.partial-${SLURM_JOB_ID}"
        rm -f -- "${temporary}"
        cp --reflink=auto -- "${source}" "${temporary}"
        [[ "$(stat -c '%s' "${temporary}")" == "${source_size}" ]] || {
            echo "staged image size mismatch: ${source}" >&2
            exit 1
        }
        mv -- "${temporary}" "${destination}"
    ) 8> "${destination}.lock"
}
cleanup_runtime_dirs() {
    rm -rf -- "${MODEL_RUNTIME_DIR}" "${TOKENIZER_RUNTIME_DIR}"
}
cleanup_before_server() {
    status=$?
    trap - EXIT
    cleanup_runtime_dirs
    rm -f -- "${RESULT_ROOT}/.active-job"
    exit "${status}"
}
trap cleanup_before_server EXIT
stage_node_local_image "${VLLM_IMAGE}" "${VLLM_RUNTIME_IMAGE}"
checkpoint_config_sha256="$(sha256sum "${CHECKPOINT_SOURCE}/config.json")"
checkpoint_config_sha256="${checkpoint_config_sha256%% *}"
tokenizer_config_sha256="$(sha256sum "${CHECKPOINT_SOURCE}/tokenizer_config.json")"
tokenizer_config_sha256="${tokenizer_config_sha256%% *}"
desired_server_config="${JOB_TMP_ROOT}/reasoning-eval-server-${SLURM_JOB_ID}.env"
cat > "${desired_server_config}" <<EOF
checkpoint=${CHECKPOINT_SOURCE}
checkpoint_config_sha256=${checkpoint_config_sha256}
tokenizer_config_sha256=${tokenizer_config_sha256}
context_length=${CONTEXT_LENGTH}
vllm_seed=${VLLM_SEED}
vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}
vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS}
vllm_reasoning_parser=qwen3
vllm_tensor_parallel_size=1
vllm_data_parallel_size=8
vllm_triton_ptxas_path=${VLLM_TRITON_PTXAS_PATH}
vllm_image=${VLLM_IMAGE}
vllm_oci_digest=${VLLM_OCI_DIGEST}
nemo_skills_image=${NEMO_SKILLS_IMAGE}
nemo_skills_oci_digest=${NEMO_SKILLS_OCI_DIGEST}
EOF
if [[ -f "${RESULT_ROOT}/server-config.env" ]]; then
    cmp --silent "${desired_server_config}" "${RESULT_ROOT}/server-config.env" || {
        diff --unified "${RESULT_ROOT}/server-config.env" "${desired_server_config}" || true
        echo "refusing to resume with a changed model-serving configuration" >&2
        exit 23
    }
else
    cp -- "${desired_server_config}" "${RESULT_ROOT}/server-config.env"
fi
python3 "${UNPAD_TOOL}" "${CHECKPOINT_SOURCE}" "${MODEL_RUNTIME_DIR}"
if [[ -f "${MODEL_RUNTIME_DIR}/config.json" ]]; then
    MODEL_MOUNTS="${MODEL_RUNTIME_DIR}:/checkpoint:ro"
    VOCAB_COMPAT_ACTION="removed-megatron-vocab-padding"
else
    MODEL_MOUNTS="${CHECKPOINT_SOURCE}:/checkpoint:ro"
    VOCAB_COMPAT_ACTION="none"
fi

mkdir -p "${TOKENIZER_RUNTIME_DIR}"
cp -- \
    "${CHECKPOINT_SOURCE}/chat_template.jinja" \
    "${CHECKPOINT_SOURCE}/config.json" \
    "${CHECKPOINT_SOURCE}/tokenizer.json" \
    "${TOKENIZER_RUNTIME_DIR}/"
TOKENIZER_EXTRA_TYPE="$(
    jq -r 'if has("extra_special_tokens") then (.extra_special_tokens | type) else "absent" end' \
        "${CHECKPOINT_SOURCE}/tokenizer_config.json"
)"
case "${TOKENIZER_EXTRA_TYPE}" in
    array)
        jq 'del(.extra_special_tokens)' \
            "${CHECKPOINT_SOURCE}/tokenizer_config.json" > "${TOKENIZER_RUNTIME_DIR}/tokenizer_config.json"
        TOKENIZER_COMPAT_ACTION="removed-legacy-extra-special-tokens-list"
        ;;
    absent|object)
        cp -- "${CHECKPOINT_SOURCE}/tokenizer_config.json" "${TOKENIZER_RUNTIME_DIR}/tokenizer_config.json"
        TOKENIZER_COMPAT_ACTION="none"
        ;;
    *)
        echo "unsupported tokenizer extra_special_tokens type: ${TOKENIZER_EXTRA_TYPE}" >&2
        exit 23
        ;;
esac

if [[ "${EVAL_MODE}" == smoke ]]; then
    NUM_REPEATS=1
    LIMIT_SAMPLES_YAML="${SMOKE_LIMIT_SAMPLES}"
    EXPECTED_OUTPUT_RECORDS="${SMOKE_LIMIT_SAMPLES}"
else
    NUM_REPEATS="${AIME_REPEATS}"
    LIMIT_SAMPLES_YAML=null
    EXPECTED_OUTPUT_RECORDS=$((30 * AIME_REPEATS))
fi

for task in "${PENDING_TASKS[@]}"; do
    partial_result="${RESULT_ROOT}/${task}.partial"
    desired_config="${JOB_TMP_ROOT}/reasoning-eval-${SLURM_JOB_ID}-${task}.yaml"
    case "${task}" in
        aime24) eval_type=nemo_skills.ns_aime2024 ;;
        aime25) eval_type=nemo_skills.ns_aime2025 ;;
        aime26) eval_type=nemo_skills.ns_aime2026 ;;
    esac
    mkdir -p "${partial_result}"
    cat > "${desired_config}" <<EOF
config:
  type: ${eval_type}
  output_dir: /results/evaluator
  params:
    limit_samples: ${LIMIT_SAMPLES_YAML}
    max_new_tokens: ${MAX_NEW_TOKENS}
    max_retries: ${MAX_RETRIES}
    parallelism: ${AIME_PARALLELISM}
    request_timeout: ${REQUEST_TIMEOUT_SECONDS}
    temperature: ${TEMPERATURE}
    top_p: ${TOP_P}
    extra:
      num_repeats: ${NUM_REPEATS}
      prompt_config: /nemo_run/code/eval_factory_prompts/math-oai.yaml
      skip_data_dir_check: true
target:
  api_endpoint:
    api_key_name: null
    model_id: qwen3-4b-rl-step-${TRAINING_STEP}
    type: chat
    url: ${API_ENDPOINT}
    adapter_config:
      params_to_add:
        chat_template_kwargs:
          enable_thinking: true
        skip_special_tokens: false
        top_k: ${TOP_K}
      use_caching: true
      caching_dir: /results/adapter-cache
      generate_html_report: false
      save_requests: true
      save_responses: true
      tracking_requests_stats: true
      log_failed_requests: true
      use_request_logging: true
      max_logged_requests: 20
      use_response_logging: true
      max_logged_responses: 20
EOF
    if [[ -f "${partial_result}/run-config.yaml" ]]; then
        cmp --silent "${desired_config}" "${partial_result}/run-config.yaml" || {
            diff --unified "${partial_result}/run-config.yaml" "${desired_config}" || true
            echo "refusing to reuse ${task} cache with a changed evaluation protocol" >&2
            exit 24
        }
    else
        cp -- "${desired_config}" "${partial_result}/run-config.yaml"
    fi
    srun \
        --exact \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=32 \
        --mem=128G \
        --container-image="${NEMO_SKILLS_IMAGE}" \
        --container-mounts="${partial_result}:/results,${NEMO_SKILLS_DATA_ROOT}:/evaluation-data,${EVALUATION_CACHE_ROOT}:/evaluation-cache" \
        --container-workdir=/results \
        --no-container-mount-home \
        --export="ALL,HF_HOME=/evaluation-cache/huggingface,NEMO_SKILLS_DATA_DIR=/evaluation-data,NEMO_EVALUATOR_TELEMETRY_LEVEL=0,XDG_CACHE_HOME=/evaluation-cache/xdg,PYTHONUNBUFFERED=1" \
        bash -lc 'nemo-evaluator run_eval --run_config /results/run-config.yaml --dry_run' \
        > "${partial_result}/rendered-config-${SLURM_JOB_ID}.txt"
done

SERVER_STEP_PID=""
stop_server() {
    if [[ -n "${SERVER_STEP_PID}" ]]; then
        if kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
            kill "${SERVER_STEP_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_STEP_PID}" 2>/dev/null || true
    fi
    SERVER_STEP_PID=""
}
cleanup() {
    status=$?
    trap - EXIT INT TERM
    stop_server
    cleanup_runtime_dirs
    rm -f -- "${RESULT_ROOT}/.active-job"
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' INT TERM

SERVED_MODEL_NAME="qwen3-4b-rl-step-${TRAINING_STEP}"
server_ready=false
VLLM_LOG=""
for ((startup_attempt = 1; startup_attempt <= VLLM_STARTUP_ATTEMPTS; startup_attempt++)); do
    VLLM_LOG="${RESULT_ROOT}/vllm-${SLURM_JOB_ID}-attempt-${startup_attempt}.log"
    # An overlapping step otherwise receives the cluster's 2 GiB step default,
    # even though the eight-GPU job owns the node's full memory allocation.  It
    # must retain the warmed container pages and all eight worker processes.
    srun \
        --overlap \
        --exact \
        --nodes=1 \
        --ntasks=1 \
        --gpus-per-node=8 \
        --cpus-per-task=96 \
        --mem=512G \
        --container-image="${VLLM_RUNTIME_IMAGE}" \
        --container-mounts="${MODEL_MOUNTS},${TOKENIZER_RUNTIME_DIR}:/runtime-tokenizer:ro,${EVALUATION_CACHE_ROOT}:/evaluation-cache,${VLLM_RUNTIME_HOOKS}:/vllm-runtime-hooks:ro" \
        --container-workdir=/ \
        --no-container-mount-home \
        --export="ALL,HF_HOME=/evaluation-cache/huggingface,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONPATH=/vllm-runtime-hooks,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,NCCL_NVLS_ENABLE=0,TRITON_PTXAS_PATH=${VLLM_TRITON_PTXAS_PATH},VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_SECONDS},VLLM_WORKER_MULTIPROC_METHOD=spawn" \
        bash -lc '
            set -euo pipefail

            # A cold Enroot filesystem is overwhelmed when eight spawned vLLM
            # workers fault the same large Python/CUDA extensions concurrently.
            # Read site-packages once before spawning them.  The workers import
            # more than torch/vLLM (for example cpuinfo) after loading the model,
            # so warming only those trees or only large files still leaves the
            # Enroot root device saturated by concurrent small random reads.  The
            # node has ample page cache, and this sequential pass replaces
            # hundreds of GiB of duplicated worker reads.
            package_root=/usr/local/lib/python3.12/site-packages
            prewarm_started=${SECONDS}
            find "${package_root}" \
                -xdev -type f -readable -exec cat -- {} + > /dev/null
            find /usr/lib/x86_64-linux-gnu \
                -xdev -type f -readable -size +1M -exec cat -- {} + > /dev/null
            echo "vLLM container page-cache prewarm completed in $((SECONDS - prewarm_started))s"

            exec vllm serve /checkpoint \
                --tokenizer /runtime-tokenizer \
                --host 0.0.0.0 \
                --port "$6" \
                --served-model-name "$1" \
                --tensor-parallel-size 1 \
                --data-parallel-size 8 \
                --dtype bfloat16 \
                --max-model-len "$2" \
                --generation-config /checkpoint \
                --gpu-memory-utilization "$3" \
                --max-num-seqs "$4" \
                --seed "$5" \
                --reasoning-parser qwen3 \
                --moe-backend auto \
                --enable-prefix-caching
        ' _ \
            "${SERVED_MODEL_NAME}" \
            "${CONTEXT_LENGTH}" \
            "${VLLM_GPU_MEMORY_UTILIZATION}" \
            "${VLLM_MAX_NUM_SEQS}" \
            "${VLLM_SEED}" \
            "${VLLM_PORT}" \
        > "${VLLM_LOG}" 2>&1 &
    SERVER_STEP_PID=$!

    server_exited=false
    for ((ready_attempt = 1; ready_attempt <= SERVER_READY_ATTEMPTS; ready_attempt++)); do
        if curl --fail --silent "${API_BASE_URL}/health" >/dev/null 2>&1; then
            server_ready=true
            break
        fi
        if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
            wait "${SERVER_STEP_PID}" 2>/dev/null || true
            SERVER_STEP_PID=""
            server_exited=true
            break
        fi
        sleep "${SERVER_READY_INTERVAL_SECONDS}"
    done
    [[ "${server_ready}" == true ]] && break
    if [[ "${server_exited}" == true ]] \
        && grep -Eq 'EADDRINUSE|address already in use' "${VLLM_LOG}" \
        && (( startup_attempt < VLLM_STARTUP_ATTEMPTS )); then
        echo "vLLM startup port collision; retrying"
        continue
    fi
    echo "vLLM failed to become healthy; last log lines follow" >&2
    stop_server
    tail -n 120 "${VLLM_LOG}" >&2 || true
    exit 25
done
[[ "${server_ready}" == true ]] || { echo "vLLM failed to start" >&2; exit 26; }

PREFLIGHT_RESPONSE="${RESULT_ROOT}/preflight-${SLURM_JOB_ID}.json"
curl \
    --fail \
    --silent \
    --show-error \
    --max-time 600 \
    --header 'Content-Type: application/json' \
    --data "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":${VLLM_PREFLIGHT_MAX_TOKENS},\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":true}}" \
    "${API_ENDPOINT}" > "${PREFLIGHT_RESPONSE}"
jq --exit-status '
    .choices[0].message
    | ((.reasoning | type) == "string")
      and ((.reasoning | length) > 0)
      and ((.content | type) == "string")
      and ((.content | length) > 0)
' "${PREFLIGHT_RESPONSE}" >/dev/null || {
    echo "preflight did not separate Qwen3 reasoning and final content" >&2
    exit 27
}

for task in "${PENDING_TASKS[@]}"; do
    partial_result="${RESULT_ROOT}/${task}.partial"
    final_result="${RESULT_ROOT}/${task}"
    case "${task}" in
        aime24) eval_type=nemo_skills.ns_aime2024 ;;
        aime25) eval_type=nemo_skills.ns_aime2025 ;;
        aime26) eval_type=nemo_skills.ns_aime2026 ;;
    esac
    provenance_file="${partial_result}/provenance-${SLURM_JOB_ID}.env"
    cat > "${provenance_file}" <<EOF
checkpoint=${CHECKPOINT_SOURCE}
checkpoint_config_sha256=${checkpoint_config_sha256}
arm_name=${ARM_NAME}
training_step=${TRAINING_STEP}
run_namespace=${RUN_NAMESPACE}
task=${task}
eval_type=${eval_type}
eval_mode=${EVAL_MODE}
protocol_name=${PROTOCOL_NAME}
temperature=${TEMPERATURE}
top_p=${TOP_P}
top_k=${TOP_K}
num_repeats=${NUM_REPEATS}
expected_output_records=${EXPECTED_OUTPUT_RECORDS}
evaluator_attempts=${EVALUATOR_ATTEMPTS}
max_new_tokens=${MAX_NEW_TOKENS}
vocab_compat_action=${VOCAB_COMPAT_ACTION}
tokenizer_compat_action=${TOKENIZER_COMPAT_ACTION}
vllm_reasoning_parser=qwen3
enable_thinking=true
vllm_image=${VLLM_IMAGE}
vllm_oci_digest=${VLLM_OCI_DIGEST}
nemo_skills_image=${NEMO_SKILLS_IMAGE}
nemo_skills_oci_digest=${NEMO_SKILLS_OCI_DIGEST}
evaluator_catalog_commit=eb3ddf2acc7f2e1fa03aeba168afea636562779c
slurm_job_id=${SLURM_JOB_ID}
started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

    evaluator_succeeded=false
    for ((evaluator_attempt = 1; evaluator_attempt <= EVALUATOR_ATTEMPTS; evaluator_attempt++)); do
        echo "Evaluating ${ARM_NAME} step=${TRAINING_STEP} task=${task} attempt=${evaluator_attempt}/${EVALUATOR_ATTEMPTS}"
        if srun \
            --overlap \
            --exact \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=32 \
            --mem=512G \
            --container-image="${NEMO_SKILLS_IMAGE}" \
            --container-mounts="${partial_result}:/results,${NEMO_SKILLS_DATA_ROOT}:/evaluation-data,${EVALUATION_CACHE_ROOT}:/evaluation-cache,${EXPORT_TOOL_DIR}:/workspace/reasoning_eval:ro" \
            --container-workdir=/results \
            --no-container-mount-home \
            --export="ALL,HF_HOME=/evaluation-cache/huggingface,NEMO_SKILLS_DATA_DIR=/evaluation-data,NEMO_EVALUATOR_TELEMETRY_LEVEL=0,XDG_CACHE_HOME=/evaluation-cache/xdg,PYTHONUNBUFFERED=1" \
            bash -lc '
                set -euo pipefail
                nemo-evaluator run_eval --run_config /results/run-config.yaml
                python3 /workspace/reasoning_eval/export_adapter_cache.py \
                    --cache-dir /results/adapter-cache \
                    --output /results/model-outputs.jsonl
            '; then
            if [[ -s "${partial_result}/model-outputs.jsonl" \
                && -s "${partial_result}/evaluator/eval-results/${task}/metrics.json" ]]; then
                attempt_output_records="$(wc -l < "${partial_result}/model-outputs.jsonl")"
                if (( attempt_output_records == EXPECTED_OUTPUT_RECORDS )) \
                    && jq --exit-status --arg task "${task}" '
                        .[$task]["pass@1"].symbolic_correct
                        | (type == "number") and (. >= 0) and (. <= 100)
                    ' "${partial_result}/evaluator/eval-results/${task}/metrics.json" >/dev/null; then
                    evaluator_succeeded=true
                    break
                fi
                echo "Evaluator attempt produced incomplete outputs for ${task}: " \
                    "records=${attempt_output_records} expected=${EXPECTED_OUTPUT_RECORDS}" >&2
            else
                echo "Evaluator attempt did not produce model outputs and metrics for ${task}" >&2
            fi
        fi
        if (( evaluator_attempt < EVALUATOR_ATTEMPTS )); then
            echo "Evaluator attempt failed for ${task}; retrying from cached partial outputs" >&2
        fi
    done
    [[ "${evaluator_succeeded}" == true ]] || {
        echo "Evaluator failed after ${EVALUATOR_ATTEMPTS} attempts for ${task}" >&2
        exit 32
    }
    [[ -s "${partial_result}/model-outputs.jsonl" ]] || {
        echo "model output export is missing for ${task}" >&2
        exit 28
    }
    [[ -s "${partial_result}/evaluator/eval-results/${task}/metrics.json" ]] || {
        echo "metrics are missing for ${task}" >&2
        exit 29
    }
    output_records="$(wc -l < "${partial_result}/model-outputs.jsonl")"
    (( output_records == EXPECTED_OUTPUT_RECORDS )) || {
        echo "${task} exported ${output_records} model outputs; expected ${EXPECTED_OUTPUT_RECORDS}" >&2
        exit 30
    }
    jq --exit-status --arg task "${task}" '
        .[$task]["pass@1"].symbolic_correct
        | (type == "number") and (. >= 0) and (. <= 100)
    ' "${partial_result}/evaluator/eval-results/${task}/metrics.json" >/dev/null || {
        echo "${task} symbolic_correct metric is missing or outside [0, 100]" >&2
        exit 31
    }
    touch "${partial_result}/_SUCCESS"
    mv -- "${partial_result}" "${final_result}"
    echo "Completed ${final_result}"
done

stop_server
rm -f -- "${RESULT_ROOT}/.active-job"
trap - EXIT INT TERM
echo "All requested AIME evaluations completed under ${RESULT_ROOT}"
