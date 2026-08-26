#!/bin/bash
#SBATCH --job-name=q3-4b-reasoning-suite
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/reasoning_eval/%x-%j.log

# Offline checkpoint evaluation for AIME, LiveCodeBench, GPQA, and IFBench. Generation
# runs against an 8-way vLLM server. The server is stopped before scoring, and
# the scorer runs without a network namespace; this is mandatory for generated
# code and keeps every verifier offline.

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"
EVAL_MODE="${EVAL_MODE:-full}"
: "${CHECKPOINT_PATH:=${HF_CKPT_DIR}/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000/iter_0004000}"
: "${RESULT_ROOT:=${OUTPUT_DIR}/reasoning_eval/qwen3-4b-base-lr2e-5-step4000/${EVAL_MODE}}"
TASKS_TEXT="${TASKS:-aime24 aime25 aime26 math500 livecodebench gpqa_diamond gpqa_main gpqa_extended ifbench}"
TASKS_TEXT="${TASKS_TEXT//,/ }"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
LCB_REPEATS="${LCB_REPEATS:-1}"
GPQA_REPEATS="${GPQA_REPEATS:-8}"
IFBENCH_REPEATS="${IFBENCH_REPEATS:-8}"
AIME_REPEATS="${AIME_REPEATS:-16}"
MATH500_REPEATS="${MATH500_REPEATS:-4}"
GENERATION_CONCURRENCY="${GENERATION_CONCURRENCY:-256}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_ATTEMPTS="${VLLM_STARTUP_ATTEMPTS:-3}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
VLLM_IMAGE="${SHARED_WS}/containers/vllm-openai-v0.20.2-70a098d9.sqsh"
VLLM_OCI_DIGEST="sha256:70a098d90dbab428a001d9e852fc0fc8d67da5beb03e7851a22247653bf35923"
EVALUATION_CACHE_ROOT="${CACHE_DIR}/reasoning_eval"
UNPAD_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/unpad_vocab.py"
VALIDATE_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/validate_checkpoint.py"

[[ "${EVAL_MODE}" == smoke || "${EVAL_MODE}" == full ]] || {
    echo "EVAL_MODE must be smoke or full" >&2
    exit 2
}
[[ "${ENABLE_THINKING}" == true || "${ENABLE_THINKING}" == false ]] || {
    echo "ENABLE_THINKING must be true or false" >&2
    exit 2
}
[[ "${VLLM_STARTUP_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "VLLM_STARTUP_ATTEMPTS must be a positive integer" >&2
    exit 2
}
export WANDB_MODE="${WANDB_MODE:-offline}"
[[ "${WANDB_MODE}" == offline ]] || {
    echo "reasoning-eval validation requires WANDB_MODE=offline" >&2
    exit 2
}
CHECKPOINT_SOURCE="$(realpath -- "${CHECKPOINT_PATH}")"
RESULT_ROOT="$(realpath -m -- "${RESULT_ROOT}")"
[[ -f "${VLLM_IMAGE}" ]] || { echo "vLLM image not found: ${VLLM_IMAGE}" >&2; exit 3; }
[[ -f "${VLLM_IMAGE}.oci-digest" \
    && "$(< "${VLLM_IMAGE}.oci-digest")" == "${VLLM_OCI_DIGEST}" ]] || {
    echo "vLLM image OCI provenance is missing or mismatched" >&2
    exit 3
}
CHECKPOINT_MANIFEST_SHA256="$(
    python3 "${VALIDATE_TOOL}" --quiet --print-manifest "${CHECKPOINT_SOURCE}"
)"
mkdir -p "${RESULT_ROOT}" "${EVALUATION_CACHE_ROOT}/huggingface"

desired_contract="${SLURM_TMPDIR:-/tmp}/reasoning-eval-suite-contract-${SLURM_JOB_ID}.env"
cat > "${desired_contract}" <<EOF
checkpoint=${CHECKPOINT_SOURCE}
checkpoint_manifest_sha256=${CHECKPOINT_MANIFEST_SHA256}
eval_mode=${EVAL_MODE}
enable_thinking=${ENABLE_THINKING}
temperature=${TEMPERATURE}
top_p=${TOP_P}
top_k=${TOP_K}
context_length=${CONTEXT_LENGTH}
lcb_repeats=${LCB_REPEATS}
gpqa_repeats=${GPQA_REPEATS}
ifbench_repeats=${IFBENCH_REPEATS}
aime_repeats=${AIME_REPEATS}
math500_repeats=${MATH500_REPEATS}
vllm_image=${VLLM_IMAGE}
vllm_oci_digest=${VLLM_OCI_DIGEST}
EOF
if [[ -f "${RESULT_ROOT}/evaluation-contract.env" ]]; then
    cmp --silent "${desired_contract}" "${RESULT_ROOT}/evaluation-contract.env" || {
        diff --unified "${RESULT_ROOT}/evaluation-contract.env" "${desired_contract}" || true
        echo "refusing to reuse RESULT_ROOT with a changed evaluation contract" >&2
        exit 3
    }
else
    if find "${RESULT_ROOT}" -mindepth 2 -type f \
        \( -name candidates.jsonl -o -name scores.jsonl -o -name _SUCCESS \) \
        -print -quit | grep -q .; then
        echo "legacy RESULT_ROOT contains task artifacts without a contract" >&2
        exit 3
    fi
    contract_partial="${RESULT_ROOT}/.evaluation-contract.partial-${SLURM_JOB_ID}"
    cp -- "${desired_contract}" "${contract_partial}"
    mv -- "${contract_partial}" "${RESULT_ROOT}/evaluation-contract.env"
fi

MODEL_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/reasoning-eval-suite-model-${SLURM_JOB_ID}"
TOKENIZER_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/reasoning-eval-suite-tokenizer-${SLURM_JOB_ID}"
python3 "${UNPAD_TOOL}" "${CHECKPOINT_SOURCE}" "${MODEL_RUNTIME_DIR}"
if [[ -f "${MODEL_RUNTIME_DIR}/config.json" ]]; then
    MODEL_MOUNTS="${MODEL_RUNTIME_DIR}:/checkpoint,${CHECKPOINT_SOURCE}:${CHECKPOINT_SOURCE}:ro"
else
    MODEL_MOUNTS="${CHECKPOINT_SOURCE}:/checkpoint:ro"
fi
mkdir -p "${TOKENIZER_RUNTIME_DIR}"
cp -- "${CHECKPOINT_SOURCE}/chat_template.jinja" "${CHECKPOINT_SOURCE}/config.json" \
    "${CHECKPOINT_SOURCE}/tokenizer.json" "${TOKENIZER_RUNTIME_DIR}/"
extra_type="$(jq -r 'if has("extra_special_tokens") then (.extra_special_tokens | type) else "absent" end' "${CHECKPOINT_SOURCE}/tokenizer_config.json")"
if [[ "${extra_type}" == array ]]; then
    jq 'del(.extra_special_tokens)' "${CHECKPOINT_SOURCE}/tokenizer_config.json" \
        > "${TOKENIZER_RUNTIME_DIR}/tokenizer_config.json"
else
    cp -- "${CHECKPOINT_SOURCE}/tokenizer_config.json" "${TOKENIZER_RUNTIME_DIR}/tokenizer_config.json"
fi

mapfile -t REQUESTED_TASKS < <(tr ' ' '\n' <<< "${TASKS_TEXT}" | sed '/^$/d')
for task in "${REQUESTED_TASKS[@]}"; do
    case "${task}" in
        aime24|aime25|aime26|math500|livecodebench|gpqa_diamond|gpqa_main|gpqa_extended|ifbench) ;;
        *) echo "unsupported task: ${task}" >&2; exit 4 ;;
    esac
done

SERVER_PID=""
stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM
SERVED_MODEL=reasoning-eval-suite-qwen3-4b
SERVER_READY=0
for startup_attempt in $(seq 1 "${VLLM_STARTUP_ATTEMPTS}"); do
    VLLM_LOG="${RESULT_ROOT}/vllm-${SLURM_JOB_ID}-attempt-${startup_attempt}.log"
    srun --overlap --exact --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=96 \
        --container-image="${VLLM_IMAGE}" \
        --container-mounts="${MODEL_MOUNTS},${TOKENIZER_RUNTIME_DIR}:/runtime-tokenizer:ro,${EVALUATION_CACHE_ROOT}:/evaluation-cache" \
        --container-workdir=/ --no-container-mount-home \
        --export="ALL,HF_HOME=/evaluation-cache/huggingface,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=1,NCCL_NVLS_ENABLE=0,VLLM_WORKER_MULTIPROC_METHOD=spawn" \
        bash -lc '
exec vllm serve /checkpoint \
    --tokenizer /runtime-tokenizer \
    --host 0.0.0.0 --port "$1" --served-model-name "$2" \
    --tensor-parallel-size 1 --data-parallel-size 8 --dtype bfloat16 \
    --max-model-len "$3" --generation-config /checkpoint \
    --gpu-memory-utilization 0.90 --max-num-seqs 64 --seed 1234 \
    --reasoning-parser qwen3 --moe-backend auto --enable-prefix-caching
' _ "${VLLM_PORT}" "${SERVED_MODEL}" "${CONTEXT_LENGTH}" > "${VLLM_LOG}" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 180); do
        if curl --fail --silent "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            SERVER_READY=1
            break
        fi
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 5
    done
    if [[ "${SERVER_READY}" == 1 ]]; then
        break
    fi
    tail -n 120 "${VLLM_LOG}" >&2
    stop_server
    if (( startup_attempt < VLLM_STARTUP_ATTEMPTS )); then
        echo "vLLM startup attempt ${startup_attempt} failed; retrying" >&2
        sleep 5
    fi
done
[[ "${SERVER_READY}" == 1 ]] || { echo "vLLM failed to start" >&2; exit 6; }

for task in "${REQUESTED_TASKS[@]}"; do
    task_root="${RESULT_ROOT}/${task}"
    mkdir -p "${task_root}"
    if [[ -f "${task_root}/_SUCCESS" ]]; then
        [[ -s "${task_root}/artifact-manifest.sha256" ]] \
            && (cd "${task_root}" && sha256sum --check artifact-manifest.sha256 >/dev/null) || {
            echo "completed result failed artifact verification: ${task_root}" >&2
            exit 7
        }
        echo "Already complete: ${task_root}"
        continue
    fi
    if [[ "${task}" == livecodebench ]]; then
        input=/data/livecodebench-lite/livecodebench-release-v6-miles-eval.jsonl
        repeats="${LCB_REPEATS}"
        max_tokens=16384
    elif [[ "${task}" =~ ^aime(24|25|26)$ ]]; then
        year="20${task#aime}"
        input="/data/aime-${year}/aime-${year}.jsonl"
        repeats="${AIME_REPEATS}"
        max_tokens=16384
    elif [[ "${task}" == math500 ]]; then
        input=/data/math-500/math-500.jsonl
        repeats="${MATH500_REPEATS}"
        max_tokens=16384
    elif [[ "${task}" == ifbench ]]; then
        input=/data/ifbench/IFBench_test_miles.jsonl
        repeats="${IFBENCH_REPEATS}"
        max_tokens=16384
    else
        split="${task#gpqa_}"
        input="/data/gpqa/gpqa-${split}-miles.jsonl"
        repeats="${GPQA_REPEATS}"
        max_tokens=16384
    fi
    limit_args=()
    [[ "${EVAL_MODE}" == full ]] || limit_args=(--limit 2)
    thinking_arg=--no-enable-thinking
    [[ "${ENABLE_THINKING}" == false ]] || thinking_arg=--enable-thinking
    host_input="${DATASET_DIR}${input#/data}"
    [[ -s "${host_input}" ]] || {
        echo "dataset is missing or empty: ${host_input} (container path ${input})" >&2
        exit 8
    }
    dataset_sha256="$(sha256sum "${host_input}")"
    dataset_sha256="${dataset_sha256%% *}"
    provenance_partial="${task_root}/.provenance.partial-${SLURM_JOB_ID}"
    cat > "${provenance_partial}" <<EOF
task=${task}
input=${input}
input_sha256=${dataset_sha256}
checkpoint=${CHECKPOINT_SOURCE}
checkpoint_manifest_sha256=${CHECKPOINT_MANIFEST_SHA256}
repeats=${repeats}
max_tokens=${max_tokens}
eval_mode=${EVAL_MODE}
enable_thinking=${ENABLE_THINKING}
temperature=${TEMPERATURE}
top_p=${TOP_P}
top_k=${TOP_K}
EOF
    mv -- "${provenance_partial}" "${task_root}/provenance.env"
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=16 \
        --container-image="${SQSH_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS},${RESULT_ROOT}:/results" \
        --container-writable --no-container-mount-home --export=ALL \
        bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
python3 experiments/tools/reasoning_eval/suite.py generate \
    --input "$1" --output "/results/$2/candidates.jsonl" \
    --endpoint "http://127.0.0.1:$3/v1/chat/completions" --model "$4" \
    --repeats "$5" --max-tokens "$6" --temperature "$7" --top-p "$8" --top-k "$9" \
    --concurrency "${10}" "${11}" "${@:12}"
' _ "${input}" "${task}" "${VLLM_PORT}" "${SERVED_MODEL}" "${repeats}" "${max_tokens}" \
            "${TEMPERATURE}" "${TOP_P}" "${TOP_K}" "${GENERATION_CONCURRENCY}" \
            "${thinking_arg}" "${limit_args[@]}"
done

stop_server

for task in "${REQUESTED_TASKS[@]}"; do
    task_root="${RESULT_ROOT}/${task}"
    if [[ -f "${task_root}/_SUCCESS" ]]; then
        continue
    fi
    if [[ "${task}" == livecodebench ]]; then
        input=/data/livecodebench-lite/livecodebench-release-v6-miles-eval.jsonl
        repeats="${LCB_REPEATS}"
    elif [[ "${task}" =~ ^aime(24|25|26)$ ]]; then
        year="20${task#aime}"
        input="/data/aime-${year}/aime-${year}.jsonl"
        repeats="${AIME_REPEATS}"
    elif [[ "${task}" == math500 ]]; then
        input=/data/math-500/math-500.jsonl
        repeats="${MATH500_REPEATS}"
    elif [[ "${task}" == ifbench ]]; then
        input=/data/ifbench/IFBench_test_miles.jsonl
        repeats="${IFBENCH_REPEATS}"
    else
        split="${task#gpqa_}"
        input="/data/gpqa/gpqa-${split}-miles.jsonl"
        repeats="${GPQA_REPEATS}"
    fi
    limit_args=()
    [[ "${EVAL_MODE}" == full ]] || limit_args=(--limit 2)
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=128 \
        --container-image="${SQSH_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS},${RESULT_ROOT}:/results,/usr/bin/bwrap:/usr/local/bin/bwrap" \
        --container-writable --no-container-mount-home --export=ALL \
        bash -lc '
set -euo pipefail
export PYTHONPATH=/root/miles:/data/ifbench:/data/ifbench-deps
export NLTK_DATA=/data/ifbench-deps/nltk_data
export LCB_ALLOW_LOCAL_EXECUTION=1
export LCB_REPO_PATH=/data/livecodebench-code
export LCB_EVAL_PROCESSES=16
PYTHON_BIN="$(realpath -- "$(command -v python3)")"
exec unshare --user --map-root-user --net --fork \
    bwrap --die-with-parent --new-session --unshare-pid --unshare-ipc --unshare-uts \
    --dir /proc --dev /dev --tmpfs /tmp --dir /home \
    --ro-bind /usr /usr --ro-bind /usr/local /usr/local --ro-bind /bin /bin \
    --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
    --ro-bind /root/miles /root/miles --ro-bind /data /data --bind /results /results \
    --chdir /root/miles "${PYTHON_BIN}" experiments/tools/reasoning_eval/suite.py score \
    --task "$1" --input "$2" --candidates "/results/$1/candidates.jsonl" \
    --output "/results/$1/scores.jsonl" --summary "/results/$1/summary.json" \
    --repeats "$3" "${@:4}"
' _ "${task}" "${input}" "${repeats}" "${limit_args[@]}"
    (
        cd "${task_root}"
        sha256sum candidates.jsonl scores.jsonl summary.json provenance.env \
            > artifact-manifest.sha256
        sha256sum --check artifact-manifest.sha256 >/dev/null
    )
    touch "${task_root}/_SUCCESS"
done

echo "All requested reasoning-suite evaluations completed under ${RESULT_ROOT}"
