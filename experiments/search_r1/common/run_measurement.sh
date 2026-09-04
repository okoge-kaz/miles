#!/bin/bash
# Shared container-side Search-R1 measurement runtime.

set -euo pipefail

cd /root/miles
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH="/root/Megatron-LM:/root/miles:/root/miles/examples/experimental/search-r1"
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

python3 -c 'import faiss, fastapi, uvicorn' >/dev/null 2>&1 || {
    echo "Search-R1 image is missing the staged faiss/fastapi/uvicorn dependencies" >&2
    exit 2
}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MEASUREMENT_MODE="${MEASUREMENT_MODE:-difficulty}"
case "${MEASUREMENT_MODE}" in
    difficulty)
        LOG_DIR="${SEARCH_R1_LOG_DIR:-/data/difficulty/search-r1-passrate-logs/${SLURM_JOB_ID}}"
        mkdir -p "${LOG_DIR}" "$(dirname -- "${OUTPUT}")" "$(dirname -- "${FILTERED_OUTPUT}")"
        ;;
    evaluation)
        : "${RESULT_ROOT:?RESULT_ROOT is required for evaluation}"
        LOG_DIR="${SEARCH_R1_LOG_DIR:-${RESULT_ROOT}/logs}"
        mkdir -p "${LOG_DIR}" "${RESULT_ROOT}"
        ;;
    *)
        echo "MEASUREMENT_MODE must be difficulty or evaluation, got ${MEASUREMENT_MODE}" >&2
        exit 2
        ;;
esac

python3 experiments/src/environments/search_r1/retrieval_server.py \
    --index /data/search-r1/e5_Flat.index \
    --corpus /data/search-r1/wiki-18.jsonl \
    --encoder /ckpt/hf/e5-base-v2 \
    --port "${RETRIEVER_PORT}" \
    --topk "${SEARCH_TOPK}" \
    --faiss-threads "${RETRIEVER_FAISS_THREADS}" \
    --batch-max-requests "${RETRIEVER_BATCH_MAX_REQUESTS}" \
    --batch-wait-ms "${RETRIEVER_BATCH_WAIT_MS}" \
    >"${LOG_DIR}/retriever.log" 2>&1 &
RETRIEVER_PID=$!

MODEL_PATH="${MODEL_PATH:-/ckpt/hf/${MODEL_NAME}}"
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${SERVER_PORT}" \
    --dp-size "${GPUS_PER_NODE}" \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --log-level "${SGLANG_LOG_LEVEL:-info}" \
    >"${LOG_DIR}/sglang.log" 2>&1 &
SERVER_PID=$!

cleanup() {
    local status=$?
    trap - EXIT
    kill "${SERVER_PID}" "${RETRIEVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    wait "${RETRIEVER_PID}" 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT

echo "waiting for Search-R1 retriever"
for attempt in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null 2>&1; then
        echo "retriever healthy after $((attempt * 10))s"
        break
    fi
    kill -0 "${RETRIEVER_PID}" 2>/dev/null || {
        echo "retriever died during startup" >&2
        tail -80 "${LOG_DIR}/retriever.log" || true
        exit 1
    }
    sleep 10
done
curl -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null || {
    echo "retriever never became healthy" >&2
    exit 1
}

probe=$(curl -sf -X POST "http://127.0.0.1:${RETRIEVER_PORT}/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"queries": ["who won the nobel prize in physics 1901"], "topk": 3}')
case "${probe}" in
    *contents*|*document*|*text*) echo "retriever content probe ok" ;;
    *) echo "retriever returned no passages: ${probe:0:300}" >&2; exit 1 ;;
esac

echo "waiting for SGLang"
for attempt in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
        echo "SGLang healthy after $((attempt * 10))s"
        break
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null || {
        echo "SGLang died during startup" >&2
        tail -80 "${LOG_DIR}/sglang.log" || true
        exit 1
    }
    sleep 10
done
curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null || {
    echo "SGLang never became healthy" >&2
    exit 1
}

measure_dataset() {
    local prompt_data=$1
    local output=$2
    local dump_responses=$3
    local limit=$4
    local limit_args=()
    [[ -z "${limit}" ]] || limit_args=(--limit "${limit}")
    python3 experiments/tools/difficulty_filter/measure_search_r1.py \
        --prompt-data "${prompt_data}" \
        --output "${output}" \
        --dump-responses "${dump_responses}" \
        --dump-limit "${DUMP_LIMIT}" \
        --model-path "${MODEL_PATH}" \
        --server-url "http://127.0.0.1:${SERVER_PORT}" \
        --search-url "http://127.0.0.1:${RETRIEVER_PORT}/retrieve" \
        --policy "${POLICY_NAME:-${MODEL_NAME}}" \
        --n-samples "${N_SAMPLES}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-turns "${MAX_TURNS}" \
        --search-topk "${SEARCH_TOPK}" \
        --concurrency "${CONCURRENCY}" \
        --search-concurrency "${SEARCH_CONCURRENCY}" \
        --search-timeout "${SEARCH_TIMEOUT}" \
        --search-max-attempts "${SEARCH_MAX_ATTEMPTS}" \
        --request-timeout "${REQUEST_TIMEOUT}" \
        --format-score "${FORMAT_SCORE}" \
        --dp-size "${GPUS_PER_NODE}" \
        "${limit_args[@]}"
}

if [[ "${MEASUREMENT_MODE}" == evaluation ]]; then
    mapfile -t benchmark_names < <(tr ' ' '\n' <<< "${BENCHMARKS}" | sed '/^$/d')
    for benchmark in "${benchmark_names[@]}"; do
        case "${benchmark}" in
            nq|hotpotqa|triviaqa|popqa|2wikimultihopqa|musique|bamboogle) ;;
            *) echo "unsupported Search-R1 benchmark: ${benchmark}" >&2; exit 3 ;;
        esac
        prompt_data="/data/search-r1/eval/${benchmark}-miles.jsonl"
        benchmark_root="${RESULT_ROOT}/${benchmark}"
        mkdir -p "${benchmark_root}"
        if [[ -f "${benchmark_root}/_SUCCESS" ]]; then
            [[ -s "${benchmark_root}/artifact-manifest.sha256" ]] \
                && (cd "${benchmark_root}" && sha256sum --check artifact-manifest.sha256 >/dev/null) || {
                echo "completed Search-R1 result failed artifact verification: ${benchmark}" >&2
                exit 4
            }
            echo "Already complete: ${benchmark_root}"
            continue
        fi
        measure_dataset \
            "${prompt_data}" \
            "${benchmark_root}/records.jsonl" \
            "${benchmark_root}/audit.jsonl" \
            "${LIMIT}"
        (
            cd "${benchmark_root}"
            sha256sum records.jsonl records.meta.json audit.jsonl > artifact-manifest.sha256
            sha256sum --check artifact-manifest.sha256 >/dev/null
        )
        touch "${benchmark_root}/_SUCCESS"
    done
    python3 experiments/search_r1/evaluation/summarize.py \
        --result-root "${RESULT_ROOT}" \
        --benchmarks "${benchmark_names[@]}" \
        --output "${RESULT_ROOT}/summary.json"
    (
        cd "${RESULT_ROOT}"
        sha256sum summary.json evaluation-contract.env > artifact-manifest.sha256
        sha256sum --check artifact-manifest.sha256 >/dev/null
    )
    touch "${RESULT_ROOT}/_SUCCESS"
else
    measure_dataset "${PROMPT_DATA}" "${OUTPUT}" "${DUMP_RESPONSES}" "${LIMIT}"
fi

if [[ "${MEASUREMENT_MODE}" == difficulty && "${APPLY_FILTER:-0}" == "1" ]]; then
    filtered_partial="${FILTERED_OUTPUT}.partial"
    rm -f "${filtered_partial}"
    python3 -m experiments.tools.difficulty_filter.apply_filter \
        --prompt-data "${PROMPT_DATA}" \
        --pass-rates "${OUTPUT}" \
        --output "${filtered_partial}" \
        --pass-rate-min "${PASS_RATE_MIN}" \
        --pass-rate-max "${PASS_RATE_MAX}" \
        --policy "${MODEL_NAME}"
    mv "${filtered_partial}" "${FILTERED_OUTPUT}"
    echo "fixed Search-R1 prompt rows: $(wc -l < "${FILTERED_OUTPUT}")"
elif [[ "${MEASUREMENT_MODE}" == difficulty ]]; then
    echo "LIMIT=${LIMIT}; measurement smoke complete, production filter intentionally skipped"
fi

kill -0 "${RETRIEVER_PID}" 2>/dev/null || { echo "retriever exited during measurement" >&2; exit 1; }
kill -0 "${SERVER_PID}" 2>/dev/null || { echo "SGLang exited during measurement" >&2; exit 1; }
