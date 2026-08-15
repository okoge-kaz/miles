#!/bin/bash
# Container-side worker for run_measure_search_r1.sbatch.

set -euo pipefail

cd /root/miles
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH="/root/Megatron-LM:/root/miles:/root/miles/examples/experimental/search-r1"
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

if ! python3 -c 'import faiss, fastapi, uvicorn' >/dev/null 2>&1; then
    pip install --no-input --quiet faiss-cpu fastapi uvicorn
fi

LOG_DIR="/data/difficulty/search-r1-passrate-logs/${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}" "$(dirname -- "${OUTPUT}")" "$(dirname -- "${FILTERED_OUTPUT}")"

python3 experiments/src/search_r1/retrieval_server.py \
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

python3 -m sglang.launch_server \
    --model-path "/ckpt/hf/${MODEL_NAME}" \
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

limit_args=()
[[ -z "${LIMIT}" ]] || limit_args=(--limit "${LIMIT}")
python3 experiments/src/offline_eval/measure_search_r1.py \
    --prompt-data "${PROMPT_DATA}" \
    --output "${OUTPUT}" \
    --dump-responses "${DUMP_RESPONSES}" \
    --dump-limit "${DUMP_LIMIT}" \
    --model-path "/ckpt/hf/${MODEL_NAME}" \
    --server-url "http://127.0.0.1:${SERVER_PORT}" \
    --search-url "http://127.0.0.1:${RETRIEVER_PORT}/retrieve" \
    --policy "${MODEL_NAME}" \
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

if [[ "${APPLY_FILTER}" == "1" ]]; then
    filtered_partial="${FILTERED_OUTPUT}.partial"
    rm -f "${filtered_partial}"
    python3 -m experiments.src.difficulty_filter.apply_filter \
        --prompt-data "${PROMPT_DATA}" \
        --pass-rates "${OUTPUT}" \
        --output "${filtered_partial}" \
        --pass-rate-min "${PASS_RATE_MIN}" \
        --pass-rate-max "${PASS_RATE_MAX}" \
        --policy "${MODEL_NAME}"
    mv "${filtered_partial}" "${FILTERED_OUTPUT}"
    echo "fixed Search-R1 prompt rows: $(wc -l < "${FILTERED_OUTPUT}")"
else
    echo "LIMIT=${LIMIT}; measurement smoke complete, production filter intentionally skipped"
fi

kill -0 "${RETRIEVER_PID}" 2>/dev/null || { echo "retriever exited during measurement" >&2; exit 1; }
kill -0 "${SERVER_PID}" 2>/dev/null || { echo "SGLang exited during measurement" >&2; exit 1; }
