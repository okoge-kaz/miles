#!/bin/bash
# Inner, container-side half of run_search_r1_eval.sbatch.

set -euo pipefail

cd /root/miles
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH="/root/Megatron-LM:/root/miles:/root/miles/examples/experimental/search-r1"
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

[[ -d "${CKPT}" ]] || { echo "no such checkpoint: ${CKPT}" >&2; exit 1; }
mkdir -p "${OUT_DIR}/audit"

EVAL_CKPT="${CKPT}"
if [[ "${UNPAD_VOCAB}" == "1" ]]; then
    EVAL_CKPT="/tmp/offline-search-r1-unpadded-${SLURM_JOB_ID}"
    python3 experiments/src/offline_eval/unpad_vocab.py "${CKPT}" "${EVAL_CKPT}"
    [[ -d "${EVAL_CKPT}" ]] || EVAL_CKPT="${CKPT}"
fi

if ! python3 -c 'import faiss, fastapi, uvicorn' >/dev/null 2>&1; then
    pip install --no-input --quiet faiss-cpu fastapi uvicorn
fi

python3 experiments/src/search_r1/retrieval_server.py \
    --index /data/search-r1/e5_Flat.index \
    --corpus /data/search-r1/wiki-18.jsonl \
    --encoder /ckpt/hf/e5-base-v2 \
    --port "${RETRIEVER_PORT}" \
    --topk "${SEARCH_TOPK}" \
    --faiss-threads "${RETRIEVER_FAISS_THREADS}" \
    --batch-max-requests "${RETRIEVER_BATCH_MAX_REQUESTS}" \
    --batch-wait-ms "${RETRIEVER_BATCH_WAIT_MS}" \
    >"${OUT_DIR}/retriever.log" 2>&1 &
RETRIEVER_PID=$!

python3 -m sglang.launch_server \
    --model-path "${EVAL_CKPT}" \
    --host 127.0.0.1 \
    --port "${SERVER_PORT}" \
    --dp-size "${GPUS_PER_NODE}" \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --log-level "${SGLANG_LOG_LEVEL:-info}" \
    >"${OUT_DIR}/sglang.log" 2>&1 &
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
        tail -80 "${OUT_DIR}/retriever.log" || true
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
        tail -80 "${OUT_DIR}/sglang.log" || true
        exit 1
    }
    sleep 10
done
curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null || {
    echo "SGLang never became healthy" >&2
    exit 1
}

pids=()
names=()
limit_args=()
[[ -z "${LIMIT}" ]] || limit_args=(--limit "${LIMIT}")
for spec in $(echo "${BENCHMARKS}" | tr '+' ' '); do
    name=${spec%%:*}
    path=${spec#*:}
    [[ "${name}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid benchmark name: ${name}" >&2; exit 1; }
    [[ -f "${path}" ]] || { echo "missing benchmark: ${path}" >&2; exit 1; }
    echo "=== ${name} (${path}) starting ==="
    SEARCH_R1_SEARCH_URL="http://127.0.0.1:${RETRIEVER_PORT}/retrieve" \
    python3 experiments/src/offline_eval/measure_search_r1.py \
        --prompt-data "${path}" \
        --output "${OUT_DIR}/${name}.jsonl" \
        --dump-responses "${OUT_DIR}/audit/${name}.jsonl" \
        --model-path "${EVAL_CKPT}" \
        --server-url "http://127.0.0.1:${SERVER_PORT}" \
        --search-url "http://127.0.0.1:${RETRIEVER_PORT}/retrieve" \
        --policy "${TAG}" \
        --n-samples "${N_SAMPLES}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-turns "${MAX_TURNS}" \
        --search-topk "${SEARCH_TOPK}" \
        --concurrency "${CONCURRENCY_PER_BENCHMARK}" \
        --search-concurrency "${CONCURRENCY_PER_BENCHMARK}" \
        --search-timeout "${SEARCH_TIMEOUT}" \
        --search-max-attempts "${SEARCH_MAX_ATTEMPTS}" \
        --request-timeout "${REQUEST_TIMEOUT}" \
        --format-score "${FORMAT_SCORE}" \
        --dp-size "${GPUS_PER_NODE}" \
        "${limit_args[@]}" \
        >"${OUT_DIR}/${name}.measure.log" 2>&1 &
    pids+=("$!")
    names+=("${name}")
done

status=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        echo "benchmark ${names[index]} failed" >&2
        status=1
    fi
done
for name in "${names[@]}"; do
    echo "=== ${name} ==="
    tail -40 "${OUT_DIR}/${name}.measure.log" || true
done
[[ "${status}" == "0" ]] || exit 1

kill -0 "${RETRIEVER_PID}" 2>/dev/null || { echo "retriever exited during evaluation" >&2; exit 1; }
kill -0 "${SERVER_PID}" 2>/dev/null || { echo "SGLang exited during evaluation" >&2; exit 1; }

echo "=== Search-R1 report ==="
python3 experiments/src/offline_eval/report_search_r1.py "${OUT_DIR}"
