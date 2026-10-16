#!/bin/bash
set -euo pipefail

: "${JUDGE_MODEL_PATH:?set the frozen model HF checkpoint or model id}"
: "${JUDGE_MODEL_REVISION:?set an immutable revision/checkpoint identity}"

command=(python3 -m sglang.launch_server
    --model-path "${JUDGE_MODEL_PATH}"
    --served-model-name "${JUDGE_MODEL_NAME:-judge}"
    --host 0.0.0.0 --port "${JUDGE_PORT:-30080}"
    --tp-size "${JUDGE_TP_SIZE:-8}"
    --context-length "${JUDGE_CONTEXT_LENGTH:-32768}"
    --mem-fraction-static "${JUDGE_MEM_FRACTION:-0.80}"
    --max-running-requests "${JUDGE_MAX_RUNNING_REQUESTS:-64}"
    --log-level warning
)

[[ -d "${JUDGE_MODEL_PATH}" ]] || command+=(--revision "${JUDGE_MODEL_REVISION}")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}" "$@"
