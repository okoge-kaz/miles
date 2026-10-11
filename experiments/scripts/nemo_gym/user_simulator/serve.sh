#!/bin/bash

set -euo pipefail

: "${TAU_SERVER_MODEL_PATH:?set the frozen model HF checkpoint or model id}"
: "${TAU_SERVER_MODEL_REVISION:?set an immutable revision/checkpoint identity}"

command=(python3 -m sglang.launch_server
    --model-path "${TAU_SERVER_MODEL_PATH}"
    --dtype bfloat16
    --served-model-name "${TAU_SERVER_MODEL_NAME:-qwen3-4b-sft-user}"
    --host 0.0.0.0 --port "${TAU_SERVER_PORT:-30180}"
    --tp-size "${TAU_SERVER_TP_SIZE:-8}"
    --context-length "${TAU_SERVER_CONTEXT_LENGTH:-32768}"
    --mem-fraction-static "${TAU_SERVER_MEM_FRACTION:-0.70}"
    --max-running-requests "${TAU_SERVER_MAX_RUNNING_REQUESTS:-64}"
    --log-level warning
)

[[ -d "${TAU_SERVER_MODEL_PATH}" ]] || command+=(--revision "${TAU_SERVER_MODEL_REVISION}")
command+=(
    --tool-call-parser "${TAU_SERVER_TOOL_CALL_PARSER:-qwen25}"
    --reasoning-parser "${TAU_SERVER_REASONING_PARSER:-qwen3}"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}" "$@"
