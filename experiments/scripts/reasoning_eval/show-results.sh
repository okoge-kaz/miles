#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

RUN_NAMESPACE="${RUN_NAMESPACE:-sr-20260819-212906}"
EVAL_MODE="${EVAL_MODE:-full}"
PROTOCOL_NAME="${PROTOCOL_NAME:-eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-thinking-t0.6-p0.95-k20-aime64-v1}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${WS}/evaluations/reasoning_eval}"
RESULT_STUDY_ROOT="${EVALUATION_ROOT}/staleness-ratio-sweep/${RUN_NAMESPACE}"
ANALYSIS_ROOT="${RESULT_STUDY_ROOT}/analysis/${PROTOCOL_NAME}/${EVAL_MODE}"

if (( $# > 1 )); then
    echo "usage: experiments/scripts/reasoning_eval/show-results.sh [RUN_NAMESPACE]" >&2
    exit 2
fi
if (( $# == 1 )); then
    RUN_NAMESPACE="$1"
    RESULT_STUDY_ROOT="${EVALUATION_ROOT}/staleness-ratio-sweep/${RUN_NAMESPACE}"
    ANALYSIS_ROOT="${RESULT_STUDY_ROOT}/analysis/${PROTOCOL_NAME}/${EVAL_MODE}"
fi

python3 "${REPO_ROOT}/experiments/tools/reasoning_eval/summarize_results.py" \
    --result-study-root "${RESULT_STUDY_ROOT}" \
    --protocol-name "${PROTOCOL_NAME}" \
    --eval-mode "${EVAL_MODE}" \
    --output-dir "${ANALYSIS_ROOT}"
python3 "${REPO_ROOT}/experiments/tools/reasoning_eval/plot_results.py" \
    --aggregate-csv "${ANALYSIS_ROOT}/aggregate-results.csv" \
    --output-dir "${ANALYSIS_ROOT}/figures"

echo
cat "${ANALYSIS_ROOT}/summary.md"
echo "Figures: ${ANALYSIS_ROOT}/figures"
