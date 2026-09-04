#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT}/experiments/env.sh"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_NAMESPACE="${RUN_NAMESPACE:-sr-20260819-212906}"
EVAL_MODE="${EVAL_MODE:-full}"
REQUESTED_PROTOCOL_NAME="${PROTOCOL_NAME:-}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
AIME_REPEATS="${AIME_REPEATS:-64}"
if [[ "${EVAL_MODE}" == smoke ]]; then
    EFFECTIVE_REPEATS=1
else
    EFFECTIVE_REPEATS="${AIME_REPEATS}"
fi
PROTOCOL_ARGS=(
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --effective-repeats "${EFFECTIVE_REPEATS}"
)
if [[ -n "${REQUESTED_PROTOCOL_NAME}" ]]; then
    PROTOCOL_ARGS+=(--protocol-name "${REQUESTED_PROTOCOL_NAME}")
fi
PROTOCOL_NAME="$(
    python3 "${REPO_ROOT}/experiments/tools/reasoning_eval/protocol.py" "${PROTOCOL_ARGS[@]}"
)"
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
if [[ -f "${RESULT_STUDY_ROOT}/grid.env" ]]; then
    source "${RESULT_STUDY_ROOT}/grid.env"
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
