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
STALENESS_ROOT="${ANALYSIS_ROOT}/staleness"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-python3}"
WANDB_PYTHON="${WANDB_PYTHON:-${ANALYSIS_PYTHON}}"
WANDB_ENTITY="${WANDB_ENTITY:-ai-horizons}"
WANDB_PROJECT="${WANDB_PROJECT:-async-rl-dapo-math}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-500}"
SKIP_WANDB_EXPORT="${SKIP_WANDB_EXPORT:-0}"

if (( $# > 1 )); then
    echo "usage: experiments/scripts/reasoning_eval/show-staleness-analysis.sh [RUN_NAMESPACE]" >&2
    exit 2
fi
if (( $# == 1 )); then
    RUN_NAMESPACE="$1"
    RESULT_STUDY_ROOT="${EVALUATION_ROOT}/staleness-ratio-sweep/${RUN_NAMESPACE}"
    ANALYSIS_ROOT="${RESULT_STUDY_ROOT}/analysis/${PROTOCOL_NAME}/${EVAL_MODE}"
    STALENESS_ROOT="${ANALYSIS_ROOT}/staleness"
fi
if [[ -f "${RESULT_STUDY_ROOT}/grid.env" ]]; then
    source "${RESULT_STUDY_ROOT}/grid.env"
fi
[[ "${BOOTSTRAP_SAMPLES}" =~ ^[0-9]+$ ]] || {
    echo "BOOTSTRAP_SAMPLES must be nonnegative" >&2
    exit 3
}
[[ "${SKIP_WANDB_EXPORT}" =~ ^[01]$ ]] || {
    echo "SKIP_WANDB_EXPORT must be 0 or 1" >&2
    exit 3
}

mkdir -p "${STALENESS_ROOT}"
"${ANALYSIS_PYTHON}" "${REPO_ROOT}/experiments/tools/reasoning_eval/summarize_results.py" \
    --result-study-root "${RESULT_STUDY_ROOT}" \
    --protocol-name "${PROTOCOL_NAME}" \
    --eval-mode "${EVAL_MODE}" \
    --output-dir "${ANALYSIS_ROOT}"
if (( SKIP_WANDB_EXPORT == 0 )); then
    "${WANDB_PYTHON}" "${REPO_ROOT}/experiments/tools/reasoning_eval/export_wandb_history.py" \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --namespace "${RUN_NAMESPACE}" \
        --output-csv "${STALENESS_ROOT}/training-history.csv"
else
    [[ -s "${STALENESS_ROOT}/training-history.csv" ]] || {
        echo "cached W&B history is missing: ${STALENESS_ROOT}/training-history.csv" >&2
        exit 4
    }
fi
analysis_args=(
    --aggregate-csv "${ANALYSIS_ROOT}/aggregate-results.csv"
    --training-history-csv "${STALENESS_ROOT}/training-history.csv"
    --output-dir "${STALENESS_ROOT}"
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
)
if [[ -s "${STALENESS_ROOT}/checkpoint-displacements.csv" \
    && -s "${STALENESS_ROOT}/checkpoint-displacements._SUCCESS" ]]; then
    analysis_args+=(
        --checkpoint-displacements-csv "${STALENESS_ROOT}/checkpoint-displacements.csv"
    )
fi
"${ANALYSIS_PYTHON}" "${REPO_ROOT}/experiments/tools/reasoning_eval/analyze_staleness.py" \
    "${analysis_args[@]}"
"${ANALYSIS_PYTHON}" "${REPO_ROOT}/experiments/tools/reasoning_eval/plot_staleness_analysis.py" \
    --checkpoint-series-csv "${STALENESS_ROOT}/checkpoint-series.csv" \
    --downstream-correlations-csv "${STALENESS_ROOT}/downstream-correlations.csv" \
    --staleness-correlations-csv "${STALENESS_ROOT}/staleness-metric-correlations.csv" \
    --wallclock-decomposition-csv "${STALENESS_ROOT}/wallclock-decomposition.csv" \
    --selected-relationships-json "${STALENESS_ROOT}/selected-relationships.json" \
    --output-dir "${STALENESS_ROOT}/figures"
"${ANALYSIS_PYTHON}" "${REPO_ROOT}/experiments/tools/reasoning_eval/plot_training_staleness.py" \
    --training-history-csv "${STALENESS_ROOT}/training-history.csv" \
    --staleness-correlations-csv "${STALENESS_ROOT}/staleness-metric-correlations.csv" \
    --output-dir "${STALENESS_ROOT}"

echo
cat "${STALENESS_ROOT}/staleness-summary.md"
echo "Analysis data: ${STALENESS_ROOT}"
echo "Figures: ${STALENESS_ROOT}/figures"
