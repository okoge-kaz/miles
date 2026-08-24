#!/bin/bash

#SBATCH --job-name=reason-eval-analysis
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/reasoning_eval/setup/%x-%j.log

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"

: "${RUN_NAMESPACE:?set RUN_NAMESPACE to the training namespace}"

EVAL_MODE="${EVAL_MODE:-full}"
PROTOCOL_NAME="${PROTOCOL_NAME:-eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-thinking-t0.6-p0.95-k20-aime64-v1}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${WS}/evaluations/reasoning_eval}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-5000}"
ANALYSIS_ROOT="${EVALUATION_ROOT}/staleness-ratio-sweep/${RUN_NAMESPACE}/analysis/${PROTOCOL_NAME}/${EVAL_MODE}"
SUCCESS_MARKER="${ANALYSIS_ROOT}/staleness/_FINAL_ANALYSIS_SUCCESS"

RUN_NAMESPACE="${RUN_NAMESPACE}" \
EVAL_MODE="${EVAL_MODE}" \
PROTOCOL_NAME="${PROTOCOL_NAME}" \
EVALUATION_ROOT="${EVALUATION_ROOT}" \
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES}" \
    "${REPO_ROOT}/experiments/scripts/reasoning_eval/show-staleness-analysis.sh"

printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SUCCESS_MARKER}"
echo "Final analysis complete: ${ANALYSIS_ROOT}"
