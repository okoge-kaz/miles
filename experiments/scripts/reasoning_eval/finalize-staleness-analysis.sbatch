#!/bin/bash
#PBS -N reason-eval-analysis
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -j oe


set -euo pipefail

REPO_ROOT="$(realpath -- "${PBS_O_WORKDIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"

: "${RUN_NAMESPACE:?set RUN_NAMESPACE to the training namespace}"

EVAL_MODE="${EVAL_MODE:-full}"
PROTOCOL_NAME="${PROTOCOL_NAME:-eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-thinking-t0.6-p0.95-k20-aime64-v1}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${WS}/evaluations/reasoning_eval}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-5000}"
ANALYSIS_ROOT="${EVALUATION_ROOT}/staleness-ratio-sweep/${RUN_NAMESPACE}/analysis/${PROTOCOL_NAME}/${EVAL_MODE}"
STALENESS_ROOT="${ANALYSIS_ROOT}/staleness"
SUCCESS_MARKER="${ANALYSIS_ROOT}/staleness/_FINAL_ANALYSIS_SUCCESS"
WANDB_ENTITY="${WANDB_ENTITY:-ai-horizons}"
WANDB_PROJECT="${WANDB_PROJECT:-async-rl-dapo-math}"
WANDB_MAX_WORKERS="${WANDB_MAX_WORKERS:-4}"
WANDB_CONTAINER_IMAGE="${WANDB_CONTAINER_IMAGE:-${CONTAINER_IMAGE}}"
WANDB_NETRC="${WANDB_NETRC:-${HOME}/.netrc}"

[[ "${WANDB_MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "WANDB_MAX_WORKERS must be positive" >&2
    exit 2
}
[[ -r "${WANDB_CONTAINER_IMAGE}" ]] || {
    echo "W&B container is not readable: ${WANDB_CONTAINER_IMAGE}" >&2
    exit 3
}

mkdir -p "${STALENESS_ROOT}/.wandb-cache"
container_mounts="${REPO_ROOT}:/root/miles:ro,${STALENESS_ROOT}:/analysis"
if [[ -r "${WANDB_NETRC}" ]]; then
    container_mounts+=",${WANDB_NETRC}:/root/.netrc:ro"
elif [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "W&B authentication is unavailable: ${WANDB_NETRC}" >&2
    exit 3
fi

miles_srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${PBS_CPU_CPUS_PER_NODE}" \
    --image="${WANDB_CONTAINER_IMAGE}" \
    --bind="${container_mounts}" \
    --no-home \
    --export=ALL,WANDB_CACHE_DIR=/analysis/.wandb-cache \
    python /root/miles/experiments/tools/reasoning_eval/export_wandb_history.py \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --namespace "${RUN_NAMESPACE}" \
        --output-csv /analysis/training-history.csv \
        --metadata-json /analysis/training-history.metadata.json \
        --max-workers "${WANDB_MAX_WORKERS}"

RUN_NAMESPACE="${RUN_NAMESPACE}" \
EVAL_MODE="${EVAL_MODE}" \
PROTOCOL_NAME="${PROTOCOL_NAME}" \
EVALUATION_ROOT="${EVALUATION_ROOT}" \
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES}" \
SKIP_WANDB_EXPORT=1 \
    "${REPO_ROOT}/experiments/scripts/reasoning_eval/show-staleness-analysis.sh"

printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SUCCESS_MARKER}"
echo "Final analysis complete: ${ANALYSIS_ROOT}"
