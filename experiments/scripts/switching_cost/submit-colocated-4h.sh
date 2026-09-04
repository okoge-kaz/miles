#!/bin/bash

# Submit the production four-hour switching-cost measurements.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiments/scripts/switching_cost/submit-colocated-4h.sh [all|qwen3-4b|qwen3-8b|qwen3-30b-a3b]

The submitted jobs explicitly enable colocated phase timing and transfer-byte
telemetry. Intrusive memory snapshots, dashboard RPCs, and event dumps remain
disabled. Set WANDB_MODE=offline to keep W&B local; the default is online.
EOF
}

case "${1:-all}" in
    all)
        MODEL_VARIANTS=(qwen3-4b qwen3-8b qwen3-30b-a3b)
        ;;
    qwen3-4b|qwen3-8b|qwen3-30b-a3b)
        MODEL_VARIANTS=("$1")
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)
cd "${REPO_ROOT}"
source experiments/env.sh

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
SWITCH_NODES="${SWITCH_NODES:-8}"
WANDB_MODE="${WANDB_MODE:-online}"

filtered_prompt_path() {
    case "$1" in
        qwen3-4b)
            echo /data/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl
            ;;
        qwen3-8b)
            echo /data/dapo-math-p10-90-qwen3-8b-base-lr1.5e-5-step4000/dapo-math-p10-90-qwen3-8b-base-lr1.5e-5-step4000.jsonl
            ;;
        qwen3-30b-a3b)
            echo /data/dapo-math-p10-90-qwen3-30b-a3b-base-lr2.0e-5-step4000/dapo-math-p10-90-qwen3-30b-a3b-base-lr2.0e-5-step4000.jsonl
            ;;
    esac
}

for model_variant in "${MODEL_VARIANTS[@]}"; do
    prompt_data=$(filtered_prompt_path "${model_variant}")
    prompt_host="${DATASET_DIR}/${prompt_data#/data/}"
    [[ -s "${prompt_host}" ]] || {
        echo "difficulty-filtered prompt file is not ready: ${prompt_host}" >&2
        echo "submit/inspect experiments/scripts/switching_cost/submit-difficulty-filters.sh" >&2
        exit 1
    }
done

for model_variant in "${MODEL_VARIANTS[@]}"; do
    prompt_data=$(filtered_prompt_path "${model_variant}")
    job_id=$(sbatch --parsable \
        -A "${ACCOUNT}" -p batch --qos=normal -N "${SWITCH_NODES}" --time=04:00:00 \
        --job-name="switch-cost-${model_variant}" \
        --export="ALL,MODEL_VARIANT=${model_variant},PROMPT_DATA=${prompt_data},WANDB_MODE=${WANDB_MODE},LOG_COLOCATE_SWITCH_METRICS=1,LOG_COLOCATE_TRANSFER_BYTES=1,LOG_MEMORY_USAGE=0,ENABLE_MILES_DASHBOARD=0,ENABLE_DUMP_DETAILS=0,RAY_DEDUP_LOGS=1" \
        experiments/scripts/switching_cost/run-colocated.sbatch)
    echo "${model_variant}: ${job_id}"
done
