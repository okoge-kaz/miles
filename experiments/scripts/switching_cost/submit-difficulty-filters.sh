#!/bin/bash

# Submit resumable policy-specific DAPO-Math filters for the 8B/30B SFT models.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiments/scripts/switching_cost/submit-difficulty-filters.sh [all|qwen3-8b|qwen3-30b-a3b]

Each model first runs a 32-prompt smoke. Two four-hour batch/interactive-QoS
half-dataset shards then run after the smoke succeeds. A CPU continuation job
resubmits incomplete shards and materializes the model-specific p10-90 dataset.
EOF
}

case "${1:-all}" in
    all)
        MODEL_VARIANTS=(qwen3-8b qwen3-30b-a3b)
        ;;
    qwen3-8b|qwen3-30b-a3b)
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

export HF_CKPT_DIR="${HF_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface}"
source experiments/env.sh

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
FILTER_SMOKE="${FILTER_SMOKE:-1}"
FILTER_MAX_ROUNDS="${FILTER_MAX_ROUNDS:-8}"

measurement_exports() {
    # Filtering is done only after both complete measurement shards are merged.
    # Explicit empty values prevent the FILTERED_OUTPUT exported by the shared
    # config from making smoke/half-shard jobs materialize an incomplete sweep.
    printf '%s' "ALL,HF_CKPT_DIR=${HF_CKPT_DIR},MODEL_VARIANT=${MODEL_VARIANT},MODEL_NAME=${MODEL_NAME},HF_MODEL_NAME=${HF_MODEL_NAME},PROMPT_DATA=${PROMPT_DATA},N_SAMPLES=16,SAMPLES_PER_REQUEST=8,MAX_NEW_TOKENS=16384,MAX_CONTEXT_LENGTH=32768,CONCURRENCY=${CONCURRENCY},MAX_RUNNING_REQUESTS=80,SGLANG_TP_SIZE=${SGLANG_TP_SIZE},SGLANG_DP_SIZE=${SGLANG_DP_SIZE},MEM_FRACTION_STATIC=0.85,RM_TYPE=deepscaler,ZERO_REWARD_ON_TRUNCATED=1,PASS_RATE_MIN=0.1,PASS_RATE_MAX=0.9,FILTERED_OUTPUT=,ANNOTATED_OUTPUT="
}

submit_model() {
    source "${SCRIPT_DIR}/difficulty-filter-config.sh"

    local checkpoint="${HF_CKPT_DIR%/}/${HF_MODEL_NAME}"
    [[ -s "${checkpoint}/config.json" ]] || {
        echo "missing SFT config: ${checkpoint}/config.json" >&2
        return 1
    }
    [[ -s "${checkpoint}/model.safetensors" || -s "${checkpoint}/model.safetensors.index.json" ]] || {
        echo "missing SFT weights: ${checkpoint}" >&2
        return 1
    }

    local smoke_dependency=()
    local smoke_job=none
    if [[ "${FILTER_SMOKE}" != 0 ]]; then
        local smoke_output="/data/difficulty/smoke/${PASS_RATES##*/}"
        local smoke_audit="/data/difficulty/smoke/${AUDIT_OUTPUT##*/}"
        smoke_job=$(sbatch --parsable \
            -A "${ACCOUNT}" -p batch --qos=interactive --time=01:00:00 \
            --job-name="dfsmoke-${FILTER_JOB_SUFFIX}" \
            --export="$(measurement_exports),OUTPUT=${smoke_output},LIMIT=32,DUMP_RESPONSES=${smoke_audit},DUMP_LIMIT=8" \
            experiments/tools/difficulty_filter/run_measure.sbatch)
        echo "${MODEL_VARIANT}: submitted smoke=${smoke_job}" >&2
        smoke_dependency=(--dependency="afterok:${smoke_job}")
    fi

    local shard_jobs=()
    local shard start end output audit job
    for shard in 0 1; do
        if [[ "${shard}" == 0 ]]; then
            start=0
            end="${HALF_PROMPTS}"
        else
            start="${HALF_PROMPTS}"
            end="${TOTAL_PROMPTS}"
        fi
        output="${PASS_RATES%.jsonl}.shard-${shard}-of-2.jsonl"
        audit="${AUDIT_OUTPUT%.jsonl}.shard-${shard}-of-2.jsonl"
        job=$(sbatch --parsable \
            -A "${ACCOUNT}" -p batch --qos=interactive --time=04:00:00 \
            --job-name="df${shard}-${FILTER_JOB_SUFFIX}" \
            "${smoke_dependency[@]}" \
            --export="$(measurement_exports),OUTPUT=${output},START_INDEX=${start},END_INDEX=${end},DUMP_RESPONSES=${audit},DUMP_LIMIT=8" \
            experiments/tools/difficulty_filter/run_measure.sbatch)
        echo "${MODEL_VARIANT}: submitted shard-${shard}=${job}" >&2
        shard_jobs+=("${job}")
    done

    local dependency
    dependency=$(IFS=:; echo "${shard_jobs[*]}")
    local continuation_job
    continuation_job=$(sbatch --parsable \
        -A "${ACCOUNT}" -p cpu --qos=cpu-interactive --time=00:20:00 \
        --job-name="dfcont-${FILTER_JOB_SUFFIX}" \
        --dependency="afterany:${dependency}" \
        --export="ALL,HF_CKPT_DIR=${HF_CKPT_DIR},MODEL_VARIANT=${MODEL_VARIANT},FILTER_ROUND=1,FILTER_MAX_ROUNDS=${FILTER_MAX_ROUNDS}" \
        experiments/scripts/switching_cost/continue-difficulty-filter.sbatch)
    echo "${MODEL_VARIANT}: submitted continuation=${continuation_job}" >&2

    echo "${MODEL_VARIANT}: smoke=${smoke_job} shards=${shard_jobs[*]} continuation=${continuation_job}"
}

for MODEL_VARIANT in "${MODEL_VARIANTS[@]}"; do
    export MODEL_VARIANT
    submit_model
done
