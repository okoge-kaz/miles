#!/bin/bash

# Submit the matched 4B/8B/30B-A3B colocated partial-rollout switch study.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiments/scripts/switching_cost/submit-partial-rollout-4h.sh [--submit] [all|qwen3-4b|qwen3-8b|qwen3-30b-a3b]

Without --submit, print the three validated sbatch commands. Each submitted
job uses eight nodes for 04:00:00 and enables partial rollout with a 192-sample
accepted batch and a 256-sample oversampling batch. Runs are logged online to
the colcoated-switching-cost W&B project.
EOF
}

SUBMIT=0
if [[ "${1:-}" == --submit ]]; then
    SUBMIT=1
    shift
fi

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
if [[ $# -gt 0 ]]; then
    shift
fi
[[ $# -eq 0 ]] || {
    usage >&2
    exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)
cd "${REPO_ROOT}"
source experiments/env.sh

# The wrapper can run from the small CPU filter coordinator. Do not export that
# allocation's limits into the eight-node GPU jobs.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
SWITCH_NODES="${SWITCH_NODES:-8}"
SFT_HF_ROOT="${SHARED_WS}/checkpoints/huggingface"
ROLLOUT_BATCH_SIZE=192
OVER_SAMPLING_BATCH_SIZE=256
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=3072
WANDB_PROJECT=colcoated-switching-cost
WANDB_GROUP="${WANDB_GROUP:-colocated-partial-rollout-$(date +%Y%m%d)}"

: "${WANDB_API_KEY:?set WANDB_API_KEY, or put an api.wandb.ai entry in ~/.netrc}"

[[ "${SWITCH_NODES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "SWITCH_NODES must be a positive integer: ${SWITCH_NODES}" >&2
    exit 1
}
(( OVER_SAMPLING_BATCH_SIZE > ROLLOUT_BATCH_SIZE )) || {
    echo "partial rollout requires oversampling greater than the accepted batch" >&2
    exit 1
}
(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT == GLOBAL_BATCH_SIZE )) || {
    echo "partial-rollout batch shape is inconsistent" >&2
    exit 1
}

select_model_assets() {
    case "$1" in
        qwen3-4b)
            DATASET_NAME=dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000
            PASSRATE_NAME=dapo-math-17k.Qwen3-4B-Base-LR2e-5-Step4000.n16-len16384-zero-trunc.passrate.jsonl
            SFT_HF_RELATIVE=Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000/iter_0004000
            SFT_MEGATRON_RELATIVE=Qwen3-4B-Base-LR2e-5-Step4000_torch_dist
            ;;
        qwen3-8b)
            DATASET_NAME=dapo-math-p10-90-qwen3-8b-base-lr1.5e-5-step4000
            PASSRATE_NAME=dapo-math-17k.Qwen3-8B-Base-LR1.5e-5-Step4000.n16-len16384-zero-trunc.passrate.jsonl
            SFT_HF_RELATIVE=Qwen3-8B-Base/LR1.5e-5-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000
            SFT_MEGATRON_RELATIVE=Qwen3-8B-Base-LR1.5e-5-Step4000_torch_dist
            ;;
        qwen3-30b-a3b)
            DATASET_NAME=dapo-math-p10-90-qwen3-30b-a3b-base-lr2.0e-5-step4000
            PASSRATE_NAME=dapo-math-17k.Qwen3-30B-A3B-Base-LR2.0e-5-Step4000.n16-len16384-zero-trunc.passrate.jsonl
            SFT_HF_RELATIVE=Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000
            SFT_MEGATRON_RELATIVE=Qwen3-30B-A3B-Base-LR2.0e-5-Step4000_torch_dist
            ;;
    esac
    PROMPT_DATA="/data/${DATASET_NAME}/${DATASET_NAME}.jsonl"
    PROMPT_HOST="${DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.jsonl"
    PASSRATE_HOST="${DATASET_DIR}/difficulty/${PASSRATE_NAME}"
    HF_CHECKPOINT_HOST="${SFT_HF_ROOT}/${SFT_HF_RELATIVE}"
    MEGATRON_CHECKPOINT_HOST="${MEGATRON_CKPT_DIR}/${SFT_MEGATRON_RELATIVE}"
}

validate_model_assets() {
    local model_variant=$1
    local passrate_lines prompt_lines

    select_model_assets "${model_variant}"
    [[ -s "${HF_CHECKPOINT_HOST}/config.json" ]] || {
        echo "${model_variant}: missing HuggingFace config: ${HF_CHECKPOINT_HOST}/config.json" >&2
        return 1
    }
    [[ -s "${HF_CHECKPOINT_HOST}/model.safetensors" \
        || -s "${HF_CHECKPOINT_HOST}/model.safetensors.index.json" ]] || {
        echo "${model_variant}: missing HuggingFace weights: ${HF_CHECKPOINT_HOST}" >&2
        return 1
    }
    [[ -s "${MEGATRON_CHECKPOINT_HOST}/latest_checkpointed_iteration.txt" \
        && "$(<"${MEGATRON_CHECKPOINT_HOST}/latest_checkpointed_iteration.txt")" == release ]] || {
        echo "${model_variant}: incomplete Megatron checkpoint: ${MEGATRON_CHECKPOINT_HOST}" >&2
        return 1
    }
    [[ -s "${PASSRATE_HOST}" && -f "${PASSRATE_HOST}.complete" ]] || {
        echo "${model_variant}: difficulty measurement is incomplete: ${PASSRATE_HOST}" >&2
        return 1
    }
    passrate_lines=$(wc -l < "${PASSRATE_HOST}")
    [[ "${passrate_lines}" -eq 17398 ]] || {
        echo "${model_variant}: expected 17398 pass-rate rows, found ${passrate_lines}: ${PASSRATE_HOST}" >&2
        return 1
    }

    [[ -s "${PROMPT_HOST}" ]] || {
        echo "${model_variant}: filtered prompt file is missing or empty: ${PROMPT_HOST}" >&2
        return 1
    }
    prompt_lines=$(wc -l < "${PROMPT_HOST}")
    (( prompt_lines >= ROLLOUT_BATCH_SIZE )) || {
        echo "${model_variant}: filtered prompt file has only ${prompt_lines} rows: ${PROMPT_HOST}" >&2
        return 1
    }
    echo "validated ${model_variant}: checkpoints=ready, pass-rate=${passrate_lines}, filtered=${prompt_lines}, prompt=${PROMPT_DATA}"
}

for model_variant in "${MODEL_VARIANTS[@]}"; do
    validate_model_assets "${model_variant}"
done

for model_variant in "${MODEL_VARIANTS[@]}"; do
    select_model_assets "${model_variant}"
    sbatch_args=(
        -A "${ACCOUNT}"
        -p batch
        -N "${SWITCH_NODES}"
        --time=04:00:00
        --job-name="switch-partial-${model_variant}"
        --export="ALL,MODEL_VARIANT=${model_variant},PROMPT_DATA=${PROMPT_DATA},WANDB_MODE=online,WANDB_PROJECT=${WANDB_PROJECT},WANDB_GROUP=${WANDB_GROUP},ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE},OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE},N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT},GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE},LOG_COLOCATE_SWITCH_METRICS=1,LOG_COLOCATE_TRANSFER_BYTES=1,LOG_MEMORY_USAGE=0,ENABLE_MILES_DASHBOARD=0,ENABLE_DUMP_DETAILS=0,RAY_DEDUP_LOGS=1"
        experiments/scripts/switching_cost/run-colocated.sbatch
    )

    if (( SUBMIT == 0 )); then
        printf 'sbatch'
        printf ' %q' "${sbatch_args[@]}"
        printf '\n'
        continue
    fi

    job_id=$(sbatch --parsable "${sbatch_args[@]}")
    echo "submitted ${model_variant}: job=${job_id} project=${WANDB_PROJECT} group=${WANDB_GROUP} partial=192/256"
done
