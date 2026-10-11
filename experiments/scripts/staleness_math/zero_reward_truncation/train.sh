#!/bin/bash

set -euo pipefail

MILES_REPO="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "${MILES_REPO}/experiments/common/training_config.sh"

# --- Model and dataset presets -----------------------------------------------
: "${MODEL_PRESET:=qwen3-4b}"
: "${DATASET_PRESET:=dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000}"

for preset_name in MODEL_PRESET DATASET_PRESET; do
    [[ "${!preset_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid ${preset_name}" >&2; exit 2; }
done
unset preset_name

source "${MILES_REPO}/experiments/scripts/models/${MODEL_PRESET}.sh"
source "${MILES_REPO}/experiments/scripts/datasets/train/${DATASET_PRESET}.sh"

# List every training setting here, including values supplied by a preset.
# Exported values and preset defaults take precedence over these fallbacks.
# Fixed official CLI values are listed in the argument blocks below.

# --- Model ------------------------------------------------------------------
: "${MODEL_NAME:?selected model preset must set MODEL_NAME}"
: "${HF_MODEL_NAME:?selected model preset must set HF_MODEL_NAME}"
: "${MODEL_PROFILE:?official scripts/models/<profile>.py architecture}"
: "${MEGATRON_TO_HF_MODE:=raw}"

# --- Experiment and placement -----------------------------------------------
export PROJECT_NAME=staleness_math
export EXPERIMENT_NAME=zero_reward_truncation
# Set none to use the built-in reward without a truncation-specific override.
export TRUNCATION_MODE="${TRUNCATION_MODE:-zero_reward}"
export PLACEMENT="${PLACEMENT:-disaggregate}"
: "${GPUS_PER_NODE:=8}"
: "${ACTOR_GPUS_PER_NODE:=${GPUS_PER_NODE}}"
if [[ "${PLACEMENT}" == disaggregate ]]; then
    : "${ACTOR_NUM_NODES:=1}"
    : "${ROLLOUT_NUM_GPUS:=$(( (MILES_JOB_NUM_NODES - ACTOR_NUM_NODES) * GPUS_PER_NODE ))}"
else
    : "${ACTOR_NUM_NODES:=${MILES_JOB_NUM_NODES}}"
    : "${ROLLOUT_NUM_GPUS:=0}"
fi

# --- Checkpoint -------------------------------------------------------------
# run_identity.sh validates these names and paths before launch.
: "${RUN_NAME:=${EXPERIMENT_NAME}-${PLACEMENT}-${MILES_JOB_ID_SHORT}}"
: "${CKPT_PATH:=/ckpt/training/v0.1.1/${PROJECT_NAME}/${EXPERIMENT_NAME}/${DATASET_TAG}/${MODEL_NAME}/${PLACEMENT}/${RUN_NAME}}"
: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=100}"
: "${SAVE_HF:=1}"
: "${HF_SAVE_INTERVAL:=}"
: "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:=1}"
: "${CLEAN_CHECKPOINT:=0}"

# --- Rollout and reward ------------------------------------------------------
: "${DATASET_TAG:?selected train dataset preset must set DATASET_TAG}"
: "${PROMPT_DATA:=}"
: "${PROMPT_DATA_CONFIG:=}"
: "${PROMPT_DATA_FILES:=}"
: "${PROMPT_DATA_RATIOS:=}"
: "${PROMPT_DATA_EPOCH_SIZE:=}"
: "${INPUT_KEY:=prompt}"
: "${LABEL_KEY:=label}"
: "${RM_TYPE:=deepscaler}"
: "${NUM_ROLLOUT:=300}"
: "${ROLLOUT_BATCH_SIZE:=192}"
: "${N_SAMPLES_PER_PROMPT:=16}"
: "${GLOBAL_BATCH_SIZE:=3072}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_CONTEXT_LEN:=32768}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"

# --- Fully async ------------------------------------------------------------
: "${MAX_WEIGHT_STALENESS:=8}"
: "${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:=30}"
: "${ASYNC_UNUSED_SAMPLES_HANDLER:=retry}"
: "${ASYNC_MAX_CONCURRENT_SAMPLES:=4096}"
# Empty concurrency uses the official rollout_batch_size * n_samples_per_prompt.
: "${PAUSE_GENERATION_MODE:=in_place}"
: "${UPDATE_WEIGHT_TRANSFER_MODE:=broadcast}"
: "${ASYNC_DATA_BUFFER_ORDER:=fifo}"
if [[ "${PLACEMENT}" == disaggregate ]]; then
    : "${ASYNC_STALENESS_METRIC:=first-prefill}"
else
    : "${ASYNC_STALENESS_METRIC:=oldest}"
fi
: "${CUSTOM_STALENESS_FUNCTION_PATH:=}"
# Legacy environment aliases are fallbacks; the new names take precedence.
: "${USE_REPLAY_DATA_BUFFER:=${USE_REPLAY_BUFFER:-0}}"
: "${REPLAY_DATA_BUFFER_ASYNC_RESTORE:=${REPLAY_BUFFER_ASYNC_RESTORE:-0}}"
: "${REPLAY_DATA_BUFFER_MODE:=rollout}"

# --- Optimizer and GRPO/TIS --------------------------------------------------
: "${LR:=1e-6}"
: "${ENTROPY_COEF:=0.00}"
: "${EPS_CLIP:=0.2}"
: "${EPS_CLIP_HIGH:=0.28}"
: "${IS_CORRECTION:=tis}"
: "${TIS_CLIP:=2.0}"
: "${TIS_CLIP_LOW:=0}"
: "${SKIP_ACTOR_FORWARD_ONLY:=1}"

# --- Parallelism and training performance -----------------------------------
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=1}"
: "${EXPERT_TENSOR_PARALLEL_SIZE:=1}"
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${TRAINING_ATTENTION_BACKEND:=fused}"

# --- SGLang -----------------------------------------------------------------
if [[ "${PLACEMENT}" == disaggregate ]]; then
    : "${ROLLOUT_NUM_GPUS_PER_ENGINE:=1}"
    : "${SGLANG_MEM_FRACTION:=0.70}"
else
    : "${ROLLOUT_NUM_GPUS_PER_ENGINE:=2}"
    : "${SGLANG_MEM_FRACTION:=0.65}"
fi
: "${SGLANG_MAX_RUNNING_REQUESTS:=}"
: "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS:=1}"

# --- Evaluation -------------------------------------------------------------
# In-run evaluation is OFF; evaluate saved checkpoints in a separate job.
export EVAL_INTERVAL=0

# --- Telemetry and W&B -------------------------------------------------------
# Rollout/trajectory dumps are OFF by default to avoid throughput overhead.
: "${DUMP_ROLLOUT_DATA:=0}"
: "${LOG_ROLLOUT_REPETITION:=0}"
: "${LOG_CATEGORY_STALENESS:=0}"
: "${LOG_POLICY_LAG_METRICS:=1}"
: "${WANDB_MODE:=online}"
: "${WANDB_PROJECT:=miles-v0.1.1-abci-math}"
: "${WANDB_TEAM:=}"

case "${TRUNCATION_MODE}" in
    zero_reward|none) ;;
    *) echo "TRUNCATION_MODE must be none or zero_reward for this recipe" >&2; exit 2 ;;
esac

miles_training_defaults

miles_recipe_training_args() {
    # Model architecture flags are resolved by the official MODEL_PROFILE loader.
    # Every other training flag, including fixed values, is owned by this recipe.
    local -a PLACEMENT_ARGS=(
        --num-gpus-per-node "${GPUS_PER_NODE}"
        --actor-num-nodes "${ACTOR_NUM_NODES}"
        --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
    )
    if [[ "${PLACEMENT}" == disaggregate ]]; then
        PLACEMENT_ARGS+=(
            --fully-async
            --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
            --async-data-buffer-capacity-factor "${ASYNC_DATA_BUFFER_CAPACITY_FACTOR}"
            --async-unused-samples-handler "${ASYNC_UNUSED_SAMPLES_HANDLER}"
            --async-data-buffer-order "${ASYNC_DATA_BUFFER_ORDER}"
            --async-staleness-metric "${ASYNC_STALENESS_METRIC}"
            --pause-generation-mode "${PAUSE_GENERATION_MODE}"
            --update-weight-transfer-mode "${UPDATE_WEIGHT_TRANSFER_MODE}"
        )
        [[ "${MAX_WEIGHT_STALENESS}" == none ]] || PLACEMENT_ARGS+=(--max-weight-staleness "${MAX_WEIGHT_STALENESS}")
        [[ -z "${ASYNC_MAX_CONCURRENT_SAMPLES}" ]] || PLACEMENT_ARGS+=(--async-max-concurrent-samples "${ASYNC_MAX_CONCURRENT_SAMPLES}")
        [[ -z "${CUSTOM_STALENESS_FUNCTION_PATH}" ]] || PLACEMENT_ARGS+=(--custom-staleness-function-path "${CUSTOM_STALENESS_FUNCTION_PATH}")
        [[ "${USE_REPLAY_DATA_BUFFER}" == 0 ]] || PLACEMENT_ARGS+=(
            --use-replay-data-buffer --replay-data-buffer-mode "${REPLAY_DATA_BUFFER_MODE}"
        )
        [[ "${REPLAY_DATA_BUFFER_ASYNC_RESTORE}" == 0 ]] || PLACEMENT_ARGS+=(--replay-data-buffer-async-restore)
        if [[ "${UPDATE_WEIGHT_TRANSFER_MODE}" == p2p ]]; then
            PLACEMENT_ARGS+=(--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine)
        fi
    else
        PLACEMENT_ARGS+=(--colocate)
    fi

    local -a CKPT_ARGS=(
        --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}"
        --ref-load "/ckpt/megatron/${MODEL_NAME}_torch_dist"
        --load "${CKPT_PATH}"
        --save "${CKPT_PATH}"
        --save-interval "${SAVE_INTERVAL}"
    )
    [[ "${SAVE_RETAIN_INTERVAL}" == none ]] || CKPT_ARGS+=(--save-retain-interval "${SAVE_RETAIN_INTERVAL}")
    [[ -z "${HF_SAVE_INTERVAL}" ]] || CKPT_ARGS+=(--hf-save-interval "${HF_SAVE_INTERVAL}")
    if [[ "${SAVE_HF}" == 1 ]]; then
        CKPT_ARGS+=(--save-hf "${CKPT_PATH}/hf/{rollout_id}")
    fi
    if [[ "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER}" == 1 ]]; then
        CKPT_ARGS+=(--use-checkpoint-opt-param-scheduler)
    fi

    local -a PROMPT_ARGS=() prompt_files=() prompt_ratios=()
    if [[ -n "${PROMPT_DATA_CONFIG}" ]]; then
        PROMPT_ARGS=(--prompt-data-config "${PROMPT_DATA_CONFIG}")
    elif [[ -n "${PROMPT_DATA_FILES}" ]]; then
        mapfile -d '' -t prompt_files < <(PYTHONPATH="${MILES_REPO}${PYTHONPATH:+:${PYTHONPATH}}" python3 -m experiments.common.dataset_args --json-array PROMPT_DATA_FILES)
        (( ${#prompt_files[@]} > 0 )) || return 2
        PROMPT_ARGS=(--prompt-data-files "${prompt_files[@]}")
        if [[ -n "${PROMPT_DATA_RATIOS}" ]]; then
            mapfile -d '' -t prompt_ratios < <(PYTHONPATH="${MILES_REPO}${PYTHONPATH:+:${PYTHONPATH}}" python3 -m experiments.common.dataset_args --json-array PROMPT_DATA_RATIOS)
            (( ${#prompt_ratios[@]} == ${#prompt_files[@]} )) || return 2
            PROMPT_ARGS+=(--prompt-data-ratios "${prompt_ratios[@]}")
        fi
    else
        PROMPT_ARGS=(--prompt-data "${PROMPT_DATA}")
    fi
    [[ -z "${PROMPT_DATA_EPOCH_SIZE}" ]] || PROMPT_ARGS+=(--prompt-data-epoch-size "${PROMPT_DATA_EPOCH_SIZE}")
    local -a ROLLOUT_ARGS=(
        --rm-type "${RM_TYPE}"
        --num-rollout "${NUM_ROLLOUT}"
        --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
        --rollout-max-response-len "${MAX_RESPONSE_LEN}"
        --rollout-max-context-len "${MAX_CONTEXT_LEN}"
        --global-batch-size "${GLOBAL_BATCH_SIZE}"
        --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
        --seed "${TRAIN_SEED}"
        --rollout-seed "${ROLLOUT_SEED}"
        --input-key "${INPUT_KEY}"
        --label-key "${LABEL_KEY}"
        --apply-chat-template
        --rollout-shuffle
        --rollout-temperature 1
        --rollout-top-p 1
        --rollout-top-k -1
        --balance-data
    )

    local -a OPTIMIZER_ARGS=(
        --optimizer adam
        --lr "${LR}"
        --lr-decay-style constant
        --weight-decay 0.1
        --adam-beta1 0.9
        --adam-beta2 0.98
    )

    [[ "${TRUNCATION_MODE}" != zero_reward ]] || ROLLOUT_ARGS+=(--custom-rm-path experiments.src.training.truncation.zero_reward)

    local -a GRPO_ARGS=(
        --advantage-estimator grpo
        --kl-coef 0
        --kl-loss-coef 0
        --entropy-coef "${ENTROPY_COEF}"
        --eps-clip "${EPS_CLIP}"
        --eps-clip-high "${EPS_CLIP_HIGH}"
    )
    if [[ "${IS_CORRECTION}" == tis ]]; then
        GRPO_ARGS+=(--use-tis --tis-clip "${TIS_CLIP}" --tis-clip-low "${TIS_CLIP_LOW}")
    fi
    if [[ "${SKIP_ACTOR_FORWARD_ONLY}" == 1 ]]; then
        GRPO_ARGS+=(--skip-actor-forward-only)
    fi

    local -a PERF_ARGS=(
        --train-backend megatron
        --bf16
        --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}"
        --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
        --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
        --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
        --attention-backend "${TRAINING_ATTENTION_BACKEND}"
        --pipeline-model-parallel-size "${PIPELINE_PARALLEL_SIZE}"
        --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE}"
        --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --use-dynamic-batch-size
        --attention-dropout 0
        --hidden-dropout 0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --sglang-chunked-prefill-size 4096
    )
    if (( TENSOR_PARALLEL_SIZE > 1 )); then
        PERF_ARGS+=(--sequence-parallel)
    fi
    if [[ "${MEGATRON_TO_HF_MODE}" == bridge ]]; then
        PERF_ARGS+=(--megatron-to-hf-mode bridge)
    fi
    [[ -z "${SGLANG_MAX_RUNNING_REQUESTS}" ]] || PERF_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
    if [[ "${PLACEMENT}" == disaggregate ]]; then
        PERF_ARGS+=(--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}')
    fi

    if [[ "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS}" == 1 ]]; then
        PERF_ARGS+=(
            --sglang-enable-response-weight-version-segments
            --sglang-weight-version 0
        )
    fi

    local -a TELEMETRY_ARGS=()
    if [[ "${LOG_ROLLOUT_REPETITION}" == 1 ]]; then
        TELEMETRY_ARGS+=(--log-rollout-repetition)
    else
        TELEMETRY_ARGS+=(--no-log-rollout-repetition)
    fi
    [[ "${LOG_CATEGORY_STALENESS}" == 0 ]] || TELEMETRY_ARGS+=(--log-category-staleness)
    [[ "${LOG_POLICY_LAG_METRICS}" == 0 ]] || TELEMETRY_ARGS+=(--log-policy-lag-metrics)
    if [[ "${DUMP_ROLLOUT_DATA}" == 1 ]]; then
        TELEMETRY_ARGS+=(
            --save-debug-rollout-data "${CKPT_PATH}/dump/rollout_{rollout_id}.pt"
            --save-debug-trajectory-data "${CKPT_PATH}/dump/trajectory_{rollout_id}.jsonl"
        )
    fi

    # In-run eval is OFF; checkpoint evaluation runs separately.
    local -a EVAL_ARGS=(--skip-eval-before-train)

    local -a WANDB_ARGS=()
    if [[ "${WANDB_MODE}" != disabled ]]; then
        WANDB_ARGS+=(
            --use-wandb
            --wandb-project "${WANDB_PROJECT}"
            --wandb-group "${RUN_NAME}"
        )
        [[ -z "${WANDB_TEAM}" ]] || WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
    fi

    printf '%s\0' \
        "${PLACEMENT_ARGS[@]}" "${CKPT_ARGS[@]}" "${PROMPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" "${GRPO_ARGS[@]}" "${PERF_ARGS[@]}" \
        "${TELEMETRY_ARGS[@]}" "${EVAL_ARGS[@]}" "${WANDB_ARGS[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "${1:-}" == --print-args ]]; then
        source "${MILES_REPO}/experiments/common/run_identity.sh"
        miles_recipe_training_args
    else
        exec bash "${MILES_REPO}/experiments/common/train.sh" "${BASH_SOURCE[0]}"
    fi
fi
