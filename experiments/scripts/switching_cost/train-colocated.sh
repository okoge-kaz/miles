#!/bin/bash

set -euo pipefail
set -x

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface

if [[ "${WANDB_MODE}" == offline || "${WANDB_MODE}" == disabled ]]; then
    unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([[ "${NVLINK_COUNT}" -gt 0 ]] && echo 1 || echo 0)

cd /root/miles
source "/root/miles/scripts/models/${MODEL_SCRIPT}"
source /root/miles/experiments/common/ray_cluster.sh

CKPT_ARGS=(
    --hf-checkpoint "/ckpt/hf/${SFT_HF_RELATIVE}"
    --ref-load "/ckpt/megatron/${SFT_MEGATRON_RELATIVE}"
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-max-context-len 32768
    --rollout-temperature 1
    --rollout-top-p 1
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout 1
    --balance-data
    --zero-reward-on-truncated
    --partial-rollout
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --use-distributed-optimizer
    --clip-grad 1.0
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
)

GRPO_ARGS=(
    --seed 1234
    --rollout-seed 42
    --advantage-estimator grpo
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --calculate-per-token-loss
    --use-tis
    --tis-clip 2.0
    --tis-clip-low 0
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE}"
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --log-probs-chunk-size -1
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
    --update-weight-transfer-mode broadcast
)
if [[ "${CHECK_WEIGHT_UPDATE_EQUAL}" != 0 ]]; then
    SGLANG_ARGS+=(--check-weight-update-equal)
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

TELEMETRY_ARGS=()
if [[ "${LOG_COLOCATE_SWITCH_METRICS}" != 0 ]]; then
    TELEMETRY_ARGS+=(--log-colocate-switch-metrics)
fi
if [[ "${LOG_COLOCATE_TRANSFER_BYTES}" != 0 ]]; then
    TELEMETRY_ARGS+=(--log-colocate-transfer-bytes)
fi
if [[ "${LOG_MEMORY_USAGE}" != 0 ]]; then
    TELEMETRY_ARGS+=(--log-memory-usage)
fi
if [[ "${ENABLE_MILES_DASHBOARD}" != 0 ]]; then
    TELEMETRY_ARGS+=(--use-miles-dashboard)
fi
if [[ "${ENABLE_DUMP_DETAILS}" != 0 ]]; then
    TELEMETRY_ARGS+=(
        --dump-details "/root/miles/experiments/outputs/switching_cost/${RUN_NAME}/dump"
        --no-dump-policy-loss-debug
        --no-dump-train-data
    )
fi

WANDB_ARGS=(
    --use-wandb
    --wandb-mode "${WANDB_MODE}"
    --wandb-dir "/root/miles/experiments/outputs/switching_cost/${RUN_NAME}/wandb"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
)

RUNTIME_ENV_JSON=$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/:/root/miles",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "RAY_DEDUP_LOGS": "${RAY_DEDUP_LOGS}",
    "no_proxy": "127.0.0.1"
  }
}
JSON
)

ray job submit --address=http://127.0.0.1:8265 \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${TELEMETRY_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
