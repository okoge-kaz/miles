#!/bin/bash

set -ex

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd /root/miles
source /root/miles/scripts/models/qwen3-8B.sh
source /root/miles/experiments/common/ray_cluster.sh

CKPT_ARGS=(
   --hf-checkpoint /ckpt/hf/Qwen3-8B
   --ref-load      /ckpt/megatron/Qwen3-8B_torch_dist
   --load          "${CKPT_PATH}"
   --save          "${CKPT_PATH}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --fully-async
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type "${RM_TYPE}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-max-context-len 32768
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --max-weight-staleness "${MAX_WEIGHT_STALENESS}"
   --pause-generation-mode "${PAUSE_GENERATION_MODE}"
)

TELEMETRY_ARGS=(
   --dump-details "${CKPT_PATH}/dump"
   --use-miles-dashboard
)
if [[ "${DUMP_TRAIN_DATA}" == "0" ]]; then
   TELEMETRY_ARGS+=(--no-dump-train-data)
else
   TELEMETRY_ARGS+=(--use-rollout-entropy)
fi

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
   --eval-top-p 1
   --eval-prompt-data
   aime24 /data/aime-2024/aime-2024.jsonl
   aime25 /data/aime-2025/aime-2025.jsonl
)
if [[ "${SKIP_EVAL_BEFORE_TRAIN}" != "0" ]]; then
   EVAL_ARGS+=(--skip-eval-before-train)
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE}"
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [[ "${CONTEXT_PARALLEL_SIZE}" -gt 1 ]]; then
   PERF_ARGS+=(--cp-comm-type a2a)
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)
if awk "BEGIN{exit !(${KL_LOSS_COEF} != 0)}"; then
   GRPO_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF}" --kl-loss-type low_var_kl)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
)
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]]; then
   SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${SGLANG_CUDA_GRAPH_MAX_BS:-}" ]]; then
   SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "off-policy-${DATASET_TAG}"
   --wandb-group "${RUN_NAME}"
   --wandb-key "${WANDB_API_KEY}"
)

RUNTIME_ENV_JSON=$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/:/root/miles",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "no_proxy": "127.0.0.1"
  }
}
JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${TELEMETRY_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
