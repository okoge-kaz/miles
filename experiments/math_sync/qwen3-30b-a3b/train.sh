#!/bin/bash

set -ex

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd /root/miles
source /root/miles/scripts/models/qwen3-30B-A3B.sh

RUN_NAME="${RUN_NAME:-math-grpo-qwen3-30b-a3b}"

source /root/miles/experiments/lib/ray_cluster.sh

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${NNODES}}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-${ACTOR_GPUS:-${GPUS_PER_NODE}}}"
TRAIN_WORLD=$(( ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE ))

_TP="${TENSOR_PARALLEL_SIZE:-4}"
_CP="${CONTEXT_PARALLEL_SIZE:-1}"
_PP=1
if (( TRAIN_WORLD % (_TP * _CP * _PP) != 0 )); then
    echo "tp${_TP} * cp${_CP} * pp${_PP} does not divide ${TRAIN_WORLD} training GPUs" >&2
    exit 1
fi
_DP=$(( TRAIN_WORLD / (_TP * _CP * _PP) ))
if (( ${GLOBAL_BATCH_SIZE:-256} % _DP != 0 )); then
    echo "global_batch_size ${GLOBAL_BATCH_SIZE:-256} is not divisible by dp ${_DP}" >&2
    exit 1
fi
echo "placement: ${ACTOR_NUM_NODES} node(s) x ${ACTOR_GPUS_PER_NODE} GPU" \
     "= ${TRAIN_WORLD} training GPUs, tp${_TP} cp${_CP} -> dp${_DP}"

TASK_FAMILY="${TASK_FAMILY:-math}"
PROMPT_DATA="${PROMPT_DATA:-/data/dapo-math-17k/dapo-math-17k.jsonl}"
DATASET_TAG="${DATASET_TAG:-$(basename "$(dirname "${PROMPT_DATA}")")}"
LR="${LR:-1e-6}"
CONFIG_TAG="${CONFIG_TAG:-async-off-${NUM_STEPS_PER_ROLLOUT:-1}step-rollout-length-$(( ${MAX_RESPONSE_LEN:-24576} / 1024 ))k-lr${LR}}"
CKPT_PATH="/ckpt/training/${TASK_FAMILY}/${DATASET_TAG}/Qwen3-30B-A3B/${CONFIG_TAG}"
echo "checkpoints: ${CKPT_PATH}"

CKPT_ARGS=(
   --hf-checkpoint /ckpt/hf/Qwen3-30B-A3B
   --ref-load      /ckpt/megatron/Qwen3-30B-A3B_torch_dist
   --load          "${CKPT_PATH}"
   --save          "${CKPT_PATH}"
   --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type "${RM_TYPE:-deepscaler}"
   --num-rollout "${NUM_ROLLOUT:-3000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN:-24576}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
   --rollout-temperature 1

   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}"
   --balance-data
)

if [[ "${DYNAMIC_SAMPLING:-1}" != "0" ]]; then
   ROLLOUT_ARGS+=(
      --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
      --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE:-$(( ${ROLLOUT_BATCH_SIZE:-32} * 2 ))}"
   )
fi

if [[ "${PARTIAL_ROLLOUT:-1}" != "0" ]]; then
   ROLLOUT_ARGS+=(--partial-rollout)
   if [[ "${MASK_OFFPOLICY:-0}" != "0" ]]; then
      ROLLOUT_ARGS+=(--mask-offpolicy-in-partial-rollout)
   fi
fi

TELEMETRY_ARGS=()
if [[ "${DUMP_DETAILS:-1}" != "0" ]]; then
   TELEMETRY_ARGS+=(--dump-details "${DUMP_DIR:-${CKPT_PATH}/dump}")
   if [[ "${USE_DASHBOARD:-1}" != "0" ]]; then
      TELEMETRY_ARGS+=(--use-miles-dashboard)
   fi
   if [[ "${ROLLOUT_ENTROPY:-1}" != "0" ]]; then
      TELEMETRY_ARGS+=(--use-rollout-entropy)
   fi
fi

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-20}"
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-16}"
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-24576}"
   --eval-top-p 1
)

if [[ -n "${EVAL_CONFIG:-}" ]]; then
   EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}")
else
   read -r -a _eval_pairs <<< "${EVAL_DATASETS:-aime24 /data/aime-2024/aime-2024.jsonl aime25 /data/aime-2025/aime-2025.jsonl}"
   EVAL_ARGS+=(--eval-prompt-data "${_eval_pairs[@]}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE:-4}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
   --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE:-8}"
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
)
if [[ "${CONTEXT_PARALLEL_SIZE:-1}" -gt 1 ]]; then
   PERF_ARGS+=(--cp-comm-type a2a)
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-rollout-routing-replay
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}"
)
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]]; then
   SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-512}")

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-off-policy-${DATASET_TAG}}"
      --wandb-group "${RUN_NAME}"
      --wandb-key "${WANDB_API_KEY}"
   )
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miles\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"127.0.0.1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
   --colocate \
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
