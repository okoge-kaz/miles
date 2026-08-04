#!/bin/bash
# Runs INSIDE the container. Single node, 8 GPUs, colocated.
#
# Smallest hybrid-thinking Qwen3 (28 layers, hidden 2048). The cheap end of
# the sweep: fits a full math run in one interactive slot, so it is the recipe
# to smoke-test a change on before spending a 30B slot on it.

set -ex

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd /root/miles
source /root/miles/scripts/models/qwen3-1.7B.sh   # MODEL_ARGS

RUN_NAME="${RUN_NAME:-math-grpo-qwen3-1.7b}"

CKPT_ARGS=(
   --hf-checkpoint /ckpt/hf/Qwen3-1.7B
   --ref-load      /ckpt/megatron/Qwen3-1.7B_torch_dist
   --load          /ckpt/training/${RUN_NAME}
   --save          /ckpt/training/${RUN_NAME}
   --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA:-/data/dapo-math-17k/dapo-math-17k.jsonl}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   # deepscaler grades the boxed answer like `math`, but first requires a
   # `</think>` (or `###Response`) delimiter and returns 0 without one
   # (rm_hub/deepscaler.py:36-44). Correct for this hybrid-thinking checkpoint; a
   # NON-THINKING model never emits the delimiter, so every response would score 0
   # and the run would look like a model that cannot learn.
   --rm-type "${RM_TYPE:-deepscaler}"
   --num-rollout "${NUM_ROLLOUT:-3000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN:-24576}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
   --rollout-temperature 1

   # invariant: rollout_batch_size * n_samples_per_prompt
   #            = global_batch_size * num_steps_per_rollout
   # Passed explicitly so arguments.py:3056 actually checks it, and so the number
   # of optimizer steps taken per rollout batch — i.e. how off-policy the later
   # steps are — is a named knob rather than a side effect of global_batch_size.
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}"
   --balance-data
)

# Dynamic sampling (DAPO): drop groups whose reward has zero variance — all
# correct or all wrong — and resample, so the batch keeps a usable advantage
# signal. In GRPO the advantage scales with sqrt(p*(1-p)), so a unanimous group
# contributes nothing while costing a full group of generation; measured on
# Qwen3-4B/DAPO-Math, 21.6 of 32 groups per batch were unanimous.
#
# ON by default. Set DYNAMIC_SAMPLING=0 to turn it off.
#
# --partial-rollout is part of the same setting, not an extra. Over-sampling
# submits OVER_SAMPLING_BATCH_SIZE groups and aborts whatever is still in flight
# once ROLLOUT_BATCH_SIZE groups have passed the filter
# (inference_rollout_train.py:143). Without --partial-rollout those aborted
# generations are dropped on the floor (inference_rollout_train.py:39); with it
# they go back to the data buffer and resume next rollout.
#
# --mask-offpolicy-in-partial-rollout then zeroes the loss over the tokens a
# resumed sample generated under the previous weights (sglang_rollout.py:313).
# This recipe has no --use-tis, so masking is the only thing keeping those
# carried-over tokens from entering the loss uncorrected.
if [[ "${DYNAMIC_SAMPLING:-1}" != "0" ]]; then
   ROLLOUT_ARGS+=(
      --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
      --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE:-$(( ${ROLLOUT_BATCH_SIZE:-32} * 2 ))}"
   )
   if [[ "${PARTIAL_ROLLOUT:-1}" != "0" ]]; then
      ROLLOUT_ARGS+=(--partial-rollout)
      [[ "${MASK_OFFPOLICY:-1}" != "0" ]] && ROLLOUT_ARGS+=(--mask-offpolicy-in-partial-rollout)
   fi
fi

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-24576}"
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-4}"
   --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE:-1}"
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # A single sample must fit in max_tokens_per_gpu * cp_size
   # (miles/backends/training_utils/data.py:473). With
   # --rollout-max-context-len 32768 that is the floor this value has to clear:
   #   12288 * 4 = 49152 >= 32768
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-12288}"
)
if [[ "${CONTEXT_PARALLEL_SIZE:-4}" -gt 1 ]]; then
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
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
   --sglang-mem-fraction-static 0.7
)

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
      --wandb-project "${WANDB_PROJECT:-miles-kazuki-fujii}"
      --wandb-group "${RUN_NAME}"
      --wandb-key "${WANDB_API_KEY}"
   )
fi

# Ray: single node, head only. train.py is submitted as a Ray job so its logs
# land in the Ray dashboard as well as this job's stdout.
ray stop --force || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus ${GPUS_PER_NODE:-8} \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${GPUS_PER_NODE:-8} \
   --colocate \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
