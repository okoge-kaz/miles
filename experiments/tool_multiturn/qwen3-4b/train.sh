#!/bin/bash
# Runs INSIDE the container. Single node, 8 GPUs, colocated.
#
# Multi-turn tool-calling RL (ReTool v2 style): the policy may call a Python
# interpreter up to --generate-max-turns times per sample, and is scored on the
# final answer. No external service and no container runtime is involved — the
# tool runs as a local subprocess with a timeout and a memory cap
# (examples/retool_v2/tool_sandbox.py).
#
# This is the last step before agentic RL: it exercises the multi-turn loop,
# token concatenation and loss masking, but nothing that needs a sandbox
# provider or an outbound network call.

set -ex

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface
# retool_v2 rides on the refactored rollout path.
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd /root/miles
source /root/miles/scripts/models/qwen3-4B.sh   # MODEL_ARGS

RUN_NAME="${RUN_NAME:-tool-multiturn-qwen3-4b}"

CKPT_ARGS=(
   --hf-checkpoint /ckpt/hf/Qwen3-4B
   --ref-load      /ckpt/megatron/Qwen3-4B_torch_dist
   --load          /ckpt/training/${RUN_NAME}
   --save          /ckpt/training/${RUN_NAME}
   --save-interval "${SAVE_INTERVAL:-20}"
)

# The four plug-points that turn the generic loop into a tool-calling agent.
# Only these differ from math_sync/qwen3-4b/train.sh.
CUSTOM_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
   --generate-tool-specs-path examples.retool_v2.tool_sandbox.tool_specs
   --generate-execute-tool-function-path examples.retool_v2.tool_sandbox.execute_tool
   --generate-tool-call-parser qwen25
   --generate-max-turns "${GENERATE_MAX_TURNS:-16}"
   --log-multi-turn
)

ROLLOUT_ARGS=(
   --prompt-data /data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --custom-rm-path examples.retool_v2.tool_sandbox.reward_func
   --reward-key score
   --num-rollout "${NUM_ROLLOUT:-3000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN:-24576}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-24576}"
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-4}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # A single sample must fit in max_tokens_per_gpu * cp_size
   # (miles/backends/training_utils/data.py:473), so long-response runs need
   # context parallelism, a bigger budget, or both.
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
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
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-passrate
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

ray stop --force || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus ${GPUS_PER_NODE:-8} \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miles\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
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
   "${CUSTOM_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
