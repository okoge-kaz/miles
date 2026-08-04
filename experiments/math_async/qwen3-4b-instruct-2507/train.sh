#!/bin/bash
# Runs INSIDE the container. Single node, 8 GPUs, DISAGGREGATED.
#
# Non-thinking instruct sibling of Qwen3-4B: identical architecture (its
# MODEL_ARGS file only re-sources qwen3-4B.sh with rotary base 5e6), but a much
# lower math ceiling, so there is real headroom to learn.
#
# RM_TYPE defaults to `math`, NOT `deepscaler`, for this checkpoint — see the
# note at --rm-type below. Measured: it answers `Answer: \boxed{37}` to a prompt
# labelled 37 and scores 0 under deepscaler.
#
# Fully-async rollout: a persistent background worker keeps generating while the
# trainer consumes finished groups (miles/rollout/fully_async_rollout.py).
#
# Three differences from the colocated recipe, per examples/fully_async/README.md:
#   1. train_async.py instead of train.py
#   2. MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 (class-based rollout API)
#   3. --fully-async
#
# Consequence: --colocate is rejected (arguments.py:53), so the 8 GPUs are split
# between the trainer and the engines.

set -ex

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd /root/miles
source /root/miles/scripts/models/qwen3-4B-Instruct-2507.sh   # MODEL_ARGS

RUN_NAME="${RUN_NAME:-math-grpo-async-qwen3-4b-instruct-2507}"

# GPU split: trainer | rollout engines. Must sum to the 8 GPUs on the node.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

CKPT_ARGS=(
   --hf-checkpoint /ckpt/hf/Qwen3-4B-Instruct-2507
   --ref-load      /ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist
   --load          /ckpt/training/${RUN_NAME}
   --save          /ckpt/training/${RUN_NAME}
   --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
   --fully-async
   --prompt-data "${PROMPT_DATA:-/data/dapo-math-17k/dapo-math-17k.jsonl}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   # `math`, not `deepscaler`: deepscaler returns 0 unless the response contains a
   # `</think>` (or `###Response`) delimiter (rm_hub/deepscaler.py:36-44), and this
   # checkpoint is non-thinking, so it never emits one. Under deepscaler the whole
   # run would score reward 0 with no error anywhere.
   --rm-type "${RM_TYPE:-math}"
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

# Dynamic sampling (DAPO). FullyAsyncRolloutFn applies the same filter at drain
# time (fully_async_rollout.py:217); a dropped group is NOT recycled, since it
# carries no usable gradient signal either way. ON by default; DYNAMIC_SAMPLING=0
# turns it off.
if [[ "${DYNAMIC_SAMPLING:-1}" != "0" ]]; then
   ROLLOUT_ARGS+=(
      --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   )
fi

# Partial rollout. A weight sync aborts whatever is generating at that moment;
# the fully-async worker hands the group's prompt samples back to the data source
# (fully_async_rollout.py:198), and because generate_and_rm mutates samples in
# place those objects still carry the tokens produced so far, so the next
# submission continues from them instead of restarting.
#
# --mask-offpolicy-in-partial-rollout zeroes the loss over that carried-over
# prefix (sglang_rollout.py:313). It is what --partial-rollout gates here: the
# continuation itself happens either way, but setting a loss_mask without
# --partial-rollout trips the assert at sglang_rollout.py:282. --use-tis below
# corrects the ratio for the tokens that ARE trained on; the prefix predates the
# current weights entirely, so it is masked rather than reweighted.
if [[ "${PARTIAL_ROLLOUT:-1}" != "0" ]]; then
   ROLLOUT_ARGS+=(--partial-rollout)
   [[ "${MASK_OFFPOLICY:-1}" != "0" ]] && ROLLOUT_ARGS+=(--mask-offpolicy-in-partial-rollout)
fi

# Off-policy bound. Groups whose oldest weight version lags the engine by more
# than this are recycled instead of trained on (fully_async_rollout.py:202-213).
# The miles default is None = no bound at all, which lets an arbitrarily stale
# group reach the optimizer; 2 keeps generation overlapped across a weight sync
# without letting a group survive more than two of them.
ROLLOUT_ARGS+=(--max-weight-staleness "${MAX_WEIGHT_STALENESS:-2}")

# Generation concurrency, decoupled from the training batch. Unset means the
# legacy bound of rollout_batch_size * n_samples_per_prompt.
if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES:-}" ]]; then
   ROLLOUT_ARGS+=(--async-max-concurrent-samples "${ASYNC_MAX_CONCURRENT_SAMPLES}")
fi

# Eval works under --fully-async. The flag only swaps the *rollout* function;
# arguments.py:36 resolves eval_function_path before that override, so eval falls
# back to the standard InferenceRolloutFn and train_async.py:59,105 calls it
# exactly like train.py does. Kept identical to math_sync so the two recipes
# report the same AIME number.
#
# Caveat: the persistent fully-async worker keeps generating during eval, so eval
# shares the SGLang engines with training rollout and takes longer here than in
# the colocated recipe.
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-24576}"
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-2}"
   --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE:-1}"
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # A single sample must fit in max_tokens_per_gpu * cp_size
   # (miles/backends/training_utils/data.py:473). With
   # --rollout-max-context-len 32768 that is the floor this value has to clear:
   #   16384 * 2 = 32768 >= 32768
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
)
if [[ "${CONTEXT_PARALLEL_SIZE:-2}" -gt 1 ]]; then
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
   # truncated importance sampling: corrects for the policy lag that async
   # generation introduces
   --use-tis
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

ray stop --force || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus ${GPUS_PER_NODE:-8} \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miles\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"127.0.0.1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
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
