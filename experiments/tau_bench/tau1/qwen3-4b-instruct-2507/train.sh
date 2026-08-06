#!/bin/bash
#
# tau-bench: the policy talks to a *user simulator* (another model, over litellm)
# while tau-bench's mock tools return database rows. Both are observations the
# policy did not produce, and unlike conv-tooluse the conversation actually
# branches on what the policy said -- this is the recipe that makes turn count a
# real independent variable.
#
# examples/experimental/tau-bench/generate_with_tau.py owns the episode loop.

set -ex

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd /root/miles
source /root/miles/scripts/models/qwen3-4B-Instruct-2507.sh
source /root/miles/experiments/common/ray_cluster.sh

# The generate/reward functions are imported by bare module name, so their
# directory has to be importable rather than just on the repo path.
export PYTHONPATH="/root/miles/examples/experimental/tau-bench:${PYTHONPATH:-}"
# litellm reads the provider key from the environment; --export=ALL carried it in.

CKPT_ARGS=(
   --hf-checkpoint /ckpt/hf/Qwen3-4B-Instruct-2507
   --ref-load      /ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist
   --load          "${CKPT_PATH}"
   --save          "${CKPT_PATH}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   # The "prompt" is a task *index*: generate_with_tau.py:140 does
   # `task_index = int(sample.prompt)` and rebuilds the episode from tau-bench's
   # own env. So no label key and no chat template -- the row carries an integer,
   # not a conversation.
   --input-key index
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   # Per *turn*, not per trajectory: the model writes a query or a short answer,
   # never a long chain. The retrieved passages dominate the token count and
   # they are not generated, so this stays small.
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   # Context has to hold max_turns worth of retrieved passages on top of the
   # question, which is what actually bounds the trajectory.
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
   --custom-generate-function-path generate_with_tau.generate
   # No --custom-rm-path: the generate function sets sample.reward itself from
   # the environment's task-success check. Pointing a reward model at these
   # samples would overwrite the only signal the episode produced.
)
if [[ "${DYNAMIC_SAMPLING:-1}" == "1" ]]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
fi

TELEMETRY_ARGS=(
   --dump-details "${CKPT_PATH}/dump"
   --use-miles-dashboard
   # Training-side entropy is a constant 0 without this: calculate_entropy is
   # `entropy_coef != 0 or observe_training_entropy` (loss_hub/losses.py) and the
   # coefficient is 0 here. Forward-only and detached, so no backward cost.
   --observe-training-entropy
   # policy_loss_debug/ is one file per micro-batch per rank, so it scales with
   # training calls rather than rollout steps: 1.17 GB in 2512 files over 12
   # rollout steps, against 287 MB of rollout dumps. Only a loss-level debug reads it.
   --no-dump-policy-loss-debug
)
if [[ "${DUMP_TRAIN_DATA}" == "0" ]]; then
   TELEMETRY_ARGS+=(--no-dump-train-data)
else
   TELEMETRY_ARGS+=(--use-rollout-entropy)
fi

# tau-bench's own splits -- the same task ids every published TAU1 number uses,
# so this is a public benchmark rather than a split of our own. Retail is the
# training domain; airline is held out entirely, which makes it the transfer
# measurement rather than an in-domain one.
EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len "${MAX_RESPONSE_LEN}"
   --eval-input-key index
   --eval-prompt-data
   tau1_retail_test  /data/tau-bench/tau1_retail_test.jsonl
   tau1_airline_test /data/tau-bench/tau1_airline_test.jsonl
)
# miles evaluates once before the first training step regardless of the interval,
# so a bring-up run reaches the eval path before it has proved the training one.
if [[ "${SKIP_EVAL_BEFORE_TRAIN:-0}" != "0" ]]; then
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
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

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

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miles:/root/miles/examples/experimental/tau-bench\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TAU_USER_MODEL_PROVIDER\": \"${TAU_USER_MODEL_PROVIDER}\",
    \"TAU_USER_MODEL\": \"${TAU_USER_MODEL}\",
    \"TAU_MAX_TURNS\": \"${TAU_MAX_TURNS}\",
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
