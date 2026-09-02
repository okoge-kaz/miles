#!/bin/bash

set -ex

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
export HAS_NVLINK

cd /root/miles
[[ "${MODEL_PROFILE}" =~ ^[A-Za-z0-9._-]+$ ]] || {
   echo "invalid MODEL_PROFILE=${MODEL_PROFILE}" >&2
   exit 1
}
MODEL_PROFILE_PATH="/root/miles/scripts/models/${MODEL_PROFILE}.sh"
[[ -r "${MODEL_PROFILE_PATH}" ]] || {
   echo "model profile is not readable: ${MODEL_PROFILE_PATH}" >&2
   exit 1
}
source "${MODEL_PROFILE_PATH}"

# ABCI exposes the full-node GPU allocation as UUIDs in
# CUDA_VISIBLE_DEVICES. Ray preserves those values in ray.get_gpu_ids(), while
# the current rollout launcher expects numeric IDs. This recipe uses every GPU
# on each exclusive node, so normalize the in-container view before Ray starts.
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
   echo "GPUS_PER_NODE must be a positive integer: ${GPUS_PER_NODE}" >&2
   exit 1
}
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$(( GPUS_PER_NODE - 1 ))")"
source /root/miles/experiments/common/ray_cluster.sh

CKPT_ARGS=(
   --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}"
   --ref-load      "/ckpt/megatron/${MODEL_NAME}_torch_dist"
   --load          "${CKPT_PATH}"
   --save          "${CKPT_PATH}"
   --save-interval "${SAVE_INTERVAL}"
   --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
)
if [[ "${SAVE_HF}" != "0" ]]; then
   CKPT_ARGS+=(--save-hf "${CKPT_PATH}/hf/{rollout_id}")
   if [[ -n "${HF_SAVE_INTERVAL:-}" ]]; then
      CKPT_ARGS+=(--hf-save-interval "${HF_SAVE_INTERVAL}")
   fi
fi

ROLLOUT_ARGS=(
   --fully-async
   --fully-async-queue-type "${QUEUE_TYPE}"
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-max-context-len 32768
   --rollout-temperature 1
   --rollout-top-p 1
   --rollout-top-k -1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
   --staleness-reference "${STALENESS_REFERENCE}"
   --pause-generation-mode "${PAUSE_GENERATION_MODE}"
)
if [[ -n "${CUSTOM_RM_PATH:-}" ]]; then
   ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
else
   ROLLOUT_ARGS+=(--rm-type "${RM_TYPE}")
fi
if [[ -n "${TOOL_KEY:-}" ]]; then
   ROLLOUT_ARGS+=(--tool-key "${TOOL_KEY}")
fi
if [[ "${ZERO_REWARD_ON_TRUNCATED}" != "0" ]]; then
   ROLLOUT_ARGS+=(--zero-reward-on-truncated)
fi
if [[ "${QUEUE_TYPE}" == queue-drop ]]; then
   ROLLOUT_ARGS+=(--fully-async-queue-factor "${QUEUE_FACTOR}")
else
   ROLLOUT_ARGS+=(--max-weight-staleness "${MAX_WEIGHT_STALENESS}")
fi
if [[ "${USE_REPLAY_BUFFER:-0}" != "0" ]]; then
   ROLLOUT_ARGS+=(--use-replay-buffer --replay-buffer-type "${REPLAY_BUFFER_TYPE:-rollout}")
fi
if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES:-}" ]]; then

   ROLLOUT_ARGS+=(--async-max-concurrent-samples "${ASYNC_MAX_CONCURRENT_SAMPLES}")
fi
if [[ -n "${DEBUG_EXIT_AFTER_ROLLOUT:-}" ]]; then
   ROLLOUT_ARGS+=(--debug-exit-after-rollout "${DEBUG_EXIT_AFTER_ROLLOUT}")
fi

TELEMETRY_ARGS=(
   --dump-details "${CKPT_PATH}/dump"
   --use-miles-dashboard
   --observe-training-entropy
)
if [[ "${DUMP_POLICY_LOSS_DEBUG:-0}" != "0" ]]; then
   TELEMETRY_ARGS+=(--dump-policy-loss-debug)
else
   TELEMETRY_ARGS+=(--no-dump-policy-loss-debug)
fi
if [[ "${DUMP_TRAIN_DATA}" == "0" ]]; then
   TELEMETRY_ARGS+=(--no-dump-train-data)
else
   TELEMETRY_ARGS+=(--use-rollout-entropy)
fi
if [[ "${LOG_SAMPLE_STALENESS_METRICS:-0}" != "0" ]]; then
   TELEMETRY_ARGS+=(--log-sample-staleness-metrics)
fi
if [[ "${LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM:-0}" != "0" ]]; then
   if [[ "${LOG_SAMPLE_STALENESS_METRICS:-0}" == "0" ]]; then
      echo "LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM requires LOG_SAMPLE_STALENESS_METRICS=1" >&2
      exit 1
   fi
   TELEMETRY_ARGS+=(--log-sample-staleness-ratio-histogram)
fi
if [[ "${LOG_UPDATE_DIAGNOSTICS:-0}" != "0" ]]; then
   TELEMETRY_ARGS+=(--log-update-diagnostics)
fi

# EVAL_INTERVAL=0 passes no --eval-interval at all, which is what leaves
# args.eval_interval None and turns both eval sites off (train.py:98,144).
# See notes/telemetry.md for why in-run eval is off by default.
EVAL_ARGS=()
if [[ "${EVAL_INTERVAL}" != "0" ]]; then
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL}"
   )
   if [[ -n "${EVAL_CONFIG:-}" ]]; then
      EVAL_ARGS+=(--eval-config "${EVAL_CONFIG}")
   else
      EVAL_ARGS+=(
         --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
         --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
         --eval-top-p 1
         --eval-prompt-data
         aime25 /data/aime-2025/aime-2025.jsonl
      )
   fi
   if [[ "${SKIP_EVAL_BEFORE_TRAIN}" != "0" ]]; then
      EVAL_ARGS+=(--skip-eval-before-train)
   fi
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE}"
   --expert-tensor-parallel-size 1

   --recompute-granularity "${RECOMPUTE_GRANULARITY}"

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"

   # trainer only: the sglang engines keep the default allocator
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)
if [[ "${RECOMPUTE_GRANULARITY}" == "full" ]]; then
   PERF_ARGS+=(--recompute-method uniform --recompute-num-layers 1)
fi
if [[ "${OVERLAP_COMM}" != "0" ]]; then
   PERF_ARGS+=(--overlap-grad-reduce --overlap-param-gather)
fi
if [[ "${CONTEXT_PARALLEL_SIZE}" -gt 1 ]]; then
   PERF_ARGS+=(--cp-comm-type a2a)
fi

GRPO_ARGS=(
   --seed         "${TRAIN_SEED}"
   --rollout-seed "${ROLLOUT_SEED}"
   --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
   --entropy-coef "${ENTROPY_COEF}"
   --eps-clip "${EPS_CLIP}"
   --eps-clip-high "${EPS_CLIP_HIGH}"
   --calculate-per-token-loss
)
if [[ "${FUSE_ONE_STEP_ACTOR_LOGPROBS}" != "0" ]]; then
   GRPO_ARGS+=(--fuse-one-step-actor-logprobs)
fi
if [[ "${VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS}" != "0" ]]; then
   if [[ "${FUSE_ONE_STEP_ACTOR_LOGPROBS}" == "0" ]]; then
      echo "VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS requires FUSE_ONE_STEP_ACTOR_LOGPROBS=1" >&2
      exit 1
   fi
   GRPO_ARGS+=(--verify-fused-one-step-actor-logprobs)
fi
TIS_BOUNDS=(--tis-clip "${TIS_CLIP}" --tis-clip-low "${TIS_CLIP_LOW}")
case "${IS_CORRECTION}" in
   none)   ;;
   tis)    GRPO_ARGS+=(--use-tis "${TIS_BOUNDS[@]}") ;;
   icepop) GRPO_ARGS+=(--use-tis "${TIS_BOUNDS[@]}"
                       --custom-tis-function-path miles.backends.training_utils.loss_hub.corrections.icepop_function) ;;
   m2po)   GRPO_ARGS+=(--use-m2po --m2po-budget "${M2PO_BUDGET}") ;;
   mis)    GRPO_ARGS+=(--use-tis
                       --custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
                       --custom-config-path "/root/miles/experiments/configs/mis/${MIS_PROFILE}.yaml") ;;
esac
case "${RATIO_DENOMINATOR}" in
   actor)            ;;
   rollout-logprobs) GRPO_ARGS+=(--use-rollout-logprobs) ;;
   old-actor)        GRPO_ARGS+=(--keep-old-actor) ;;
esac
if [[ -n "${EPS_CLIP_C}" ]]; then
   GRPO_ARGS+=(--eps-clip-c "${EPS_CLIP_C}")
fi
if [[ "${USE_OPSM}" != "0" ]]; then
   GRPO_ARGS+=(--use-opsm --opsm-delta "${OPSM_DELTA}")
fi
if awk "BEGIN{exit !(${KL_LOSS_COEF} != 0)}"; then
   GRPO_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF}" --kl-loss-type low_var_kl)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --clip-grad 1.0
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.999
   --adam-eps 1e-8
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
if [[ -n "${SGLANG_MOE_RUNNER_BACKEND:-}" ]]; then
   SGLANG_ARGS+=(--sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND}")
fi
if [[ "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS:-0}" != "0" ]]; then
   SGLANG_ARGS+=(--sglang-enable-response-weight-version-segments)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend "${TRAINING_ATTENTION_BACKEND}"
)

# WANDB_API_KEY is resolved on the compute host and inherited by the container.
# Never put it in this argv: both shell xtrace and Ray's "Running entrypoint"
# line are persisted in job logs.
WANDB_ARGS=(
   --use-wandb
   --wandb-team "${WANDB_TEAM}"
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${RUN_NAME}"
)

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

env_vars = {
    "PYTHONPATH": "/root/Megatron-LM/:/root/miles",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": os.environ["HAS_NVLINK"],
    "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE") or "0",
    "MILES_NCCL_TRANSPORT": os.environ.get("MILES_NCCL_TRANSPORT") or "system",
    "WANDB_MODE": os.environ.get("WANDB_MODE") or "online",
    "CODE_EXEC_SANDBOX": os.environ.get("CODE_EXEC_SANDBOX") or "bubblewrap",
    "CODE_EXEC_CONCURRENCY": os.environ.get("CODE_EXEC_CONCURRENCY") or "16",
    "CODE_EXEC_MAX_TESTS": os.environ.get("CODE_EXEC_MAX_TESTS") or "50",
    "REASONING_GYM_DEPS_PATH": (
        os.environ.get("REASONING_GYM_DEPS_PATH") or "/data/reasoning-gym-deps"
    ),
    "no_proxy": "127.0.0.1",
}

# An empty NCCL selector is not equivalent to an unset selector. In system
# mode these remain absent so NCCL can choose its internal IB implementation.
for name in ("NCCL_NET", "NCCL_NET_PLUGIN", "NCCL_TUNER_PLUGIN", "FI_PROVIDER"):
    if value := os.environ.get(name):
        env_vars[name] = value

print(json.dumps({"env_vars": env_vars}, separators=(",", ":")))
PY
)"

# The replay-buffer validation analyzer uses this marker for end-to-end startup
# from a ready Ray cluster through the first rollout and optimizer metrics.  It
# intentionally excludes scheduler wait and Ray cluster formation, which are
# unrelated to the replay representation being compared.
printf 'MILES_RL_ENTRY_EPOCH=%s\n' "${EPOCHREALTIME}"

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
