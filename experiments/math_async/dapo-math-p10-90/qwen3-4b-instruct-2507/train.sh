#!/bin/bash

set -ex

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd /root/miles
source /root/miles/scripts/models/qwen3-4B-Instruct-2507.sh
source /root/miles/experiments/common/ray_cluster.sh

CKPT_ARGS=(
   --hf-checkpoint /ckpt/hf/Qwen3-4B-Instruct-2507
   --ref-load      /ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist
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
   --rollout-top-p 1
   --rollout-top-k -1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
   --max-weight-staleness "${MAX_WEIGHT_STALENESS}"
   --staleness-reference "${STALENESS_REFERENCE}"
   --pause-generation-mode "${PAUSE_GENERATION_MODE}"
)
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
   --no-dump-policy-loss-debug
)
if [[ "${DUMP_TRAIN_DATA}" == "0" ]]; then
   TELEMETRY_ARGS+=(--no-dump-train-data)
else
   TELEMETRY_ARGS+=(--use-rollout-entropy)
fi

# EVAL_INTERVAL=0 passes no --eval-interval at all, which is what leaves
# args.eval_interval None and turns both eval sites off (train.py:98,144).
# See notes/telemetry.md for why in-run eval is off by default.
EVAL_ARGS=()
if [[ "${EVAL_INTERVAL}" != "0" ]]; then
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL}"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
      --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
      --eval-top-p 1
      --eval-prompt-data
      aime25 /data/aime-2025/aime-2025.jsonl
   )
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

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT:-off-policy-${DATASET_TAG}}"
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
