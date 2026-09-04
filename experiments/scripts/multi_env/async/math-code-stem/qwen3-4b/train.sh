#!/bin/bash
# In-container Math + Code + STEM RL entry point. Configuration is supplied by
# the adjacent run.sbatch; every train_async.py argument group is defined here.

set -euxo pipefail

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
HAS_NVLINK=$([[ "${NVLINK_COUNT}" -gt 0 ]] && echo 1 || echo 0)

cd /root/miles
[[ "${MODEL_PROFILE}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "invalid MODEL_PROFILE=${MODEL_PROFILE}" >&2
    exit 1
}
MODEL_ARGS_LINE="$(
    python3 miles/utils/external_utils/model_args_utils.py "${MODEL_PROFILE}"
)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
source /root/miles/experiments/common/ray_cluster.sh

CKPT_ARGS=(
    --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}"
    --ref-load "/ckpt/megatron/${MODEL_NAME}_torch_dist"
    --load "${CKPT_PATH}"
    --save "${CKPT_PATH}"
    --save-interval "${SAVE_INTERVAL}"
    --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
)
if [[ "${SAVE_HF}" != 0 ]]; then
    CKPT_ARGS+=(--save-hf "${CKPT_PATH}/hf/{rollout_id}")
    [[ -z "${HF_SAVE_INTERVAL:-}" ]] || CKPT_ARGS+=(--hf-save-interval "${HF_SAVE_INTERVAL}")
fi

readonly VERIFIED_CUSTOM_RM_PATH=experiments.src.reward_sets.math_code_stem.reward
[[ -z "${RM_TYPE:-}" && "${CUSTOM_RM_PATH:-}" == "${VERIFIED_CUSTOM_RM_PATH}" ]] || {
    echo "Multi-environment RL requires the validated custom reward ${VERIFIED_CUSTOM_RM_PATH}" >&2
    exit 1
}

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
ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
[[ -z "${TOOL_KEY:-}" ]] || ROLLOUT_ARGS+=(--tool-key "${TOOL_KEY}")
[[ "${ZERO_REWARD_ON_TRUNCATED}" == 0 ]] || ROLLOUT_ARGS+=(--zero-reward-on-truncated)
if [[ "${QUEUE_TYPE}" == queue-drop ]]; then
    ROLLOUT_ARGS+=(--fully-async-queue-factor "${QUEUE_FACTOR}")
else
    ROLLOUT_ARGS+=(--max-weight-staleness "${MAX_WEIGHT_STALENESS}")
fi
if [[ "${USE_REPLAY_BUFFER}" != 0 ]]; then
    ROLLOUT_ARGS+=(--use-replay-buffer --replay-buffer-type "${REPLAY_BUFFER_TYPE}")
fi
[[ -z "${ASYNC_MAX_CONCURRENT_SAMPLES:-}" ]] || \
    ROLLOUT_ARGS+=(--async-max-concurrent-samples "${ASYNC_MAX_CONCURRENT_SAMPLES}")
[[ -z "${DEBUG_EXIT_AFTER_ROLLOUT:-}" ]] || \
    ROLLOUT_ARGS+=(--debug-exit-after-rollout "${DEBUG_EXIT_AFTER_ROLLOUT}")

TELEMETRY_ARGS=(
    --dump-details "${CKPT_PATH}/dump"
    --use-miles-dashboard
    --observe-training-entropy
)
if [[ "${DUMP_POLICY_LOSS_DEBUG}" != 0 ]]; then
    TELEMETRY_ARGS+=(--dump-policy-loss-debug)
else
    TELEMETRY_ARGS+=(--no-dump-policy-loss-debug)
fi
if [[ "${DUMP_TRAIN_DATA}" == 0 ]]; then
    TELEMETRY_ARGS+=(--no-dump-train-data)
else
    TELEMETRY_ARGS+=(--use-rollout-entropy)
fi
[[ "${LOG_SAMPLE_STALENESS_METRICS}" == 0 ]] || TELEMETRY_ARGS+=(--log-sample-staleness-metrics)
if [[ "${LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM}" != 0 ]]; then
    [[ "${LOG_SAMPLE_STALENESS_METRICS}" != 0 ]] || {
        echo "LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM requires LOG_SAMPLE_STALENESS_METRICS=1" >&2
        exit 1
    }
    TELEMETRY_ARGS+=(--log-sample-staleness-ratio-histogram)
fi
[[ "${LOG_UPDATE_DIAGNOSTICS}" == 0 ]] || TELEMETRY_ARGS+=(--log-update-diagnostics)

[[ "${EVAL_INTERVAL}" == 0 ]] || {
    echo "multi-environment evaluation is offline; EVAL_INTERVAL must be 0" >&2
    exit 1
}

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
    --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)
[[ "${RECOMPUTE_GRANULARITY}" != full ]] || \
    PERF_ARGS+=(--recompute-method uniform --recompute-num-layers 1)
[[ "${OVERLAP_COMM}" == 0 ]] || PERF_ARGS+=(--overlap-grad-reduce --overlap-param-gather)
[[ "${CONTEXT_PARALLEL_SIZE}" -le 1 ]] || PERF_ARGS+=(--cp-comm-type a2a)

GRPO_ARGS=(
    --seed "${TRAIN_SEED}"
    --rollout-seed "${ROLLOUT_SEED}"
    --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
    --entropy-coef "${ENTROPY_COEF}"
    --eps-clip "${EPS_CLIP}"
    --eps-clip-high "${EPS_CLIP_HIGH}"
    --calculate-per-token-loss
)
[[ "${FUSE_ONE_STEP_ACTOR_LOGPROBS}" == 0 ]] || GRPO_ARGS+=(--fuse-one-step-actor-logprobs)
if [[ "${VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS}" != 0 ]]; then
    [[ "${FUSE_ONE_STEP_ACTOR_LOGPROBS}" != 0 ]] || {
        echo "VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS requires FUSE_ONE_STEP_ACTOR_LOGPROBS=1" >&2
        exit 1
    }
    GRPO_ARGS+=(--verify-fused-one-step-actor-logprobs)
fi
TIS_BOUNDS=(--tis-clip "${TIS_CLIP}" --tis-clip-low "${TIS_CLIP_LOW}")
case "${IS_CORRECTION}" in
    none) ;;
    tis) GRPO_ARGS+=(--use-tis "${TIS_BOUNDS[@]}") ;;
    icepop)
        GRPO_ARGS+=(--use-tis "${TIS_BOUNDS[@]}" \
            --custom-tis-function-path miles.backends.training_utils.loss_hub.corrections.icepop_function)
        ;;
    m2po) GRPO_ARGS+=(--use-m2po --m2po-budget "${M2PO_BUDGET}") ;;
    mis)
        GRPO_ARGS+=(--use-tis \
            --custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp \
            --custom-config-path "/root/miles/experiments/configs/mis/${MIS_PROFILE}.yaml")
        ;;
esac
case "${RATIO_DENOMINATOR}" in
    actor) ;;
    rollout-logprobs) GRPO_ARGS+=(--use-rollout-logprobs) ;;
    old-actor) GRPO_ARGS+=(--keep-old-actor) ;;
esac
[[ -z "${EPS_CLIP_C}" ]] || GRPO_ARGS+=(--eps-clip-c "${EPS_CLIP_C}")
[[ "${USE_OPSM}" == 0 ]] || GRPO_ARGS+=(--use-opsm --opsm-delta "${OPSM_DELTA}")
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
[[ -z "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]] || \
    SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
[[ -z "${SGLANG_CUDA_GRAPH_MAX_BS:-}" ]] || \
    SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
[[ "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS}" == 0 ]] || \
    SGLANG_ARGS+=(--sglang-enable-response-weight-version-segments)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend "${TRAINING_ATTENTION_BACKEND}"
)
WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-off-policy-${DATASET_TAG}}"
    --wandb-group "${RUN_NAME}"
)

RUNTIME_ENV_JSON=$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/:/root/miles",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_IB_DISABLE": "${NCCL_IB_DISABLE:-0}",
    "NCCL_NET": "${NCCL_NET:-}",
    "NCCL_NET_PLUGIN": "${NCCL_NET_PLUGIN:-}",
    "NCCL_TUNER_PLUGIN": "${NCCL_TUNER_PLUGIN:-}",
    "FI_PROVIDER": "${FI_PROVIDER:-}",
    "MILES_NCCL_TRANSPORT": "${MILES_NCCL_TRANSPORT:-system}",
    "WANDB_MODE": "${WANDB_MODE:-online}",
    "CODE_EXEC_SANDBOX": "${CODE_EXEC_SANDBOX}",
    "CODE_EXEC_CONCURRENCY": "${CODE_EXEC_CONCURRENCY}",
    "CODE_EXEC_MAX_TESTS": "${CODE_EXEC_MAX_TESTS}",
    "REASONING_GYM_DEPS_PATH": "${REASONING_GYM_DEPS_PATH:-/data/reasoning-gym-deps}",
    "no_proxy": "127.0.0.1"
  }
}
JSON
)

printf 'MILES_RL_ENTRY_EPOCH=%s\n' "${EPOCHREALTIME}"
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
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
