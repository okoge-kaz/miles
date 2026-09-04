#!/bin/bash
# In-container AReaL Tau2 RL entry point for the agentic SFT iteration 953 policy.

set -euxo pipefail

export PATH="${PATH}:/opt/tau3/.venv/bin"
export PYTHONUNBUFFERED=1
export HF_HOME=/root/.cache/huggingface
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export TAU2_DATA_DIR=/opt/tau3/data
export AREAL_TAU2_ROOT=/data/areal-tau2-data
export LOGURU_LEVEL="${TAU_LOG_LEVEL}"
export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:/opt/tau3/src:/opt/tau3/.venv/lib/python3.12/site-packages:/root/Megatron-LM/:/root/miles"

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
HAS_NVLINK=$([[ "${NVLINK_COUNT}" -gt 0 ]] && echo 1 || echo 0)
export HAS_NVLINK

cd /root/miles
[[ "${ROLLOUT_SEMANTICS}" == stateful_multi_turn_user_simulator_environment ]]
[[ "${CUSTOM_GENERATE_FUNCTION_PATH}" == experiments.src.environments.areal_tau2.generator.generate ]]
if [[ "${ALLOW_TAU_REPLAY_ABLATION}" == 0 ]]; then
    [[ "${USE_REPLAY_BUFFER}" == 1 && "${REPLAY_BUFFER_TYPE}" == inflight ]] || {
        echo "AReaL Tau2 requires multi-turn inflight replay outside an ablation" >&2
        exit 1
    }
fi
[[ -z "${RM_TYPE:-}" && -z "${CUSTOM_RM_PATH:-}" ]] || {
    echo "AReaL Tau2 obtains terminal reward from its custom generator" >&2
    exit 1
}
[[ "${EVAL_INTERVAL}" == 0 ]] || {
    echo "Tau v3 evaluation is held out and offline; EVAL_INTERVAL must be 0" >&2
    exit 1
}
[[ "${MODEL_PROFILE}" =~ ^[A-Za-z0-9._-]+$ ]]
MODEL_ARGS_LINE="$(
    python3 miles/utils/external_utils/model_args_utils.py "${MODEL_PROFILE}"
)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"

python3 - <<'PY'
import importlib.metadata
from pathlib import Path

assert importlib.metadata.version("tau2") == "1.0.1"
assert Path("/opt/tau3/data/tau2/domains/airline/policy.md").is_file()
assert Path("/data/areal-tau2-data/miles-tau2-rl-train.jsonl").is_file()
import miles
import tau2

print("Miles + Tau2 v1.0.1 training runtime: ok")
PY

source /root/miles/experiments/common/ray_cluster.sh

[[ -s "${PROMPT_DATA}" ]] || {
    echo "AReaL Tau2 training data is missing: ${PROMPT_DATA}" >&2
    exit 1
}

CKPT_ARGS=(
    --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}"
    --load "${CKPT_PATH}"
    --save "${CKPT_PATH}"
    --save-interval "${SAVE_INTERVAL}"
    --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
)
[[ -z "${REF_LOAD}" ]] || CKPT_ARGS+=(--ref-load "${REF_LOAD}")
[[ -z "${SAVE_TRIGGER_SENTINEL}" ]] || \
    CKPT_ARGS+=(--save-trigger-sentinel "${SAVE_TRIGGER_SENTINEL}")
if [[ "${SAVE_HF}" != 0 ]]; then
    CKPT_ARGS+=(--save-hf "${CKPT_PATH}/hf/{rollout_id}")
    [[ -z "${HF_SAVE_INTERVAL:-}" ]] || CKPT_ARGS+=(--hf-save-interval "${HF_SAVE_INTERVAL}")
fi

ROLLOUT_ARGS=(
    --fully-async
    --fully-async-queue-type "${QUEUE_TYPE}"
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-max-context-len "${MAX_CONTEXT_LEN}"
    --rollout-temperature 1
    --rollout-top-p 1
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
    --balance-data
    --staleness-reference "${STALENESS_REFERENCE}"
    --pause-generation-mode "${PAUSE_GENERATION_MODE}"
    --async-max-concurrent-samples "${ASYNC_MAX_CONCURRENT_SAMPLES}"
    --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"
    --tau-max-turns "${TAU_MAX_TURNS}"
    --tau-max-steps "${TAU_MAX_STEPS}"
    --tau-user-provider "${TAU_USER_PROVIDER}"
    --tau-user-model "${TAU_USER_MODEL}"
    --tau-user-max-tokens "${TAU_USER_MAX_TOKENS}"
    --tau-user-temperature "${TAU_USER_TEMPERATURE}"
    --tau-user-top-p "${TAU_USER_TOP_P}"
    --tau-user-request-timeout "${TAU_USER_REQUEST_TIMEOUT}"
    --tau-user-max-retries "${TAU_USER_MAX_RETRIES}"
    --tau-tool-call-parser "${TAU_TOOL_CALL_PARSER}"
)
if [[ "${QUEUE_TYPE}" == queue-drop ]]; then
    ROLLOUT_ARGS+=(--fully-async-queue-factor "${QUEUE_FACTOR}")
else
    ROLLOUT_ARGS+=(--max-weight-staleness "${MAX_WEIGHT_STALENESS}")
fi
if [[ "${USE_REPLAY_BUFFER}" != 0 ]]; then
    ROLLOUT_ARGS+=(--use-replay-buffer --replay-buffer-type "${REPLAY_BUFFER_TYPE}")
    ROLLOUT_ARGS+=(--replay-buffer-keep-last "${REPLAY_BUFFER_KEEP_LAST}")
fi
[[ "${LOG_REPLAY_RESUME_METRICS}" == 0 ]] || ROLLOUT_ARGS+=(--log-replay-resume-metrics)
[[ "${ZERO_REWARD_ON_TRUNCATED}" == 0 ]] || ROLLOUT_ARGS+=(--zero-reward-on-truncated)
[[ "${ZERO_LOSS_ON_TRUNCATED}" == 0 ]] || ROLLOUT_ARGS+=(--zero-loss-on-truncated)
[[ "${TAU_OVERLAP_DB_RESTORE_WITH_PREFILL}" == 0 ]] || \
    ROLLOUT_ARGS+=(--tau-overlap-db-restore-with-prefill)
[[ "${TAU_LOG_OVERHEAD}" == 0 ]] || ROLLOUT_ARGS+=(--tau-log-overhead)
[[ -z "${DEBUG_EXIT_AFTER_ROLLOUT:-}" ]] || \
    ROLLOUT_ARGS+=(--debug-exit-after-rollout "${DEBUG_EXIT_AFTER_ROLLOUT}")
if [[ -n "${DEBUG_FAIL_AFTER_ROLLOUT:-}" ]]; then
    ROLLOUT_ARGS+=(
        --debug-fail-after-rollout "${DEBUG_FAIL_AFTER_ROLLOUT}"
        --debug-failure-marker "${DEBUG_FAILURE_MARKER}"
        --debug-failure-min-outstanding-groups "${DEBUG_FAILURE_MIN_OUTSTANDING_GROUPS}"
        --debug-failure-min-completed-groups "${DEBUG_FAILURE_MIN_COMPLETED_GROUPS}"
        --debug-failure-min-inflight-groups "${DEBUG_FAILURE_MIN_INFLIGHT_GROUPS}"
        --debug-failure-min-inflight-tokens "${DEBUG_FAILURE_MIN_INFLIGHT_TOKENS}"
        --debug-failure-min-regenerate-groups "${DEBUG_FAILURE_MIN_REGENERATE_GROUPS}"
    )
fi

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
[[ "${LOG_SAMPLE_STALENESS_METRICS}" == 0 ]] || \
    TELEMETRY_ARGS+=(
        --log-sample-staleness-metrics
        --sample-staleness-max-bin "${SAMPLE_STALENESS_MAX_BIN}"
    )
if [[ "${LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM}" != 0 ]]; then
    [[ "${LOG_SAMPLE_STALENESS_METRICS}" != 0 ]]
    TELEMETRY_ARGS+=(--log-sample-staleness-ratio-histogram)
fi
[[ "${LOG_UPDATE_DIAGNOSTICS}" == 0 ]] || TELEMETRY_ARGS+=(--log-update-diagnostics)

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
TIS_BOUNDS=(--tis-clip "${TIS_CLIP}" --tis-clip-low "${TIS_CLIP_LOW}")
case "${IS_CORRECTION}" in
    none) ;;
    tis) GRPO_ARGS+=(--use-tis "${TIS_BOUNDS[@]}") ;;
    icepop)
        GRPO_ARGS+=(--use-tis "${TIS_BOUNDS[@]}" \
            --custom-tis-function-path miles.backends.training_utils.loss_hub.corrections.icepop_function)
        ;;
    m2po) GRPO_ARGS+=(--use-m2po --m2po-budget "${M2PO_BUDGET}") ;;
    *) echo "unsupported IS_CORRECTION=${IS_CORRECTION}" >&2; exit 1 ;;
esac
case "${RATIO_DENOMINATOR}" in
    actor) ;;
    rollout-logprobs) GRPO_ARGS+=(--use-rollout-logprobs) ;;
    old-actor) GRPO_ARGS+=(--keep-old-actor) ;;
    *) echo "unsupported RATIO_DENOMINATOR=${RATIO_DENOMINATOR}" >&2; exit 1 ;;
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

# API keys are inherited by raylets before this job is submitted. They are
# deliberately absent from the Ray runtime JSON and dashboard metadata.
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

keys = (
    "PATH",
    "NCCL_IB_DISABLE",
    "NCCL_NET",
    "NCCL_NET_PLUGIN",
    "NCCL_TUNER_PLUGIN",
    "FI_PROVIDER",
    "MILES_NCCL_TRANSPORT",
    "NVIDIA_INFERENCE_BASE_URL",
    "WANDB_MODE",
)
env = {key: os.environ.get(key, "") for key in keys}
env.update(
    {
        "PYTHONPATH": (
            "/usr/local/lib/python3.12/dist-packages:/opt/tau3/src:"
            "/opt/tau3/.venv/lib/python3.12/site-packages:"
            "/root/Megatron-LM/:/root/miles"
        ),
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "TAU2_DATA_DIR": "/opt/tau3/data",
        "AREAL_TAU2_ROOT": "/data/areal-tau2-data",
        "LOGURU_LEVEL": os.environ["TAU_LOG_LEVEL"],
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": os.environ["HAS_NVLINK"],
        "no_proxy": "127.0.0.1",
    }
)
print(json.dumps({"env_vars": env}, separators=(",", ":")))
PY
)"

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
