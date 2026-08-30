#!/bin/bash
# In-container SWE-ReBench V2/SWE-Gym RL entry point. Configuration is supplied
# by the adjacent run.sbatch; E2B credentials belong only to the Harbor server.

set -euo pipefail

: "${HARBOR_RUN_SECRET:?SWE rollout workers require the Harbor /run bearer secret}"
(( ${#HARBOR_RUN_SECRET} >= 32 && ${#HARBOR_RUN_SECRET} <= 4096 )) && \
    [[ "$HARBOR_RUN_SECRET" != *$'\r'* && "$HARBOR_RUN_SECRET" != *$'\n'* ]] || {
    echo "HARBOR_RUN_SECRET must be 32-4096 characters without CR/LF" >&2
    exit 2
}

export PYTHONBUFFERED=16
export HF_HOME=/root/.cache/huggingface
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
HAS_NVLINK=$([[ "${NVLINK_COUNT}" -gt 0 ]] && echo 1 || echo 0)
export HAS_NVLINK

cd /root/miles
MODEL_PROFILE_PATH="/root/miles/scripts/models/${MODEL_PROFILE}.sh"
[[ -r "${MODEL_PROFILE_PATH}" ]] || {
    echo "model profile is not readable: ${MODEL_PROFILE_PATH}" >&2
    exit 1
}
source "${MODEL_PROFILE_PATH}"
source /root/miles/experiments/common/ray_cluster.sh

[[ -s "${PROMPT_DATA}" ]] || {
    echo "SWE prompt data is missing: ${PROMPT_DATA}" >&2
    exit 1
}
case "${PROMPT_DATA}" in
    /data/miles-swe/admitted/*-train.jsonl) ;;
    *)
        echo "SWE RL only accepts datasets promoted under /data/miles-swe/admitted" >&2
        exit 1
        ;;
esac
[[ -s "${ADMISSION_SUMMARY}" ]] || {
    echo "SWE admission summary is missing: ${ADMISSION_SUMMARY}" >&2
    exit 1
}
python3 - "${PROMPT_DATA}" "${ADMISSION_SUMMARY}" "${SWE_AGENT_NAME}" <<'PY'
import hashlib
import json
import sys

prompt_path, summary_path, expected_agent = sys.argv[1:]
with open(summary_path, encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("schema_version") != "miles-swe-admitted-dataset-v1":
    raise SystemExit("SWE admission summary has an unsupported schema")
if summary.get("artifact_stage") != "environment-admitted" or (
    summary.get("environment_admitted") is not True
):
    raise SystemExit("SWE prompt data has not passed environment admission")
if summary.get("output") != prompt_path:
    raise SystemExit("SWE admission summary is bound to a different output path")

digest = hashlib.sha256()
row_count = 0
task_bindings = {}
with open(prompt_path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != summary.get("output_sha256"):
    raise SystemExit("SWE prompt dataset differs from its admission summary")

with open(prompt_path, encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        row_count += 1
        row = json.loads(line)
        actual = row.get("metadata", {}).get("agent_name")
        if actual != expected_agent:
            raise SystemExit(
                f"SWE row {line_number} agent_name={actual!r} does not match "
                f"admitted agent {expected_agent!r}"
            )
        metadata = row.get("metadata") or {}
        instance_id = metadata.get("instance_id")
        task_digest = (metadata.get("swe_task") or {}).get("task_digest")
        if not isinstance(instance_id, str) or not isinstance(task_digest, str):
            raise SystemExit(f"SWE row {line_number} lacks a task binding")
        previous = task_bindings.setdefault(instance_id, task_digest)
        if previous != task_digest:
            raise SystemExit(f"SWE row {line_number} conflicts with its task binding")
if row_count <= 0 or row_count != summary.get("admitted_rows"):
    raise SystemExit("SWE admitted row count differs from its admission summary")
task_ids = sorted(task_bindings)
task_pairs = [[task_id, task_bindings[task_id]] for task_id in task_ids]

def stable_digest(value):
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

if (
    len(task_ids) != summary.get("admitted_unique_tasks")
    or stable_digest(task_ids) != summary.get("task_set_sha256")
    or stable_digest(task_pairs) != summary.get("task_binding_sha256")
    or not isinstance(summary.get("task_runtime_sha256"), str)
    or len(summary["task_runtime_sha256"]) != 64
    or any(
        character not in "0123456789abcdef"
        for character in summary["task_runtime_sha256"]
    )
):
    raise SystemExit("SWE prompt task set differs from its admission summary")
PY
[[ "${CUSTOM_RM_PATH}" == experiments.src.reward_sets.swe.reward ]] || {
    echo "SWE RL only permits the fail-closed Harbor reward" >&2
    exit 1
}
[[ "${CUSTOM_GENERATE_FUNCTION_PATH}" == miles.rollout.generate_hub.agentic_tool_call.generate ]] || {
    echo "SWE RL requires agentic_tool_call.generate" >&2
    exit 1
}
[[ "${CUSTOM_AGENT_FUNCTION_PATH}" == swe_agent_function.run ]] || {
    echo "SWE RL requires the validated Harbor /run client" >&2
    exit 1
}
[[ "${USE_REPLAY_BUFFER}" == 0 ]] || {
    echo "SWE replay is disabled until fresh and resume E2B validation both pass" >&2
    exit 1
}
[[ "${EVAL_INTERVAL}" == 0 ]] || {
    echo "SWE evaluation is an external executable benchmark; EVAL_INTERVAL must be 0" >&2
    exit 1
}

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
    CKPT_ARGS+=(--hf-save-interval "${HF_SAVE_INTERVAL}")
fi

ROLLOUT_ARGS=(
    --fully-async
    --fully-async-queue-type "${QUEUE_TYPE}"
    --max-weight-staleness "${MAX_WEIGHT_STALENESS}"
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --max-seq-len "${MAX_SEQUENCE_LEN}"
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
    --custom-agent-function-path "${CUSTOM_AGENT_FUNCTION_PATH}"
    --custom-rm-path "${CUSTOM_RM_PATH}"
    --rollout-function-path "${ROLLOUT_FUNCTION_PATH}"
    --dynamic-sampling-filter-path "${DYNAMIC_FILTER_PATH}"
    --tito-model "${TITO_MODEL}"
    --tito-allowed-append-roles user tool
    --use-session-server
    --session-server-port 30000
)
[[ "${ZERO_REWARD_ON_TRUNCATED}" == 0 ]] || ROLLOUT_ARGS+=(--zero-reward-on-truncated)
[[ -z "${DEBUG_EXIT_AFTER_ROLLOUT}" ]] || \
    ROLLOUT_ARGS+=(--debug-exit-after-rollout "${DEBUG_EXIT_AFTER_ROLLOUT}")

TELEMETRY_ARGS=(
    --dump-details "${CKPT_PATH}/dump"
    --use-miles-dashboard
    --observe-training-entropy
    --no-dump-policy-loss-debug
    --no-dump-train-data
)
[[ "${LOG_SAMPLE_STALENESS_METRICS}" == 0 ]] || \
    TELEMETRY_ARGS+=(--log-sample-staleness-metrics)
[[ "${LOG_UPDATE_DIAGNOSTICS}" == 0 ]] || TELEMETRY_ARGS+=(--log-update-diagnostics)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --recompute-granularity "${RECOMPUTE_GRANULARITY}"
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

GRPO_ARGS=(
    --seed "${TRAIN_SEED}"
    --rollout-seed "${ROLLOUT_SEED}"
    --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
    --entropy-coef "${ENTROPY_COEF}"
    --eps-clip "${EPS_CLIP}"
    --eps-clip-high "${EPS_CLIP_HIGH}"
    --calculate-per-token-loss
    --use-tis
    --tis-clip "${TIS_CLIP}"
    --tis-clip-low "${TIS_CLIP_LOW}"
)
[[ "${FUSE_ONE_STEP_ACTOR_LOGPROBS}" == 0 ]] || GRPO_ARGS+=(--fuse-one-step-actor-logprobs)
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
    --sglang-tool-call-parser "${SGLANG_TOOL_CALL_PARSER}"
    --sglang-reasoning-parser "${SGLANG_REASONING_PARSER}"
    --use-miles-router
    --sglang-router-port 31000
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

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

names = (
    "AGENT_MODEL_NAME",
    "AGENT_SERVER_URL",
    "FI_PROVIDER",
    "HARBOR_CLIENT_ID",
    "LD_LIBRARY_PATH",
    "MILES_NCCL_TRANSPORT",
    "NCCL_IB_DISABLE",
    "NCCL_NET",
    "NCCL_NET_PLUGIN",
    "NCCL_TUNER_PLUGIN",
    "PYTHON_DOTENV_DISABLED",
    "WANDB_MODE",
)
env_vars = {name: os.environ.get(name, "") for name in names}
env_vars.update(
    {
        "PYTHONPATH": "/root/Megatron-LM/:/root/miles/examples/swe-agent:/root/miles",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": os.environ["HAS_NVLINK"],
        "no_proxy": "127.0.0.1",
    }
)
print(json.dumps({"env_vars": env_vars}, separators=(",", ":")))
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
