#!/bin/bash

set -euo pipefail

: "${VALIDATION_MODE:?legacy or fused}"
: "${VALIDATION_LOAD:?checkpoint directory}"
: "${VALIDATION_DUMP:?fixed rollout dump}"
: "${VALIDATION_OUTPUT:?output directory}"
: "${VALIDATION_NUM_ROLLOUT:?number of rollout batches}"
: "${VALIDATION_SAVE:=0}"
: "${VALIDATION_VERIFY:=0}"
: "${VALIDATION_DUMPER:=0}"
: "${VALIDATION_DETERMINISTIC:=0}"
: "${VALIDATION_SAVE_TRAIN_DATA:=0}"
: "${VALIDATION_FULLY_ASYNC:=0}"
: "${VALIDATION_STALENESS_METRICS:=0}"
: "${VALIDATION_STALENESS_RATIO_HISTOGRAM:=0}"
: "${VALIDATION_DISABLE_KL_CHECKER:=0}"
: "${VALIDATION_ROLLOUT_BATCH_SIZE:=192}"
: "${VALIDATION_N_SAMPLES_PER_PROMPT:=16}"
: "${VALIDATION_GLOBAL_BATCH_SIZE:=$((VALIDATION_ROLLOUT_BATCH_SIZE * VALIDATION_N_SAMPLES_PER_PROMPT))}"
: "${VALIDATION_MAX_TOKENS_PER_GPU:=32768}"
: "${VALIDATION_CODE_ROOT:=/root/miles}"

cd "${VALIDATION_CODE_ROOT}"
source "${VALIDATION_CODE_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"

mkdir -p "${VALIDATION_OUTPUT}"

ray stop --force || true
node_ip=$(getent hosts "$(hostname)" | awk '{print $1; exit}')
ray start --head --node-ip-address "${node_ip}" --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

checkpoint_args=(
    --hf-checkpoint /ckpt/hf/Qwen3-4B-Instruct-2507
    --ref-load /ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist
    --load "${VALIDATION_LOAD}"
)
if [[ "${VALIDATION_SAVE}" == 1 ]]; then
    checkpoint_args+=(
        --save "${VALIDATION_LOAD}"
        --save-interval 1000000
    )
fi

rollout_args=(
    --load-debug-rollout-data "${VALIDATION_DUMP}"
    --prompt-data /data/dapo-math-p10-90/dapo-math-p10-90.jsonl
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --start-rollout-id 0
    --num-rollout "${VALIDATION_NUM_ROLLOUT}"
    --rollout-batch-size "${VALIDATION_ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${VALIDATION_N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 32768
    --rollout-max-context-len 32768
    --rollout-temperature 1
    --rollout-top-p 1
    --rollout-top-k -1
    --global-batch-size "${VALIDATION_GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout 1
    --balance-data
)
driver="${VALIDATION_CODE_ROOT}/train.py"
if [[ "${VALIDATION_FULLY_ASYNC}" == 1 ]]; then
    rollout_args+=(--fully-async)
    driver="${VALIDATION_CODE_ROOT}/train_async.py"
fi

optimizer_args=(
    --optimizer adam
    --clip-grad 1.0
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
)

grpo_args=(
    --seed 1234
    --rollout-seed 42
    --advantage-estimator grpo
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --calculate-per-token-loss
    --use-tis
    --tis-clip 2.0
    --tis-clip-low 0.0
)
case "${VALIDATION_MODE}" in
    legacy) ;;
    fused) grpo_args+=(--fuse-one-step-actor-logprobs) ;;
    *) echo "VALIDATION_MODE must be legacy or fused" >&2; exit 1 ;;
esac
if [[ "${VALIDATION_VERIFY}" == 1 ]]; then
    grpo_args+=(--verify-fused-one-step-actor-logprobs)
fi

perf_args=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${VALIDATION_MAX_TOKENS_PER_GPU}"
)
train_env_vars='{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
if [[ "${VALIDATION_DETERMINISTIC}" == 1 ]]; then
    perf_args+=(--deterministic-mode)
    train_env_vars='{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","NCCL_ALGO":"Ring","NVTE_ALLOW_NONDETERMINISTIC_ALGO":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"}'
fi
perf_args+=(--train-env-vars "${train_env_vars}")

telemetry_args=(
    --observe-training-entropy
    --no-dump-policy-loss-debug
    --no-dump-train-data
)
if [[ "${VALIDATION_STALENESS_METRICS}" == 1 ]]; then
    if [[ "${VALIDATION_FULLY_ASYNC}" != 1 ]]; then
        echo "VALIDATION_STALENESS_METRICS=1 requires VALIDATION_FULLY_ASYNC=1" >&2
        exit 1
    fi
    telemetry_args+=(--log-sample-staleness-metrics)
fi
if [[ "${VALIDATION_STALENESS_RATIO_HISTOGRAM}" == 1 ]]; then
    if [[ "${VALIDATION_STALENESS_METRICS}" != 1 ]]; then
        echo "VALIDATION_STALENESS_RATIO_HISTOGRAM=1 requires VALIDATION_STALENESS_METRICS=1" >&2
        exit 1
    fi
    telemetry_args+=(--log-sample-staleness-ratio-histogram)
fi
if [[ "${VALIDATION_DUMPER}" == 1 ]]; then
    telemetry_args+=(
        --dumper-dir "${VALIDATION_OUTPUT}/dumper"
        --dumper-fwd-bwd
        enable=1
        enable_value=0
        enable_grad=0
        enable_model_value=0
        enable_model_grad=1
        include_parallel_rank_in_filename=1
    )
fi
if [[ "${VALIDATION_SAVE_TRAIN_DATA}" == 1 ]]; then
    telemetry_args+=(--save-debug-train-data "${VALIDATION_OUTPUT}/train_data/{rollout_id}_{rank}.pt")
fi

misc_args=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --actor-num-nodes 1
    --actor-num-gpus-per-node 8
    --rollout-num-gpus 0
    --ci-test
    --ci-disable-logprobs-checker
    --ci-save-grad-norm "${VALIDATION_OUTPUT}/grad_norm-{rollout_id}-{step_id}.pt"
)
if [[ "${VALIDATION_DISABLE_KL_CHECKER}" == 1 ]]; then
    misc_args+=(--ci-disable-kl-checker)
fi

runtime_env="{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM/:${VALIDATION_CODE_ROOT}\",\"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\":\"1\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"1\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"no_proxy\":\"127.0.0.1\"}}"
if [[ "${VALIDATION_DETERMINISTIC}" == 1 ]]; then
    runtime_env="{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM/:${VALIDATION_CODE_ROOT}\",\"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\":\"1\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"1\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"NCCL_ALGO\":\"Ring\",\"NVTE_ALLOW_NONDETERMINISTIC_ALGO\":\"0\",\"CUBLAS_WORKSPACE_CONFIG\":\":4096:8\",\"no_proxy\":\"127.0.0.1\"}}"
fi

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${runtime_env}" \
    -- python3 "${driver}" \
    "${MODEL_ARGS[@]}" \
    "${checkpoint_args[@]}" \
    "${rollout_args[@]}" \
    "${optimizer_args[@]}" \
    "${grpo_args[@]}" \
    "${perf_args[@]}" \
    "${telemetry_args[@]}" \
    "${misc_args[@]}"

ray stop --force || true
