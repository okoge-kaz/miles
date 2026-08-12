#!/bin/bash

set -euo pipefail

: "${LIVE_LOAD:?checkpoint directory}"
: "${LIVE_OUTPUT:?output directory}"
: "${LIVE_MODE:=fused}"
: "${LIVE_NUM_ROLLOUT:=3}"
: "${LIVE_ROLLOUT_BATCH_SIZE:=8}"
: "${LIVE_N_SAMPLES_PER_PROMPT:=16}"
: "${LIVE_GLOBAL_BATCH_SIZE:=$((LIVE_ROLLOUT_BATCH_SIZE * LIVE_N_SAMPLES_PER_PROMPT))}"
: "${LIVE_MAX_RESPONSE_LEN:=512}"

cd /root/miles
source /root/miles/scripts/models/qwen3-4B-Instruct-2507.sh

mkdir -p "${LIVE_OUTPUT}"

ray stop --force || true
node_ip=$(getent hosts "$(hostname)" | awk '{print $1; exit}')
ray start --head --node-ip-address "${node_ip}" --num-gpus 8 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

checkpoint_args=(
    --hf-checkpoint /ckpt/hf/Qwen3-4B-Instruct-2507
    --ref-load /ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist
    --load "${LIVE_LOAD}"
)

rollout_args=(
    --fully-async
    --prompt-data /data/dapo-math-p10-90/dapo-math-p10-90.jsonl
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --start-rollout-id 0
    --num-rollout "${LIVE_NUM_ROLLOUT}"
    --rollout-batch-size "${LIVE_ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${LIVE_N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${LIVE_MAX_RESPONSE_LEN}"
    --rollout-max-context-len 4096
    --rollout-temperature 1
    --rollout-top-p 1
    --rollout-top-k -1
    --global-batch-size "${LIVE_GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout 1
    --balance-data
    --max-weight-staleness 1
    --staleness-reference prefill
    --pause-generation-mode in_place
    --async-max-concurrent-samples "${LIVE_GLOBAL_BATCH_SIZE}"
)

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
case "${LIVE_MODE}" in
    legacy) ;;
    fused) grpo_args+=(--fuse-one-step-actor-logprobs) ;;
    *) echo "LIVE_MODE must be legacy or fused" >&2; exit 1 ;;
esac

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
    --max-tokens-per-gpu 8192
    --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

telemetry_args=(
    --dump-details "${LIVE_OUTPUT}/dump"
    --observe-training-entropy
    --no-dump-policy-loss-debug
    --no-dump-train-data
)

sglang_args=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.70
)

misc_args=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --actor-num-nodes 1
    --actor-num-gpus-per-node 4
    --rollout-num-gpus 4
)

runtime_env='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM/:/root/miles","MILES_EXPERIMENTAL_ROLLOUT_REFACTOR":"1","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"1","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","no_proxy":"127.0.0.1"}}'

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${runtime_env}" \
    -- python3 train_async.py \
    "${MODEL_ARGS[@]}" \
    "${checkpoint_args[@]}" \
    "${rollout_args[@]}" \
    "${optimizer_args[@]}" \
    "${grpo_args[@]}" \
    "${perf_args[@]}" \
    "${telemetry_args[@]}" \
    "${sglang_args[@]}" \
    "${misc_args[@]}"

ray stop --force || true
