#!/bin/bash
set -euo pipefail

cd /root/miles
export PYTHONUNBUFFERED=1
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH=/root/miles:/root/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
bash experiments/container/apply_oci_sglang_provenance.sh
python experiments/src/math/lightning_sglang_compat.py

source /root/miles/experiments/common/ray_cluster.sh
# The base architecture has the same dimensions as Nano 30B; Bridge reads the
# actual hybrid layer pattern and Mamba dimensions from the Lightning HF config.
read -r -a MODEL_ARGS <<< "$(python -m miles.utils.external_utils.model_args_utils nemotron-3-nano-30b-a3b)"
CKPT_ARGS=(
    --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}"
    --ref-load "/ckpt/megatron/${MODEL_NAME}_torch_dist"
    --load "${CKPT_PATH}"
    --save "${CKPT_PATH}"
    # Preserve the constant-LR scheduler when extending the total rollout count.
    --use-checkpoint-opt-param-scheduler
    --save-interval "${SAVE_INTERVAL}"
    --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
    --save-hf "${CKPT_PATH}/hf/{rollout_id}"
    --hf-save-interval "${HF_SAVE_INTERVAL}"
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --fully-async
    --fully-async-queue-type "${QUEUE_TYPE}"
    --max-weight-staleness "${MAX_WEIGHT_STALENESS}"
    --staleness-reference "${STALENESS_REFERENCE}"
    --training-buffer-queue-size "${TRAINING_BUFFER_QUEUE_SIZE}"
    --async-max-concurrent-samples "${ASYNC_MAX_CONCURRENT_SAMPLES}"
    --pause-generation-mode "${PAUSE_GENERATION_MODE}"
    --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
    --use-replay-buffer
    --replay-buffer-type "${REPLAY_BUFFER_TYPE}"
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": true}'
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
    --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-max-context-len "${MAX_CONTEXT_LEN}"
    --rollout-temperature 1
    --rollout-top-p "${ROLLOUT_TOP_P}"
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
    --balance-data
)
if [[ -n "${DEBUG_EXIT_AFTER_ROLLOUT}" ]]; then
    ROLLOUT_ARGS+=(--debug-exit-after-rollout "${DEBUG_EXIT_AFTER_ROLLOUT}")
fi

TELEMETRY_ARGS=(
    --dump-details "${CKPT_PATH}/dump"
    --use-miles-dashboard
    --no-dump-policy-loss-debug
    --no-dump-train-data
    --log-sample-staleness-metrics
    --sample-staleness-max-bin 8
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --log-probs-chunk-size 128
    --use-rollout-routing-replay
)

GRPO_ARGS=(
    --seed "${TRAIN_SEED}"
    --rollout-seed "${ROLLOUT_SEED}"
    --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
    --advantage-clip-low "${ADVANTAGE_CLIP_LOW}"
    --advantage-clip-high "${ADVANTAGE_CLIP_HIGH}"
    # One optimizer update: reuse the training forward, including rollout routing replay.
    --fuse-one-step-actor-logprobs
    --entropy-coef 0
    --eps-clip "${EPS_CLIP}"
    --eps-clip-high "${EPS_CLIP_HIGH}"
    --calculate-per-token-loss
    # Async uses the actor denominator with truncated importance sampling.
    --use-tis
    --tis-clip "${TIS_CLIP}"
    --tis-clip-low "${TIS_CLIP_LOW}"
)
if [[ "${USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER}" == 1 ]]; then
    GRPO_ARGS+=(
        --use-train-rollout-logprob-sequence-filter
        --train-rollout-logprob-sequence-filter-threshold "${TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER_THRESHOLD}"
    )
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --clip-grad 1
    --lr "${LR}"
    --lr-decay-style constant
    --lr-warmup-iters 0
    --weight-decay 0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
    --sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
    --sglang-enable-response-weight-version-segments
)

MISC_ARGS=(
    --attention-dropout 0
    --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend auto
)
if [[ "${OCI_R3_VALIDATION:-0}" == 1 ]]; then
    MISC_ARGS+=(--custom-megatron-init-path tests.manual.oci_validation.replay_routing_probe.install)
fi

# EVAL_INTERVAL=0 is enforced by run.sbatch; evaluate the retained HF exports offline.
WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${RUN_NAME}"
)

RUNTIME_ENV_JSON=$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/miles:/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
    "no_proxy": "127.0.0.1"
  }
}
JSON
)

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --num-gpus-per-node "${GPUS_PER_NODE}" \
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
