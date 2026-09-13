#!/bin/bash
set -euo pipefail
cd /root/miles
export PYTHONUNBUFFERED=1
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH=/root/miles:/root/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
python experiments/src/math/lightning_sglang_compat.py
source /root/miles/experiments/common/ray_cluster.sh
# The base architecture has the same dimensions as Nano 30B; Bridge reads the
# actual hybrid layer pattern and Mamba dimensions from the Lightning HF config.
read -r -a MODEL_ARGS <<< "$(python -m miles.utils.external_utils.model_args_utils nemotron-3-nano-30b-a3b)"
CKPT_ARGS=(
    --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}"
    --ref-load "/ckpt/megatron/${MODEL_NAME}_torch_dist"
    --load "${CKPT_PATH}" --save "${CKPT_PATH}"
    # Preserve the constant-LR scheduler when extending the total rollout count.
    --use-checkpoint-opt-param-scheduler
    --save-interval "${SAVE_INTERVAL}" --save-retain-interval "${SAVE_RETAIN_INTERVAL}"
    --save-hf "${CKPT_PATH}/hf/{rollout_id}" --hf-save-interval "${HF_SAVE_INTERVAL}"
    --megatron-to-hf-mode bridge
)
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}" --input-key prompt --label-key label
    --apply-chat-template --apply-chat-template-kwargs '{"enable_thinking": true}'
    --rollout-shuffle --rm-type math --zero-reward-on-truncated
    --num-rollout "${NUM_ROLLOUT}" --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
    --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}" --rollout-max-context-len "${MAX_CONTEXT_LEN}"
    --rollout-temperature 1 --rollout-top-p 0.95 --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}" --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" --balance-data
)
PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_PARALLEL_SIZE}" --sequence-parallel
    --pipeline-model-parallel-size 1 --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_PARALLEL_SIZE}" --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" --log-probs-chunk-size 128
    --use-rollout-routing-replay
)
EVAL_ARGS=()
if (( EVAL_INTERVAL > 0 )); then
    EVAL_ARGS=(--eval-interval "${EVAL_INTERVAL}" --eval-prompt-data aime2024 "${EVAL_PROMPT_DATA}"
        --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}" --eval-temperature 1 --eval-top-p 0.95)
    if (( SKIP_EVAL_BEFORE_TRAIN )); then EVAL_ARGS+=(--skip-eval-before-train); fi
fi
WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-lightning-dapo-math}" --wandb-group "${RUN_NAME}")
fi
ray job submit --address=http://127.0.0.1:8265 \
    --runtime-env-json '{"env_vars":{"PYTHONPATH":"/root/miles:/root/Megatron-LM","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"1","no_proxy":"127.0.0.1"}}' \
    -- python3 train.py \
    --actor-num-nodes "${ACTOR_NUM_NODES}" --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --num-gpus-per-node "${GPUS_PER_NODE}" --colocate \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" "${WANDB_ARGS[@]}" \
    --seed "${TRAIN_SEED}" --rollout-seed "${ROLLOUT_SEED}" \
    --advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss \
    --optimizer adam --lr "${LR}" --lr-decay-style constant --weight-decay 0 \
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1 \
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}" \
    --sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND}" \
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}" --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}" \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend auto \
    --dump-details "${CKPT_PATH}/dump" --use-miles-dashboard --no-dump-policy-loss-debug --no-dump-train-data
