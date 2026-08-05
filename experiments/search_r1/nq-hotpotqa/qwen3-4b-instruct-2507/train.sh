#!/bin/bash
#
# Search-R1: the policy emits <search>query</search>, a local retriever answers
# with wiki-18 passages, those tokens are appended with loss_mask 0, and decoding
# resumes. `examples/experimental/search-r1/generate_with_search.py` owns the
# loop; this recipe only wires it up.
#
# Why this workload is here and not just another tool-use set: the observation is
# *long*. A Python interpreter returns a number, a retriever returns three
# passages, so each turn prepends thousands of prefill tokens to a short
# generation. It is the only recipe whose rollout is prefill-bound, which is
# exactly the axis the off-policy sweep wants to vary.

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
export PYTHONPATH="/root/miles/examples/experimental/search-r1:${PYTHONPATH:-}"

# --- retriever ---------------------------------------------------------------
# One process per node, holding the 65 GB FAISS index and the e5 query encoder.
# It has to be up before the first rollout: generate_with_search treats a failed
# search as an empty observation, so a missing retriever does not crash the run,
# it silently trains on unanswerable prompts.
if [[ "${SLURM_NODEID:-0}" == "0" ]]; then
    # Not in the image. faiss-cpu on purpose: the index is searched on CPU so the
    # GPUs stay with the policy, and the GPU build would want its own CUDA stack.
    pip install --no-input --quiet faiss-cpu fastapi uvicorn \
        || { echo "retriever dep install failed"; exit 1; }

    python3 /root/miles/experiments/src/search_r1/retrieval_server.py \
        --index /data/search-r1/e5_Flat.index \
        --corpus /data/search-r1/wiki-18.jsonl \
        --encoder /ckpt/hf/e5-base-v2 \
        --port "${RETRIEVER_PORT}" \
        --topk "${SEARCH_TOPK}" &
    RETRIEVER_PID=$!
    trap 'kill ${RETRIEVER_PID} 2>/dev/null || true' EXIT

    echo "waiting for retriever on :${RETRIEVER_PORT}"
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null 2>&1 && break
        kill -0 "${RETRIEVER_PID}" 2>/dev/null || { echo "retriever died during startup"; exit 1; }
        sleep 10
    done
    curl -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null || { echo "retriever never came up"; exit 1; }
fi

CKPT_ARGS=(
   --hf-checkpoint /ckpt/hf/Qwen3-4B-Instruct-2507
   --ref-load      /ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist
   --load          "${CKPT_PATH}"
   --save          "${CKPT_PATH}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   # Search-R1's parquet carries the question under `prompt` and the answer set
   # under `reward_model`, and the prompt is already a plain templated string --
   # hence no --apply-chat-template here.
   --input-key prompt
   --label-key reward_model
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
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)
if [[ "${DYNAMIC_SAMPLING:-1}" == "1" ]]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
fi

TELEMETRY_ARGS=(
   --dump-details "${CKPT_PATH}/dump"
   --use-miles-dashboard
)
if [[ "${DUMP_TRAIN_DATA}" == "0" ]]; then
   TELEMETRY_ARGS+=(--no-dump-train-data)
else
   TELEMETRY_ARGS+=(--use-rollout-entropy)
fi

# The seven FlashRAG QA sets Search-R1 reports, so the numbers are comparable to
# published ones instead of to a held-out split of our own.
EVAL_ARGS=(
   --eval-interval 20
   --eval-config /root/miles/experiments/configs/eval_search_r1.yaml
)

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
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miles:/root/miles/examples/experimental/search-r1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SEARCH_R1_SEARCH_URL\": \"http://127.0.0.1:${RETRIEVER_PORT}/retrieve\",
    \"SEARCH_R1_MAX_TURNS\": \"${SEARCH_MAX_TURNS}\",
    \"SEARCH_R1_TOPK\": \"${SEARCH_TOPK}\",
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
