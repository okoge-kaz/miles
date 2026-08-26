#!/bin/bash
#
# Shared Search-R1 worker for the sync and fully-async recipes. The policy emits
# <search>query</search>, a local retriever answers
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
if [[ "${PLACEMENT}" == "async" ]]; then
    export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
else
    export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=0
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

cd /root/miles
source /root/miles/scripts/models/qwen3-4B-Instruct-2507.sh
source /root/miles/experiments/common/ray_cluster.sh

# The custom generation function is imported by bare module name, so its
# directory has to be importable rather than just on the repo path.
export PYTHONPATH="/root/miles/examples/experimental/search-r1:${PYTHONPATH:-}"

# --- retriever ---------------------------------------------------------------
# One process on the Ray head, holding the 65 GB FAISS index and e5 encoder.
# It has to be up before the first rollout. Runtime request failures are retried
# and then abort the affected trajectory, but a dead server would otherwise
# waste every rollout attempt.
# The address the rollout uses. RAY_HEAD_IP is node 0's routable address, which
# is what a Ray actor can reach; loopback is not, even co-located.
RETRIEVER_HOST="${RAY_HEAD_IP:-127.0.0.1}"

if [[ "${SLURM_NODEID:-0}" == "0" ]]; then
    # faiss-cpu on purpose: the index is searched on CPU so the GPUs stay with
    # the policy. A cluster-specific image can bake these packages and avoid
    # requiring package-index egress at job startup.
    if ! python3 -c 'import faiss, fastapi, uvicorn' 2>/dev/null; then
        pip install --no-input --quiet faiss-cpu fastapi uvicorn \
            || { echo "retriever dep install failed"; exit 1; }
    fi

    python3 /root/miles/experiments/src/search_r1/retrieval_server.py \
        --index /data/search-r1/e5_Flat.index \
        --corpus /data/search-r1/wiki-18.jsonl \
        --encoder /ckpt/hf/e5-base-v2 \
        --port "${RETRIEVER_PORT}" \
        --topk "${SEARCH_TOPK}" \
        --faiss-threads "${RETRIEVER_FAISS_THREADS}" \
        --batch-max-requests "${RETRIEVER_BATCH_MAX_REQUESTS}" \
        --batch-wait-ms "${RETRIEVER_BATCH_WAIT_MS}" &
    RETRIEVER_PID=$!

    cleanup() {
        local status=$?
        trap - EXIT
        touch "${RAY_DONE_FLAG}" 2>/dev/null || true
        if [[ -n "${RAY_CLIENT_PID:-}" ]] && kill -0 "${RAY_CLIENT_PID}" 2>/dev/null; then
            kill "${RAY_CLIENT_PID}" 2>/dev/null || true
            wait "${RAY_CLIENT_PID}" 2>/dev/null || true
        fi
        if kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
            kill "${RETRIEVER_PID}" 2>/dev/null || true
        fi
        wait "${RETRIEVER_PID}" 2>/dev/null || true
        exit "${status}"
    }
    # ray_cluster.sh installs its own EXIT trap. Compose that notification with
    # retriever cleanup instead of replacing it (which would strand worker nodes
    # in a multi-node variant of this recipe).
    trap cleanup EXIT

    echo "waiting for retriever on ${RETRIEVER_HOST}:${RETRIEVER_PORT}"
    for _ in $(seq 1 240); do
        curl -sf "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health" >/dev/null 2>&1 && break
        kill -0 "${RETRIEVER_PID}" 2>/dev/null || { echo "retriever died during startup"; exit 1; }
        sleep 10
    done
    curl -sf "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/health" >/dev/null || { echo "retriever never came up"; exit 1; }

    # /health only says the process is alive. Probe the real endpoint on the
    # address the rollout will use, and require passages back.
    #
    # This check exists because the failure it catches is silent: a bring-up run
    # on 2026-08-05 had a healthy retriever that the Ray actors could not reach
    # (loopback bind), and every search returned an empty observation. The run
    # completed and reward came back 0.328 -- earned
    # entirely from the model's parametric memory, with the retriever unused.
    # Without this probe that looks like a working Search-R1 run.
    _probe=$(curl -sf -X POST "http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/retrieve" \
        -H 'Content-Type: application/json' \
        -d '{"queries": ["who won the nobel prize in physics 1901"], "topk": 3}' || true)
    case "${_probe}" in
        *contents*|*document*|*text*) echo "retriever probe ok" ;;
        *) echo "retriever returned no passages from ${RETRIEVER_HOST}:${RETRIEVER_PORT}"
           echo "  response: ${_probe:0:300}"
           echo "  a reachable-but-useless retriever trains on unanswerable prompts silently"
           exit 1 ;;
    esac
fi

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
   --prompt-data "${PROMPT_DATA}"
   # Search-R1 rows carry the question under `prompt` and the answer set under
   # `reward_model`. `prompt` is a one-message chat *list*, not a string,
   # so --apply-chat-template is required: without it data.py hands the raw list
   # through untouched (data.py:220-226) and generate_with_search tokenizes a
   # Python list as if it were text.
   --apply-chat-template
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
   # Keep the closing action tag in text/tokens/logprobs. generate_with_search
   # parses it to decide whether to retrieve or finish.
   --rollout-stop '</search>' '</answer>'
   # Context has to hold max_turns worth of retrieved passages on top of the
   # question, which is what actually bounds the trajectory.
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-temperature 1
   --rollout-top-p 1
   --rollout-top-k -1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --balance-data
   --custom-generate-function-path generate_with_search.generate
   --rm-type search_r1
   --search-r1-format-score "${SEARCH_FORMAT_SCORE}"
)
if [[ "${PLACEMENT}" == "async" ]]; then
   ROLLOUT_ARGS+=(
      --fully-async
      --fully-async-queue-type "${QUEUE_TYPE}"
      --training-buffer-queue-size "${TRAINING_BUFFER_QUEUE_SIZE}"
      --staleness-reference "${STALENESS_REFERENCE}"
      --pause-generation-mode "${PAUSE_GENERATION_MODE}"
   )
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
fi
# Difficulty selection is entirely offline. Deliberately pass neither a reward
# filter nor the abort-only filter here: both use the online top-up path and make
# generated trajectories per update depend on run-time outcomes. Retriever
# health is enforced by the endpoint probe and process watchdog below; aborted
# trajectory metrics must remain zero in a valid run.

TELEMETRY_ARGS=(
   --dump-details "${CKPT_PATH}/dump"
   --use-miles-dashboard
)
if [[ "${OBSERVE_TRAINING_ENTROPY:-0}" != "0" ]]; then
   # Forward-only and detached, but still material on long retrieved contexts.
   TELEMETRY_ARGS+=(--observe-training-entropy)
fi
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

# EVAL_INTERVAL=0 deliberately passes no eval arguments.  Search-R1 evaluation
# starts a tool environment and thousands of multi-turn episodes, so it belongs
# in run_search_r1_eval.sbatch against the HF snapshots above, outside the
# wall-clock path being measured.
EVAL_ARGS=()
if [[ "${EVAL_INTERVAL}" != "0" ]]; then
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL}"
      --eval-config /root/miles/experiments/configs/eval_search_r1.yaml
   )
   if [[ "${SKIP_EVAL_BEFORE_TRAIN:-0}" != "0" ]]; then
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
)
if [[ "${PLACEMENT}" == "async" ]]; then
   # Trainer only: SGLang keeps its default allocator. Colocated training uses
   # torch_memory_saver, which is incompatible with expandable segments.
   PERF_ARGS+=(--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}')
fi
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
)
if [[ "${FUSE_ONE_STEP_ACTOR_LOGPROBS:-0}" != "0" ]]; then
   GRPO_ARGS+=(--fuse-one-step-actor-logprobs)
fi
if [[ "${VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS:-0}" != "0" ]]; then
   if [[ "${FUSE_ONE_STEP_ACTOR_LOGPROBS:-0}" == "0" ]]; then
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
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]]; then
   SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${SGLANG_CUDA_GRAPH_MAX_BS:-}" ]]; then
   SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
fi
if [[ "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS:-0}" != "0" ]]; then
   SGLANG_ARGS+=(--sglang-enable-response-weight-version-segments)
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
   --wandb-project "async-search-r1"
   --wandb-group "${RUN_NAME}"
   # wandb reads WANDB_API_KEY from the inherited environment. Passing it as a
   # CLI argument would expose it in Megatron's full argument dump.
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miles:/root/miles/examples/experimental/search-r1\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"${MILES_EXPERIMENTAL_ROLLOUT_REFACTOR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SEARCH_R1_SEARCH_URL\": \"http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/retrieve\",
    \"SEARCH_R1_MAX_TURNS\": \"${SEARCH_MAX_TURNS}\",
    \"SEARCH_R1_TOPK\": \"${SEARCH_TOPK}\",
    \"SEARCH_R1_SEARCH_CONCURRENCY\": \"${SEARCH_CONCURRENCY}\",
    \"SEARCH_R1_SEARCH_TIMEOUT\": \"${SEARCH_TIMEOUT}\",
    \"SEARCH_R1_SEARCH_MAX_ATTEMPTS\": \"${SEARCH_MAX_ATTEMPTS}\",
    \"SEARCH_R1_FORMAT_SCORE\": \"${SEARCH_FORMAT_SCORE}\",
    \"SEARCH_R1_RETURN_LOGPROB\": \"1\",
    \"no_proxy\": \"${RETRIEVER_HOST},127.0.0.1,localhost\"
  }
}"

# Keep the full command out of shell xtrace. W&B gets its API key only from the
# inherited environment, while Ray still streams the training logs normally.
if [[ "${PLACEMENT}" == "async" ]]; then
   ENTRYPOINT_ARGS=(
      python3 train_async.py
      --actor-num-nodes "${ACTOR_NUM_NODES}"
      --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
      --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   )
else
   ENTRYPOINT_ARGS=(
      python3 train.py
      --actor-num-nodes "${ACTOR_NUM_NODES}"
      --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
      --colocate
   )
fi
set +x
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${ENTRYPOINT_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${TELEMETRY_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" &
RAY_CLIENT_PID=$!

# A missing retriever must fail the whole job. Aborted trajectories are useful
# for a transient request failure, but they are not a substitute for a live
# environment: otherwise a run can keep answering from parametric memory after
# retrieval has disappeared.
while kill -0 "${RAY_CLIENT_PID}" 2>/dev/null && kill -0 "${RETRIEVER_PID}" 2>/dev/null; do
    sleep 5
done

if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    retriever_status=0
    wait "${RETRIEVER_PID}" || retriever_status=$?
    echo "retriever exited unexpectedly with status ${retriever_status}; terminating the Ray job client"
    kill "${RAY_CLIENT_PID}" 2>/dev/null || true
    wait "${RAY_CLIENT_PID}" 2>/dev/null || true
    exit 1
fi

ray_status=0
wait "${RAY_CLIENT_PID}" || ray_status=$?
exit "${ray_status}"
