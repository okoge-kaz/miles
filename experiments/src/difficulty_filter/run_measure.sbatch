#!/bin/bash
#SBATCH --job-name=passrate
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/difficulty/%x-%j.log

# Measure a policy's per-prompt pass rate over a whole prompt file.
#
#   sbatch -A coreai_horizon_dilations \
#     experiments/src/difficulty_filter/run_measure.sbatch
#
# Knobs (all --export=ALL environment variables):
#   MODEL_NAME       policy/artifact name                 (Qwen3-4B-Instruct-2507)
#   HF_MODEL_NAME    HF dir under /ckpt/hf                (defaults to MODEL_NAME)
#   PROMPT_DATA      in-container prompt jsonl
#   OUTPUT           in-container output jsonl (resumable)
#   N_SAMPLES        samples per prompt                    (8, matches training)
#   SAMPLES_PER_REQUEST split each group into requests of this size (unset = N_SAMPLES)
#   MAX_NEW_TOKENS   generation cap; MUST match the recipe    (24576)
#                    --rollout-max-response-len
#   MAX_CONTEXT_LENGTH total prompt+response cap; matches --rollout-max-context-len
#   CUSTOM_RM_PATH   dotted path to a custom verifier; wins over RM_TYPE
#   TOOL_KEY         where per-row tools live               (tools)
#                    Top-level `tools` is what --tool-key reads during training;
#                    the Nemotron converters put them under metadata instead and
#                    both are accepted. Tools missing from the rendered prompt
#                    means there is no function to call, so every row scores 0.
#   INPUT_KEY        prompt field                           (prompt)
#   LABEL_KEY        label field                            (label)
#   EXTRA_PIP        '+'-separated packages to install first (verifier deps)
#   PREFLIGHT        strict | warn | off   (default strict)
#   RM_TYPE          verifier; MUST match the recipe        (math)
#                    `deepscaler` returns 0 unless the response contains
#                    `</think>`, so it zeroes every non-thinking model.
#   ZERO_REWARD_ON_TRUNCATED  1 makes finish_reason=length score 0 (default 0)
#   CONCURRENCY      prompts in flight                     (96)
#   TOP_P / TOP_K    sampling cutoffs                      (1.0 / -1)
#   MAX_RUNNING_REQUESTS  sequences admitted per replica    (96)
#   SGLANG_LOG_LEVEL sglang verbosity                      (info)
#                    Keep at info: the Decode-batch lines carry token usage,
#                    running-req and gen throughput, which are the only way to
#                    tell a healthy sweep from one silently retracting.
#   DUMP_RESPONSES   audit file: full responses + rewards for DUMP_LIMIT prompts
#   FILTERED_OUTPUT  optional filtered JSONL produced after measurement completes
#   ANNOTATED_OUTPUT optional full JSONL with difficulty metadata merged in
#   PASS_RATE_MIN/MAX filtering window                    (0.2 / 0.8)
#   START_INDEX/END_INDEX contiguous source range          (0 / all)
#   LIMIT            stop after N prompts (smoke runs)
#
# Inference only: this reads the HF weights that SGLang serves and never touches
# the torch_dist checkpoint, so it can run before or in parallel with training.
#
# Partition: `interactive` (4 h, up to 2 nodes / 16 GPUs) is the fast lane and
# this job needs one node, so it is the default. The partition selects the QoS
# on this cluster (interactive -> p_interactive), so no --qos is needed.
#
# Resumable by design. The output jsonl is appended and flushed per prompt and a
# rerun skips indices already present, so hitting the 4 h wall just means
# resubmitting the same command.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
HF_MODEL_NAME="${HF_MODEL_NAME:-${MODEL_NAME}}"
PROMPT_DATA="${PROMPT_DATA:-/data/dapo-math-17k/dapo-math-17k.jsonl}"
OUTPUT="${OUTPUT:-/data/difficulty/dapo-math-17k.${MODEL_NAME}.passrate.jsonl}"
N_SAMPLES="${N_SAMPLES:-8}"
SAMPLES_PER_REQUEST_ARG=""
[[ -n "${SAMPLES_PER_REQUEST:-}" ]] && SAMPLES_PER_REQUEST_ARG="--samples-per-request ${SAMPLES_PER_REQUEST}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24576}"
MAX_CONTEXT_ARG=""
[[ -n "${MAX_CONTEXT_LENGTH:-}" ]] && MAX_CONTEXT_ARG="--max-context-length ${MAX_CONTEXT_LENGTH}"
CONCURRENCY="${CONCURRENCY:-96}"
# Sequences admitted per dp replica. One prompt expands to N_SAMPLES
# sequences, so the client-side CONCURRENCY is in prompts while this is in
# sequences: keeping them equal makes total in-flight sequences
# CONCURRENCY * N_SAMPLES spread over GPUS_PER_NODE replicas, i.e.
# CONCURRENCY sequences each. Sized against the KV budget below.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
RM_TYPE="${RM_TYPE:-math}"
ZERO_TRUNCATED_ARG=""
[[ "${ZERO_REWARD_ON_TRUNCATED:-0}" != 0 ]] && ZERO_TRUNCATED_ARG="--zero-reward-on-truncated"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
PASS_RATE_MIN="${PASS_RATE_MIN:-0.2}"
PASS_RATE_MAX="${PASS_RATE_MAX:-0.8}"
START_INDEX="${START_INDEX:-0}"
END_INDEX_ARG=""
[[ -n "${END_INDEX:-}" ]] && END_INDEX_ARG="--end-index ${END_INDEX}"
SERVER_PORT="${SERVER_PORT:-30000}"
LIMIT_ARG=""
[[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"
DUMP_ARG=""
[[ -n "${DUMP_RESPONSES:-}" ]] && DUMP_ARG="--dump-responses ${DUMP_RESPONSES} --dump-limit ${DUMP_LIMIT:-8}"
FILTER_OUTPUT_ARGS=""
[[ -n "${FILTERED_OUTPUT:-}" ]] && FILTER_OUTPUT_ARGS="${FILTER_OUTPUT_ARGS} --output ${FILTERED_OUTPUT}"
[[ -n "${ANNOTATED_OUTPUT:-}" ]] && FILTER_OUTPUT_ARGS="${FILTER_OUTPUT_ARGS} --output-annotated ${ANNOTATED_OUTPUT}"

mkdir -p "${OUTPUT_DIR}/difficulty" "${DATASET_DIR}/difficulty"

SKIP_MEASUREMENT=0
if [[ "${OUTPUT}" == /data/* ]]; then
    OUTPUT_HOST="${DATASET_DIR}/${OUTPUT#/data/}"
    if [[ -f "${OUTPUT_HOST}.complete" ]]; then
        echo "skip completed measurement: ${OUTPUT_HOST}"
        exit 0
    fi
    if [[ -f "${OUTPUT_HOST}.measured" ]]; then
        echo "reuse completed measurement: ${OUTPUT_HOST}"
        SKIP_MEASUREMENT=1
    fi
fi

# This job can be submitted by a small CPU coordinator via --export=ALL.  Slurm
# then leaves the coordinator's SLURM_MEM_PER_NODE / SLURM_CPUS_PER_TASK in the
# batch environment even though this allocation owns the full GPU node.  If the
# nested step inherits those stale values, SGLang is cgroup-limited to 2 GiB and
# one of its DP workers is SIGKILLed while CUDA graphs are captured.  Command-line
# resource requests take precedence over the stale environment values.
srun \
    --cpus-per-task="${SLURM_JOB_CPUS_PER_NODE:-1}" \
    --mem=0 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc "
set -euo pipefail
cd /root/miles
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH=/root/Megatron-LM:/root/miles
export PYTHONUNBUFFERED=1

# Verifier dependencies the image does not ship. '--rm-type ifbench' clones
# allenai/IFBench at runtime and imports its evaluation_lib, which needs nltk and
# friends; miles points at examples/eval_multi_task/requirements_ifbench.txt for
# these but that file is not in this checkout, so they are named here instead.
# '+'-separated, not comma or space: sbatch --export treats commas as the
# separator between VAR=VALUE pairs (a comma list silently delivers only the
# first package) and a space ends the argument entirely.
if [[ -n "\${EXTRA_PIP:-}" ]]; then
    _pkgs=\$(echo \${EXTRA_PIP} | tr '+' ' ')
    echo \"installing verifier deps: \${_pkgs}\"
    pip install --no-input --quiet \${_pkgs} || { echo 'verifier dep install failed'; exit 1; }
fi

if [[ ${SKIP_MEASUREMENT} == 0 ]]; then
    # dp-size 8: eight independent replicas behind one endpoint. Qwen3-4B fits on a
    # single H100, so data parallelism beats tensor parallelism here -- no
    # cross-GPU collectives on the critical path of a pure-inference sweep.
    #
    # KV budget per replica: mem-fraction-static 0.85 * 80 GB = 68 GB, minus ~8 GB
    # of bf16 weights, leaves ~60 GB. Qwen3-4B keeps 36 layers * 8 KV heads * 128
    # dims * 2 (K,V) * 2 bytes = 144 KB per token, so ~410k tokens fit. Measured
    # mean response is ~4.1k tokens, hence ~96 concurrent sequences per replica.
    # Oversubscribing does not fail loudly -- SGLang retracts and re-prefills
    # requests, which burns compute silently (1635 retractions at 384x8=3072).
    python3 -m sglang.launch_server \
        --model-path /ckpt/hf/${HF_MODEL_NAME} \
        --host 127.0.0.1 --port ${SERVER_PORT} \
        --dp-size ${GPUS_PER_NODE} --tp-size 1 \
        --mem-fraction-static 0.85 \
        --max-running-requests ${MAX_RUNNING_REQUESTS} \
        --log-level ${SGLANG_LOG_LEVEL:-info} &
    SERVER_PID=\$!
    trap 'kill \${SERVER_PID} 2>/dev/null || true' EXIT

    echo 'waiting for sglang to come up...'
    max_gpu_processes=0
    for i in \$(seq 1 180); do
        if curl -sf http://127.0.0.1:${SERVER_PORT}/health >/dev/null 2>&1; then
            echo \"sglang healthy after \${i}0s\"
            break
        fi
        kill -0 \${SERVER_PID} 2>/dev/null || { echo 'sglang died during startup'; exit 1; }

        # The DP controller can remain alive indefinitely after one scheduler is
        # SIGKILLed.  Detect a GPU process disappearing after all replicas have
        # attached, so the dependency chain retries instead of wasting four hours.
        gpu_processes=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*\$/d' | wc -l)
        (( gpu_processes > max_gpu_processes )) && max_gpu_processes=\${gpu_processes}
        if (( max_gpu_processes >= ${GPUS_PER_NODE} && gpu_processes < ${GPUS_PER_NODE} )); then
            echo \"sglang lost a DP worker during startup: expected=${GPUS_PER_NODE} current=\${gpu_processes}\" >&2
            exit 1
        fi
        sleep 10
    done
    curl -sf http://127.0.0.1:${SERVER_PORT}/health >/dev/null || { echo 'sglang never became healthy'; exit 1; }

    python3 experiments/src/difficulty_filter/measure_pass_rate.py \
        --prompt-data ${PROMPT_DATA} \
        --output ${OUTPUT} \
        --model-path /ckpt/hf/${HF_MODEL_NAME} \
        --server-url http://127.0.0.1:${SERVER_PORT} \
        --rm-type ${RM_TYPE} \
        ${ZERO_TRUNCATED_ARG} \
        ${CUSTOM_RM_PATH:+--custom-rm-path ${CUSTOM_RM_PATH}} \
        ${PREFLIGHT:+--preflight ${PREFLIGHT}} \
        --policy ${MODEL_NAME} \
        ${REWARD_KEY:+--reward-key ${REWARD_KEY}} \
        ${TOOL_KEY:+--tool-key ${TOOL_KEY}} \
        ${INPUT_KEY:+--input-key ${INPUT_KEY}} \
        ${LABEL_KEY:+--label-key ${LABEL_KEY}} \
        --n-samples ${N_SAMPLES} \
        ${SAMPLES_PER_REQUEST_ARG} \
        --temperature ${TEMPERATURE} \
        --top-p ${TOP_P} \
        --top-k ${TOP_K} \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        ${MAX_CONTEXT_ARG} \
        --concurrency ${CONCURRENCY} \
        --start-index ${START_INDEX} \
        ${END_INDEX_ARG} \
        --pass-rate-min ${PASS_RATE_MIN} \
        --pass-rate-max ${PASS_RATE_MAX} \
        ${LIMIT_ARG} ${DUMP_ARG}
    touch ${OUTPUT}.measured
fi

if [[ -n \"${FILTER_OUTPUT_ARGS}\" ]]; then
    python3 -m experiments.src.difficulty_filter.apply_filter \
        --prompt-data ${PROMPT_DATA} \
        --pass-rates ${OUTPUT} \
        ${FILTER_OUTPUT_ARGS} \
        --pass-rate-min ${PASS_RATE_MIN} \
        --pass-rate-max ${PASS_RATE_MAX}
fi
touch ${OUTPUT}.complete
"
