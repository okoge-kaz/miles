#!/bin/bash
#SBATCH --job-name=passrate
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/difficulty/%x-%j.log

# Measure a policy's per-prompt pass rate over a whole prompt file.
#
#   sbatch -A coreai_horizon_dilations \
#     experiments/src/difficulty_filter/run_measure.sbatch
#
# Knobs (all --export=ALL environment variables):
#   MODEL_NAME       HF dir under /ckpt/hf                (Qwen3-4B-Instruct-2507)
#   PROMPT_DATA      in-container prompt jsonl
#   OUTPUT           in-container output jsonl (resumable)
#   N_SAMPLES        samples per prompt                    (8, matches training)
#   MAX_NEW_TOKENS   generation cap; MUST match the recipe    (24576)
#                    --rollout-max-response-len
#   RM_TYPE          verifier; MUST match the recipe        (math)
#                    `deepscaler` returns 0 unless the response contains
#                    `</think>`, so it zeroes every non-thinking model.
#   CONCURRENCY      prompts in flight                     (96)
#   MAX_RUNNING_REQUESTS  sequences admitted per replica    (96)
#   SGLANG_LOG_LEVEL sglang verbosity                      (info)
#                    Keep at info: the Decode-batch lines carry token usage,
#                    running-req and gen throughput, which are the only way to
#                    tell a healthy sweep from one silently retracting.
#   DUMP_RESPONSES   audit file: full responses + rewards for DUMP_LIMIT prompts
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
PROMPT_DATA="${PROMPT_DATA:-/data/dapo-math-17k/dapo-math-17k.jsonl}"
OUTPUT="${OUTPUT:-/data/difficulty/dapo-math-17k.${MODEL_NAME}.passrate.jsonl}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24576}"
CONCURRENCY="${CONCURRENCY:-96}"
# Sequences admitted per dp replica. One prompt expands to N_SAMPLES
# sequences, so the client-side CONCURRENCY is in prompts while this is in
# sequences: keeping them equal makes total in-flight sequences
# CONCURRENCY * N_SAMPLES spread over GPUS_PER_NODE replicas, i.e.
# CONCURRENCY sequences each. Sized against the KV budget below.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
RM_TYPE="${RM_TYPE:-math}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SERVER_PORT="${SERVER_PORT:-30000}"
LIMIT_ARG=""
[[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"
DUMP_ARG=""
[[ -n "${DUMP_RESPONSES:-}" ]] && DUMP_ARG="--dump-responses ${DUMP_RESPONSES} --dump-limit ${DUMP_LIMIT:-8}"

mkdir -p "${OUTPUT_DIR}/difficulty" "${DATASET_DIR}/difficulty"

srun \
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
    --model-path /ckpt/hf/${MODEL_NAME} \
    --host 127.0.0.1 --port ${SERVER_PORT} \
    --dp-size ${GPUS_PER_NODE} --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-running-requests ${MAX_RUNNING_REQUESTS} \
    --log-level ${SGLANG_LOG_LEVEL:-info} &
SERVER_PID=\$!
trap 'kill \${SERVER_PID} 2>/dev/null || true' EXIT

echo 'waiting for sglang to come up...'
for i in \$(seq 1 180); do
    if curl -sf http://127.0.0.1:${SERVER_PORT}/health >/dev/null 2>&1; then
        echo \"sglang healthy after \${i}0s\"
        break
    fi
    kill -0 \${SERVER_PID} 2>/dev/null || { echo 'sglang died during startup'; exit 1; }
    sleep 10
done
curl -sf http://127.0.0.1:${SERVER_PORT}/health >/dev/null || { echo 'sglang never became healthy'; exit 1; }

python3 experiments/src/difficulty_filter/measure_pass_rate.py \
    --prompt-data ${PROMPT_DATA} \
    --output ${OUTPUT} \
    --model-path /ckpt/hf/${MODEL_NAME} \
    --server-url http://127.0.0.1:${SERVER_PORT} \
    --rm-type ${RM_TYPE} \
    --policy ${MODEL_NAME} \
    ${REWARD_KEY:+--reward-key ${REWARD_KEY}} \
    --n-samples ${N_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --concurrency ${CONCURRENCY} \
    ${LIMIT_ARG} ${DUMP_ARG}
"
