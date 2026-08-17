#!/bin/bash
#SBATCH --job-name=format-pilot
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/format-pilot/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
PROMPT_DATA="${PROMPT_DATA:-/data/dapo-math-p10-90/dapo-math-p10-90.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-42}"
SERVER_PORT="${SERVER_PORT:-30000}"
PILOT_TAG="${PILOT_TAG:-instruct-2507-prompt-only-${SLURM_JOB_ID}}"
PILOT_OUTPUT="/root/miles/experiments/outputs/format-pilot/${PILOT_TAG}"

mkdir -p "${OUTPUT_DIR}/format-pilot"

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

python3 -m sglang.launch_server \\
    --model-path /ckpt/hf/${MODEL_NAME} \\
    --host 127.0.0.1 --port ${SERVER_PORT} \\
    --dp-size ${GPUS_PER_NODE} --tp-size 1 \\
    --mem-fraction-static 0.85 \\
    --max-running-requests ${MAX_RUNNING_REQUESTS} \\
    --log-level info &
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

python3 experiments/src/format_pilot/measure_prompt_formats.py \\
    --prompt-data ${PROMPT_DATA} \\
    --model-path /ckpt/hf/${MODEL_NAME} \\
    --server-url http://127.0.0.1:${SERVER_PORT} \\
    --output-dir ${PILOT_OUTPUT} \\
    --num-prompts ${NUM_PROMPTS} \\
    --n-samples ${N_SAMPLES} \\
    --temperature ${TEMPERATURE} \\
    --top-p ${TOP_P} \\
    --top-k ${TOP_K} \\
    --max-new-tokens ${MAX_NEW_TOKENS} \\
    --concurrency ${CONCURRENCY} \\
    --seed ${SEED}
"
