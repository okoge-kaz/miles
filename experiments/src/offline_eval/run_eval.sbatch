#!/bin/bash
#SBATCH --job-name=offline-eval
#SBATCH --partition=batch,batch_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/offline_eval/%x-%j.log

# Evaluate one checkpoint on every AIME year, offline.
# Procedure, benchmark choice and failure modes: notes/offline-eval.md
#
# Knobs:
#   CKPT           HF checkpoint dir, in-container (required)
#   BENCHMARKS     '+'-separated <name>:<path> pairs. '+' because sbatch
#                  --export splits on commas.
#   N_SAMPLES      samples per problem (16)
#   MAX_NEW_TOKENS generation cap (32768)
#   TEMPERATURE / TOP_P  sampling (1.0 / 1.0, matching training)
#   RM_TYPE        verifier; must match the policy (math for non-thinking)
#   OUT_DIR        where the per-benchmark jsonl goes
#   EXTRA_MOUNTS   appended to CONTAINER_MOUNTS; required, see the note
#   UNPAD_VOCAB    1: unpad CKPT into node-local /tmp before loading (0)

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${CKPT:?set CKPT to an in-container HF checkpoint dir, e.g. /ckpt/training/.../hf/120}"

BENCHMARKS="${BENCHMARKS:-aime24:/data/aime-2024/aime-2024.jsonl+aime25:/data/aime-2025/aime-2025.jsonl+aime26:/data/aime-2026/aime-2026.jsonl}"
N_SAMPLES="${N_SAMPLES:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
RM_TYPE="${RM_TYPE:-math}"
CONCURRENCY="${CONCURRENCY:-96}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
SERVER_PORT="${SERVER_PORT:-30000}"
UNPAD_VOCAB="${UNPAD_VOCAB:-0}"

# One directory per checkpoint, named after it, so a sweep's results do not mix.
TAG="${TAG:-$(echo "${CKPT}" | sed 's|^/ckpt/training/||; s|/|_|g')}"
OUT_DIR="${OUT_DIR:-/data/offline_eval/${TAG}}"

mkdir -p "${OUTPUT_DIR}/offline_eval" "${DATASET_DIR}/offline_eval"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc "
set -euo pipefail
cd /root/miles
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH=/root/Megatron-LM:/root/miles
export PYTHONUNBUFFERED=1

[[ -d ${CKPT} ]] || { echo \"no such checkpoint: ${CKPT}\"; exit 1; }
mkdir -p ${OUT_DIR}

EVAL_CKPT=${CKPT}
if [[ '${UNPAD_VOCAB}' == 1 ]]; then
    EVAL_CKPT=/tmp/offline-eval-unpadded-${SLURM_JOB_ID}
    python3 experiments/src/offline_eval/unpad_vocab.py ${CKPT} \${EVAL_CKPT}
    [[ -d \${EVAL_CKPT} ]] || EVAL_CKPT=${CKPT}
fi

python3 -m sglang.launch_server \
    --model-path \${EVAL_CKPT} \
    --host 127.0.0.1 --port ${SERVER_PORT} \
    --dp-size ${GPUS_PER_NODE} --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-running-requests ${MAX_RUNNING_REQUESTS} \
    --log-level ${SGLANG_LOG_LEVEL:-info} &
SERVER_PID=\$!
trap 'kill \${SERVER_PID} 2>/dev/null || true' EXIT

echo 'waiting for sglang to come up...'
for i in \$(seq 1 180); do
    curl -sf http://127.0.0.1:${SERVER_PORT}/health >/dev/null 2>&1 && { echo \"sglang healthy after \${i}0s\"; break; }
    kill -0 \${SERVER_PID} 2>/dev/null || { echo 'sglang died during startup'; exit 1; }
    sleep 10
done
curl -sf http://127.0.0.1:${SERVER_PORT}/health >/dev/null || { echo 'sglang never became healthy'; exit 1; }

pids=""
for spec in \$(echo '${BENCHMARKS}' | tr '+' ' '); do
    name=\${spec%%:*}
    path=\${spec#*:}
    if [[ ! -f \${path} ]]; then
        echo \"SKIP \${name}: no such file \${path}\"
        continue
    fi
    echo \"=== \${name} (\${path}) starting ===\"
    # Backgrounded, not sequential -- see the note.
    python3 experiments/src/difficulty_filter/measure_pass_rate.py \
        --prompt-data \${path} \
        --output ${OUT_DIR}/\${name}.jsonl \
        --model-path \${EVAL_CKPT} \
        --server-url http://127.0.0.1:${SERVER_PORT} \
        --rm-type ${RM_TYPE} \
        --policy \${name} \
        --n-samples ${N_SAMPLES} \
        --temperature ${TEMPERATURE} \
        --top-p ${TOP_P} \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --concurrency ${CONCURRENCY} \
        --preflight ${PREFLIGHT:-strict} > ${OUT_DIR}/\${name}.measure.log 2>&1 &
    pids=\"\${pids} \$!\"
done

rc=0
for pid in \${pids}; do wait \${pid} || rc=1; done
for spec in \$(echo '${BENCHMARKS}' | tr '+' ' '); do
    name=\${spec%%:*}
    [[ -f ${OUT_DIR}/\${name}.measure.log ]] && { echo \"=== \${name} ===\"; tail -40 ${OUT_DIR}/\${name}.measure.log; }
done
[[ \${rc} -eq 0 ]] || { echo 'a benchmark failed'; exit 1; }

echo '=== report ==='
python3 experiments/src/offline_eval/report.py ${OUT_DIR}
"
