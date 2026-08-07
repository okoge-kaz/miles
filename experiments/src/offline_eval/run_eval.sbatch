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
#
#   sbatch -A coreai_horizon_dilations \
#     --export=ALL,CKPT=/ckpt/training/math/dapo-math-p10-90/Qwen3-4B-Instruct-2507/<config>/hf/120 \
#     experiments/src/offline_eval/run_eval.sbatch
#
# This is the path that produces reported numbers. It is deliberately separate
# from the in-training eval, which is AIME-2025 only at n=8 and exists to show
# that a run is learning:
#
#   * the generation budget is 32768, not the 24576 the training recipe uses.
#     13.75% of AIME-2024 truncates at 24576 and a truncated sample scores 0
#     under every rule-based verifier, so the in-training number is depressed by
#     the budget rather than by the model.
#   * n is 16 rather than 8, over four years rather than one. 30 problems put the
#     standard error at ~9 points; 120 problems put it at ~4.5.
#
#     All four years carry the same instruction wrapper, asserted at staging time by
#     prepare_aime.py --template-reference. AIME-2024 was the exception until
#     2026-08-05 (bare problem text, so the boxed answer the verifier grades was
#     never asked for); aime-2024.nowrapper.jsonl is that older file, and scores
#     measured against it do not compare to scores measured now.
#   * sampling can be set to whatever the comparison needs. Qwen's own card
#     reports AIME25 = 47.4 at temperature 0.7 / top_p 0.8, and the training
#     recipe generates at 1.0 / 1.0.
#
# Knobs:
#   CKPT           HF checkpoint dir, in-container (required). Written by
#                  --save-hf at every --save-interval.
# AIME 2023 is deliberately NOT in the default set: it is in the training data.
# Normalized verbatim matching against the prompt files gives
#   dapo-math-17k   23/30 of AIME-2023   (and 1/30 of AIME-2026)
#   dapo-math-p10-90 11/30               <- the set actually trained on
#   deepscaler       26/30
# so an AIME-2023 score is a memorisation score, not a generalisation one.
# 2024/2025 are clean against all three. 2026 is clean against the filtered
# p10-80 set but not against unfiltered dapo-math-17k, so re-check the overlap
# whenever the difficulty window is widened. Matching is exact-after-normalisation
# and therefore a LOWER bound: paraphrases and number-swapped variants are missed.
#
#   BENCHMARKS     '+'-separated <name>:<path> pairs. Default is the AIME years
#                  present on disk. '+' because sbatch --export splits on commas.
#   N_SAMPLES      samples per problem (16)
#   MAX_NEW_TOKENS generation cap (32768)
#   TEMPERATURE / TOP_P  sampling (1.0 / 1.0, matching training)
#   RM_TYPE        verifier; must match the policy (math for non-thinking)
#   OUT_DIR        where the per-benchmark jsonl goes
#   EXTRA_MOUNTS   appended to CONTAINER_MOUNTS. unpad_vocab.py symlinks the
#                  shards it did not rewrite, so the source tree has to be
#                  visible inside the container at the same host path.
#
# Resumable: measure_pass_rate.py appends per prompt and skips indices already
# present, so re-submitting after a wall-clock kill continues.

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

# dp-size 8, tp-size 1: a 4B model fits on one H100, so eight independent
# replicas beat one tensor-parallel engine for a pure-inference sweep -- no
# cross-GPU collectives on the critical path.
python3 -m sglang.launch_server \
    --model-path ${CKPT} \
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
    # Backgrounded, not sequential. A benchmark is 30 prompts, so its last few
    # long generations leave the eight replicas nearly idle; run sequentially
    # that tail is paid once per year. Overlapping them keeps the engines fed and
    # costs nothing, since 90 prompts still fits under --concurrency.
    python3 experiments/src/difficulty_filter/measure_pass_rate.py \
        --prompt-data \${path} \
        --output ${OUT_DIR}/\${name}.jsonl \
        --model-path ${CKPT} \
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
