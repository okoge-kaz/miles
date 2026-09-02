#!/bin/bash
#PBS -N prep-math-code-stem
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -j oe

# Build an equal-domain training mixture for the 4-node multi-environment run.
# Benchmark evaluation rows are rejected by the merger and never enter this file.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles

OUT=/data/nemotron-performance-transfer
python3 -m experiments.src.datasets.common.merge \
    --input \
        /data/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl \
        "${OUT}/nemotron3-nano-competitive-code-train.jsonl" \
        "${OUT}/nemotron3-nano-knowledge-mcqa-reasoning-gym-train.jsonl" \
    --output "${OUT}/math-code-stem-balanced-train.jsonl" \
    --rows-per-input 10891 \
    --seed 42 \
    --default-verifier math \
    --default-source dapo-math-filtered \
    --require-verifiers math python_code mcqa_regex reasoning_gym \
    --summary "${OUT}/math-code-stem-balanced-summary.json"

python3 -m experiments.src.datasets.common.audit \
    --input "${OUT}/math-code-stem-balanced-train.jsonl" \
    --expected-rows 32673 \
    --require-verifiers math python_code mcqa_regex reasoning_gym \
    --output "${OUT}/math-code-stem-balanced-audit.json"
'
