#!/bin/bash
#PBS -N prep-code-rl-eval
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -j oe

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

SOURCE=/data/nemotron-rl-comp-coding
OUT=/data/nemotron-performance-transfer
python3 -m experiments.src.datasets.nemotron.convert \
    --dataset competitive-coding \
    --input "${SOURCE}/validation.jsonl" \
    --output "${OUT}/code-validation-miles.jsonl" \
    --summary "${OUT}/code-validation-summary.json"
python3 -m experiments.src.datasets.common.audit \
    --input "${OUT}/code-validation-miles.jsonl" \
    --expected-rows 322 \
    --require-verifiers python_code \
    --output "${OUT}/code-validation-audit.json"
python3 -m experiments.src.datasets.common.merge \
    --input "${OUT}/code-validation-miles.jsonl" \
    --output "${OUT}/code-validation-64-miles.jsonl" \
    --rows-per-input 64 \
    --seed 42 \
    --require-verifiers python_code \
    --summary "${OUT}/code-validation-64-summary.json"
python3 -m experiments.src.datasets.common.audit \
    --input "${OUT}/code-validation-64-miles.jsonl" \
    --expected-rows 64 \
    --require-verifiers python_code \
    --output "${OUT}/code-validation-64-audit.json"
'
