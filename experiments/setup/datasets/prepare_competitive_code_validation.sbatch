#!/bin/bash
#SBATCH --job-name=prep-domain-rl-eval
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
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
