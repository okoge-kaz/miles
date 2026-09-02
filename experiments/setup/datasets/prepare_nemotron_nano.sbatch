#!/bin/bash
#PBS -N prep-nemotron-nano
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=08:00:00
#PBS -j oe

# Restore the two masked math components and split the Nano blend into:
#   miles-train-static.jsonl       83,013 rows with local deterministic rewards
#   miles-train-workbench.jsonl    10,229 rows requiring a NeMo Gym Workbench server
#   miles-unverifiable.jsonl       published source rows with no ground truth

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

NANO=/data/nemotron-3-nano-rl-training-blend
python3 -m experiments.src.datasets.nemotron.restore \
    --input "${NANO}/train.jsonl" \
    --output "${NANO}/train-restored.jsonl" \
    --dapo /data/dapo-math-17k-byted/data/dapo-math-17k.parquet \
    --skywork /data/skywork-or1-rl/data/math-*.parquet

python3 -m experiments.src.datasets.nemotron.convert \
    --dataset nano-blend \
    --input "${NANO}/train-restored.jsonl" \
    --output "${NANO}/miles-train-static.jsonl" \
    --environment-output "${NANO}/miles-train-workbench.jsonl" \
    --unverifiable-output "${NANO}/miles-unverifiable.jsonl" \
    --summary "${NANO}/miles-conversion-summary.json"

test "$(wc -l < "${NANO}/train-restored.jsonl")" -eq 93244
test "$(wc -l < "${NANO}/miles-train-static.jsonl")" -eq 83013
test "$(wc -l < "${NANO}/miles-train-workbench.jsonl")" -eq 10229
test "$(wc -l < "${NANO}/miles-unverifiable.jsonl")" -eq 2
python3 -m experiments.src.datasets.common.audit \
    --input "${NANO}/miles-train-static.jsonl" \
    --expected-rows 83013 \
    --require-verifiers math mcqa_regex python_code json_schema ifeval_g \
    --output "${NANO}/miles-audit.json"
'
