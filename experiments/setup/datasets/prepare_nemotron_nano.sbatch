#!/bin/bash
#SBATCH --job-name=prep-nemotron-nano
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Restore the two masked math components and split the Nano blend into:
#   miles-train-static.jsonl       83,013 rows with local deterministic rewards
#   miles-train-workbench.jsonl    10,229 rows requiring a NeMo Gym Workbench server
#   miles-unverifiable.jsonl       published source rows with no ground truth

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
