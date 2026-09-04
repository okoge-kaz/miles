#!/bin/bash
#SBATCH --job-name=prep-livecodebench
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Convert LiveCodeBench release_v6/latest into the Miles prompt schema. This is
# evaluation-only: using these benchmark answers for training contaminates the
# metric. The private tests stay encoded exactly as published.

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
LCB=/data/livecodebench-lite
python3 -m experiments.src.datasets.nemotron.convert \
    --dataset livecodebench \
    --input "${LCB}"/test.jsonl "${LCB}"/test2.jsonl "${LCB}"/test3.jsonl \
            "${LCB}"/test4.jsonl "${LCB}"/test5.jsonl "${LCB}"/test6.jsonl \
    --output "${LCB}/livecodebench-release-v6-miles-eval.jsonl" \
    --summary "${LCB}/miles-conversion-summary.json"
python3 -m experiments.src.datasets.common.audit \
    --input "${LCB}/livecodebench-release-v6-miles-eval.jsonl" \
    --require-verifiers livecodebench \
    --require-eval-only \
    --output "${LCB}/miles-audit.json"
'
