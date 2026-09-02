#!/bin/bash
#PBS -N prep-livecodebench
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -j oe

# Convert LiveCodeBench release_v6/latest into the Miles prompt schema. This is
# evaluation-only: using these benchmark answers for training contaminates the
# metric. The private tests stay encoded exactly as published.

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
