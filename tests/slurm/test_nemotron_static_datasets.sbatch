#!/bin/bash
#PBS -N test-nemotron-static-data
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
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
export WANDB_MODE=offline

python3 -m compileall -q \
    experiments/src/protocols \
    experiments/src/datasets/common/audit.py \
    experiments/src/datasets/common/merge.py \
    experiments/src/datasets/nemotron
pytest -q tests/fast/experiments/test_nemotron_static_datasets.py
'
