#!/bin/bash
#PBS -N test-notes
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=USER,WANDB_MODE=offline \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles

python3 -m pytest -q tests/fast/experiments/test_notes_consistency.py
'
