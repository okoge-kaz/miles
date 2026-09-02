#!/bin/bash
#PBS -N miles-check-tau3-image
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

# Compatibility validation for the former Enroot Tau image derivation. Tau is
# now pinned by container/miles.def and shipped in the single Miles SIF.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

[[ -r "${CONTAINER_IMAGE}" ]] || {
    echo "missing CONTAINER_IMAGE=${CONTAINER_IMAGE}; run experiments/setup.sh container --submit" >&2
    exit 1
}

singularity exec --no-eval --no-home "${CONTAINER_IMAGE}" /opt/tau3/.venv/bin/python -c '
import importlib.metadata
from tau2.runner.helpers import load_tasks

assert importlib.metadata.version("tau2") == "1.0.1"
assert len(load_tasks("retail", "train")) == 74
assert len(load_tasks("airline", "train")) == 30
assert len(load_tasks("telecom", "train")) == 74
print("Tau three v1.0.1 Gym runtime: ok")
'
echo "Tau three dependencies are present in ${CONTAINER_IMAGE}"
