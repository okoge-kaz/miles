#!/bin/bash
#PBS -N taurb-summary
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)}"
cd "${REPO_ROOT}"

: "${VALIDATION_MANIFEST:?}"
: "${VALIDATION_SUMMARY:?}"

python3 experiments/tools/replay_buffer_validation/tau2/analyze.py \
    "${VALIDATION_MANIFEST}" \
    --output "${VALIDATION_SUMMARY}"
