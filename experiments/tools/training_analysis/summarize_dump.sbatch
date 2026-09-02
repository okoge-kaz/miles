#!/bin/bash
#PBS -N training-dump-summary
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="$(realpath -- "${PBS_O_WORKDIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"
: "${DUMP_DIR:?set DUMP_DIR to the host-side dump directory}"

DUMP_DIR="$(realpath -- "${DUMP_DIR}")"
OUTPUT_NAME="${OUTPUT_NAME:-training-summary.json}"
REQUESTED_STALENESS_BOUND="${REQUESTED_STALENESS_BOUND:-4}"
[[ "${OUTPUT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "OUTPUT_NAME must be a filename without directory components" >&2
    exit 2
}
[[ "${REQUESTED_STALENESS_BOUND}" =~ ^[0-9]+$ ]] || {
    echo "REQUESTED_STALENESS_BOUND must be a non-negative integer" >&2
    exit 2
}

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},${DUMP_DIR}:/dump" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
python3 experiments/tools/training_analysis/summarize_dump.py /dump \
    --requested-staleness-bound "$1" \
    --output "/dump/$2"
' _ "${REQUESTED_STALENESS_BOUND}" "${OUTPUT_NAME}"

echo "summary written to ${DUMP_DIR}/${OUTPUT_NAME}"
