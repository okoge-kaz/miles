#!/bin/bash
#SBATCH --job-name=training-dump-summary
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
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

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${DUMP_DIR}:/dump" \
    --container-writable \
    --no-container-mount-home \
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
