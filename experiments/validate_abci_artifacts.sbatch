#!/bin/bash
#PBS -N miles-v01-artifacts
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -j oe

# Run large replay deserialization on a CPU compute allocation.
set -euo pipefail
cd "${PBS_O_WORKDIR:?submit from the Miles checkout}"
export TRAIN_CKPT_DIR="${PWD}/experiments/outputs/abci-validation/checkpoints"
export MILES_REPO="${MILES_REPO:-${PWD}}"
source "${MILES_REPO}/experiments/env.sh"
miles_srun --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},${PWD}/experiments/outputs/abci-validation:/validation" \
    --no-home --export=ALL bash -lc '
set -euo pipefail
cd /root/miles
provenance_args=()
if [[ "${REQUIRE_COMPLETE_POLICY_PROVENANCE:-0}" == 1 ]]; then
    provenance_args+=(--require-complete-policy-provenance)
fi
PYTHONPATH=/root/miles python3 -m experiments.tools.analyze_abci_artifacts \
    --checkpoint "/ckpt/training/${VALIDATION_CHECKPOINT_RELATIVE:?}" \
    --output "/validation/${ARTIFACT_REPORT_NAME:?}" "${provenance_args[@]}"
'
