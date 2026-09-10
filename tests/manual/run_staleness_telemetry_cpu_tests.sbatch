#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=miles-staleness-telemetry-cpu
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=experiments/outputs/staleness-telemetry-cpu-%j.log

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
source "${REPO_ROOT}/experiments/env.sh"
test -r "${SQSH_IMAGE}"

# Pure CPU telemetry checks; executor/backend imports requiring CUDA belong in
# run_oci_validation.sbatch. Removed experiments/src parsers are not recreated.
srun --container-image="${SQSH_IMAGE}" \
    --container-mounts="${REPO_ROOT}:/root/miles" \
    --container-workdir=/root/miles --no-container-mount-home bash -lc '
    set -euo pipefail
    export PYTHONPATH=/root/miles HF_HUB_OFFLINE=1
    python -m pytest -q tests/fast/experiments \
        tests/fast/rollout/test_queue_policy.py \
        tests/fast/rollout/test_queue_telemetry.py \
        tests/fast/rollout/test_recycle_compute_metrics.py \
        tests/fast/rollout/test_partial_rollout_telemetry.py
'
