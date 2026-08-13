#!/bin/bash
#SBATCH --job-name=queue-real-ray
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=experiments/outputs/queue-real-ray-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    bash -lc '
        set -euo pipefail
        cd /root/miles
        pytest -q \
            tests/fast/ray/rollout/real_ray/test_rollout_manager.py::TestGenerate::test_invokes_rollout_fn_with_correct_input_and_returns_dp_split
    '
