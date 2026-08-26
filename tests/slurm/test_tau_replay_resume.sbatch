#!/bin/bash
#SBATCH --job-name=test-tau-replay
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
export WANDB_MODE=offline

pytest -q tests/fast/experiments/test_tau_replay_resume.py
pytest -q tests/fast/rollout/test_replay_buffer.py \
    -k "dataset_fingerprint_includes_tau_environment_configuration or \
prepared_batch_is_a_warm_resume_hit_and_active_prompt_is_regenerated or \
replay_buffer_checksum_fingerprint_and_retention"
pytest -q tests/fast/test_train_async_replay_buffer.py::test_replay_buffer_commit_order
'
