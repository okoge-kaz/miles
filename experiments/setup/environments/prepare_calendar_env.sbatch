#!/bin/bash
#SBATCH --job-name=prep-calendar-local-env
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Convert every Calendar v2 row and solve it once with the local verifier. This
# job does not install, import, or start NeMo Gym.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

CALENDAR_ROOT="${DATASET_DIR}/nemotron-rl-ifollow-calendar-v2"
srun --container-image="${SQSH_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable --no-container-mount-home --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
python3 -m experiments.src.datasets.calendar.prepare \
    --train-input /data/nemotron-rl-ifollow-calendar-v2/train.jsonl \
    --validation-input /data/nemotron-rl-ifollow-calendar-v2/validation.jsonl \
    --train-output /data/nemotron-rl-ifollow-calendar-v2/miles-local-env-train.jsonl \
    --validation-output /data/nemotron-rl-ifollow-calendar-v2/miles-local-env-validation.jsonl
test "$(wc -l < /data/nemotron-rl-ifollow-calendar-v2/miles-local-env-train.jsonl)" -eq 9659
test "$(wc -l < /data/nemotron-rl-ifollow-calendar-v2/miles-local-env-validation.jsonl)" -eq 256
'

printf 'Calendar local env prepared under %s\n' "${CALENDAR_ROOT}"
