#!/bin/bash
#PBS -N prep-calendar-local-env
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Convert every Calendar v2 row and solve it once with the local verifier. This
# job does not install, import, or start NeMo Gym.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

CALENDAR_ROOT="${DATASET_DIR}/nemotron-rl-ifollow-calendar-v2"
miles_srun --image="${CONTAINER_IMAGE}" --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs --no-home --export=ALL \
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
