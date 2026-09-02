#!/bin/bash
#PBS -N prep-workplace-local-env
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Convert Workplace rows while using only the resource-specific modules and CSV
# fixtures from the pinned source tree. No NeMo Gym package or server is used.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

readonly WORKPLACE_COMMIT=48d5b9c01e3fc59a49f19674d0034a6f06396074
readonly RESOURCE_ROOT="${DATASET_DIR}/nemo-gym-code"
[[ "$(git -C "${RESOURCE_ROOT}" rev-parse HEAD)" == "${WORKPLACE_COMMIT}" ]] || {
    echo "Workplace resource source is not pinned to ${WORKPLACE_COMMIT}" >&2
    exit 1
}

miles_srun --image="${CONTAINER_IMAGE}" --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs --no-home --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
python3 -m experiments.src.datasets.workplace.prepare \
    --train-input /data/nemotron-rl-workplace-assistant/train.jsonl \
    --validation-input /data/nemotron-rl-workplace-assistant/validation.jsonl \
    --train-output /data/nemotron-rl-workplace-assistant/miles-local-env-train.jsonl \
    --validation-output /data/nemotron-rl-workplace-assistant/miles-local-env-validation.jsonl
test "$(wc -l < /data/nemotron-rl-workplace-assistant/miles-local-env-train.jsonl)" -eq 1255
test "$(wc -l < /data/nemotron-rl-workplace-assistant/miles-local-env-validation.jsonl)" -eq 545
python3 - <<"PY"
from experiments.src.environments.workplace.runtime import create_tool_environment

environment = create_tool_environment()
assert len(environment["functions"]) == 27
assert len(environment["schemas"]) == 27
print("Workplace standalone environment: 27 tools loaded without a server")
PY
'
