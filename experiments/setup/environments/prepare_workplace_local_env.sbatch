#!/bin/bash
#SBATCH --job-name=prep-workplace-local-env
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Convert Workplace rows while using only the resource-specific modules and CSV
# fixtures from the pinned source tree. No NeMo Gym package or server is used.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

readonly WORKPLACE_COMMIT=48d5b9c01e3fc59a49f19674d0034a6f06396074
readonly RESOURCE_ROOT="${DATASET_DIR}/nemo-gym-code"
[[ "$(git -C "${RESOURCE_ROOT}" rev-parse HEAD)" == "${WORKPLACE_COMMIT}" ]] || {
    echo "Workplace resource source is not pinned to ${WORKPLACE_COMMIT}" >&2
    exit 1
}

srun --container-image="${SQSH_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable --no-container-mount-home --export=ALL \
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
