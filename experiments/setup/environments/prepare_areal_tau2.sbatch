#!/bin/bash
#SBATCH --job-name=prepare-areal-tau2
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log
#SBATCH --export=NIL

# Validate all AReaL tasks/DBs against Tau2 v1.0.1 and precompute each task's
# official gold terminal state. This is a one-time cost, not rollout overhead.

set -euo pipefail

USER="${USER:-$(id -un)}"
WANDB_MODE=disabled
export USER WANDB_MODE

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

readonly DATA_ROOT="${DATASET_DIR}/areal-tau2-data"
TAU3_SQSH_IMAGE="${TAU3_SQSH_IMAGE:-${CONTAINER_DIR}/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0-tau3-1.0.1.sqsh}"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -r "${TAU3_SQSH_IMAGE}" ]] || { echo "Tau image not found: ${TAU3_SQSH_IMAGE}" >&2; exit 1; }
[[ -r "${DATA_ROOT}/tau2_rl_train.jsonl" ]] || { echo "AReaL Tau2 RL source is not staged" >&2; exit 1; }
[[ -r "${DOTENV_MASK}" ]]

srun \
    --container-image="${TAU3_SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro" \
    --container-workdir=/root/miles \
    --container-writable \
    --no-container-mount-home \
    --export=USER,WANDB_MODE \
    bash -lc '
set -euo pipefail
export PATH=/opt/tau3/.venv/bin:${PATH}
export PYTHONPATH=/root/miles
export TAU2_DATA_DIR=/opt/tau3/data
export AREAL_TAU2_ROOT=/data/areal-tau2-data
export LOGURU_LEVEL=ERROR

python3 -m experiments.src.datasets.areal_tau2.prepare \
    --dataset-root /data/areal-tau2-data \
    --input /data/areal-tau2-data/tau2_rl_train.jsonl \
    --output /data/areal-tau2-data/miles-tau2-rl-train.jsonl \
    --summary /data/areal-tau2-data/miles-tau2-rl-summary.json \
    --workers 16

python3 - <<"PY"
import json
from pathlib import Path

root = Path("/data/areal-tau2-data")
summary = json.loads((root / "miles-tau2-rl-summary.json").read_text())
assert summary["rows"] == 1982
assert summary["domain_counts"] == {"airline": 1148, "retail": 563, "telecom": 271}
assert summary["interaction_mode"] == "stateful_multi_turn_user_simulator_environment"
assert summary["user_simulator"] is True
assert summary["verifier"] == "areal_tau2_environment"
assert summary["environment_policy"] == "areal-tau2-user-simulator-terminal-state-rl"
assert summary["tau_package_version"] == "1.0.1"
assert len(summary["db_sha256"]) == 9
assert summary["schedule"]["updates_by_rollout_batch_size_and_epoch"]["32"] == {
    "5": 310,
    "6": 372,
}
assert sum(1 for _ in (root / "miles-tau2-rl-train.jsonl").open()) == 1982
print(json.dumps(summary, indent=2, sort_keys=True))
PY
'
