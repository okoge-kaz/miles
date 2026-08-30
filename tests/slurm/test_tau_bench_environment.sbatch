#!/bin/bash
#SBATCH --job-name=test-tau3-environment
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

# Exercise the pinned Tau three Gym lifecycle and every prepared task identity
# without making an external model request.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
export WANDB_MODE=offline
source "${REPO_ROOT}/experiments/env.sh"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
TAU3_SQSH_IMAGE="${TAU3_SQSH_IMAGE:-${CONTAINER_DIR}/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0-tau3-1.0.1.sqsh}"
[[ -r "${TAU3_SQSH_IMAGE}" ]]

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
export NVIDIA_INFERENCE_API_KEY=non-secret-offline-test

/opt/tau3/.venv/bin/python - <<"PY"
import json
from pathlib import Path

from experiments.src.environments.tau_bench.runtime import TauSession, TauUserConfig, _official_task
from tau2.data_model.message import UserMessage
import tau2.user.user_simulator as user_module

rows = [
    json.loads(line)
    for line in Path("/data/tau-bench/tau3-test-miles.jsonl").read_text().splitlines()
]
assert len(rows) == 100
for row in rows:
    assert row["metadata"]["eval_only"] is True
    _official_task(row["metadata"])

responses = iter(("I need help with an order.", "###STOP###"))


def fake_generate(**_kwargs):
    return UserMessage(role="user", content=next(responses))


user_module.generate = fake_generate
metadata = json.loads(
    Path("/data/tau-bench/tau3-retail-test-miles.jsonl").read_text().splitlines()[0]
)["metadata"]
session = TauSession(metadata, TauUserConfig(provider="nvidia", model="offline-test"), max_steps=20)
initial = session.reset()
assert [message["role"] for message in initial.messages] == ["assistant", "user"]
assert any(tool["function"]["name"] == "done" for tool in initial.tools)
step = session.step("I can help. What is your order ID?")
assert step.terminated is True
assert step.reward == 0.0
assert step.reward_info["reward"] == 0.0
assert step.simulation_run["messages"]
session.close()
print("Tau three Gym: 100 held-out task identities and terminal reward lifecycle=ok")
PY
'
