#!/bin/bash
#PBS -N test-tau3-environment
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Exercise the pinned Tau three Gym lifecycle and every prepared task identity
# without making an external model request.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
export WANDB_MODE=offline
source "${REPO_ROOT}/experiments/env.sh"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -r "${CONTAINER_IMAGE}" ]]

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro" \
    --cwd=/root/miles \
    --writable-tmpfs \
    --no-home \
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
