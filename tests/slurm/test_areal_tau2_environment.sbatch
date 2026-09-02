#!/bin/bash
#PBS -N test-areal-tau2-env
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Validate the exact prepared AReaL split and exercise the custom official Gym
# lifecycle without making an external user-model request.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
export WANDB_MODE=offline
source "${REPO_ROOT}/experiments/env.sh"

DATA_ROOT="${DATASET_DIR}/areal-tau2-data"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -r "${CONTAINER_IMAGE}" ]]
[[ -r "${DATA_ROOT}/miles-tau2-rl-train.jsonl" ]]
[[ -r "${DATA_ROOT}/miles-tau2-rl-summary.json" ]]

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro" \
    --cwd=/root/miles \
    --writable-tmpfs \
    --no-home \
    --export=USER,WANDB_MODE \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles

python3 -m pytest -q \
    tests/fast/experiments/test_areal_tau2.py \
    tests/fast/experiments/test_experiment_src_layout.py \
    tests/fast/experiments/test_notes_consistency.py \
    tests/fast/experiments/test_setup_layout.py \
    tests/fast/experiments/test_tau_bench.py \
    tests/fast/experiments/test_tau_bench_dataset.py \
    tests/fast/experiments/test_workplace_dataset.py \
    tests/fast/experiments/test_workplace_env.py \
    tests/fast/ray/rollout/test_train_data_conversion.py \
    tests/fast/rollout/test_replay_buffer.py \
    tests/fast/utils/test_arguments.py

export PATH=${PATH}:/opt/tau3/.venv/bin
export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/opt/tau3/src:/opt/tau3/.venv/lib/python3.12/site-packages:/root/miles
export TAU2_DATA_DIR=/opt/tau3/data
export AREAL_TAU2_ROOT=/data/areal-tau2-data
export NVIDIA_INFERENCE_API_KEY=non-secret-offline-test
export LOGURU_LEVEL=ERROR

python3 - <<"PY"
import huggingface_hub
import sys
import tau2
import transformers

assert not sys.executable.startswith("/opt/tau3/")
assert tau2.__file__.startswith("/opt/tau3/src/")
assert huggingface_hub.__file__.startswith("/usr/local/lib/python3.12/dist-packages/")
assert transformers.__file__.startswith("/usr/local/lib/python3.12/dist-packages/")
print("Miles/Tau Python dependency precedence: ok")
PY

python3 - <<"PY"
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.src.environments.areal_tau2.runtime import (
    AReaLTau2Session,
    _validate_metadata,
    compute_expected_state,
    create_domain_environment,
    load_task,
)
from experiments.src.environments.tau_bench.continuation import build_continuation
from experiments.src.environments.tau_bench.runtime import TauUserConfig
from experiments.src.protocols.areal_tau2 import AREAL_TAU2_EXPECTED_COUNTS
from tau2.data_model.message import UserMessage
from tau2.data_model.tasks import RewardType
import tau2.user.user_simulator as user_module

root = Path("/data/areal-tau2-data")
data_path = root / "miles-tau2-rl-train.jsonl"
summary = json.loads((root / "miles-tau2-rl-summary.json").read_text())
rows = [json.loads(line) for line in data_path.read_text().splitlines() if line]

assert len(rows) == 1982 == summary["rows"]
assert summary["dataset_repo"] == "inclusionAI/AReaL-tau2-data"
assert summary["dataset_revision"] == "1322eae337f836fe0e19bae14dab1eefc26bc983"
assert summary["verifier"] == "areal_tau2_environment"
assert summary["environment_policy"] == "areal-tau2-user-simulator-terminal-state-rl"
assert summary["tau_package_version"] == "1.0.1"
assert len(summary["db_sha256"]) == 9
assert Counter(row["metadata"]["tau_domain"] for row in rows) == Counter(
    AREAL_TAU2_EXPECTED_COUNTS
)
assert len({row["metadata"]["source_row_id"] for row in rows}) == 1982
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == summary["output_sha256"]

for row in rows:
    metadata = row["metadata"]
    _validate_metadata(metadata)
    task = load_task(metadata["tau_task"])
    assert RewardType.NL_ASSERTION not in task.evaluation_criteria.reward_basis

probe = next(
    row
    for row in rows
    if row["metadata"]["tau_domain"] == "retail"
    and row["metadata"]["tau_task"]["evaluation_criteria"].get("actions")
)
metadata = probe["metadata"]
left = create_domain_environment(
    metadata["tau_domain"],
    dataset_root=root,
    db_relative_path=metadata["tau_db_path"],
    db_sha256=metadata["tau_db_sha256"],
)
right = create_domain_environment(
    metadata["tau_domain"],
    dataset_root=root,
    db_relative_path=metadata["tau_db_path"],
    db_sha256=metadata["tau_db_sha256"],
)
assert left.get_db_hash() == right.get_db_hash()
assert left.tools.db is not right.tools.db

agent_hash, user_hash, errors = compute_expected_state(
    metadata["tau_task"],
    domain=metadata["tau_domain"],
    dataset_root=root,
    db_relative_path=metadata["tau_db_path"],
    db_sha256=metadata["tau_db_sha256"],
)
assert agent_hash == metadata["tau_expected_agent_db_hash"]
assert user_hash == metadata["tau_expected_user_db_hash"]
assert errors == metadata["tau_gold_replay_errors"]

responses = iter(
    (
        "Hello, I need help with the continuation request.",
        "Hello, I need help with the terminal request.",
        "###STOP###",
    )
)
user_generation_calls = []


def fake_generate(*_args, **_kwargs):
    response = next(responses)
    user_generation_calls.append(response)
    return UserMessage(role="user", content=response)


user_module.generate = fake_generate

continuation_metadata = rows[0]["metadata"]
continuation_task = load_task(continuation_metadata["tau_task"])
gold_actions = [
    action
    for action in continuation_task.evaluation_criteria.actions
    if action.requestor == "assistant"
]
assert len(gold_actions) >= 4
assert gold_actions[3].name == "update_reservation_flights"

session = AReaLTau2Session(
    continuation_metadata,
    TauUserConfig(provider="nvidia", model="offline-test"),
    max_steps=20,
)
try:
    session.reset()
    initial_snapshot = session.snapshot()
    for action in gold_actions[:4]:
        step = session.step(
            json.dumps(
                {
                    "id": action.action_id,
                    "name": action.name,
                    "arguments": action.arguments,
                    "requestor": action.requestor,
                },
                separators=(",", ":"),
            )
        )
        assert not step.terminated
        assert not step.truncated
    saved_snapshot = session.snapshot()
    assert saved_snapshot["agent_db_hash"] != initial_snapshot["agent_db_hash"]
finally:
    session.close()

continuation = build_continuation(
    saved_snapshot,
    policy_turns=4,
    turn_start_response_length=0,
    policy_response_prefix="",
    policy_prefix_response_tokens=0,
)
restored = AReaLTau2Session(
    continuation_metadata,
    TauUserConfig(provider="nvidia", model="offline-test"),
    max_steps=20,
    continuation=continuation,
)
try:
    restored.reset()
    replayed_snapshot = restored.snapshot()
    assert replayed_snapshot["message_history_sha256"] == saved_snapshot["message_history_sha256"]
    assert replayed_snapshot["agent_db_hash"] == saved_snapshot["agent_db_hash"]
    assert replayed_snapshot["user_db_hash"] == saved_snapshot["user_db_hash"]
    assert replayed_snapshot["orchestrator_step_count"] == saved_snapshot["orchestrator_step_count"]
    assert replayed_snapshot["orchestrator_num_errors"] == saved_snapshot["orchestrator_num_errors"]
    assert len(user_generation_calls) == 1
finally:
    restored.close()

session = AReaLTau2Session(
    metadata,
    TauUserConfig(provider="nvidia", model="offline-test"),
    max_steps=20,
)
try:
    initial = session.reset()
    assert initial.messages
    assert any(tool["function"]["name"] == "done" for tool in initial.tools)
    step = session.step("I cannot complete the requested changes.")
    assert step.terminated is True
    assert step.truncated is False
    assert step.reward in {0.0, 1.0}
    assert step.reward_info["reward"] == step.reward
    assert step.simulation_run["messages"]
finally:
    session.close()

print(
    "AReaL Tau2: 1,982 tasks validated; mutating event-log continuation, "
    "DB-hash restore, and offline user-simulator terminal lifecycle=ok"
)
PY
'
