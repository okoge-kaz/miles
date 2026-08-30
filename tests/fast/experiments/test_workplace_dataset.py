from __future__ import annotations

import json
import subprocess
import sys

from experiments.src.datasets.workplace.prepare import adapt_workplace
from experiments.src.environments.workplace.runtime import WORKPLACE_RESOURCE_COMMIT


def _row() -> dict:
    return {
        "id": 7,
        "category": "workplace_assistant_email",
        "responses_create_params": {
            "input": [{"role": "user", "content": "Delete the email."}],
            "tools": [
                {
                    "type": "function",
                    "name": "email_delete_email",
                    "parameters": {"type": "object"},
                }
            ],
        },
        "ground_truth": [
            {"name": "email_delete_email", "arguments": '{"email_id":"00000259"}'}
        ],
    }


def test_workplace_conversion_is_deterministic_and_preserves_state_target():
    row = _row()
    first = adapt_workplace(row, eval_only=False)
    second = adapt_workplace(row, eval_only=False)

    assert first == second
    assert first["metadata"]["verifier"] == "workplace_environment"
    assert first["metadata"]["interaction_mode"] == "single_turn_multi_step_environment"
    assert first["metadata"]["conversation_turns"] == 1
    assert first["metadata"]["stateful_environment"] is True
    assert first["metadata"]["expected_actions"] == row["ground_truth"]
    assert first["metadata"]["workplace_resource_commit"] == WORKPLACE_RESOURCE_COMMIT
    assert first["metadata"]["runtime_dependency"] == (
        "pinned-standalone-resource-modules-no-nemo-gym-server"
    )
    assert json.loads(first["label"]) == row["ground_truth"]
    assert first["tools"][0]["function"]["name"] == "email_delete_email"


def test_workplace_validation_rows_are_marked_eval_only():
    converted = adapt_workplace(_row(), eval_only=True)
    assert converted["metadata"]["eval_only"] is True


def test_workplace_dataset_import_does_not_load_rl_runtime():
    program = """
import sys
import experiments.src.datasets.workplace.prepare
assert not any(name == 'miles' or name.startswith('miles.') for name in sys.modules)
assert 'experiments.src.environments.workplace.generator' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True)
