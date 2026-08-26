from __future__ import annotations

from experiments.src.datasets.workplace.prepare import adapt_workplace
import pytest

from experiments.src.environments.workplace.runtime import (
    DEFAULT_RESOURCE_ROOT,
    create_tool_environment,
    execute_action,
)
from experiments.src.environments.workplace.verifier import score_action_trajectory


def test_workplace_adapter_preserves_multistep_state_target():
    row = {
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
        "ground_truth": [{"name": "email_delete_email", "arguments": '{"email_id":"00000259"}'}],
    }
    converted = adapt_workplace(row, eval_only=False)
    assert converted["metadata"]["verifier"] == "workplace_environment"
    assert converted["metadata"]["expected_actions"] == row["ground_truth"]
    assert converted["metadata"]["runtime_dependency"] == (
        "pinned-standalone-resource-modules-no-nemo-gym-server"
    )


def test_workplace_local_environment_and_state_match():
    resource_utils = (
        DEFAULT_RESOURCE_ROOT / "resources_servers" / "workplace_assistant" / "utils.py"
    )
    if not resource_utils.is_file():
        with pytest.raises(RuntimeError, match="standalone Workplace resource modules are missing"):
            create_tool_environment()
        return

    environment = create_tool_environment()
    assert len(environment["functions"]) == 27
    result = execute_action(environment, "company_directory_find_email_address", {"name": "carlos"})
    assert any("carlos" in address for address in result)
    expected = [
        {
            "name": "email_reply_email",
            "arguments": {
                "email_id": "00000057",
                "body": "Thanks for the update - I will get back to you tomorrow.",
            },
        }
    ]
    assert score_action_trajectory(expected, expected) == 1.0
    assert score_action_trajectory([], expected) == 0.0
