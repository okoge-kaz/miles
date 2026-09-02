"""Result-contract, reward, and aggregation tests for Harbor SWE tasks."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.evaluators.swe import (
    _evaluation_is_valid,
    _harbor_agent_model,
    _load_tasks,
    _require_server_ready,
    _required_process_secret,
    _run_trial,
    _safe_service_url,
)
from experiments.src.environments.swe.result import (
    HarborSWEOutcome,
    SWEEvaluationTrial,
    summarize_trials,
)
from experiments.src.reward_sets.swe import reward
from miles.rollout.harbor.auth import (
    derive_harbor_health_bearer,
    derive_harbor_run_bearer,
)


def _trial(instance_id: str, reward_value: float | None) -> SWEEvaluationTrial:
    return SWEEvaluationTrial(
        instance_id=instance_id,
        trial_index=0,
        status_code=200 if reward_value is not None else -1,
        elapsed_seconds=1.0,
        reward=reward_value,
        exit_status="Submitted" if reward_value is not None else "ClientError",
        eval_report={"resolved": reward_value == 1.0, "reward": reward_value}
        if reward_value is not None
        else {},
        agent_metrics={},
        error=None if reward_value is not None else "network unavailable",
    )


def test_harbor_outcome_accepts_binary_verifier_result() -> None:
    outcome = HarborSWEOutcome.from_mapping(
        {
            "reward": 1,
            "exit_status": "Submitted",
            "eval_report": {"resolved": True, "reward": 1},
            "agent_metrics": {"turns": 12},
        }
    )

    assert outcome.reward == 1.0
    assert outcome.eval_report == {"resolved": True, "reward": 1}


def test_harbor_outcome_drops_private_reports_and_untrusted_agent_metadata() -> None:
    outcome = HarborSWEOutcome.from_mapping(
        {
            "reward": 0,
            "exit_status": "Submitted",
            "eval_report": {
                "reward": 0,
                "expected_count": 4,
                "expected_test_ids": ["private::test"],
            },
            "agent_metrics": {
                "turns": 3,
                "total_tool_time": 1.5,
                "private_agent_note": "model-controlled",
                "eval_time": float("nan"),
            },
        }
    )

    assert outcome.eval_report == {
        "reward": 0.0,
        "resolved": False,
        "expected_count": 4,
    }
    assert outcome.agent_metrics == {"turns": 3, "total_tool_time": 1.5}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"reward": None, "eval_report": {}},
        {"reward": float("nan"), "eval_report": {}},
        {"reward": -1, "eval_report": {}},
        {
            "reward": 0.75,
            "exit_status": "Submitted",
            "eval_report": {"reward": 0.75},
        },
        {"reward": 1, "exit_status": "Submitted", "eval_report": None},
        {"reward": 0, "exit_status": "Submitted", "eval_report": {}},
        {
            "reward": 1,
            "exit_status": "Submitted",
            "eval_report": {"reward": 0},
        },
    ],
)
def test_harbor_outcome_rejects_ungraded_or_invalid_results(payload: dict) -> None:
    with pytest.raises(ValueError):
        HarborSWEOutcome.from_mapping(payload)


def test_swe_reward_is_restricted_and_uses_validated_harbor_reward() -> None:
    sample = SimpleNamespace(
        metadata={
            "verifier": "swe_environment",
            "reward": 1,
            "exit_status": "Submitted",
            "eval_report": {"resolved": True, "reward": 1},
            "agent_metrics": {},
        }
    )

    assert asyncio.run(reward(None, sample)) == 1.0


def test_swe_reward_rejects_missing_environment_result() -> None:
    sample = SimpleNamespace(metadata={"verifier": "swe_environment"})

    with pytest.raises(ValueError, match="ungraded"):
        asyncio.run(reward(None, sample))


def test_swe_summary_excludes_infrastructure_failures_from_reward_denominator() -> None:
    summary = summarize_trials(
        [
            _trial("task-a", 1.0),
            _trial("task-a", 0.0),
            _trial("task-b", None),
        ],
        task_count=2,
    )

    assert summary["graded_trials"] == 2
    assert summary["infrastructure_failures"] == 1
    assert summary["exact_success_rate"] == 0.5
    assert summary["task_any_success_rate"] == 0.5
    assert not _evaluation_is_valid(
        summary,
        maximum_infrastructure_failures=0,
    )
    assert _evaluation_is_valid(
        summary,
        maximum_infrastructure_failures=1,
    )
    assert not _evaluation_is_valid(
        {**summary, "graded_trials": 0, "infrastructure_failures": 0},
        maximum_infrastructure_failures=0,
    )


def test_swe_evaluator_requires_authenticated_ready_server() -> None:
    task_binding = ("a" * 64, "b" * 64, "c" * 64, 1)

    class Response:
        status = 200

        async def json(self) -> dict:
            return {
                "status": "ok",
                "ready": True,
                "task_set_sha256": task_binding[0],
                "task_binding_sha256": task_binding[1],
                "task_runtime_sha256": task_binding[2],
                "task_count": task_binding[3],
            }

    class ResponseContext:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *_args):
            return None

    class Session:
        request: dict | None = None

        def get(self, url: str, *, headers: dict):
            self.request = {"url": url, "headers": headers}
            return ResponseContext()

    session = Session()
    asyncio.run(
        _require_server_ready(
            "http://127.0.0.1:11000",
            session=session,
            run_master_secret="r" * 32,
            timeout_seconds=1,
            expected_task_binding=task_binding,
        )
    )

    assert session.request == {
        "url": "http://127.0.0.1:11000/health",
        "headers": {
            "Authorization": "Bearer "
            + derive_harbor_health_bearer("r" * 32)
        },
    }


def test_swe_evaluator_rejects_credentials_in_service_urls() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        _safe_service_url("https://token@example.invalid/v1", name="--base-url")

    assert _safe_service_url("http://127.0.0.1:30000/v1/", name="--base-url") == (
        "http://127.0.0.1:30000/v1"
    )


def test_swe_evaluator_uses_same_litellm_provider_contract_as_training() -> None:
    assert _harbor_agent_model("model") == "openai/model"
    assert _harbor_agent_model("Qwen/Qwen3-4B") == "openai/Qwen/Qwen3-4B"
    assert _harbor_agent_model("openai/model") == "openai/model"

    for invalid in ("", " model", "model ", "model name", "model\nname"):
        with pytest.raises(ValueError, match="--model"):
            _harbor_agent_model(invalid)


def test_swe_evaluator_requires_process_environment_run_secret(monkeypatch) -> None:
    monkeypatch.delenv("HARBOR_RUN_SECRET", raising=False)
    with pytest.raises(ValueError, match="HARBOR_RUN_SECRET"):
        _required_process_secret("HARBOR_RUN_SECRET")

    monkeypatch.setenv("HARBOR_RUN_SECRET", "r" * 32)
    assert _required_process_secret("HARBOR_RUN_SECRET") == "r" * 32


def test_swe_evaluator_sends_only_task_scoped_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 200

        async def text(self) -> str:
            return json.dumps(
                {
                    "reward": 1,
                    "exit_status": "Submitted",
                    "eval_report": {"reward": 1},
                    "agent_metrics": {},
                }
            )

    class ResponseContext:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *_args):
            return None

    class Session:
        def __init__(self) -> None:
            self.request: dict | None = None

        def post(self, url: str, *, json: dict, headers: dict):
            self.request = {"url": url, "json": json, "headers": headers}
            return ResponseContext()

    session = Session()
    monkeypatch.setattr(
        "experiments.src.evaluators.swe.secrets.token_hex",
        lambda _size: "1" * 32,
    )
    args = SimpleNamespace(
        server_url="http://127.0.0.1:11000",
        base_url="http://127.0.0.1:30000/v1",
        model="model",
        agent_name="terminus-2",
        temperature=0.0,
        top_p=1.0,
        max_response_tokens=16384,
        max_sequence_length=32768,
    )
    result = asyncio.run(
        _run_trial(
            args,
            session,
            asyncio.Semaphore(1),
            instance_id="task-1",
            task_digest="d" * 64,
            trial_index=0,
            run_master_secret="r" * 32,
            client_id="eval-client",
        )
    )

    assert result.reward == 1.0
    assert session.request is not None
    expected_bearer = derive_harbor_run_bearer(
        "r" * 32,
        instance_id="task-1",
        client_id="eval-client",
        request_id="1" * 32,
    )
    assert session.request["headers"] == {
        "Authorization": f"Bearer {expected_bearer}"
    }
    assert session.request["json"]["model"] == "openai/model"
    assert "r" * 32 not in session.request["headers"].values()
    assert "r" * 32 not in session.request["json"].values()


def test_swe_evaluator_requires_and_preserves_task_digest(tmp_path) -> None:
    data = tmp_path / "eval.jsonl"
    digest = "a" * 64
    data.write_text(
        json.dumps(
            {
                "prompt": "fix it",
                "metadata": {
                    "instance_id": "task-a",
                    "swe_task": {"task_digest": digest},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _load_tasks(data, limit=None) == [("task-a", digest)]


def test_swe_evaluator_rejects_missing_task_digest(tmp_path) -> None:
    data = tmp_path / "eval.jsonl"
    data.write_text(
        json.dumps({"metadata": {"instance_id": "task-a", "swe_task": {}}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="task_digest"):
        _load_tasks(data, limit=None)


def test_swe_evaluator_rejects_conflicting_duplicate_digest(tmp_path) -> None:
    data = tmp_path / "eval.jsonl"
    rows = [
        {
            "metadata": {
                "instance_id": "task-a",
                "swe_task": {"task_digest": character * 64},
            }
        }
        for character in ("a", "b")
    ]
    data.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting task digests"):
        _load_tasks(data, limit=None)


def test_swe_pbs_exports_only_fixed_required_environment_names() -> None:
    repository = Path(__file__).parents[3]
    recipe = (
        repository
        / "experiments/scripts/swe/async/swe-rebench-v2-swe-gym/qwen3-4b/run.sbatch"
    )
    train = recipe.with_name("train.sh")
    evaluation = repository / "experiments/scripts/swe/eval/swebench-verified/run.sbatch"
    recipe_text = recipe.read_text(encoding="utf-8")
    train_text = train.read_text(encoding="utf-8")
    evaluation_text = evaluation.read_text(encoding="utf-8")

    assert "--export=ALL" not in recipe_text
    assert "--export=ALL" not in evaluation_text
    training_names = re.search(
        r"TRAINING_EXPORT_NAMES=\((.*?)\n\)",
        recipe_text,
        flags=re.DOTALL,
    )
    assert training_names is not None
    exported_names = set(training_names.group(1).split())
    assert "HARBOR_RUN_SECRET" in exported_names
    assert "HARBOR_ADMIN_SECRET" not in exported_names
    assert "RAY_DASHBOARD_HOST" in exported_names
    referenced_names = set(
        re.findall(
            r"\$\{([A-Z][A-Z0-9_]*)",
            train_text
            + (repository / "experiments/common/ray_cluster.sh").read_text(
                encoding="utf-8"
            ),
        )
    )
    locally_assigned = set(
        re.findall(
            r"(?m)^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=",
            train_text
            + (repository / "experiments/common/ray_cluster.sh").read_text(
                encoding="utf-8"
            ),
        )
    )
    scheduler_or_optional = {
        "BASH_SOURCE",
        "EPOCHREALTIME",
        "MODEL_ARGS",
        "RAY_TEMP_DIR",
        "MILES_JOB_NUM_NODES",
        "MILES_NODE_RANK",
        "WANDB_PROJECT",
    }
    externally_required = referenced_names - locally_assigned - scheduler_or_optional
    assert externally_required <= exported_names
    runtime_env = train_text.split('RUNTIME_ENV_JSON="', 1)[1].split(")\n\n", 1)[0]
    assert "HARBOR_RUN_SECRET" not in runtime_env
    assert "HARBOR_ADMIN_SECRET" not in runtime_env
    assert '"HARBOR_CLIENT_ID"' in runtime_env
    assert "set -x" not in train_text
    assert "set -eux" not in train_text
    assert "--export=HARBOR_CLIENT_ID,HARBOR_RUN_SECRET," in evaluation_text


def test_swe_rebench_selector_does_not_silently_alias_full_to_filtered() -> None:
    recipe = (
        Path(__file__).parents[3]
        / "experiments/scripts/swe/async/swe-rebench-v2-swe-gym/qwen3-4b/run.sbatch"
    ).read_text(encoding="utf-8")

    assert ': "${SWE_DATASET:=swe-rebench-v2}"' in recipe
    assert "Qwen3-4B-Instruct-2507" not in recipe
    assert "MODEL_NAME=Qwen3-4B-Base-LR2e-5-Step4000" in recipe
    assert re.search(
        r"swe-rebench-v2\)\s+DATASET_TAG=swe-rebench-v2\s+;;",
        recipe,
    )
    assert re.search(
        r"swe-rebench-v2-filtered-verified\)\s+DATASET_TAG=swe-rebench-v2-filtered-verified\s+;;",
        recipe,
    )
