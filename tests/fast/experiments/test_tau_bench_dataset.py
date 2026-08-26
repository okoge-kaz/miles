from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.src.datasets.tau_bench.prepare import (
    partition_reward_verified_rows,
    reservoir_sample,
)
from experiments.src.environments.tau_bench.task_identity import (
    TAU_COMMIT,
    _task_digest,
    _task_dict,
    validate_task_identity,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_tau_dataset_import_does_not_load_miles_or_rollout_generator():
    program = """
import sys
import experiments.src.datasets.tau_bench.prepare
assert not any(name == 'miles' or name.startswith('miles.') for name in sys.modules)
assert 'experiments.src.environments.tau_bench.generator' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_tau_reservoir_sample_is_deterministic_and_rejects_eval_rows(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_rows(
        source,
        [{"prompt": str(index), "metadata": {"verifier": "expert_action"}} for index in range(20)],
    )
    first = reservoir_sample(source, count=5, seed=42)
    second = reservoir_sample(source, count=5, seed=42)
    assert first == second
    assert len({row["prompt"] for row in first}) == 5

    contaminated = tmp_path / "contaminated.jsonl"
    _write_rows(contaminated, [{"prompt": "held out", "metadata": {"eval_only": True}}])
    with pytest.raises(ValueError, match="eval_only"):
        reservoir_sample(contaminated, count=1, seed=42)


def test_tau_task_identity_is_pinned_and_fail_closed():
    task = _FakeTask(3)
    metadata = {
        "tau_commit": TAU_COMMIT,
        "tau_task_index": 3,
        "tau_task_sha256": _task_digest(_task_dict(task)),
    }
    validate_task_identity(metadata, task)
    with pytest.raises(ValueError, match="digest"):
        validate_task_identity({**metadata, "tau_task_sha256": "wrong"}, task)


class _FakeTask:
    def __init__(self, index: int):
        self.index = index
        self.actions = [f"action-{index}"]

    def model_dump(self) -> dict:
        return {"index": self.index, "actions": self.actions}


def test_tau_reward_audit_rejects_no_op_positive_tasks(monkeypatch):
    tasks = [_FakeTask(0), _FakeTask(1)]
    environment = type("FakeEnv", (), {"tasks": tasks})()
    rows = [
        {
            "prompt": str(index),
            "metadata": {
                "tau_commit": TAU_COMMIT,
                "tau_task_index": index,
                "tau_task_sha256": _task_digest(_task_dict(task)),
            },
        }
        for index, task in enumerate(tasks)
    ]

    def fake_reward(unused_environment, task_index, actions):
        if actions:
            return 1.0
        return float(task_index == 1)

    monkeypatch.setattr(
        "experiments.src.datasets.tau_bench.prepare._official_reward",
        fake_reward,
    )
    verified, rejected = partition_reward_verified_rows(rows, environment)

    assert [row["metadata"]["tau_task_index"] for row in verified] == [0]
    assert verified[0]["metadata"]["tau_reward_verified"] is True
    assert rejected == [{"task_index": 1, "no_op_reward": 1.0, "ground_truth_reward": 1.0}]
