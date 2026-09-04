from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.src.datasets.tau_bench.prepare import (
    EXPECTED_COUNTS,
    NON_EVAL_OUTPUTS,
    _write_jsonl,
    load_tau_rows,
    prepare,
    validate_split_contract,
)
from experiments.src.environments.tau_bench.task_identity import (
    TAU_COMMIT,
    TAU_RELEASE,
    TAU_VERIFIER,
    task_digest,
    task_dict,
    validate_task_identity,
)


class FakeTask:
    def __init__(self, task_id: str) -> None:
        self.id = task_id

    def model_dump(self, *, mode: str) -> dict[str, str]:
        assert mode == "json"
        return {"id": self.id, "scenario": f"scenario-{self.id}"}


def _row(domain: str, split: str, task_id: str) -> dict:
    return {"metadata": {"tau_domain": domain, "tau_split": split, "tau_task_id": task_id}}


def test_tau_dataset_import_does_not_load_miles_or_rollout_generator() -> None:
    program = """
import sys
import experiments.src.datasets.tau_bench.prepare
assert not any(name == 'miles' or name.startswith('miles.') for name in sys.modules)
assert 'experiments.src.environments.tau_bench.generator' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_tau_three_release_counts_are_explicit() -> None:
    assert EXPECTED_COUNTS == {
        ("retail", "train"): 74,
        ("retail", "test"): 40,
        ("retail", "base"): 114,
        ("airline", "train"): 30,
        ("airline", "test"): 20,
        ("airline", "base"): 50,
        ("telecom", "train"): 74,
        ("telecom", "test"): 40,
        ("telecom", "base"): 114,
    }


def test_load_tau_rows_records_stable_task_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    tasks = [FakeTask(str(index)) for index in range(40)]
    monkeypatch.setattr("experiments.src.datasets.tau_bench.prepare._load_tasks", lambda *_args: tasks)

    rows = load_tau_rows("retail", "test")

    assert len(rows) == 40
    metadata = rows[3]["metadata"]
    assert metadata == {
        "source": "tau3-retail",
        "verifier": TAU_VERIFIER,
        "tau_domain": "retail",
        "tau_split": "test",
        "tau_task_id": "3",
        "tau_task_sha256": task_digest(task_dict(tasks[3])),
        "tau_release": TAU_RELEASE,
        "tau_commit": TAU_COMMIT,
        "eval_only": True,
        "official_compat_only": False,
    }


def test_tau_task_identity_is_pinned_and_fail_closed() -> None:
    task = FakeTask("3")
    metadata = {
        "verifier": TAU_VERIFIER,
        "tau_release": TAU_RELEASE,
        "tau_commit": TAU_COMMIT,
        "tau_domain": "retail",
        "tau_split": "test",
        "tau_task_id": "3",
        "tau_task_sha256": task_digest(task_dict(task)),
    }
    validate_task_identity(metadata, task)
    with pytest.raises(ValueError, match="digest"):
        validate_task_identity({**metadata, "tau_task_sha256": "wrong"}, task)
    with pytest.raises(ValueError, match="task ID"):
        validate_task_identity({**metadata, "tau_task_id": "4"}, task)


def test_split_contract_requires_disjoint_train_test_and_exact_base_union() -> None:
    rows = {
        ("retail", "train"): [_row("retail", "train", "0")],
        ("retail", "test"): [_row("retail", "test", "1")],
        ("retail", "base"): [_row("retail", "base", "0"), _row("retail", "base", "1")],
        ("airline", "train"): [_row("airline", "train", "0")],
        ("airline", "test"): [_row("airline", "test", "1")],
        ("airline", "base"): [_row("airline", "base", "0"), _row("airline", "base", "1")],
        ("telecom", "train"): [_row("telecom", "train", "0")],
        ("telecom", "test"): [_row("telecom", "test", "1")],
        ("telecom", "base"): [_row("telecom", "base", "0"), _row("telecom", "base", "1")],
    }
    validate_split_contract(rows)
    rows[("retail", "test")] = [_row("retail", "test", "0")]
    with pytest.raises(ValueError, match="train/test overlap"):
        validate_split_contract(rows)


def test_atomic_jsonl_writer_is_deterministic(tmp_path: Path) -> None:
    rows = [{"id": 1}, {"id": 2}]
    output = tmp_path / "rows.jsonl"
    assert _write_jsonl(output, rows) == 2
    first = output.read_bytes()
    assert _write_jsonl(output, rows) == 2
    assert output.read_bytes() == first
    assert [json.loads(line) for line in output.read_text().splitlines()] == rows


def test_prepare_materializes_only_held_out_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_load_tasks(domain: str, split: str) -> list[FakeTask]:
        train_count = EXPECTED_COUNTS[(domain, "train")]
        test_count = EXPECTED_COUNTS[(domain, "test")]
        train = [FakeTask(str(index)) for index in range(train_count)]
        test = [FakeTask(str(train_count + index)) for index in range(test_count)]
        return {"train": train, "test": test, "base": [*train, *test]}[split]

    monkeypatch.setattr("experiments.src.datasets.tau_bench.prepare._load_tasks", fake_load_tasks)
    for name in NON_EVAL_OUTPUTS:
        (tmp_path / name).write_text("stale\n", encoding="utf-8")

    summary = prepare(tmp_path)

    assert summary["counts"] == {
        "tau3-retail-test-miles.jsonl": 40,
        "tau3-airline-test-miles.jsonl": 20,
        "tau3-telecom-test-miles.jsonl": 40,
        "tau3-test-miles.jsonl": 100,
    }
    assert summary["evaluation_policy"]["train_and_base_materialized"] is False
    assert summary["evaluation_policy"]["training_dataset"] == "inclusionAI/AReaL-tau2-data"
    assert summary["evaluation_policy"]["training_split"] == "tau2_rl_train.jsonl"
    assert summary["evaluation_policy"]["training_environment"] == (
        "stateful_multi_turn_user_simulator_environment"
    )
    assert all(not (tmp_path / name).exists() for name in NON_EVAL_OUTPUTS)
