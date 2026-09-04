"""Focused schema and leakage tests for executable SWE dataset adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from experiments.src.datasets.common.io import read_rows
from experiments.src.datasets.swe import prepare as prepare_module
from experiments.src.datasets.swe.prepare import prepare
from experiments.src.datasets.swe.schema import normalize_swe_row

_BASE_COMMIT = "1" * 40
_GOLD_COMMIT = "2" * 40


def _install_test_source_lock(
    tmp_path: Path,
    monkeypatch,
    *,
    source: Path,
    dataset_id: str,
    schema: str,
    split: str = "train",
    holdout_ids: list[str] | None = None,
    holdout_rows: list[dict] | None = None,
) -> Path:
    holdout = tmp_path / "holdout.jsonl"
    if holdout_rows is None:
        holdout_rows = [
            {
                "instance_id": instance_id,
                "repo": "holdout/repo",
                "base_commit": "a" * 40,
            }
            for instance_id in (holdout_ids or ["holdout__repo-1"])
        ]
    holdout.write_text(
        "".join(json.dumps(row) + "\n" for row in holdout_rows),
        encoding="utf-8",
    )
    lock = {
        "schema_version": "miles-swe-source-lock-v1",
        "downstream_holdout": {
            "dataset_id": "test/holdout",
            "revision": "a" * 40,
            "rows": len(holdout_rows),
            "file": {
                "name": holdout.name,
                "sha256": hashlib.sha256(holdout.read_bytes()).hexdigest(),
                "size": holdout.stat().st_size,
            },
        },
        "sources": {
            "test-source": {
                "dataset_id": dataset_id,
                "revision": "b" * 40,
                "usage": "train",
                "schemas": [schema],
                "row_source_datasets": [dataset_id],
                "splits": [split],
                "artifacts": [
                    {
                        "files": [
                            {
                                "name": source.name,
                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                "size": source.stat().st_size,
                            }
                        ]
                    }
                ],
            }
        },
    }
    lock_path = tmp_path / "swe-source-lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(prepare_module, "_SOURCE_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        prepare_module,
        "_SOURCE_LOCK_SHA256",
        hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    )
    return holdout


def _r2e_row() -> dict:
    return {
        "repo_name": "orange3",
        "docker_image": f"example/orange3:{_GOLD_COMMIT}",
        "commit_hash": _GOLD_COMMIT,
        "problem_statement": "Fix the widget migration bug.",
        "expected_output_json": json.dumps(
            {"TestWidget.test_migration": "PASSED", "TestWidget.test_fast": "PASSED"}
        ),
    }


def _swe_gym_row() -> dict:
    return {
        "instance_id": "pandas-dev__pandas-12345",
        "repo": "pandas-dev/pandas",
        "base_commit": _BASE_COMMIT,
        "problem_statement": "Correct the dataframe behavior.",
        "patch": "gold must not enter Miles rows",
        "test_patch": "diff --git a/test.py b/test.py\n",
        "FAIL_TO_PASS": ["pandas/tests/test_frame.py::test_fix"],
        "PASS_TO_PASS": ["pandas/tests/test_frame.py::test_existing"],
        "split": "train",
    }


def _swe_rebench_row() -> dict:
    return {
        "instance_id": "python-markdown__markdown-1529",
        "repo": "Python-Markdown/markdown",
        "base_commit": _BASE_COMMIT,
        "problem_statement": "Fix list parsing.",
        "patch": "gold must not enter Miles rows",
        "test_patch": "diff --git a/tests/test_list.py b/tests/test_list.py\n",
        "FAIL_TO_PASS": ["tests/test_list.py::test_nested"],
        "PASS_TO_PASS": ["tests/test_core.py::test_basic"],
        "image_name": "docker.io/swerebenchv2/python-markdown-markdown:1529-base",
        "language": "Python",
        "install_config": {
            "test_cmd": "pytest -q",
            "log_parser": "pytest",
            "install": [],
        },
    }


def _nemotron_wrapper(instance: dict, dataset_name: str) -> dict:
    return {
        "agent_ref": {"type": "responses_api_agents", "name": "swe_agents_train"},
        "responses_create_params": {
            "input": [],
            "metadata": {
                "dataset_name": dataset_name,
                "instance_id": instance.get("instance_id"),
                "instance_dict": json.dumps(instance),
            },
        },
    }


def test_r2e_normalization_uses_opaque_routing_id_and_exact_map() -> None:
    task = normalize_swe_row(
        _r2e_row(),
        dataset_id="R2E-Gym/R2E-Gym-V1",
        usage="train",
    )

    assert task.instance_id.startswith("r2e-")
    assert _GOLD_COMMIT not in task.instance_id
    assert task.source_schema == "r2e-gym-v1"
    assert task.verifier["kind"] == "r2e-expected-pytest-map-v1"
    assert task.verifier["gold_commit"] == _GOLD_COMMIT
    rendered_training_row = json.dumps(task.to_miles_row(), sort_keys=True)
    assert _GOLD_COMMIT not in rendered_training_row
    assert "TestWidget.test_migration" not in rendered_training_row


def test_r2e_public_ids_and_bindings_are_random_not_gold_derived() -> None:
    first = normalize_swe_row(
        _r2e_row(),
        dataset_id="R2E-Gym/R2E-Gym-V1",
        usage="train",
    )
    second = normalize_swe_row(
        _r2e_row(),
        dataset_id="R2E-Gym/R2E-Gym-V1",
        usage="train",
    )

    assert first.instance_id != second.instance_id
    assert first.task_binding != second.task_binding
    assert _GOLD_COMMIT not in first.instance_id
    assert first.to_task_manifest()["content_digest"] not in json.dumps(
        first.to_miles_row(),
        sort_keys=True,
    )


def test_nemotron_wrapper_keeps_published_id_only_in_private_metadata() -> None:
    row = _r2e_row() | {
        "instance_id": f"numpy__numpy-{_GOLD_COMMIT}",
        "repo": "numpy/numpy",
        "base_commit": _BASE_COMMIT,
    }
    task = normalize_swe_row(
        _nemotron_wrapper(row, "R2E-Gym/R2E-Gym-Subset"),
        usage="train",
    )

    assert task.instance_id.startswith("r2e-")
    assert _GOLD_COMMIT not in task.instance_id
    assert task.repo == "numpy/numpy"
    assert task.base_commit == _BASE_COMMIT
    assert task.source_metadata["published_instance_id"] == (
        f"numpy__numpy-{_GOLD_COMMIT}"
    )
    assert _GOLD_COMMIT not in json.dumps(task.to_miles_row(), sort_keys=True)


def test_nemotron_pivot_action_is_not_mislabeled_as_full_swe() -> None:
    pivot = {
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
        },
        "responses_create_params": {"input": [], "metadata": {}},
        "expected_action": {"type": "function_call", "name": "shell_command"},
    }

    with pytest.raises(ValueError, match="not a full environment task"):
        normalize_swe_row(pivot, usage="train")


def test_swe_gym_uses_published_image_convention_and_private_verifier() -> None:
    task = normalize_swe_row(
        _swe_gym_row(),
        dataset_id="SWE-Gym/SWE-Gym",
        usage="train",
    )
    miles_row = task.to_miles_row()
    manifest = task.to_task_manifest()

    assert task.source_image == (
        "docker.io/xingyaoww/sweb.eval.x86_64."
        "pandas-dev_s_pandas-12345:latest"
    )
    assert manifest["verifier"]["fail_to_pass"] == [
        "pandas/tests/test_frame.py::test_fix"
    ]
    rendered_training_row = json.dumps(miles_row, sort_keys=True)
    assert "gold must not enter Miles rows" not in rendered_training_row
    assert "test_frame.py::test_fix" not in rendered_training_row
    assert task.source_image not in rendered_training_row
    assert manifest["task_digest"] == miles_row["metadata"]["swe_task"]["task_digest"]


def test_swe_rebench_v2_preserves_image_install_and_test_semantics() -> None:
    task = normalize_swe_row(
        _swe_rebench_row(),
        dataset_id="PrimeIntellect/SWE-rebench-V2-Filtered-Verified",
        usage="train",
    )

    assert task.source_schema == "swe-rebench-v2"
    assert task.verifier["kind"] == "swe-rebench-v2"
    assert task.verifier["install_config"]["test_cmd"] == "pytest -q"
    assert task.source_image.startswith("docker.io/swerebenchv2/")


def test_prime_filtered_rebench_restores_published_upstream_image() -> None:
    row = _swe_rebench_row()
    row["image_name"] = "prime/primeintellect/python-markdown-markdown:1529-base"

    task = normalize_swe_row(
        row,
        dataset_id="PrimeIntellect/SWE-rebench-V2-Filtered-Verified",
        usage="train",
    )

    assert task.source_image == (
        "docker.io/swerebenchv2/python-markdown-markdown:1529-base"
    )
    assert task.source_metadata["published_source_image"] == row["image_name"]
    assert task.source_metadata["image_reference_transform"] == (
        "prime-filtered-to-upstream-dockerhub-v1"
    )


def test_prime_internal_rebench_alias_is_rejected_for_other_sources() -> None:
    row = _swe_rebench_row()
    row["image_name"] = "prime/primeintellect/python-markdown-markdown:1529-base"

    with pytest.raises(ValueError, match="accepted only for the pinned"):
        normalize_swe_row(
            row,
            dataset_id="nebius/SWE-rebench-V2",
            usage="train",
        )


def test_swe_rebench_v2_accepts_sequential_test_commands() -> None:
    row = _swe_rebench_row()
    row["install_config"] = row["install_config"] | {
        "test_cmd": ["pytest tests/unit -q", "pytest tests/integration -q"]
    }

    task = normalize_swe_row(
        row,
        dataset_id="nebius/SWE-rebench-V2",
        usage="train",
    )

    assert task.verifier["install_config"]["test_cmd"] == [
        "pytest tests/unit -q",
        "pytest tests/integration -q",
    ]


def test_swebench_verified_is_eval_only() -> None:
    row = _swe_gym_row()
    with pytest.raises(ValueError, match="refusing to emit training rows"):
        normalize_swe_row(
            row,
            dataset_id="princeton-nlp/SWE-bench_Verified",
            usage="train",
        )

    task = normalize_swe_row(
        row,
        dataset_id="princeton-nlp/SWE-bench_Verified",
        usage="eval",
    )
    assert task.eval_only is True


def test_prepare_deduplicates_private_manifests_but_preserves_train_sampling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(json.dumps(_swe_gym_row()) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"
    manifest = tmp_path / "tasks.jsonl"
    holdout = _install_test_source_lock(
        tmp_path,
        monkeypatch,
        source=source,
        dataset_id="SWE-Gym/SWE-Gym",
        schema="swe-gym",
    )
    args = argparse.Namespace(
        input=[str(source)],
        source_name="test-source",
        output=output,
        task_manifest=manifest,
        downstream_holdout=holdout,
        dataset_id="SWE-Gym/SWE-Gym",
        usage="train",
        agent_name="mini-swe-agent",
        include_schema=None,
        on_invalid="error",
        limit=None,
    )

    summary = prepare(args)

    assert summary["rows"] == 2
    assert summary["unique_tasks"] == 1
    assert summary["duplicate_rows"] == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1
    assert manifest.stat().st_mode & 0o077 == 0
    assert summary["artifact_stage"] == "schema-normalized"
    assert summary["environment_admitted"] is False


def test_prepare_quarantines_invalid_rows_with_owner_only_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "input.jsonl"
    invalid = _swe_gym_row()
    invalid["instance_id"] = "pandas-dev__pandas-54321"
    invalid["problem_statement"] = ""
    source.write_text(
        json.dumps(_swe_gym_row()) + "\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )
    holdout = _install_test_source_lock(
        tmp_path,
        monkeypatch,
        source=source,
        dataset_id="SWE-Gym/SWE-Gym",
        schema="swe-gym",
    )
    invalid_report = tmp_path / "invalid.private.jsonl"
    args = argparse.Namespace(
        input=[str(source)],
        source_name="test-source",
        output=tmp_path / "candidate.jsonl",
        task_manifest=tmp_path / "tasks.private.jsonl",
        downstream_holdout=holdout,
        r2e_id_map=None,
        invalid_report=invalid_report,
        dataset_id="SWE-Gym/SWE-Gym",
        usage="train",
        agent_name="terminus-2",
        include_schema=["swe-gym"],
        on_invalid="quarantine",
        limit=None,
    )

    summary = prepare(args)

    report = json.loads(invalid_report.read_text(encoding="utf-8"))
    assert summary["rows"] == 1
    assert summary["invalid_rows"] == 1
    assert summary["invalid_reasons"] == {"missing-required-field": 1}
    assert summary["invalid_report_sha256"] == hashlib.sha256(
        invalid_report.read_bytes()
    ).hexdigest()
    assert invalid_report.stat().st_mode & 0o077 == 0
    assert report["schema_version"] == "miles-swe-invalid-row-v1"
    assert report["source_index"] == 1
    assert len(report["source_locator_sha256"]) == 64
    assert report["reason"] == "missing-required-field"
    assert "gold must not enter Miles rows" not in json.dumps(report, sort_keys=True)


def test_prepare_refuses_silent_invalid_row_discard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(_swe_gym_row()) + "\n", encoding="utf-8")
    holdout = _install_test_source_lock(
        tmp_path,
        monkeypatch,
        source=source,
        dataset_id="SWE-Gym/SWE-Gym",
        schema="swe-gym",
    )
    args = argparse.Namespace(
        input=[str(source)],
        source_name="test-source",
        output=tmp_path / "candidate.jsonl",
        task_manifest=tmp_path / "tasks.private.jsonl",
        downstream_holdout=holdout,
        r2e_id_map=None,
        invalid_report=None,
        dataset_id="SWE-Gym/SWE-Gym",
        usage="train",
        agent_name="terminus-2",
        include_schema=["swe-gym"],
        on_invalid="quarantine",
        limit=None,
    )

    with pytest.raises(ValueError, match="requires an owner-only --invalid-report"):
        prepare(args)


def test_prepare_persists_random_r2e_route_and_binding_across_reruns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "r2e.jsonl"
    source.write_text(json.dumps(_r2e_row()) + "\n", encoding="utf-8")
    output = tmp_path / "train.jsonl"
    manifest = tmp_path / "tasks.private.jsonl"
    id_map = tmp_path / "r2e-ids.private.json"
    holdout = _install_test_source_lock(
        tmp_path,
        monkeypatch,
        source=source,
        dataset_id="R2E-Gym/R2E-Gym-V1",
        schema="r2e-gym-v1",
    )
    args = argparse.Namespace(
        input=[str(source)],
        source_name="test-source",
        output=output,
        task_manifest=manifest,
        downstream_holdout=holdout,
        r2e_id_map=id_map,
        dataset_id="R2E-Gym/R2E-Gym-V1",
        usage="train",
        agent_name="mini-swe-agent",
        include_schema=None,
        on_invalid="error",
        limit=None,
    )

    prepare(args)
    first_row = json.loads(output.read_text(encoding="utf-8"))
    prepare(args)
    second_row = json.loads(output.read_text(encoding="utf-8"))

    assert first_row["metadata"]["instance_id"] == second_row["metadata"]["instance_id"]
    assert first_row["metadata"]["swe_task"]["task_digest"] == (
        second_row["metadata"]["swe_task"]["task_digest"]
    )
    assert id_map.stat().st_mode & 0o077 == 0
    rendered_map = id_map.read_text(encoding="utf-8")
    assert _GOLD_COMMIT not in first_row["metadata"]["instance_id"]
    assert _GOLD_COMMIT not in json.dumps(first_row, sort_keys=True)
    assert "miles-r2e-id-map-v3" in rendered_map
    assert "source_content_digest" in rendered_map


def test_prepare_rejects_changed_r2e_contents_for_same_gold_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "r2e.jsonl"
    changed = _r2e_row() | {
        "problem_statement": "A conflicting problem for the same gold commit.",
        "expected_output_json": json.dumps({"different_test": "PASSED"}),
    }
    source.write_text(
        json.dumps(_r2e_row()) + "\n" + json.dumps(changed) + "\n",
        encoding="utf-8",
    )
    holdout = _install_test_source_lock(
        tmp_path,
        monkeypatch,
        source=source,
        dataset_id="R2E-Gym/R2E-Gym-V1",
        schema="r2e-gym-v1",
    )
    args = argparse.Namespace(
        input=[str(source)],
        source_name="test-source",
        output=tmp_path / "train.jsonl",
        task_manifest=tmp_path / "tasks.private.jsonl",
        downstream_holdout=holdout,
        r2e_id_map=tmp_path / "r2e-ids.private.json",
        dataset_id="R2E-Gym/R2E-Gym-V1",
        usage="train",
        agent_name="terminus-2",
        include_schema=None,
        on_invalid="error",
        limit=None,
    )
    with pytest.raises(ValueError, match="conflicting task contents"):
        prepare(args)


def test_prepare_rejects_dataset_relabeling_and_forbidden_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "swe-gym.jsonl"
    source.write_text(json.dumps(_swe_gym_row() | {"split": "test"}) + "\n", encoding="utf-8")
    holdout = _install_test_source_lock(
        tmp_path,
        monkeypatch,
        source=source,
        dataset_id="SWE-Gym/SWE-Gym",
        schema="swe-gym",
    )
    args = argparse.Namespace(
        input=[str(source)],
        source_name="test-source",
        output=tmp_path / "train.jsonl",
        task_manifest=tmp_path / "tasks.private.jsonl",
        downstream_holdout=holdout,
        r2e_id_map=None,
        dataset_id="princeton-nlp/SWE-bench_Verified",
        usage="train",
        agent_name="terminus-2",
        include_schema=["swe-gym"],
        on_invalid="error",
        limit=None,
    )

    with pytest.raises(ValueError, match="--dataset-id must be"):
        prepare(args)

    args.dataset_id = "SWE-Gym/SWE-Gym"
    with pytest.raises(ValueError, match="forbidden split"):
        prepare(args)


def test_prepare_excludes_downstream_holdout_overlap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "swe-gym.jsonl"
    source.write_text(json.dumps(_swe_gym_row()) + "\n", encoding="utf-8")
    holdout = _install_test_source_lock(
        tmp_path,
        monkeypatch,
        source=source,
        dataset_id="SWE-Gym/SWE-Gym",
        schema="swe-gym",
        holdout_ids=[_swe_gym_row()["instance_id"]],
    )
    args = argparse.Namespace(
        input=[str(source)],
        source_name="test-source",
        output=tmp_path / "train.jsonl",
        task_manifest=tmp_path / "tasks.private.jsonl",
        downstream_holdout=holdout,
        r2e_id_map=None,
        dataset_id="SWE-Gym/SWE-Gym",
        usage="train",
        agent_name="terminus-2",
        include_schema=["swe-gym"],
        on_invalid="error",
        limit=None,
    )

    summary = prepare(args)

    assert summary["rows"] == 0
    assert summary["downstream_holdout_excluded_rows"] == 1
    assert not args.output.read_text(encoding="utf-8")
    assert not args.task_manifest.read_text(encoding="utf-8")


def test_prepare_excludes_same_repo_base_under_a_different_instance_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "swe-gym.jsonl"
    source.write_text(json.dumps(_swe_gym_row()) + "\n", encoding="utf-8")
    holdout = _install_test_source_lock(
        tmp_path,
        monkeypatch,
        source=source,
        dataset_id="SWE-Gym/SWE-Gym",
        schema="swe-gym",
        holdout_rows=[
            {
                "instance_id": "different__published-id-999",
                "repo": "https://github.com/PANDAS-DEV/pandas.git",
                "base_commit": _BASE_COMMIT.upper(),
            }
        ],
    )
    args = argparse.Namespace(
        input=[str(source)],
        source_name="test-source",
        output=tmp_path / "train.jsonl",
        task_manifest=tmp_path / "tasks.private.jsonl",
        downstream_holdout=holdout,
        r2e_id_map=None,
        dataset_id="SWE-Gym/SWE-Gym",
        usage="train",
        agent_name="terminus-2",
        include_schema=["swe-gym"],
        on_invalid="error",
        limit=None,
    )

    summary = prepare(args)

    assert summary["rows"] == 0
    assert summary["downstream_holdout_excluded_rows"] == 1


def test_invalid_r2e_expected_map_fails_closed() -> None:
    row = _r2e_row() | {"expected_output_json": "not json"}
    with pytest.raises(ValueError, match="invalid JSON"):
        normalize_swe_row(row, dataset_id="R2E-Gym/R2E-Gym-V1")


def test_hugging_face_arrow_stream_is_read_without_materializing_table(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "train.arrow"
    table = pyarrow.Table.from_pylist([_swe_rebench_row()])
    with pyarrow.OSFile(str(path), "wb") as sink:
        with pyarrow.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    assert list(read_rows([path])) == [_swe_rebench_row()]
