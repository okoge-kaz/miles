"""Tests for the SWE candidate-to-admitted dataset promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest

from experiments.src.datasets.swe.schema import normalize_swe_row
from experiments.src.environments.swe import finalize_admitted
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock

finalize = finalize_admitted.finalize

_BASE_COMMIT = "1" * 40
_IMAGE = "docker.io/swerebenchv2/example@sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def _restore_private_directory_modes(tmp_path: Path):
    yield
    for current, directory_names, _ in os.walk(
        tmp_path,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        if not current_path.is_symlink():
            current_path.chmod(0o700)
        for name in directory_names:
            child = current_path / name
            if not child.is_symlink():
                child.chmod(0o700)


def _task(name: str = "example__project-1"):
    return normalize_swe_row(
        {
            "instance_id": name,
            "repo": "example/project",
            "base_commit": _BASE_COMMIT,
            "problem_statement": f"Fix {name} without changing tests.",
            "patch": ("diff --git a/src/example.py b/src/example.py\n--- a/src/example.py\n+++ b/src/example.py\n"),
            "test_patch": ("diff --git a/tests/test_example.py b/tests/test_example.py\n--- a/tests/test_example.py\n+++ b/tests/test_example.py\n"),
            "FAIL_TO_PASS": ["tests/test_example.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_example.py::test_existing"],
            "image_name": "prime/primeintellect/example:latest",
            "install_config": {
                "test_cmd": "pytest -q",
                "log_parser": "parse_log_pytest",
            },
        },
        dataset_id="PrimeIntellect/SWE-rebench-V2-Filtered-Verified",
        usage="train",
    )


def _locked_manifest(task) -> dict:
    value = task.to_task_manifest()
    requested = value["sandbox"]["source_image"]
    input_digest = value["content_digest"]
    value["sandbox"]["source_image"] = _IMAGE
    value["sandbox"]["image_lock"] = {
        "schema_version": "miles-oci-image-lock-v1",
        "source_image_requested": requested,
        "source_image_resolved": _IMAGE,
        "input_content_digest": input_digest,
        "index_digest": None,
        "child_manifest_digest": "sha256:" + "a" * 64,
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    digest_payload = dict(value)
    digest_payload.pop("content_digest")
    digest_payload.pop("task_digest")
    value["content_digest"] = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return value


def _write_private_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_task(root: Path, manifest: dict) -> None:
    task_dir = root / manifest["instance_id"]
    tests_dir = task_dir / "tests"
    environment_dir = task_dir / "environment"
    tests_dir.mkdir(parents=True)
    environment_dir.mkdir()
    task_toml = f'''schema_version = "1.3"

[metadata]
task_digest = "{manifest["task_digest"]}"
source_dataset = "{manifest["source_dataset"]}"
source_schema = "{manifest["source_schema"]}"

[verifier]
environment_mode = "separate"
user = 0

[[verifier.collect]]
required = true

[verifier.environment]
network_mode = "no-network"
docker_image = "{manifest["sandbox"]["source_image"]}"

[agent]
user = 1000

[environment]
network_mode = "no-network"
'''
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")
    (task_dir / "instruction.md").write_text(
        manifest["problem_statement"].rstrip() + "\n\nModify only the repository implementation.\n",
        encoding="utf-8",
    )
    (tests_dir / ".harbor-e2b-late-tests").write_text("late\n", encoding="utf-8")
    (tests_dir / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tests_dir / "model_path_policy.json").write_text("{}\n", encoding="utf-8")
    (environment_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    for path in sorted(task_dir.rglob("*"), reverse=True):
        path.chmod(0o500 if path.is_dir() or path.name == "test.sh" else 0o400)
    task_dir.chmod(0o500)
    root.chmod(0o700)


def _template_evidence() -> dict[str, dict[str, str]]:
    shared = {
        "template_id": "template-shared",
        "build_id": "build-shared",
        "alias_sha256": "2" * 64,
        "template_identity_sha256": "3" * 64,
    }
    agent = {
        "template_id": "template-agent",
        "build_id": "build-agent",
        "alias_sha256": "4" * 64,
        "template_identity_sha256": "5" * 64,
    }
    return {
        "source": {**shared, "sandbox_id": "sandbox-source"},
        "agent": {**agent, "sandbox_id": "sandbox-agent"},
        "empty_verifier": {**shared, "sandbox_id": "sandbox-empty"},
        "oracle_verifier": {**shared, "sandbox_id": "sandbox-oracle"},
    }


def _semantic_admission(manifest: dict, task_dir: Path) -> dict:
    image_lock = manifest["sandbox"]["image_lock"]
    return {
        "schema_version": "miles-swe-rebench-admission-v1",
        "instance_id": manifest["instance_id"],
        "source_schema": "swe-rebench-v2",
        "task_digest": manifest["task_digest"],
        "input_content_digest": image_lock["input_content_digest"],
        "locked_content_digest": manifest["content_digest"],
        "content_digest": manifest["content_digest"],
        "source_image_requested": image_lock["source_image_requested"],
        "source_image_resolved": _IMAGE,
        "source_image": _IMAGE,
        "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
        "base_commit": manifest["base_commit"],
        "base_tree": "6" * 40,
        "oracle_patch_sha256": hashlib.sha256(
            manifest["solution"]["oracle_patch"].encode("utf-8")
        ).hexdigest(),
        "test_patch_sha256": hashlib.sha256(
            manifest["verifier"]["test_patch"].encode("utf-8")
        ).hexdigest(),
        "model_path_policy_sha256": hashlib.sha256(
            (task_dir / "tests" / "model_path_policy.json").read_bytes()
        ).hexdigest(),
        "admitted_task_tree_sha256": materialize_module._task_tree_sha256(task_dir),
        "rebench_commit": materialize_module._REBENCH_COMMIT,
        "log_parsers_sha256": materialize_module._REBENCH_LOG_PARSERS_SHA256,
        "constants_sha256": materialize_module._REBENCH_CONSTANTS_SHA256,
        "eval_sha256": materialize_module._REBENCH_EVAL_SHA256,
        "template_evidence": _template_evidence(),
        "checks": dict(materialize_module._REPOSITORY_ADMISSION_CHECKS),
    }


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "candidate": tmp_path / "candidate.jsonl",
        "manifest": tmp_path / "tasks.private.jsonl",
        "materialization_evidence": tmp_path / "materialization.private.jsonl",
        "semantic_admission_manifest": [tmp_path / "admission.private.jsonl"],
        "tasks_dir": tmp_path / "tasks",
        "output": tmp_path / "admitted" / "train.jsonl",
        "task_ids_output": tmp_path / "admitted" / "train.task-ids.txt",
        "summary": tmp_path / "admitted" / "summary.json",
        "allow_subset": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _fixture(tmp_path: Path, *, duplicate: bool = False):
    task = _task()
    manifest = _locked_manifest(task)
    candidate = task.to_miles_row(agent_name="terminus-2")
    args = _args(tmp_path)
    args.tasks_dir.mkdir(mode=0o700)
    _write_task(args.tasks_dir, manifest)
    task_dir = args.tasks_dir / manifest["instance_id"]
    admission = _semantic_admission(manifest, task_dir)
    _write_private_jsonl(args.semantic_admission_manifest[0], [admission])
    evidence = {
        "schema_version": "miles-swe-materialization-evidence-v1",
        "instance_id": manifest["instance_id"],
        "source_schema": manifest["source_schema"],
        "task_digest": manifest["task_digest"],
        "content_digest": manifest["content_digest"],
        "source_image": manifest["sandbox"]["source_image"],
        "private_manifest_record_sha256": hashlib.sha256(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "task_tree_sha256": materialize_module._task_tree_sha256(task_dir),
        "semantic_admission_record_sha256": hashlib.sha256(
            json.dumps(
                admission,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "checks": {
            "production_materialization": True,
            "immutable_image": True,
            "semantic_admission": True,
            "late_verifier_upload": True,
        },
    }
    _write_private_jsonl(args.materialization_evidence, [evidence])
    rows = [candidate, candidate] if duplicate else [candidate]
    _write_private_jsonl(args.candidate, rows)
    _write_private_jsonl(args.manifest, [manifest])
    return task, manifest, candidate, args


def test_finalize_emits_only_public_rows_and_preserves_sampling(tmp_path: Path) -> None:
    task, manifest, candidate, args = _fixture(tmp_path, duplicate=True)

    summary = finalize(args)

    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert rows == [candidate, candidate]
    assert summary["artifact_stage"] == "environment-admitted"
    assert summary["environment_admitted"] is True
    assert summary["candidate_rows"] == 2
    assert summary["admitted_rows"] == 2
    assert summary["admitted_unique_tasks"] == 1
    assert summary["task_ids_count"] == 1
    assert args.task_ids_output.read_text(encoding="utf-8") == (
        f"{manifest['instance_id']}\n"
    )
    assert summary["task_ids_sha256"] == hashlib.sha256(
        args.task_ids_output.read_bytes()
    ).hexdigest()
    assert summary["task_runtime_sha256"] == finalize_admitted._stable_digest(
        [
            [
                manifest["instance_id"],
                manifest["task_digest"],
                materialize_module._task_tree_sha256(
                    args.tasks_dir / manifest["instance_id"]
                ),
            ]
        ]
    )
    assert args.output.stat().st_mode & 0o077 == 0
    assert args.task_ids_output.stat().st_mode & 0o077 == 0
    rendered = args.output.read_text(encoding="utf-8")
    assert manifest["solution"]["oracle_patch"] not in rendered
    assert manifest["verifier"]["test_patch"] not in rendered
    assert task.source_image not in rendered


def test_finalize_requires_explicit_subset_without_replacing_output(
    tmp_path: Path,
) -> None:
    _, _, candidate, args = _fixture(tmp_path)
    missing = _task("example__project-2")
    missing_manifest = _locked_manifest(missing)
    _write_private_jsonl(args.candidate, [candidate, missing.to_miles_row(agent_name="terminus-2")])
    first_manifest = json.loads(args.manifest.read_text().splitlines()[0])
    _write_private_jsonl(args.manifest, [first_manifest, missing_manifest])
    args.output.parent.mkdir(mode=0o700)
    args.output.write_text("previous-safe-output\n", encoding="utf-8")
    args.output.chmod(0o600)

    with pytest.raises(ValueError, match="were not materialized"):
        finalize(args)

    assert args.output.read_text(encoding="utf-8") == "previous-safe-output\n"

    args.allow_subset = True
    summary = finalize(args)
    assert summary["excluded_unique_tasks"] == 1
    assert len(args.output.read_text(encoding="utf-8").splitlines()) == 1


def test_finalize_rejects_mutable_manifest_image(tmp_path: Path) -> None:
    _, manifest, _, args = _fixture(tmp_path)
    manifest["sandbox"]["source_image"] = "docker.io/example/project:latest"
    digest_payload = dict(manifest)
    digest_payload.pop("content_digest")
    digest_payload.pop("task_digest")
    manifest["content_digest"] = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _write_private_jsonl(args.manifest, [manifest])

    with pytest.raises(ValueError, match="not locked to a digest"):
        finalize(args)


def test_finalize_rejects_untrusted_image_publisher(tmp_path: Path) -> None:
    _, manifest, _, args = _fixture(tmp_path)
    manifest["sandbox"]["source_image"] = (
        "docker.io/attacker/example@sha256:" + "a" * 64
    )
    manifest.pop("content_digest")
    manifest.pop("task_digest")
    manifest["task_digest"] = "f" * 64
    digest_payload = dict(manifest)
    digest_payload.pop("task_digest")
    manifest["content_digest"] = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _write_private_jsonl(args.manifest, [manifest])

    with pytest.raises(ValueError, match="publisher policy"):
        finalize(args)


def test_finalize_rejects_private_e2b_build_context(tmp_path: Path) -> None:
    _, _, _, args = _fixture(tmp_path)
    tests_dir = next(args.tasks_dir.iterdir()) / "tests"
    tests_dir.chmod(0o700)
    dockerfile = tests_dir / "Dockerfile"
    dockerfile.write_text("FROM private\n", encoding="utf-8")
    dockerfile.chmod(0o400)
    tests_dir.chmod(0o500)

    with pytest.raises(ValueError, match="must not be an E2B template build context"):
        finalize(args)


def test_finalize_rejects_candidate_prompt_drift(tmp_path: Path) -> None:
    _, _, candidate, args = _fixture(tmp_path)
    candidate["prompt"] = "A different problem statement."
    _write_private_jsonl(args.candidate, [candidate])

    with pytest.raises(ValueError, match="prompt differs"):
        finalize(args)


def test_finalize_rejects_candidate_source_drift(tmp_path: Path) -> None:
    _, _, candidate, args = _fixture(tmp_path)
    candidate["metadata"]["source"] = "private/oracle-notes"
    _write_private_jsonl(args.candidate, [candidate])

    with pytest.raises(ValueError, match="source differs"):
        finalize(args)


def test_finalize_rejects_unbound_semantic_admission(tmp_path: Path) -> None:
    _, _, _, args = _fixture(tmp_path)
    admission = json.loads(args.semantic_admission_manifest[0].read_text())
    admission["base_tree"] = "f" * 40
    _write_private_jsonl(args.semantic_admission_manifest[0], [admission])

    with pytest.raises(ValueError, match="semantic record is absent"):
        finalize(args)


def test_finalize_rejects_task_tree_drift_after_materialization(
    tmp_path: Path,
) -> None:
    _, _, _, args = _fixture(tmp_path)
    instruction = next(args.tasks_dir.iterdir()) / "instruction.md"
    instruction.chmod(0o600)
    instruction.write_text(
        instruction.read_text(encoding="utf-8") + "post-admission drift\n",
        encoding="utf-8",
    )
    instruction.chmod(0o400)

    with pytest.raises(ValueError, match="differs from admission evidence"):
        finalize(args)


def test_finalize_rejects_writable_task_tree_after_materialization(
    tmp_path: Path,
) -> None:
    _, _, _, args = _fixture(tmp_path)
    instruction = next(args.tasks_dir.iterdir()) / "instruction.md"
    instruction.chmod(0o600)

    with pytest.raises(PermissionError, match="not read-only"):
        finalize(args)


def test_finalize_rejects_symlinks_in_task_tree(tmp_path: Path) -> None:
    _, _, _, args = _fixture(tmp_path)
    task_dir = next(args.tasks_dir.iterdir())
    link = task_dir / "agent-visible-secret"
    task_dir.chmod(0o700)
    os.symlink(task_dir / "instruction.md", link)
    task_dir.chmod(0o500)

    try:
        with pytest.raises(ValueError, match="symlink/special"):
            finalize(args)
    finally:
        task_dir.chmod(0o700)
        link.unlink(missing_ok=True)
        task_dir.chmod(0o500)


def test_finalize_rejects_hardlinked_private_input(tmp_path: Path) -> None:
    _, _, _, args = _fixture(tmp_path)
    linked = tmp_path / "candidate-copy.jsonl"
    os.link(args.candidate, linked)

    with pytest.raises(ValueError, match="must not be hard-linked"):
        finalize(args)


def test_finalize_rejects_candidate_symlink_swap_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, args = _fixture(tmp_path)
    original_read = finalize_admitted._read_jsonl
    swapped = False

    def swap_after_first_row(
        path: Path,
        *,
        name: str,
    ):
        nonlocal swapped
        for row in original_read(path, name=name):
            if path == args.candidate and not swapped:
                swapped = True
                parked = tmp_path / "candidate-before-swap.jsonl"
                args.candidate.replace(parked)
                args.candidate.symlink_to(parked.name)
            yield row

    monkeypatch.setattr(finalize_admitted, "_read_jsonl", swap_after_first_row)
    with pytest.raises(RuntimeError, match="invalid candidate prompt dataset"):
        finalize(args)


def test_finalize_rejects_hardlinks_in_task_tree(tmp_path: Path) -> None:
    _, _, _, args = _fixture(tmp_path)
    task_dir = next(args.tasks_dir.iterdir())
    task_dir.chmod(0o700)
    os.link(task_dir / "instruction.md", task_dir / "agent-visible-copy")
    task_dir.chmod(0o500)

    with pytest.raises(ValueError, match="symlink/special"):
        finalize(args)
