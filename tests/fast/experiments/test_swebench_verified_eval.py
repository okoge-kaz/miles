"""End-to-end contracts for hardened-local SWE-bench Verified evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest

from experiments.src.datasets.swe.schema import normalize_swe_row
from experiments.src.environments.swe import admit_swebench_verified
from experiments.src.environments.swe import finalize_admitted
from experiments.src.environments.swe import finalize_evaluation
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock
from experiments.src.evaluators import swe as swe_evaluator
from experiments.src.evaluators.swe import _load_tasks

_BASE = "1" * 40
_RESOLVED_IMAGE = (
    "docker.io/swebench/sweb.eval.x86_64.django_1776_django-12345@sha256:"
    + "a" * 64
)


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


def _task():
    return normalize_swe_row(
        {
            "instance_id": "django__django-12345",
            "repo": "django/django",
            "version": "4.1",
            "base_commit": _BASE,
            "problem_statement": "Fix the documented Django behavior.",
            "patch": (
                "diff --git a/django/core/checks.py b/django/core/checks.py\n"
                "--- a/django/core/checks.py\n"
                "+++ b/django/core/checks.py\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
            "test_patch": (
                "diff --git a/tests/checks/test_registry.py "
                "b/tests/checks/test_registry.py\n"
                "--- a/tests/checks/test_registry.py\n"
                "+++ b/tests/checks/test_registry.py\n"
                "@@ -1 +1 @@\n-assert False\n+assert True\n"
            ),
            "FAIL_TO_PASS": ["checks.test_registry.RegistryTests.test_fix"],
            "PASS_TO_PASS": ["checks.test_registry.RegistryTests.test_existing"],
            "split": "test",
        },
        dataset_id="princeton-nlp/SWE-bench_Verified",
        usage="eval",
    )


def _locked_manifest() -> dict:
    value = _task().to_task_manifest()
    requested = value["sandbox"]["source_image"]
    input_digest = value["content_digest"]
    value["sandbox"]["source_image"] = _RESOLVED_IMAGE
    value["sandbox"]["image_lock"] = {
        "schema_version": oci_image_lock.LOCK_SCHEMA,
        "source_image_requested": requested,
        "source_image_resolved": _RESOLVED_IMAGE,
        "input_content_digest": input_digest,
        "index_digest": None,
        "child_manifest_digest": "sha256:" + "a" * 64,
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    value["content_digest"] = oci_image_lock._stable_digest_without_bindings(value)
    return value


def _write_private_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _public_row(manifest: dict) -> dict:
    return {
        "prompt": manifest["problem_statement"],
        "label": "",
        "metadata": {
            "instance_id": manifest["instance_id"],
            "agent_name": "terminus-2",
            "source": manifest["source_dataset"],
            "verifier": "swe_environment",
            "swe_task": {
                "schema_version": "miles-swe-task-v1",
                "source_dataset": manifest["source_dataset"],
                "source_schema": manifest["source_schema"],
                "task_id": manifest["instance_id"],
                "task_digest": manifest["task_digest"],
                "eval_only": True,
            },
        },
    }


def _fake_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "official-swebench"
    harness = root / "swebench" / "harness"
    harness.mkdir(parents=True)
    files = {
        "constants.py": "MAP_REPO_VERSION_TO_SPECS = {}\nNON_TEST_EXTS = []\n",
        "log_parsers.py": "MAP_REPO_TO_PARSER = {}\n",
        "grading.py": "def get_eval_tests_report(*args): return {}\n",
    }
    constants = {
        "constants.py": "_SWEBENCH_CONSTANTS_SHA256",
        "log_parsers.py": "_SWEBENCH_LOG_PARSERS_SHA256",
        "grading.py": "_SWEBENCH_GRADING_SHA256",
    }
    harness_hashes = {}
    for name, content in files.items():
        path = harness / name
        path.write_text(content, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        harness_hashes[name] = digest
        monkeypatch.setattr(
            materialize_module,
            constants[name],
            digest,
        )
    monkeypatch.setattr(
        swe_evaluator,
        "_SWEBENCH_HARNESS_FILES",
        harness_hashes,
    )
    return root


def _materialize_args(
    manifest: Path,
    output: Path,
    harness: Path,
    **overrides,
) -> argparse.Namespace:
    values = {
        "manifest": manifest,
        "output": output,
        "admission_evidence": None,
        "r2e_execution_log_parser": None,
        "r2e_admission_manifest": None,
        "swe_rebench_log_parsers": None,
        "swe_rebench_constants": None,
        "swe_rebench_eval": None,
        "swe_rebench_admission_manifest": None,
        "swe_gym_harness_root": None,
        "swe_gym_admission_manifest": None,
        "swebench_harness_root": harness,
        "swebench_verified_admission_manifest": None,
        "allow_mutable_images": False,
        "allow_unadmitted_r2e_dry_run": False,
        "allow_unadmitted_swe_rebench_dry_run": False,
        "allow_unadmitted_swe_gym_dry_run": False,
        "allow_unadmitted_swebench_verified_dry_run": True,
        "limit": None,
        "summary": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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


def _admission(manifest: dict, task_dir: Path) -> dict:
    image_lock = manifest["sandbox"]["image_lock"]
    adapter = admit_swebench_verified.SwebenchVerifiedAdapter(
        harness_root=Path("unused")
    )
    return {
        "schema_version": admit_swebench_verified.ADMISSION_SCHEMA,
        "instance_id": manifest["instance_id"],
        "source_schema": "swebench",
        "task_digest": manifest["task_digest"],
        "input_content_digest": image_lock["input_content_digest"],
        "locked_content_digest": manifest["content_digest"],
        "content_digest": manifest["content_digest"],
        "source_image_requested": image_lock["source_image_requested"],
        "source_image_resolved": _RESOLVED_IMAGE,
        "source_image": _RESOLVED_IMAGE,
        "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
        "base_commit": manifest["base_commit"],
        "base_tree": "6" * 40,
        "oracle_patch_sha256": hashlib.sha256(
            manifest["solution"]["oracle_patch"].encode()
        ).hexdigest(),
        "test_patch_sha256": hashlib.sha256(
            manifest["verifier"]["test_patch"].encode()
        ).hexdigest(),
        "model_path_policy_sha256": hashlib.sha256(
            (task_dir / "tests" / "model_path_policy.json").read_bytes()
        ).hexdigest(),
        "admitted_task_tree_sha256": materialize_module._task_tree_sha256(
            task_dir
        ),
        **adapter.admission_metadata(manifest),
        "template_evidence": _template_evidence(),
        "checks": dict(materialize_module._REPOSITORY_ADMISSION_CHECKS),
    }


def test_verified_image_policy_is_exact_and_eval_only() -> None:
    manifest = _locked_manifest()
    oci_image_lock.validate_task_image_policy(manifest)

    wrong = json.loads(json.dumps(manifest))
    wrong["eval_only"] = False
    with pytest.raises(ValueError, match="eval-only publisher policy"):
        oci_image_lock.validate_task_image_policy(wrong)

    wrong = json.loads(json.dumps(manifest))
    wrong["sandbox"]["source_image"] = (
        "docker.io/xingyaoww/sweb.eval.x86_64.django_1776_django-12345"
        "@sha256:" + "a" * 64
    )
    with pytest.raises(ValueError, match="eval-only publisher policy"):
        oci_image_lock.validate_task_image_policy(wrong)


def test_hardened_path_policy_denies_tests_but_allows_implementation() -> None:
    policy = materialize_module._model_path_policy({"source_schema": "swebench"})
    assert policy["schema_version"] == "miles-swe-model-path-policy-v2"
    assert policy["policy_mode"] == "deny-sensitive-paths"
    assert "tests" in policy["denied_components"]
    assert materialize_module._denied_model_path("tests/test_feature.py") is True
    assert materialize_module._denied_model_path("django/core/checks.py") is False


def test_verified_producer_to_evaluator_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _fake_harness(tmp_path, monkeypatch)
    manifest = _locked_manifest()
    manifest_path = tmp_path / "verified.private.jsonl"
    _write_private_jsonl(manifest_path, [manifest])

    admission_tree = tmp_path / "admission-tree"
    materialize_module.materialize(
        _materialize_args(manifest_path, admission_tree, harness)
    )
    admission = _admission(
        manifest,
        admission_tree / manifest["instance_id"],
    )
    admission_path = tmp_path / "verified.admission.private.jsonl"
    _write_private_jsonl(admission_path, [admission])

    tasks = tmp_path / "production-tasks"
    evidence = tmp_path / "verified.materialization.private.jsonl"
    summary = materialize_module.materialize(
        _materialize_args(
            manifest_path,
            tasks,
            harness,
            admission_evidence=evidence,
            swebench_verified_admission_manifest=admission_path,
            allow_unadmitted_swebench_verified_dry_run=False,
        )
    )
    assert summary["schemas"] == {"swebench": 1}

    candidate = tmp_path / "verified.candidate.jsonl"
    _write_private_jsonl(candidate, [_public_row(manifest)])
    output = tmp_path / "eval" / "verified-hardened.jsonl"
    output_summary = tmp_path / "eval" / "verified-hardened.summary.json"
    finalize_args = argparse.Namespace(
        candidate=candidate,
        manifest=manifest_path,
        materialization_evidence=evidence,
        semantic_admission_manifest=[admission_path],
        tasks_dir=tasks,
        output=output,
        task_ids_output=tmp_path / "eval" / "verified-hardened.task-ids.txt",
        summary=output_summary,
        allow_subset=False,
    )
    finalized = finalize_evaluation.finalize(finalize_args)

    assert finalized["evaluation_only"] is True
    assert finalized["official_comparable"] is False
    assert finalized["score_semantics"] == (
        "hardened-local-not-official-comparable-v1"
    )
    assert finalized["task_runtime_sha256"] == finalize_admitted._stable_digest(
        [
            [
                manifest["instance_id"],
                manifest["task_digest"],
                materialize_module._task_tree_sha256(
                    tasks / manifest["instance_id"]
                ),
            ]
        ]
    )
    assert _load_tasks(
        output,
        limit=None,
        input_summary=output_summary,
    ) == [(manifest["instance_id"], manifest["task_digest"])]
    row = json.loads(output.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="cannot enter RL training"):
        finalize_admitted._validate_public_row(row)


def test_evaluator_rejects_unbound_or_mislabeled_summary(tmp_path: Path) -> None:
    manifest = _locked_manifest()
    data = tmp_path / "eval.jsonl"
    _write_private_jsonl(data, [_public_row(manifest)])
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "miles-swe-evaluation-dataset-v1",
                "output_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    summary.chmod(0o600)
    with pytest.raises(ValueError, match="not an admitted Verified artifact"):
        _load_tasks(data, limit=None, input_summary=summary)


def test_evaluator_rejects_unsafe_bound_input_file(tmp_path: Path) -> None:
    manifest = _locked_manifest()
    target = tmp_path / "eval-target.jsonl"
    _write_private_jsonl(target, [_public_row(manifest)])
    data = tmp_path / "eval.jsonl"
    data.symlink_to(target)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": (
                    "miles-swebench-verified-hardened-local-evaluation-v1"
                ),
                "artifact_stage": (
                    "hardened-local-environment-admitted-evaluation"
                ),
                "environment_admitted": True,
                "evaluation_only": True,
                "official_comparable": False,
                "score_semantics": (
                    "hardened-local-not-official-comparable-v1"
                ),
                "source_dataset": "princeton-nlp/SWE-bench_Verified",
                "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    summary.chmod(0o600)

    with pytest.raises(ValueError, match="input is missing or unsafe"):
        _load_tasks(data, limit=None, input_summary=summary)
