"""Promote only materialized, immutable SWE tasks into Miles RL input.

Dataset normalization deliberately emits owner-only ``not-admitted`` prompt
rows.  This module joins those rows to the exact private task manifest and to
the Harbor task tree produced by the production materializer.  It never copies
private verifier fields into the resulting prompt JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import tomllib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from experiments.src.datasets.swe.schema import SCHEMA_VERSION
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,511}@sha256:[0-9a-f]{64}")
_SAFE_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,239}")
_PUBLIC_ROW_KEYS = {"label", "metadata", "prompt"}
_PUBLIC_METADATA_KEYS = {
    "agent_name",
    "instance_id",
    "source",
    "swe_task",
    "verifier",
}
_PUBLIC_TASK_KEYS = {
    "eval_only",
    "schema_version",
    "source_dataset",
    "source_schema",
    "task_digest",
    "task_id",
}
_EVIDENCE_SCHEMA = "miles-swe-materialization-evidence-v1"
_EVIDENCE_CHECKS = {
    "production_materialization": True,
    "immutable_image": True,
    "semantic_admission": True,
    "late_verifier_upload": True,
}
_R2E_ADMISSION_CHECKS: dict[str, bool | int] = {
    "publisher_namespace_policy": True,
    "source_head_matches_base": True,
    "unique_parent_matches_base": True,
    "empty_reward": 0,
    "oracle_reward": 1,
    "runtime_smoke": True,
    "tool_smoke": True,
    "no_new_privileges": True,
    "effective_capabilities_zero": True,
    "suid_sgid_absent": True,
    "file_capabilities_absent": True,
    "fresh_separate_verifier": True,
    "source_network_denied": True,
    "agent_network_denied": True,
    "empty_verifier_network_denied": True,
    "oracle_verifier_network_denied": True,
    "late_verifier_tests_absent_before_upload": True,
    "gold_history_absent": True,
    "known_gold_artifacts_absent": True,
    "gold_blob_content_absent": True,
    "gold_commit_text_absent": True,
}
_TEMPLATE_EVIDENCE_FIELDS = {
    "template_id",
    "build_id",
    "alias_sha256",
    "template_identity_sha256",
    "sandbox_id",
}
_TEMPLATE_ROLES = {"source", "agent", "empty_verifier", "oracle_verifier"}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    """Write a public-only RL dataset bound to production Harbor tasks."""
    _validate_distinct_paths(args)
    _validate_owner_only_file(args.candidate, name="candidate prompt dataset")
    _validate_owner_only_file(args.manifest, name="private task manifest")
    _validate_owner_only_file(
        args.materialization_evidence,
        name="materialization admission evidence",
    )
    if not args.semantic_admission_manifest:
        raise ValueError("at least one --semantic-admission-manifest is required")
    for path in args.semantic_admission_manifest:
        _validate_owner_only_file(path, name="semantic admission manifest")
    _validate_task_root(args.tasks_dir)
    input_paths = (
        args.candidate,
        args.manifest,
        args.materialization_evidence,
        *args.semantic_admission_manifest,
    )
    input_fingerprints = {
        path: oci_image_lock._capture_private_fingerprint(path)
        for path in input_paths
    }

    def require_inputs_unchanged() -> None:
        for path, fingerprint in input_fingerprints.items():
            oci_image_lock._assert_private_unchanged(path, fingerprint)

    manifests = _load_manifests(args.manifest)
    semantic_admissions = _load_semantic_admissions(args.semantic_admission_manifest)
    evidence = _load_materialization_evidence(
        args.materialization_evidence,
        manifests=manifests,
        semantic_admissions=semantic_admissions,
    )
    admitted = _load_materialized_tasks(
        args.tasks_dir,
        manifests,
        evidence=evidence,
    )
    if not admitted:
        raise ValueError(f"no production Harbor tasks found under {args.tasks_dir}")

    rows_seen = 0
    rows_written = 0
    candidate_task_ids: set[str] = set()
    admitted_task_ids: set[str] = set()

    def validate_rows(*, count: bool) -> Iterator[dict[str, Any]]:
        nonlocal rows_seen, rows_written
        for row in _read_jsonl(args.candidate, name="candidate prompt dataset"):
            if count:
                rows_seen += 1
            instance_id, task_digest = _validate_public_row(row)
            if count:
                candidate_task_ids.add(instance_id)
            manifest = manifests.get(instance_id)
            if manifest is None:
                raise ValueError(f"candidate {instance_id!r} has no matching private manifest")
            if task_digest != manifest["task_digest"]:
                raise ValueError(f"candidate task binding differs from private manifest: {instance_id}")
            if row["prompt"] != manifest["problem_statement"]:
                raise ValueError(f"candidate prompt differs from private manifest: {instance_id}")
            if row["metadata"]["source"] != manifest["source_dataset"]:
                raise ValueError(
                    f"candidate source differs from private manifest: {instance_id}"
                )
            swe_task = row["metadata"]["swe_task"]
            if swe_task["source_dataset"] != manifest["source_dataset"] or swe_task["source_schema"] != manifest["source_schema"] or swe_task["task_id"] != instance_id:
                raise ValueError(f"candidate provenance differs from private manifest: {instance_id}")
            if instance_id not in admitted:
                continue
            if count:
                admitted_task_ids.add(instance_id)
                rows_written += 1
            yield row

    for _ in validate_rows(count=True):
        pass
    require_inputs_unchanged()
    missing_task_ids = sorted(candidate_task_ids - admitted_task_ids)
    extra_task_ids = sorted(set(admitted) - candidate_task_ids)
    if extra_task_ids:
        raise ValueError(f"materialized task tree contains tasks absent from the candidate dataset: {extra_task_ids[:5]}")
    if missing_task_ids and not args.allow_subset:
        raise ValueError(f"{len(missing_task_ids)} candidate tasks were not materialized; rerun admission/materialization, or pass --allow-subset to make the safe-subset decision explicit")
    if rows_written == 0:
        raise ValueError("no candidate rows matched production Harbor tasks")

    temporary_output, output_digest = _stage_private_jsonl(
        args.output,
        validate_rows(count=False),
    )
    task_ids = sorted(admitted_task_ids)
    try:
        temporary_task_ids, task_ids_digest = _stage_private_text(
            args.task_ids_output,
            "".join(f"{instance_id}\n" for instance_id in task_ids),
        )
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    try:
        require_inputs_unchanged()
        os.replace(temporary_output, args.output)
        os.replace(temporary_task_ids, args.task_ids_output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_task_ids.unlink(missing_ok=True)
        raise

    summary = {
        "schema_version": "miles-swe-admitted-dataset-v1",
        "artifact_stage": "environment-admitted",
        "environment_admitted": True,
        "candidate": str(args.candidate),
        "private_manifest": str(args.manifest),
        "materialization_evidence": str(args.materialization_evidence),
        "semantic_admission_manifests": [str(path) for path in args.semantic_admission_manifest],
        "tasks_dir": str(args.tasks_dir),
        "output": str(args.output),
        "task_ids_output": str(args.task_ids_output),
        "task_ids_count": len(task_ids),
        "task_ids_sha256": task_ids_digest,
        "candidate_rows": rows_seen,
        "admitted_rows": rows_written,
        "candidate_unique_tasks": len(candidate_task_ids),
        "admitted_unique_tasks": len(admitted_task_ids),
        "excluded_unique_tasks": len(missing_task_ids),
        "allow_subset": bool(args.allow_subset),
        "output_sha256": output_digest,
        "task_set_sha256": _stable_digest(sorted(admitted_task_ids)),
        "task_binding_sha256": _stable_digest(
            [
                [instance_id, manifests[instance_id]["task_digest"]]
                for instance_id in sorted(admitted_task_ids)
            ]
        ),
        "task_runtime_sha256": _stable_digest(
            [
                [
                    instance_id,
                    manifests[instance_id]["task_digest"],
                    evidence[instance_id]["task_tree_sha256"],
                ]
                for instance_id in sorted(admitted_task_ids)
            ]
        ),
        "model_path_policy_scope": (
            "pilot training admission: oracle-touched implementation paths only; "
            "not applied to official downstream evaluation"
        ),
    }
    _write_private_json(args.summary, summary)
    return summary


def _load_manifests(path: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for value in _read_jsonl(path, name="private task manifest"):
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported private task schema: {value.get('schema_version')!r}")
        instance_id = _required_string(value, "instance_id")
        if _SAFE_INSTANCE_ID.fullmatch(instance_id) is None or instance_id in {".", ".."}:
            raise ValueError(f"unsafe SWE instance_id: {instance_id!r}")
        task_digest = _required_string(value, "task_digest")
        content_digest = _required_string(value, "content_digest")
        if _SHA256.fullmatch(task_digest) is None or _SHA256.fullmatch(content_digest) is None:
            raise ValueError(f"invalid task/content digest for {instance_id}")
        digest_payload = dict(value)
        digest_payload.pop("task_digest", None)
        digest_payload.pop("content_digest", None)
        if _stable_digest(digest_payload) != content_digest:
            raise ValueError(f"private manifest content digest mismatch: {instance_id}")
        sandbox = value.get("sandbox")
        source_image = sandbox.get("source_image") if isinstance(sandbox, dict) else None
        if not isinstance(source_image, str) or _IMMUTABLE_IMAGE.fullmatch(source_image) is None:
            raise ValueError(f"private manifest image was not locked to a digest: {instance_id}")
        oci_image_lock.validate_task_image_policy(value)
        if instance_id in manifests:
            raise ValueError(f"duplicate private task manifest: {instance_id}")
        _required_string(value, "problem_statement")
        _required_string(value, "source_dataset")
        _required_string(value, "source_schema")
        manifests[instance_id] = value
    if not manifests:
        raise ValueError(f"private task manifest is empty: {path}")
    return manifests


def _load_materialized_tasks(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    *,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    tasks: dict[str, Path] = {}
    for task_dir in sorted(root.iterdir()):
        if task_dir.name.startswith("."):
            raise ValueError(f"partial/hidden task entry under {root}: {task_dir.name}")
        _require_real_directory(task_dir, name="Harbor task")
        manifest = manifests.get(task_dir.name)
        if manifest is None:
            raise ValueError(f"Harbor task has no private manifest: {task_dir.name}")
        _validate_materialized_task(task_dir, manifest)
        admission = evidence.get(task_dir.name)
        if admission is None:
            raise ValueError(f"Harbor task has no production materialization evidence: {task_dir.name}")
        if materialize_module._task_tree_sha256(task_dir) != admission[
            "task_tree_sha256"
        ]:
            raise ValueError(f"Harbor task tree differs from admission evidence: {task_dir.name}")
        tasks[task_dir.name] = task_dir
    missing_trees = sorted(set(evidence) - set(tasks))
    if missing_trees:
        raise ValueError(f"materialization evidence has no corresponding Harbor task tree: {missing_trees[:5]}")
    return tasks


def _load_semantic_admissions(paths: list[Path]) -> dict[str, dict[str, Any]]:
    admissions: dict[str, dict[str, Any]] = {}
    for path in paths:
        for value in _read_jsonl(path, name="semantic admission manifest"):
            _validate_semantic_admission_record(value)
            digest = _stable_digest(value)
            previous = admissions.get(digest)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting semantic admission record: {digest}")
            admissions[digest] = value
    if not admissions:
        raise ValueError("semantic admission manifests are empty")
    return admissions


def _validate_semantic_admission_record(value: dict[str, Any]) -> None:
    schema = value.get("schema_version")
    if schema == "miles-r2e-admission-v1":
        _validate_r2e_semantic_admission(value)
    elif schema == "miles-swe-rebench-admission-v1":
        _validate_repository_semantic_admission(value, environment="rebench")
    elif schema == "miles-swe-gym-admission-v1":
        _validate_repository_semantic_admission(value, environment="swe-gym")
    elif schema == "miles-swebench-verified-hardened-local-admission-v1":
        _validate_repository_semantic_admission(
            value,
            environment="swebench-verified",
        )
    else:
        raise ValueError(f"unsupported semantic admission schema: {schema!r}")


def _validate_r2e_semantic_admission(value: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "instance_id",
        "task_digest",
        "input_content_digest",
        "locked_content_digest",
        "content_digest",
        "source_image_requested",
        "source_image_resolved",
        "source_image",
        "image_publisher_policy",
        "base_commit",
        "oracle_patch_sha256",
        "admitted_task_tree_sha256",
        "e2b_sandbox_evidence",
        "checks",
    }
    if set(value) != required:
        raise ValueError("R2E semantic admission field set is invalid")
    _validate_semantic_identity(value, expected_source_schema=None)
    _require_sha256(value, "oracle_patch_sha256")
    _require_sha256(value, "admitted_task_tree_sha256")
    checks = value.get("checks")
    if checks != _R2E_ADMISSION_CHECKS:
        raise ValueError("R2E semantic admission live checks are invalid")
    _validate_template_roles(value, key="e2b_sandbox_evidence", name="R2E")


def _validate_repository_semantic_admission(
    value: dict[str, Any],
    *,
    environment: str,
) -> None:
    common = {
        "schema_version",
        "instance_id",
        "source_schema",
        "task_digest",
        "input_content_digest",
        "locked_content_digest",
        "content_digest",
        "source_image_requested",
        "source_image_resolved",
        "source_image",
        "image_publisher_policy",
        "base_commit",
        "base_tree",
        "oracle_patch_sha256",
        "test_patch_sha256",
        "model_path_policy_sha256",
        "admitted_task_tree_sha256",
        "template_evidence",
        "checks",
    }
    if environment == "rebench":
        expected_source = "swe-rebench-v2"
        pins = {
            "rebench_commit": materialize_module._REBENCH_COMMIT,
            "log_parsers_sha256": materialize_module._REBENCH_LOG_PARSERS_SHA256,
            "constants_sha256": materialize_module._REBENCH_CONSTANTS_SHA256,
            "eval_sha256": materialize_module._REBENCH_EVAL_SHA256,
        }
    elif environment == "swe-gym":
        expected_source = "swe-gym"
        pins = {
            "dataset_revision": materialize_module._SWE_GYM_DATASET_REVISION,
            "harness_commit": materialize_module._SWE_GYM_HARNESS_COMMIT,
            "harness_version": materialize_module._SWE_GYM_HARNESS_VERSION,
            "constants_sha256": materialize_module._SWE_GYM_CONSTANTS_SHA256,
            "log_parsers_sha256": materialize_module._SWE_GYM_LOG_PARSERS_SHA256,
            "grading_sha256": materialize_module._SWE_GYM_GRADING_SHA256,
            "harbor_adapter_commit": materialize_module._SWE_GYM_HARBOR_COMMIT,
            "harbor_adapter_sha256": (
                materialize_module._SWE_GYM_HARBOR_ADAPTER_SHA256
            ),
        }
    elif environment == "swebench-verified":
        expected_source = "swebench"
        pins = {
            "dataset_revision": (
                materialize_module._SWEBENCH_VERIFIED_DATASET_REVISION
            ),
            "harness_repository": (
                materialize_module._SWEBENCH_HARNESS_REPOSITORY
            ),
            "harness_commit": materialize_module._SWEBENCH_HARNESS_COMMIT,
            "harness_version": materialize_module._SWEBENCH_HARNESS_VERSION,
            "constants_sha256": materialize_module._SWEBENCH_CONSTANTS_SHA256,
            "log_parsers_sha256": (
                materialize_module._SWEBENCH_LOG_PARSERS_SHA256
            ),
            "grading_sha256": materialize_module._SWEBENCH_GRADING_SHA256,
            "score_semantics": (
                materialize_module._SWEBENCH_HARDENED_SCORE_SEMANTICS
            ),
            "harbor_adapter_commit": materialize_module._SWE_GYM_HARBOR_COMMIT,
            "harbor_adapter_sha256": (
                materialize_module._SWE_GYM_HARBOR_ADAPTER_SHA256
            ),
        }
    else:
        raise ValueError(f"unsupported repository admission environment: {environment}")
    if set(value) != common | set(pins):
        raise ValueError(f"{environment} semantic admission field set is invalid")
    _validate_semantic_identity(value, expected_source_schema=expected_source)
    for key in (
        "base_tree",
        "oracle_patch_sha256",
        "test_patch_sha256",
        "model_path_policy_sha256",
        "admitted_task_tree_sha256",
    ):
        pattern = re.fullmatch(r"[0-9a-f]{40}", _required_string(value, key))
        if key != "base_tree":
            pattern = _SHA256.fullmatch(_required_string(value, key))
        if pattern is None:
            raise ValueError(f"semantic admission {key} is invalid")
    if any(value.get(key) != expected for key, expected in pins.items()):
        raise ValueError(f"{environment} semantic admission pins are invalid")
    if value.get("checks") != materialize_module._REPOSITORY_ADMISSION_CHECKS:
        raise ValueError(f"{environment} semantic admission live checks are invalid")
    _validate_template_roles(value, key="template_evidence", name=environment)


def _validate_semantic_identity(
    value: dict[str, Any],
    *,
    expected_source_schema: str | None,
) -> None:
    instance_id = _required_string(value, "instance_id")
    if _SAFE_INSTANCE_ID.fullmatch(instance_id) is None:
        raise ValueError("semantic admission instance_id is invalid")
    for key in (
        "task_digest",
        "input_content_digest",
        "locked_content_digest",
        "content_digest",
    ):
        _require_sha256(value, key)
    if value["locked_content_digest"] != value["content_digest"]:
        raise ValueError("semantic admission locked-content binding is invalid")
    base_commit = _required_string(value, "base_commit")
    if re.fullmatch(r"[0-9a-f]{40}", base_commit) is None:
        raise ValueError("semantic admission base commit is invalid")
    requested = _required_string(value, "source_image_requested")
    resolved = _required_string(value, "source_image_resolved")
    source_image = _required_string(value, "source_image")
    if not requested or resolved != source_image or _IMMUTABLE_IMAGE.fullmatch(
        source_image
    ) is None:
        raise ValueError("semantic admission image binding is invalid")
    if value.get("image_publisher_policy") != oci_image_lock.IMAGE_PUBLISHER_POLICY:
        raise ValueError("semantic admission image publisher policy is invalid")
    if expected_source_schema is not None and value.get(
        "source_schema"
    ) != expected_source_schema:
        raise ValueError("semantic admission source schema is invalid")


def _validate_template_roles(
    value: dict[str, Any],
    *,
    key: str,
    name: str,
) -> None:
    evidence = value.get(key)
    if not isinstance(evidence, dict) or set(evidence) != _TEMPLATE_ROLES:
        raise ValueError(f"{name} semantic admission template roles are invalid")
    phases: dict[str, dict[str, Any]] = {}
    for role in _TEMPLATE_ROLES:
        phase = evidence.get(role)
        if not isinstance(phase, dict) or set(phase) != _TEMPLATE_EVIDENCE_FIELDS:
            raise ValueError(f"{name} semantic admission template fields are invalid")
        for digest_key in ("alias_sha256", "template_identity_sha256"):
            _require_sha256(phase, digest_key)
        for id_key in ("template_id", "build_id", "sandbox_id"):
            identifier = _required_string(phase, id_key)
            if re.fullmatch(r"[A-Za-z0-9_-]{6,128}", identifier) is None:
                raise ValueError(f"{name} semantic admission E2B ID is invalid")
        phases[role] = phase
    immutable = (
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
    )
    for role in ("empty_verifier", "oracle_verifier"):
        if any(phases["source"][field] != phases[role][field] for field in immutable):
            raise ValueError(f"{name} source/verifier template pin differs")
    sandbox_ids = {phase["sandbox_id"] for phase in phases.values()}
    if len(sandbox_ids) != len(phases):
        raise ValueError(f"{name} semantic admission reused an E2B sandbox")


def _require_sha256(value: dict[str, Any], key: str) -> str:
    digest = _required_string(value, key)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"semantic admission {key} is invalid")
    return digest


def _load_materialization_evidence(
    path: Path,
    *,
    manifests: dict[str, dict[str, Any]],
    semantic_admissions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for value in _read_jsonl(path, name="materialization admission evidence"):
        if value.get("schema_version") != _EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported materialization evidence schema: {value.get('schema_version')!r}")
        instance_id = _required_string(value, "instance_id")
        manifest = manifests.get(instance_id)
        if manifest is None:
            raise ValueError(f"materialization evidence has no private manifest: {instance_id}")
        for key in ("task_digest", "content_digest"):
            field = _required_string(value, key)
            if _SHA256.fullmatch(field) is None or field != manifest[key]:
                raise ValueError(f"materialization evidence {key} mismatch: {instance_id}")
        source_image = _required_string(value, "source_image")
        if source_image != manifest["sandbox"]["source_image"]:
            raise ValueError(f"materialization evidence source image mismatch: {instance_id}")
        if value.get("source_schema") != manifest["source_schema"]:
            raise ValueError(f"materialization evidence source schema mismatch: {instance_id}")
        manifest_sha256 = _required_string(
            value,
            "private_manifest_record_sha256",
        )
        if manifest_sha256 != _stable_digest(manifest):
            raise ValueError(f"materialization evidence private-record mismatch: {instance_id}")
        tree_sha256 = _required_string(value, "task_tree_sha256")
        admission_sha256 = _required_string(
            value,
            "semantic_admission_record_sha256",
        )
        if _SHA256.fullmatch(tree_sha256) is None or _SHA256.fullmatch(admission_sha256) is None:
            raise ValueError(f"materialization evidence contains an invalid digest: {instance_id}")
        admission = semantic_admissions.get(admission_sha256)
        if admission is None:
            raise ValueError(f"materialization evidence semantic record is absent: {instance_id}")
        _validate_semantic_admission_binding(admission, manifest)
        if admission["admitted_task_tree_sha256"] != tree_sha256:
            raise ValueError(
                "materialization evidence tree differs from live semantic "
                f"admission: {instance_id}"
            )
        if value.get("checks") != _EVIDENCE_CHECKS:
            raise ValueError(f"materialization evidence lacks production checks: {instance_id}")
        if instance_id in evidence:
            raise ValueError(f"duplicate materialization evidence: {instance_id}")
        evidence[instance_id] = value
    if not evidence:
        raise ValueError(f"materialization admission evidence is empty: {path}")
    return evidence


def _validate_semantic_admission_binding(
    admission: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    instance_id = manifest["instance_id"]
    for key in ("instance_id", "task_digest", "content_digest", "base_commit"):
        if admission.get(key) != manifest[key]:
            raise ValueError(f"semantic admission {key} mismatch: {instance_id}")
    if admission.get("locked_content_digest") != manifest["content_digest"]:
        raise ValueError(f"semantic admission locked content mismatch: {instance_id}")
    schema_sources = {
        "miles-r2e-admission-v1": "r2e-gym-v1",
        "miles-swe-rebench-admission-v1": "swe-rebench-v2",
        "miles-swe-gym-admission-v1": "swe-gym",
        "miles-swebench-verified-hardened-local-admission-v1": "swebench",
    }
    if schema_sources[admission["schema_version"]] != manifest["source_schema"]:
        raise ValueError(f"semantic admission environment mismatch: {instance_id}")
    sandbox = manifest["sandbox"]
    source_image = sandbox["source_image"]
    if (
        admission.get("source_image") != source_image
        or admission.get("source_image_resolved") != source_image
    ):
        raise ValueError(f"semantic admission source image mismatch: {instance_id}")
    image_lock = sandbox.get("image_lock")
    if not isinstance(image_lock, dict) or image_lock.get(
        "schema_version"
    ) != "miles-oci-image-lock-v1":
        raise ValueError(f"semantic admission has no OCI lock binding: {instance_id}")
    if (
        admission.get("source_image_requested")
        != image_lock.get("source_image_requested")
        or admission.get("input_content_digest")
        != image_lock.get("input_content_digest")
        or image_lock.get("source_image_resolved") != source_image
    ):
        raise ValueError(f"semantic admission OCI provenance mismatch: {instance_id}")
    solution = manifest.get("solution")
    oracle_patch = solution.get("oracle_patch") if isinstance(solution, dict) else None
    if not isinstance(oracle_patch, str) or admission.get(
        "oracle_patch_sha256"
    ) != hashlib.sha256(oracle_patch.encode("utf-8")).hexdigest():
        raise ValueError(f"semantic admission oracle patch mismatch: {instance_id}")
    if admission["schema_version"] != "miles-r2e-admission-v1":
        verifier = manifest.get("verifier")
        test_patch = verifier.get("test_patch") if isinstance(verifier, dict) else None
        if not isinstance(test_patch, str) or admission.get(
            "test_patch_sha256"
        ) != hashlib.sha256(test_patch.encode("utf-8")).hexdigest():
            raise ValueError(f"semantic admission test patch mismatch: {instance_id}")


def _validate_materialized_task(task_dir: Path, manifest: dict[str, Any]) -> None:
    for path in [task_dir, *task_dir.rglob("*")]:
        metadata = path.lstat()
        mode = metadata.st_mode
        is_directory = stat.S_ISDIR(mode)
        is_regular = stat.S_ISREG(mode) and metadata.st_nlink == 1
        if stat.S_ISLNK(mode) or not (is_directory or is_regular):
            raise ValueError(f"Harbor task contains a symlink/special file: {path}")
        if metadata.st_uid != os.getuid():
            raise PermissionError(f"Harbor task entry is owned by another user: {path}")
        if mode & 0o077:
            raise PermissionError(f"Harbor task is not owner-only: {path}")
        private_mode = stat.S_IMODE(mode)
        expected_modes = {0o500} if is_directory else {0o400, 0o500}
        if private_mode not in expected_modes:
            raise PermissionError(f"Harbor task is not read-only: {path}")

    task_toml_path = task_dir / "task.toml"
    _validate_owner_only_file(task_toml_path, name="Harbor task.toml")
    config = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    metadata = config.get("metadata")
    verifier = config.get("verifier")
    agent = config.get("agent")
    environment = config.get("environment")
    if not all(isinstance(value, dict) for value in (metadata, verifier, agent, environment)):
        raise ValueError(f"Harbor task lacks required config tables: {task_dir.name}")
    assert isinstance(metadata, dict)
    assert isinstance(verifier, dict)
    assert isinstance(agent, dict)
    assert isinstance(environment, dict)
    if metadata.get("task_digest") != manifest["task_digest"] or metadata.get("source_dataset") != manifest["source_dataset"] or metadata.get("source_schema") != manifest["source_schema"]:
        raise ValueError(f"Harbor task metadata binding mismatch: {task_dir.name}")
    verifier_environment = verifier.get("environment")
    source_image = manifest["sandbox"]["source_image"]
    if (
        verifier.get("environment_mode") != "separate"
        or verifier.get("user") != 0
        or not isinstance(verifier_environment, dict)
        or verifier_environment.get("network_mode") != "no-network"
        or verifier_environment.get("docker_image") != source_image
        or environment.get("network_mode") != "no-network"
        or agent.get("user") != 1000
    ):
        raise ValueError(f"Harbor task is not the admitted separate/no-network layout: {task_dir.name}")
    collect = verifier.get("collect")
    if not isinstance(collect, list) or len(collect) != 1 or not isinstance(collect[0], dict) or collect[0].get("required") is not True:
        raise ValueError(f"Harbor task lacks fail-closed artifact collection: {task_dir.name}")

    tests_dir = task_dir / "tests"
    marker = tests_dir / ".harbor-e2b-late-tests"
    _require_real_directory(tests_dir, name="private verifier package")
    _validate_owner_only_file(marker, name="late verifier marker")
    if (tests_dir / "Dockerfile").exists():
        raise ValueError(f"private verifier files must not be an E2B template build context: {task_dir.name}")
    instruction = task_dir / "instruction.md"
    _validate_owner_only_file(instruction, name="Harbor instruction")
    expected_prefix = manifest["problem_statement"].rstrip() + "\n"
    if not instruction.read_text(encoding="utf-8").startswith(expected_prefix):
        raise ValueError(f"Harbor instruction/prompt binding mismatch: {task_dir.name}")


def _validate_public_row(
    row: dict[str, Any],
    *,
    expected_eval_only: bool = False,
) -> tuple[str, str]:
    if set(row) != _PUBLIC_ROW_KEYS:
        raise ValueError(f"candidate row has non-public top-level keys: {sorted(set(row))}")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or "\0" in prompt:
        raise ValueError("candidate prompt must be non-empty text")
    if row.get("label") != "":
        raise ValueError("SWE candidate label must be empty")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != _PUBLIC_METADATA_KEYS:
        raise ValueError("candidate metadata contains missing/private fields")
    instance_id = metadata.get("instance_id")
    if not isinstance(instance_id, str) or _SAFE_INSTANCE_ID.fullmatch(instance_id) is None or instance_id in {".", ".."}:
        raise ValueError("candidate metadata.instance_id is unsafe")
    if metadata.get("agent_name") != "terminus-2":
        raise ValueError("candidate metadata.agent_name must be terminus-2")
    if metadata.get("verifier") != "swe_environment":
        raise ValueError("candidate metadata.verifier must be swe_environment")
    swe_task = metadata.get("swe_task")
    if not isinstance(swe_task, dict) or set(swe_task) != _PUBLIC_TASK_KEYS:
        raise ValueError("candidate metadata.swe_task contains missing/private fields")
    task_digest = swe_task.get("task_digest")
    if not isinstance(task_digest, str) or _SHA256.fullmatch(task_digest) is None:
        raise ValueError("candidate task digest is invalid")
    if swe_task.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate schema version is unsupported")
    if swe_task.get("eval_only") is not expected_eval_only:
        if expected_eval_only:
            raise ValueError("SWE evaluation input must contain only evaluation tasks")
        raise ValueError("evaluation-only SWE tasks cannot enter RL training")
    return instance_id, task_digest


def _validate_task_root(path: Path) -> None:
    _require_real_directory(path, name="Harbor task root")
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"Harbor task root must be owner-only: {path}")


def _require_real_directory(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{name} is missing: {path}") from exc
    mode = metadata.st_mode
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise ValueError(f"{name} must be a real directory: {path}")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"{name} must be owned by the current user: {path}")


def _validate_owner_only_file(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{name} is missing: {path}") from exc
    mode = metadata.st_mode
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise ValueError(f"{name} must be a regular file: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"{name} must not be hard-linked: {path}")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"{name} must be owned by the current user: {path}")
    if mode & 0o077:
        raise PermissionError(f"{name} must be owner-only (chmod 600): {path}")


def _read_jsonl(path: Path, *, name: str) -> Iterator[dict[str, Any]]:
    try:
        yield from oci_image_lock._read_jsonl(path)
    except (ValueError, PermissionError, RuntimeError) as exc:
        raise type(exc)(f"invalid {name}: {exc}") from exc


def _stage_private_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> tuple[Path, str]:
    parent = _prepare_private_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=parent,
    )
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            for row in rows:
                rendered = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.write(rendered)
                digest.update(rendered.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return Path(temporary_name), digest.hexdigest()


def _stage_private_text(path: Path, content: str) -> tuple[Path, str]:
    parent = _prepare_private_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return Path(temporary_name), hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    parent = _prepare_private_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _prepare_private_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"output parent must be a real directory: {path.parent}")
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        _validate_owner_only_file(path, name="existing output")
    return path.parent


def _validate_distinct_paths(args: argparse.Namespace) -> None:
    named_paths = {
        "candidate": args.candidate,
        "manifest": args.manifest,
        "materialization-evidence": args.materialization_evidence,
        "output": args.output,
        "task-ids-output": args.task_ids_output,
        "summary": args.summary,
    }
    for index, path in enumerate(args.semantic_admission_manifest):
        named_paths[f"semantic-admission-manifest-{index}"] = path
    resolved: dict[Path, str] = {}
    for name, path in named_paths.items():
        normalized = path.absolute()
        previous = resolved.get(normalized)
        if previous is not None:
            raise ValueError(f"--{name} and --{previous} must use distinct paths")
        resolved[normalized] = name


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip() or "\0" in result:
        raise ValueError(f"required text field {key!r} is missing")
    return result.strip()


def _stable_digest(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--materialization-evidence", type=Path, required=True)
    parser.add_argument(
        "--semantic-admission-manifest",
        type=Path,
        action="append",
        required=True,
        help="owner-only schema-specific live admission JSONL; repeat as needed",
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="explicitly emit only candidates that passed task materialization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(finalize(parse_args()), sort_keys=True))
