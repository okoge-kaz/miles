"""Bind hardened-local Verified rows to live-admitted immutable Harbor tasks."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from experiments.src.environments.swe import finalize_admitted
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock

_DATASET_ID = "princeton-nlp/SWE-bench_Verified"
_SOURCE_SCHEMA = "swebench"


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    """Write a gold-free Verified input bound to production Harbor tasks."""
    _validate_distinct_paths(args)
    finalize_admitted._validate_owner_only_file(
        args.candidate,
        name="evaluation candidate dataset",
    )
    finalize_admitted._validate_owner_only_file(
        args.manifest,
        name="private task manifest",
    )
    finalize_admitted._validate_owner_only_file(
        args.materialization_evidence,
        name="materialization admission evidence",
    )
    if not args.semantic_admission_manifest:
        raise ValueError("at least one --semantic-admission-manifest is required")
    for path in args.semantic_admission_manifest:
        finalize_admitted._validate_owner_only_file(
            path,
            name="semantic admission manifest",
        )
    finalize_admitted._validate_task_root(args.tasks_dir)
    input_paths = (
        args.candidate,
        args.manifest,
        args.materialization_evidence,
        *args.semantic_admission_manifest,
    )
    fingerprints = {
        path: oci_image_lock._capture_private_fingerprint(path)
        for path in input_paths
    }

    def require_inputs_unchanged() -> None:
        for path, fingerprint in fingerprints.items():
            oci_image_lock._assert_private_unchanged(path, fingerprint)

    manifests = finalize_admitted._load_manifests(args.manifest)
    _validate_verified_manifests(manifests)
    semantic_admissions = finalize_admitted._load_semantic_admissions(
        args.semantic_admission_manifest
    )
    evidence = finalize_admitted._load_materialization_evidence(
        args.materialization_evidence,
        manifests=manifests,
        semantic_admissions=semantic_admissions,
    )
    materialized = finalize_admitted._load_materialized_tasks(
        args.tasks_dir,
        manifests,
        evidence=evidence,
    )
    if not materialized:
        raise ValueError(f"no production Harbor tasks found under {args.tasks_dir}")

    rows, task_ids = _validated_rows(
        args.candidate,
        manifests=manifests,
        materialized=materialized,
        allow_subset=args.allow_subset,
    )
    require_inputs_unchanged()
    temporary_output, output_sha256 = finalize_admitted._stage_private_jsonl(
        args.output,
        rows(),
    )
    task_ids_content = "".join(f"{instance_id}\n" for instance_id in task_ids)
    try:
        temporary_task_ids, task_ids_sha256 = finalize_admitted._stage_private_text(
            args.task_ids_output,
            task_ids_content,
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
        "schema_version": "miles-swebench-verified-hardened-local-evaluation-v1",
        "artifact_stage": "hardened-local-environment-admitted-evaluation",
        "environment_admitted": True,
        "evaluation_only": True,
        "official_comparable": False,
        "score_semantics": materialize_module._SWEBENCH_HARDENED_SCORE_SEMANTICS,
        "source_dataset": _DATASET_ID,
        "source_revision": materialize_module._SWEBENCH_VERIFIED_DATASET_REVISION,
        "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
        "harness_repository": materialize_module._SWEBENCH_HARNESS_REPOSITORY,
        "harness_commit": materialize_module._SWEBENCH_HARNESS_COMMIT,
        "harness_version": materialize_module._SWEBENCH_HARNESS_VERSION,
        "harness_files_sha256": {
            "constants.py": materialize_module._SWEBENCH_CONSTANTS_SHA256,
            "grading.py": materialize_module._SWEBENCH_GRADING_SHA256,
            "log_parsers.py": materialize_module._SWEBENCH_LOG_PARSERS_SHA256,
        },
        "candidate": str(args.candidate),
        "private_manifest": str(args.manifest),
        "materialization_evidence": str(args.materialization_evidence),
        "semantic_admission_manifests": [
            str(path) for path in args.semantic_admission_manifest
        ],
        "tasks_dir": str(args.tasks_dir),
        "output": str(args.output),
        "output_sha256": output_sha256,
        "rows": len(task_ids),
        "unique_tasks": len(task_ids),
        "task_ids_output": str(args.task_ids_output),
        "task_ids_sha256": task_ids_sha256,
        "task_set_sha256": finalize_admitted._stable_digest(task_ids),
        "task_binding_sha256": finalize_admitted._stable_digest(
            [
                [instance_id, manifests[instance_id]["task_digest"]]
                for instance_id in task_ids
            ]
        ),
        "task_runtime_sha256": finalize_admitted._stable_digest(
            [
                [
                    instance_id,
                    manifests[instance_id]["task_digest"],
                    evidence[instance_id]["task_tree_sha256"],
                ]
                for instance_id in task_ids
            ]
        ),
        "allow_subset": bool(args.allow_subset),
        "model_path_policy_scope": (
            "hardened-local evaluation: deny tests, configuration, CI, and toolchain "
            "paths; allow repository implementation paths"
        ),
        "model_path_policy_schema": (
            materialize_module._EVAL_MODEL_PATH_POLICY_SCHEMA
        ),
    }
    finalize_admitted._write_private_json(args.summary, summary)
    return summary


def _validate_verified_manifests(manifests: dict[str, dict[str, Any]]) -> None:
    for instance_id, manifest in manifests.items():
        if (
            manifest.get("source_schema") != _SOURCE_SCHEMA
            or manifest.get("source_dataset") != _DATASET_ID
            or manifest.get("eval_only") is not True
            or manifest.get("source_metadata", {}).get("split") != "test"
        ):
            raise ValueError(
                f"private manifest is not official eval-only SWE-bench Verified: "
                f"{instance_id}"
            )


def _validated_rows(
    candidate: Path,
    *,
    manifests: dict[str, dict[str, Any]],
    materialized: dict[str, Path],
    allow_subset: bool,
) -> tuple[Callable[[], Iterator[dict[str, Any]]], list[str]]:
    candidate_ids: set[str] = set()
    selected_ids: set[str] = set()

    def rows(*, collect: bool) -> Iterator[dict[str, Any]]:
        seen: set[str] = set()
        for row in finalize_admitted._read_jsonl(
            candidate,
            name="evaluation candidate dataset",
        ):
            instance_id, task_digest = finalize_admitted._validate_public_row(
                row,
                expected_eval_only=True,
            )
            if instance_id in seen:
                raise ValueError(f"duplicate SWE evaluation task: {instance_id}")
            seen.add(instance_id)
            if collect:
                candidate_ids.add(instance_id)
            manifest = manifests.get(instance_id)
            if manifest is None:
                raise ValueError(
                    f"evaluation candidate has no private manifest: {instance_id}"
                )
            _validate_row_binding(row, manifest, task_digest=task_digest)
            if instance_id not in materialized:
                continue
            if collect:
                selected_ids.add(instance_id)
            yield row

    for _ in rows(collect=True):
        pass
    extra = sorted(set(materialized) - candidate_ids)
    if extra:
        raise ValueError(
            "materialized evaluation tasks are absent from the candidate dataset: "
            f"{extra[:5]}"
        )
    missing = sorted(candidate_ids - selected_ids)
    if missing and not allow_subset:
        raise ValueError(
            f"{len(missing)} Verified tasks were not materialized; rerun admission or "
            "pass --allow-subset explicitly"
        )
    if not selected_ids:
        raise ValueError("no Verified rows matched production Harbor tasks")

    def emit() -> Iterator[dict[str, Any]]:
        yield from rows(collect=False)

    return emit, sorted(selected_ids)


def _validate_row_binding(
    row: dict[str, Any],
    manifest: dict[str, Any],
    *,
    task_digest: str,
) -> None:
    instance_id = manifest["instance_id"]
    swe_task = row["metadata"]["swe_task"]
    if (
        task_digest != manifest["task_digest"]
        or row["prompt"] != manifest["problem_statement"]
        or row["metadata"]["source"] != _DATASET_ID
        or swe_task.get("source_dataset") != _DATASET_ID
        or swe_task.get("source_schema") != _SOURCE_SCHEMA
        or swe_task.get("task_id") != instance_id
    ):
        raise ValueError(f"evaluation row/private task binding mismatch: {instance_id}")


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
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = finalize(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
