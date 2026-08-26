"""Prepare gold-free Miles SWE rows and private Harbor task manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import expand_paths, read_rows
from experiments.src.datasets.swe.schema import normalize_swe_row

_SOURCE_LOCK_PATH = (
    Path(__file__).resolve().parents[3] / "setup" / "manifests" / "swe_sources.lock.json"
)
_SOURCE_LOCK_SHA256 = "beef58a52af263c9bf883112b1d558ae22e8e096688ba32101fbb4b7b70fbc63"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.resolve() == args.task_manifest.resolve():
        raise ValueError("--output and --task-manifest must be different paths")
    r2e_id_map_path = getattr(args, "r2e_id_map", None)
    if r2e_id_map_path is not None and r2e_id_map_path.resolve() in {
        args.output.resolve(),
        args.task_manifest.resolve(),
    }:
        raise ValueError("--r2e-id-map must be distinct from output paths")
    invalid_report_path = getattr(args, "invalid_report", None)
    if args.on_invalid == "quarantine" and invalid_report_path is None:
        raise ValueError("--on-invalid quarantine requires an owner-only --invalid-report")
    if args.on_invalid == "error" and invalid_report_path is not None:
        raise ValueError("--invalid-report is only valid with --on-invalid quarantine")
    if invalid_report_path is not None and invalid_report_path.resolve() in {
        args.output.resolve(),
        args.task_manifest.resolve(),
        r2e_id_map_path.resolve() if r2e_id_map_path is not None else None,
    }:
        raise ValueError("--invalid-report must be distinct from every output path")
    source_name = getattr(args, "source_name", None)
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("--source-name is required and must select the pinned SWE source lock")
    source_lock, source_spec = _load_source_spec(source_name)
    _validate_source_arguments(args, source_spec)
    paths = expand_paths(args.input)
    artifact_digest = _validate_source_artifacts(paths, source_spec)
    downstream_holdout = _load_downstream_holdout(
        getattr(args, "downstream_holdout", None),
        source_lock,
        usage=args.usage,
    )
    r2e_id_map = _load_r2e_id_map(r2e_id_map_path)
    persisted_bindings = _load_task_bindings(args.task_manifest)
    output_partial = _partial_path(args.output)
    manifest_partial = _partial_path(args.task_manifest)
    id_map_partial = _partial_path(r2e_id_map_path) if r2e_id_map_path is not None else None
    invalid_report_partial = (
        _partial_path(invalid_report_path) if invalid_report_path is not None else None
    )
    seen_tasks: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    row_count = manifest_count = duplicate_count = invalid_count = excluded_count = 0
    holdout_excluded_count = 0
    try:
        with _open_private_new(output_partial) as output_handle, _open_private_new(
            manifest_partial
        ) as manifest_handle, _open_optional_private_new(
            invalid_report_partial
        ) as invalid_report_handle:
            for source_index, row in enumerate(read_rows(paths)):
                if args.limit is not None and row_count >= args.limit:
                    break
                try:
                    task = normalize_swe_row(
                        row,
                        dataset_id=args.dataset_id,
                        usage=args.usage,
                    )
                except ValueError as exc:
                    if args.on_invalid == "error":
                        raise ValueError(f"invalid source row {source_index}: {exc}") from exc
                    reason = _invalid_reason_code(str(exc))
                    invalid_reasons[reason] += 1
                    _write_jsonl(
                        invalid_report_handle,
                        {
                            "schema_version": "miles-swe-invalid-row-v1",
                            "source_index": source_index,
                            "source_locator_sha256": _source_locator_sha256(
                                artifact_digest,
                                source_index,
                            ),
                            "reason": reason,
                            "detail": str(exc),
                        },
                    )
                    invalid_count += 1
                    continue
                _validate_task_provenance(task, source_spec)
                if _task_overlaps_holdout(task, downstream_holdout):
                    holdout_excluded_count += 1
                    continue
                include_schema = getattr(args, "include_schema", None)
                if include_schema and task.source_schema not in include_schema:
                    excluded_count += 1
                    continue
                if task.source_schema == "r2e-gym-v1":
                    if r2e_id_map_path is None:
                        raise ValueError(
                            "R2E-Gym preparation requires a private --r2e-id-map so random "
                            "routing IDs remain stable across reruns"
                        )
                    map_key = _r2e_map_key(task)
                    source_content_digest = _r2e_source_content_digest(task)
                    existing_route = r2e_id_map.get(map_key)
                    if existing_route is None:
                        existing_ids = {route[0] for route in r2e_id_map.values()}
                        existing_bindings = {route[1] for route in r2e_id_map.values()}
                        if task.instance_id in existing_ids or task.task_binding in existing_bindings:
                            raise ValueError("random R2E routing ID collision")
                        r2e_id_map[map_key] = (
                            task.instance_id,
                            task.task_binding,
                            source_content_digest,
                        )
                    else:
                        if existing_route[2] != source_content_digest:
                            raise ValueError(
                                "R2E source identity maps to conflicting task contents; "
                                "the problem, verifier, image, or source metadata changed"
                            )
                        task = replace(
                            task,
                            instance_id=existing_route[0],
                            task_binding=existing_route[1],
                        )
                manifest = task.to_task_manifest()
                content_digest = str(manifest["content_digest"])
                persisted_binding = persisted_bindings.get(content_digest)
                if persisted_binding is not None and task.source_schema != "r2e-gym-v1":
                    task = replace(task, task_binding=persisted_binding)
                    manifest = task.to_task_manifest()
                else:
                    persisted_bindings[content_digest] = task.task_binding
                previous_digest = seen_tasks.get(task.instance_id)
                if previous_digest is not None and previous_digest != content_digest:
                    raise ValueError(
                        f"instance_id {task.instance_id!r} maps to conflicting task manifests"
                    )
                if previous_digest is None:
                    _write_jsonl(manifest_handle, manifest)
                    seen_tasks[task.instance_id] = content_digest
                    manifest_count += 1
                else:
                    duplicate_count += 1
                if args.usage == "eval" and previous_digest is not None:
                    continue
                _write_jsonl(output_handle, task.to_miles_row(agent_name=args.agent_name))
                source_counts[task.source_dataset] += 1
                row_count += 1
        if invalid_report_partial is not None:
            os.replace(invalid_report_partial, invalid_report_path)
        if id_map_partial is not None:
            _write_r2e_id_map(id_map_partial, r2e_id_map)
            os.replace(id_map_partial, r2e_id_map_path)
        os.replace(output_partial, args.output)
        os.replace(manifest_partial, args.task_manifest)
    except Exception:
        output_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        if id_map_partial is not None:
            id_map_partial.unlink(missing_ok=True)
        if invalid_report_partial is not None:
            invalid_report_partial.unlink(missing_ok=True)
        raise
    invalid_report_digest = (
        _file_sha256(invalid_report_path) if invalid_report_path is not None else None
    )
    return {
        "inputs": [str(path) for path in paths],
        "output": str(args.output),
        "task_manifest": str(args.task_manifest),
        "usage": args.usage,
        "source_name": source_name,
        "source_dataset": source_spec["dataset_id"],
        "source_revision": source_spec["revision"],
        "source_lock_sha256": _SOURCE_LOCK_SHA256,
        "source_artifact_digest": artifact_digest,
        "rows": row_count,
        "unique_tasks": manifest_count,
        "duplicate_rows": duplicate_count,
        "invalid_rows": invalid_count,
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "invalid_report": str(invalid_report_path) if invalid_report_path is not None else None,
        "invalid_report_sha256": invalid_report_digest,
        "excluded_rows": excluded_count,
        "downstream_holdout_excluded_rows": holdout_excluded_count,
        "r2e_id_map": str(r2e_id_map_path) if r2e_id_map_path is not None else None,
        "sources": dict(sorted(source_counts.items())),
        "artifact_stage": "schema-normalized",
        "environment_admitted": False,
    }


def _partial_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".partial")


def _load_source_spec(source_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _SOURCE_LOCK_PATH.is_file() or _SOURCE_LOCK_PATH.is_symlink():
        raise ValueError(f"pinned SWE source lock is missing: {_SOURCE_LOCK_PATH}")
    actual_digest = _file_sha256(_SOURCE_LOCK_PATH)
    if actual_digest != _SOURCE_LOCK_SHA256:
        raise ValueError(
            "pinned SWE source lock checksum mismatch: "
            f"expected {_SOURCE_LOCK_SHA256}, got {actual_digest}"
        )
    value = json.loads(_SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "miles-swe-source-lock-v1":
        raise ValueError("invalid pinned SWE source lock schema")
    sources = value.get("sources")
    if not isinstance(sources, dict) or source_name not in sources:
        raise ValueError(f"unknown pinned SWE source: {source_name!r}")
    source_spec = sources[source_name]
    if not isinstance(source_spec, dict):
        raise ValueError(f"invalid pinned SWE source specification: {source_name!r}")
    return value, source_spec


def _validate_source_arguments(args: argparse.Namespace, source_spec: dict[str, Any]) -> None:
    dataset_id = source_spec.get("dataset_id")
    row_sources = source_spec.get("row_source_datasets")
    schemas = source_spec.get("schemas")
    splits = source_spec.get("splits")
    if (
        not isinstance(dataset_id, str)
        or not isinstance(row_sources, list)
        or not row_sources
        or any(not isinstance(value, str) for value in row_sources)
        or not isinstance(schemas, list)
        or not schemas
        or any(not isinstance(value, str) for value in schemas)
        or not isinstance(splits, list)
        or not splits
        or any(not isinstance(value, str) for value in splits)
    ):
        raise ValueError("pinned SWE source specification is incomplete")
    if args.usage != source_spec.get("usage"):
        raise ValueError(
            f"pinned source {dataset_id} permits usage={source_spec.get('usage')!r}, "
            f"not {args.usage!r}"
        )
    expected_dataset_argument = dataset_id if dataset_id in row_sources else None
    if args.dataset_id != expected_dataset_argument:
        raise ValueError(
            f"--dataset-id must be {expected_dataset_argument!r} for pinned source {dataset_id}"
        )
    include_schema = getattr(args, "include_schema", None)
    if include_schema and not set(include_schema).issubset(set(schemas)):
        raise ValueError(
            f"requested schemas {include_schema!r} are not permitted for pinned source {dataset_id}"
        )


def _validate_source_artifacts(
    paths: list[Path],
    source_spec: dict[str, Any],
) -> str:
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("pinned SWE inputs must be regular, non-symlink files")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("pinned SWE inputs contain duplicate basenames")
    actual_names = {path.name for path in paths}
    artifacts = source_spec.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("pinned SWE source has no artifact groups")
    selected_files: list[dict[str, Any]] | None = None
    for artifact in artifacts:
        files = artifact.get("files") if isinstance(artifact, dict) else None
        if not isinstance(files, list) or not files:
            raise ValueError("pinned SWE artifact group has no files")
        expected_names = {
            value.get("name") for value in files if isinstance(value, dict)
        }
        if len(expected_names) != len(files) or any(not isinstance(name, str) for name in expected_names):
            raise ValueError("pinned SWE artifact group has invalid filenames")
        if expected_names == actual_names:
            selected_files = files
            break
    if selected_files is None:
        raise ValueError(
            f"input file set is not an accepted artifact for {source_spec.get('dataset_id')}: "
            f"{sorted(actual_names)}"
        )
    expected_by_name = {value["name"]: value for value in selected_files}
    verified: list[tuple[str, str]] = []
    for path in sorted(paths, key=lambda value: value.name):
        expected = expected_by_name[path.name]
        if path.stat().st_size != expected.get("size"):
            raise ValueError(f"pinned SWE input size mismatch: {path}")
        digest = _file_sha256(path)
        if digest != expected.get("sha256"):
            raise ValueError(f"pinned SWE input checksum mismatch: {path}")
        verified.append((path.name, digest))
    encoded = json.dumps(verified, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_downstream_holdout(
    path: Path | None,
    source_lock: dict[str, Any],
    *,
    usage: str,
) -> tuple[set[str], set[tuple[str, str]]]:
    if usage == "eval":
        if path is not None:
            raise ValueError("--downstream-holdout is only valid for training preparation")
        return set(), set()
    if path is None:
        raise ValueError("training SWE preparation requires --downstream-holdout")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"downstream holdout must be a regular file: {path}")
    holdout = source_lock.get("downstream_holdout")
    file_spec = holdout.get("file") if isinstance(holdout, dict) else None
    if not isinstance(file_spec, dict):
        raise ValueError("pinned SWE source lock has no downstream holdout")
    if path.name != file_spec.get("name"):
        raise ValueError(f"unexpected downstream holdout filename: {path.name}")
    if path.stat().st_size != file_spec.get("size") or _file_sha256(path) != file_spec.get(
        "sha256"
    ):
        raise ValueError("downstream holdout payload does not match the pinned benchmark")
    instance_ids: set[str] = set()
    repo_bases: set[tuple[str, str]] = set()
    row_count = 0
    for row in read_rows([path]):
        row_count += 1
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("downstream holdout row has no instance_id")
        if instance_id in instance_ids:
            raise ValueError(f"duplicate downstream holdout instance_id: {instance_id}")
        instance_ids.add(instance_id)
        repo_base = _normalized_repo_base(row.get("repo"), row.get("base_commit"))
        if repo_base is None:
            raise ValueError(
                f"downstream holdout row {instance_id!r} has no valid repo/base_commit identity"
            )
        repo_bases.add(repo_base)
    if row_count != holdout.get("rows"):
        raise ValueError(
            f"downstream holdout row count mismatch: expected {holdout.get('rows')}, "
            f"got {row_count}"
        )
    return instance_ids, repo_bases


def _validate_task_provenance(task, source_spec: dict[str, Any]) -> None:
    if task.source_dataset not in source_spec["row_source_datasets"]:
        raise ValueError(
            f"normalized task claims unpinned source dataset {task.source_dataset!r}"
        )
    if task.source_schema not in source_spec["schemas"]:
        raise ValueError(f"normalized task has unpinned schema {task.source_schema!r}")
    split = task.source_metadata.get("split")
    if split not in source_spec["splits"]:
        raise ValueError(
            f"task {task.instance_id!r} has forbidden split {split!r}; "
            f"allowed={source_spec['splits']!r}"
        )


def _task_overlaps_holdout(
    task,
    holdout: tuple[set[str], set[tuple[str, str]]],
) -> bool:
    holdout_instance_ids, holdout_repo_bases = holdout
    if not holdout_instance_ids and not holdout_repo_bases:
        return False
    candidate_ids = {task.instance_id}
    published_id = task.source_metadata.get("published_instance_id")
    if isinstance(published_id, str):
        candidate_ids.add(published_id)
    if not candidate_ids.isdisjoint(holdout_instance_ids):
        return True
    repo_base = _normalized_repo_base(task.repo, task.base_commit)
    return repo_base is not None and repo_base in holdout_repo_bases


def _normalized_repo_base(repo: Any, base_commit: Any) -> tuple[str, str] | None:
    if not isinstance(repo, str) or not isinstance(base_commit, str):
        return None
    normalized_repo = repo.strip().lower()
    for prefix in (
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
    ):
        if normalized_repo.startswith(prefix):
            normalized_repo = normalized_repo[len(prefix) :]
            break
    normalized_repo = normalized_repo.rstrip("/")
    if normalized_repo.endswith(".git"):
        normalized_repo = normalized_repo[:-4]
    normalized_commit = base_commit.strip().lower()
    if (
        not normalized_repo
        or normalized_repo.count("/") != 1
        or re.fullmatch(r"[0-9a-f]{40}", normalized_commit) is None
    ):
        return None
    return normalized_repo, normalized_commit


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _open_private_new(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


class _NullWriter:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        return None

    def write(self, _value: str) -> None:
        raise RuntimeError("invalid rows cannot be discarded without a quarantine report")


def _open_optional_private_new(path: Path | None):
    return _NullWriter() if path is None else _open_private_new(path)


def _write_private_text(path: Path, value: str) -> None:
    partial = _partial_path(path)
    try:
        with _open_private_new(partial) as handle:
            handle.write(value)
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _source_locator_sha256(artifact_digest: str, source_index: int) -> str:
    encoded = json.dumps(
        {
            "source_artifact_digest": artifact_digest,
            "source_index": source_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _invalid_reason_code(detail: str) -> str:
    if detail.startswith("required SWE field "):
        return "missing-required-field"
    if detail.startswith("required SWE commit field "):
        return "missing-required-commit"
    if "must be a full 40-character hexadecimal commit" in detail:
        return "invalid-commit"
    if "requires install_config.test_cmd and log_parser" in detail:
        return "incomplete-install-config"
    if "requires install_config" in detail:
        return "missing-install-config"
    if "has no FAIL_TO_PASS tests" in detail:
        return "missing-fail-to-pass-tests"
    if "has no expected test outcomes" in detail:
        return "missing-r2e-expected-outcomes"
    if "unknown test statuses" in detail:
        return "invalid-r2e-test-status"
    if "invalid JSON" in detail:
        return "invalid-json"
    if detail.startswith("cannot identify executable SWE schema"):
        return "unsupported-schema"
    if detail.startswith("Nemotron SWE row is not a full environment task"):
        return "unsupported-nemotron-task"
    return "normalization-error"


def _load_r2e_id_map(path: Path | None) -> dict[str, tuple[str, str, str]]:
    if path is None or not path.exists():
        return {}
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"R2E ID map must be a regular file: {path}")
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"R2E ID map must be owner-only (chmod 600): {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "miles-r2e-id-map-v3":
        raise ValueError(f"invalid R2E ID map schema: {path}")
    routes = value.get("routes")
    if not isinstance(routes, dict):
        raise ValueError(f"invalid R2E ID routes: {path}")
    result: dict[str, tuple[str, str, str]] = {}
    for fingerprint, route in routes.items():
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            raise ValueError(f"invalid private R2E fingerprint in {path}")
        if not isinstance(route, dict):
            raise ValueError(f"invalid private R2E route in {path}")
        routing_id = route.get("instance_id")
        task_binding = route.get("task_digest")
        source_content_digest = route.get("source_content_digest")
        if not isinstance(routing_id, str) or re.fullmatch(r"r2e-[0-9a-f]{32}", routing_id) is None:
            raise ValueError(f"invalid random R2E routing ID in {path}")
        if not isinstance(task_binding, str) or re.fullmatch(r"[0-9a-f]{64}", task_binding) is None:
            raise ValueError(f"invalid random R2E task binding in {path}")
        if (
            not isinstance(source_content_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_content_digest) is None
        ):
            raise ValueError(f"invalid private R2E source content digest in {path}")
        result[fingerprint] = (routing_id, task_binding, source_content_digest)
    if len({route[0] for route in result.values()}) != len(result):
        raise ValueError(f"R2E ID map contains duplicate routing IDs: {path}")
    if len({route[1] for route in result.values()}) != len(result):
        raise ValueError(f"R2E ID map contains duplicate task bindings: {path}")
    return result


def _r2e_map_key(task) -> str:
    gold_commit = task.verifier.get("gold_commit")
    if not isinstance(gold_commit, str):
        raise ValueError("R2E task has no private gold commit for ID mapping")
    encoded = json.dumps(
        [task.source_dataset, task.repo, gold_commit],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _r2e_source_content_digest(task) -> str:
    value = {
        "schema_version": "miles-r2e-source-identity-v1",
        "source_dataset": task.source_dataset,
        "source_schema": task.source_schema,
        "repo": task.repo,
        "problem_statement": task.problem_statement,
        "base_commit": task.base_commit,
        "source_image": task.source_image,
        "oracle_patch": task.oracle_patch,
        "verifier": task.verifier,
        "source_metadata": task.source_metadata,
        "eval_only": task.eval_only,
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_r2e_id_map(
    path: Path,
    routes: dict[str, tuple[str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "miles-r2e-id-map-v3",
        "routes": {
            fingerprint: {
                "instance_id": route[0],
                "task_digest": route[1],
                "source_content_digest": route[2],
            }
            for fingerprint, route in sorted(routes.items())
        },
    }
    with _open_private_new(path) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_task_bindings(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o077:
        raise PermissionError(f"existing private task manifest is not owner-only: {path}")
    bindings: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            content_digest = value.get("content_digest") if isinstance(value, dict) else None
            task_digest = value.get("task_digest") if isinstance(value, dict) else None
            if (
                isinstance(content_digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", content_digest)
                and isinstance(task_digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", task_digest)
            ):
                previous = bindings.get(content_digest)
                if previous is not None and previous != task_digest:
                    raise ValueError(f"conflicting persisted task bindings in {path}")
                bindings[content_digest] = task_digest
    return bindings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="input JSONL/parquet paths or globs")
    parser.add_argument(
        "--source-name",
        required=True,
        help="entry in the checksum-pinned experiments/setup SWE source lock",
    )
    parser.add_argument("--output", type=Path, required=True, help="gold-free Miles prompt JSONL")
    parser.add_argument(
        "--task-manifest",
        type=Path,
        required=True,
        help="private Harbor materialization manifest; never use as prompt data",
    )
    parser.add_argument(
        "--dataset-id",
        help="official source dataset ID; required when raw rows omit dataset_name",
    )
    parser.add_argument("--usage", choices=("train", "eval"), required=True)
    parser.add_argument(
        "--downstream-holdout",
        type=Path,
        help="pinned SWE-bench Verified parquet; mandatory for training overlap exclusion",
    )
    parser.add_argument("--agent-name", default="terminus-2")
    parser.add_argument(
        "--r2e-id-map",
        type=Path,
        help="private persisted random routing-ID map; required for R2E-Gym",
    )
    parser.add_argument(
        "--include-schema",
        action="append",
        choices=("r2e-gym-v1", "swe-rebench-v2", "swe-gym", "swebench"),
        help="emit only these normalized schemas; repeat to select more than one",
    )
    parser.add_argument("--on-invalid", choices=("error", "quarantine"), default="error")
    parser.add_argument(
        "--invalid-report",
        type=Path,
        help=(
            "owner-only JSONL evidence for quarantined source rows; required with "
            "--on-invalid quarantine"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = _parse_args()
    summary = prepare(args)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.summary:
        _write_private_text(args.summary, rendered + "\n")


if __name__ == "__main__":
    main()
