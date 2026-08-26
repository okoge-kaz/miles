"""Materialize private SWE manifests as isolated Harbor task directories.

The agent image contains only a pruned repository base. Harbor collects the
agent's patch after the agent exits. A separate verifier starts directly from
the same admitted immutable source image with no network; only then does the
pinned Harbor overlay atomically upload the private ``tests/`` package.

Every production environment requires owner-only live-admission records for
the exact enriched private task and immutable image. Raw mutable-image rows are
never promoted by this command; the admission producer must emit a separately
bound admitted private manifest first. SWE-bench Verified is evaluation-only
and uses a separately finalized hardened-local artifact.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments.src.datasets.swe.schema import SCHEMA_VERSION
from experiments.src.environments.swe import oci_image_lock
from experiments.src.environments.swe import timeouts

_ASSET_DIR = Path(__file__).with_name("templates")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}")
_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,239}")
_SUPPORTED = {"r2e-gym-v1", "swe-gym", "swe-rebench-v2", "swebench"}
_NPM_RUNTIME_POLICY = "npm-node-modules-v2"
_PYTHON_EDITABLE_RUNTIME_POLICY = "python-editable-metadata-v1"
_NO_RUNTIME_POLICY = "none"
_INVALID_PATCH = "MILES_SWE_AGENT_STATE_INVALID"
_LATE_TESTS_MARKER = ".harbor-e2b-late-tests"
_MODEL_PATH_POLICY_SCHEMA = "miles-swe-model-path-policy-v1"
_EVAL_MODEL_PATH_POLICY_SCHEMA = "miles-swe-model-path-policy-v2"
_DENIED_MODEL_BASENAMES = {
    ".gitattributes",
    ".gitmodules",
    "dockerfile",
    "makefile",
    "conftest.py",
    "environment.yml",
    "package-lock.json",
    "package.json",
    "poetry.lock",
    "pyproject.toml",
    "pytest.ini",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "sitecustomize.py",
    "tox.ini",
    "usercustomize.py",
    "uv.lock",
}
_DENIED_MODEL_COMPONENTS = {
    ".circleci",
    ".devcontainer",
    ".github",
    ".gitlab",
    "test",
    "tests",
    "testing",
}
_R2E_PARSER_COMMIT = "0d94c4eb9431cd195c55a7ea3abd54006c9a1735"
_R2E_PARSER_SHA256 = "395f637f4b8d68160948f95097f861506f123978852a9da77258aa3ba3fe1904"
_REBENCH_COMMIT = "c71902a8cf8d2b725f63d51f199f4d3e56f68d2d"
_REBENCH_LOG_PARSERS_SHA256 = "a717b03efde1cb79dfb11e2a57d0262c0057d352a347a9fb09667ef6e5f6f20c"
_REBENCH_CONSTANTS_SHA256 = "823dd1ef512d363ed5d4dce05d70f22d7f93b25722cda5b0971f17010f5168a5"
_REBENCH_EVAL_SHA256 = "4768c0c3e2adf3540c2228f819f4b073e4665ada06fa00f2234a1f7620d69eda"
_SWE_GYM_HARNESS_COMMIT = "16dd480cce9b27bf111a362d280881c6def5d2a7"
_SWE_GYM_HARNESS_VERSION = "2.0.13"
_SWE_GYM_CONSTANTS_SHA256 = "5bd655172c9a9dfcb494d9c52bcbcc0d79a59cd20c39b8bbdc41ab8b0a9baf14"
_SWE_GYM_LOG_PARSERS_SHA256 = "6d1bce4088dc4d7cb614783dcd63eeb5c20478906158a82cc97ce0b886367b7c"
_SWE_GYM_GRADING_SHA256 = "51eb584b85ffbc245042332a080ff20ba2529c712e76db14da16f246b0e0675d"
_SWE_GYM_HARBOR_COMMIT = "2ce5ba2af33a00c9fba0463f6403313996373f85"
_SWE_GYM_HARBOR_ADAPTER_SHA256 = "26244e254bff3b4a363c73d3995fc6231aec8b4ffcec4b6d5e0723a507c993bc"
_SWE_GYM_DATASET_REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
_SWEBENCH_HARNESS_REPOSITORY = "SWE-bench/SWE-bench"
_SWEBENCH_HARNESS_COMMIT = "7336033d65d32ec62f9ce2419aa8f3a757b06ce2"
_SWEBENCH_HARNESS_VERSION = "2.0.13"
_SWEBENCH_CONSTANTS_SHA256 = (
    "6d189fcea0459897741eb241407b25e728467d3120f9feb8deaf2c61c030bc3e"
)
_SWEBENCH_LOG_PARSERS_SHA256 = (
    "43ce7f06a562177ef82126f547e40fe19c51874148aae6027760d8b49b74bc89"
)
_SWEBENCH_GRADING_SHA256 = (
    "c953793204d52a7ac67b197e0479987efcdf13f381f87b685d711b2e156e3ce3"
)
_SWEBENCH_HARDENED_SCORE_SEMANTICS = (
    "hardened-local-not-official-comparable-v1"
)
_SWEBENCH_VERIFIED_DATASET_REVISION = (
    "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
)
_MATERIALIZATION_EVIDENCE_SCHEMA = "miles-swe-materialization-evidence-v1"
_REPOSITORY_ADMISSION_CHECKS: dict[str, bool | int] = {
    "publisher_namespace_policy": True,
    "registry_digest_resolved": True,
    "agent_verifier_same_image_digest": True,
    "source_head_matches_base": True,
    "agent_history_scrubbed": True,
    "verifier_history_scrubbed": True,
    "hidden_test_patch_isolated": True,
    "late_private_verifier_upload": True,
    "model_path_policy_enforced": True,
    "safe_patch_policy_enforced": True,
    "official_test_command_parity": True,
    "official_exact_parser_parity": True,
    "empty_reward": 0,
    "oracle_reward": 1,
    "runtime_smoke": True,
    "tool_smoke": True,
    "no_new_privileges": True,
    "effective_capabilities_zero": True,
    "fresh_separate_verifier": True,
    "fresh_template_id_pinned": True,
    "agent_public_network_blocked": True,
    "verifier_public_network_blocked": True,
    "tracked_gold_history_absent": True,
}
_REPOSITORY_ADMISSION_BASE_FIELDS = {
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


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    """Build task trees atomically from a private, owner-only manifest."""
    _validate_private_input(args.manifest)
    evidence_path = getattr(args, "admission_evidence", None)
    if evidence_path is not None:
        if (
            getattr(args, "allow_mutable_images", False)
            or getattr(args, "allow_unadmitted_r2e_dry_run", False)
            or getattr(args, "allow_unadmitted_swe_rebench_dry_run", False)
            or getattr(args, "allow_unadmitted_swe_gym_dry_run", False)
            or getattr(
                args,
                "allow_unadmitted_swebench_verified_dry_run",
                False,
            )
        ):
            raise ValueError(
                "production admission evidence cannot be issued with dry-run flags"
            )
        evidence_resolved = evidence_path.resolve()
        output_resolved = args.output.resolve()
        if evidence_resolved == output_resolved or output_resolved in evidence_resolved.parents:
            raise ValueError("admission evidence must remain outside the Harbor task tree")
    r2e_admissions = _load_r2e_admissions(getattr(args, "r2e_admission_manifest", None))
    swe_rebench_admissions = _load_swe_rebench_admissions(
        getattr(args, "swe_rebench_admission_manifest", None)
    )
    swe_gym_admissions = _load_swe_gym_admissions(getattr(args, "swe_gym_admission_manifest", None))
    swebench_verified_admissions = _load_swebench_verified_admissions(
        getattr(args, "swebench_verified_admission_manifest", None)
    )
    r2e_parser = _validate_dependency(
        getattr(args, "r2e_execution_log_parser", None),
        expected_sha256=_R2E_PARSER_SHA256,
        name="official R2E-Gym execution-log parser",
    )
    official_parsers = _validate_dependency(
        getattr(args, "swe_rebench_log_parsers", None),
        expected_sha256=_REBENCH_LOG_PARSERS_SHA256,
        name="official SWE-rebench log parser",
    )
    official_constants = _validate_dependency(
        getattr(args, "swe_rebench_constants", None),
        expected_sha256=_REBENCH_CONSTANTS_SHA256,
        name="official SWE-rebench status definitions",
    )
    official_rebench_eval = _validate_dependency(
        getattr(args, "swe_rebench_eval", None),
        expected_sha256=_REBENCH_EVAL_SHA256,
        name="official SWE-rebench evaluator",
    )
    swe_gym_harness = _validate_swe_gym_harness(getattr(args, "swe_gym_harness_root", None))
    swebench_harness = _validate_swebench_harness(
        getattr(args, "swebench_harness_root", None)
    )
    _ensure_private_directory(args.output)
    count = 0
    schema_counts: dict[str, int] = {}
    task_ids: set[str] = set()
    evidence_rows: list[dict[str, Any]] = []
    for manifest in _read_manifest(args.manifest):
        if args.limit is not None and count >= args.limit:
            break
        validated = _validate_manifest(
            manifest,
            allow_mutable_images=getattr(args, "allow_mutable_images", False),
            allow_unadmitted_r2e=getattr(args, "allow_unadmitted_r2e_dry_run", False),
            r2e_admissions=r2e_admissions,
            allow_unadmitted_swe_rebench=getattr(
                args,
                "allow_unadmitted_swe_rebench_dry_run",
                False,
            ),
            swe_rebench_admissions=swe_rebench_admissions,
            allow_unadmitted_swe_gym=getattr(
                args,
                "allow_unadmitted_swe_gym_dry_run",
                False,
            ),
            swe_gym_admissions=swe_gym_admissions,
            allow_unadmitted_swebench_verified=getattr(
                args,
                "allow_unadmitted_swebench_verified_dry_run",
                False,
            ),
            swebench_verified_admissions=swebench_verified_admissions,
        )
        semantic_admission = None
        if evidence_path is not None:
            semantic_admission = _semantic_admission(
                validated,
                r2e_admissions=r2e_admissions,
                swe_rebench_admissions=swe_rebench_admissions,
                swe_gym_admissions=swe_gym_admissions,
                swebench_verified_admissions=swebench_verified_admissions,
            )
        instance_id = validated["instance_id"]
        if instance_id in task_ids:
            raise ValueError(f"duplicate private task manifest for {instance_id!r}")
        target = args.output / instance_id
        if target.exists():
            raise FileExistsError(f"refusing to replace existing Harbor task: {target}")
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{instance_id}.partial-", dir=args.output))
        try:
            _write_task(
                temp_dir,
                validated,
                r2e_parser=r2e_parser,
                official_parsers=official_parsers,
                official_constants=official_constants,
                official_rebench_eval=official_rebench_eval,
                swe_gym_harness=swe_gym_harness,
                swebench_harness=swebench_harness,
            )
            _seal_private_task_tree(temp_dir)
            os.replace(temp_dir, target)
        except BaseException as failure:
            try:
                _remove_private_tree(temp_dir)
            except BaseException as cleanup_failure:
                raise cleanup_failure from failure
            raise
        task_ids.add(instance_id)
        count += 1
        schema = validated["source_schema"]
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
        if evidence_path is not None:
            if semantic_admission is None:
                raise AssertionError("semantic admission unexpectedly missing")
            evidence_rows.append(
                _materialization_evidence(target, validated, semantic_admission)
            )
    if count == 0:
        raise ValueError(f"no task records found in {args.manifest}")
    summary = {
        "manifest": str(args.manifest),
        "output": str(args.output),
        "tasks": count,
        "schemas": dict(sorted(schema_counts.items())),
        "separate_verifier": True,
        "agent_network": "no-network",
        "verifier_network": "no-network",
        "immutable_images_required": not getattr(args, "allow_mutable_images", False),
        "r2e_admission_required": True,
        "swe_rebench_admission_required": True,
        "swe_gym_admission_required": True,
        "swebench_verified_admission_required": True,
        "swe_gym_image_inventory": "live digest admission required; stale 2396/38 claim rejected",
        "task_tree_modes": "directories=0500; files=0400 or executable=0500",
        "model_path_policy_scope": (
            "pilot training admission: oracle-touched implementation paths only; "
            "hardened Verified eval: deny sensitive paths; neither score is an "
            "unmodified official run_evaluation score"
        ),
    }
    if evidence_path is not None:
        _write_admission_evidence(evidence_path, evidence_rows)
        summary["admission_evidence"] = str(evidence_path)
        summary["admission_evidence_records"] = len(evidence_rows)
    return summary


def _ensure_private_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise PermissionError(f"private output directory is unsafe: {path}")
    else:
        path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)


def _seal_private_task_tree(task_dir: Path) -> None:
    """Seal a materialized task as an owner-readable, immutable input tree."""
    for path in [task_dir, *task_dir.rglob("*")]:
        metadata = path.lstat()
        is_directory = stat.S_ISDIR(metadata.st_mode)
        is_regular = stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
        if stat.S_ISLNK(metadata.st_mode) or not (is_directory or is_regular):
            raise ValueError(f"materialized task contains an unsafe entry: {path}")
        if metadata.st_uid != os.getuid():
            raise PermissionError(f"materialized task entry is owned by another user: {path}")
        if is_directory:
            sealed_mode = 0o500
        else:
            sealed_mode = 0o500 if metadata.st_mode & stat.S_IXUSR else 0o400
        path.chmod(sealed_mode)


def _remove_private_tree(root: Path) -> None:
    """Make owned directories deletable without following links, then remove them."""
    for current, directory_names, _ in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        metadata = current_path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise PermissionError(f"private tree contains an unsafe directory: {current_path}")
        current_path.chmod(0o700)
        for name in directory_names:
            child = current_path / name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode):
                continue
            if (
                not stat.S_ISDIR(child_metadata.st_mode)
                or child_metadata.st_uid != os.getuid()
            ):
                raise PermissionError(f"private tree contains an unsafe directory: {child}")
            child.chmod(0o700)
    shutil.rmtree(root)


def _read_manifest(path: Path) -> Iterable[dict[str, Any]]:
    before = os.lstat(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"private manifest changed before open: {path}")
        handle = os.fdopen(descriptor, encoding="utf-8")
        descriptor = -1
        with handle:
            try:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSON on {path}:{line_number}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"private manifest row {line_number} must be an object"
                        )
                    yield value
            finally:
                finished = os.fstat(handle.fileno())
                path_after = os.lstat(path)
                if (
                    (opened.st_dev, opened.st_ino, opened.st_size)
                    != (finished.st_dev, finished.st_ino, finished.st_size)
                    or (finished.st_dev, finished.st_ino)
                    != (path_after.st_dev, path_after.st_ino)
                    or opened.st_mtime_ns != finished.st_mtime_ns
                    or opened.st_ctime_ns != finished.st_ctime_ns
                ):
                    raise RuntimeError(
                        f"private manifest changed while reading: {path}"
                    )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_private_input(path: Path) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"private manifest must be a regular file: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"private manifest must not be hard-linked: {path}")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"private manifest must be owned by this user: {path}")
    if metadata.st_mode & 0o077:
        raise PermissionError(f"private manifest must not be group/world accessible (chmod 600): {path}")


def _validate_dependency(
    path: Path | None,
    *,
    expected_sha256: str | None,
    name: str,
) -> Path | None:
    if path is None:
        return None
    content = _read_stable_regular_file(path, name=name)
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(f"{name} checksum mismatch: expected {expected_sha256.lower()}, got {digest}")
    return path


def _validate_swe_gym_harness(root: Path | None) -> dict[str, Path] | None:
    if root is None:
        return None
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"official SWE-Gym harness root must be a real directory: {root}")
    harness = root / "swegym" / "harness"
    dependencies = {
        "constants": (harness / "constants.py", _SWE_GYM_CONSTANTS_SHA256),
        "log_parsers": (harness / "log_parsers.py", _SWE_GYM_LOG_PARSERS_SHA256),
        "grading": (harness / "grading.py", _SWE_GYM_GRADING_SHA256),
    }
    validated: dict[str, Path] = {}
    for name, (path, digest) in dependencies.items():
        result = _validate_dependency(
            path,
            expected_sha256=digest,
            name=f"official SWE-Gym {_SWE_GYM_HARNESS_VERSION} {name}",
        )
        if result is None:
            raise AssertionError("required SWE-Gym dependency unexpectedly resolved to None")
        validated[name] = result
    return validated


def _validate_swebench_harness(root: Path | None) -> dict[str, Path] | None:
    if root is None:
        return None
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"official SWE-bench harness root must be a real directory: {root}")
    harness = root / "swebench" / "harness"
    dependencies = {
        "constants": (harness / "constants.py", _SWEBENCH_CONSTANTS_SHA256),
        "log_parsers": (
            harness / "log_parsers.py",
            _SWEBENCH_LOG_PARSERS_SHA256,
        ),
        "grading": (harness / "grading.py", _SWEBENCH_GRADING_SHA256),
    }
    validated: dict[str, Path] = {}
    for name, (path, digest) in dependencies.items():
        result = _validate_dependency(
            path,
            expected_sha256=digest,
            name=f"official SWE-bench {_SWEBENCH_HARNESS_VERSION} {name}",
        )
        if result is None:
            raise AssertionError(
                "required official SWE-bench dependency unexpectedly resolved to None"
            )
        validated[name] = result
    return validated


def _validate_manifest(
    value: dict[str, Any],
    *,
    allow_mutable_images: bool,
    allow_unadmitted_r2e: bool,
    r2e_admissions: dict[tuple[str, str], dict[str, Any]],
    allow_unadmitted_swe_rebench: bool,
    swe_rebench_admissions: dict[tuple[str, str], dict[str, Any]],
    allow_unadmitted_swe_gym: bool,
    swe_gym_admissions: dict[tuple[str, str], dict[str, Any]],
    allow_unadmitted_swebench_verified: bool = False,
    swebench_verified_admissions: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported private manifest schema: {value.get('schema_version')!r}")
    instance_id = _required_string(value, "instance_id")
    if _INSTANCE_ID.fullmatch(instance_id) is None or instance_id in {".", ".."}:
        raise ValueError(f"unsafe Harbor instance_id: {instance_id!r}")
    source_schema = _required_string(value, "source_schema")
    if source_schema not in _SUPPORTED:
        raise ValueError(f"{source_schema!r} has no isolated local verifier; supported={sorted(_SUPPORTED)}")
    oci_image_lock.validate_task_image_policy(value)
    source_image = _required_string(_required_object(value, "sandbox"), "source_image")
    if _IMAGE.fullmatch(source_image) is None:
        raise ValueError(f"unsafe source image reference for {instance_id}: {source_image!r}")
    if not allow_mutable_images and re.search(r"@sha256:[0-9a-f]{64}$", source_image) is None:
        raise ValueError(f"source image for {instance_id} is mutable; resolve it to name@sha256:digest before production materialization (or use --allow-mutable-images only for dry-run)")
    base_commit = value.get("base_commit")
    if base_commit is not None and (not isinstance(base_commit, str) or _COMMIT.fullmatch(base_commit) is None):
        raise ValueError(f"invalid base_commit for {instance_id}")
    task_digest = _required_string(value, "task_digest")
    if _DIGEST.fullmatch(task_digest) is None:
        raise ValueError(f"invalid task_digest for {instance_id}")
    content_digest = _required_string(value, "content_digest")
    if _DIGEST.fullmatch(content_digest) is None:
        raise ValueError(f"invalid private content_digest for {instance_id}")
    digest_payload = dict(value)
    digest_payload.pop("task_digest", None)
    digest_payload.pop("content_digest", None)
    if _stable_digest(digest_payload) != content_digest:
        raise ValueError(f"private manifest content digest mismatch for {instance_id}")
    if source_schema == "r2e-gym-v1" and not allow_unadmitted_r2e:
        admission = r2e_admissions.get((content_digest, source_image))
        if admission is None:
            raise ValueError("R2E-Gym is not admitted for production: an owner-only admission record must bind this exact content_digest and immutable source image after live HEAD/parent, empty=0, oracle=1, runtime, and leakage checks")
        _validate_r2e_admission_binding(value, admission)
    if source_schema == "swe-rebench-v2" and not allow_unadmitted_swe_rebench:
        admission = swe_rebench_admissions.get((content_digest, source_image))
        if admission is None:
            raise ValueError(
                "SWE-ReBench-V2 is not admitted for production: an owner-only "
                "semantic record must bind the exact locked task after live "
                "empty=0/oracle=1, parser, isolation, and leakage checks"
            )
        _validate_repository_admission_binding(
            value,
            admission,
            name="SWE-ReBench-V2",
        )
    if source_schema == "swe-gym" and not allow_unadmitted_swe_gym:
        admission = swe_gym_admissions.get((content_digest, source_image))
        if admission is None or admission.get("instance_id") != instance_id:
            raise ValueError("SWE-Gym is not admitted for production: an owner-only admission record must bind this exact instance/content/image after registry digest, history, hidden-test, exact-parser, empty=0, and oracle=1 checks")
        if admission.get("task_digest") != task_digest:
            raise ValueError(f"SWE-Gym admission task binding mismatch for {instance_id}")
        _validate_repository_admission_binding(
            value,
            admission,
            name="SWE-Gym",
        )
    if source_schema == "swebench" and not allow_unadmitted_swebench_verified:
        admission = (swebench_verified_admissions or {}).get(
            (content_digest, source_image)
        )
        if admission is None or admission.get("instance_id") != instance_id:
            raise ValueError(
                "SWE-bench Verified is not admitted for evaluation: an owner-only "
                "admission record must bind this exact eval-only task after registry, "
                "empty=0, oracle=1, harness, isolation, and leakage checks"
            )
        if admission.get("task_digest") != task_digest:
            raise ValueError(
                f"SWE-bench Verified admission task binding mismatch for {instance_id}"
            )
        _validate_repository_admission_binding(
            value,
            admission,
            name="SWE-bench Verified",
        )
    problem_statement = _required_string(value, "problem_statement")
    verifier = _required_object(value, "verifier")
    solution = value.get("solution") or {}
    if not isinstance(solution, dict):
        raise ValueError(f"solution must be an object for {instance_id}")
    oracle_patch = solution.get("oracle_patch")
    if oracle_patch is not None and not isinstance(oracle_patch, str):
        raise ValueError(f"oracle_patch must be text or null for {instance_id}")
    return {
        **value,
        "instance_id": instance_id,
        "source_schema": source_schema,
        "source_image": source_image,
        "base_commit": base_commit,
        "task_digest": task_digest,
        "content_digest": content_digest,
        "problem_statement": problem_statement,
        "verifier": verifier,
        "oracle_patch": oracle_patch,
        "_private_manifest_record_sha256": _stable_digest(value),
    }


def _validate_r2e_admission_binding(
    manifest: dict[str, Any],
    admission: dict[str, Any],
) -> None:
    instance_id = _required_string(manifest, "instance_id")
    sandbox = _required_object(manifest, "sandbox")
    source_image = _required_string(sandbox, "source_image")
    task_digest = _required_string(manifest, "task_digest")
    content_digest = _required_string(manifest, "content_digest")
    if admission.get("instance_id") != instance_id:
        raise ValueError(f"R2E admission instance binding mismatch for {instance_id}")
    if admission.get("task_digest") != task_digest:
        raise ValueError(f"R2E admission task binding mismatch for {instance_id}")
    if admission.get("content_digest") != content_digest:
        raise ValueError(f"R2E admission content binding mismatch for {instance_id}")
    if admission.get("source_image") != source_image or admission.get(
        "source_image_resolved"
    ) != source_image:
        raise ValueError(f"R2E admission image binding mismatch for {instance_id}")
    base_commit = _required_string(manifest, "base_commit")
    if admission.get("base_commit") != base_commit:
        raise ValueError(f"R2E admission base binding mismatch for {instance_id}")
    solution = _required_object(manifest, "solution")
    oracle_patch = solution.get("oracle_patch")
    if (
        not isinstance(oracle_patch, str)
        or not oracle_patch.strip()
        or "\0" in oracle_patch
    ):
        raise ValueError("required text field 'oracle_patch' is missing or invalid")
    if admission.get("oracle_patch_sha256") != hashlib.sha256(
        oracle_patch.encode("utf-8")
    ).hexdigest():
        raise ValueError(f"R2E admission oracle binding mismatch for {instance_id}")

    image_lock = _required_object(sandbox, "image_lock")
    if image_lock.get("schema_version") != "miles-oci-image-lock-v1":
        raise ValueError(f"R2E admitted task has an invalid OCI lock for {instance_id}")
    requested = _required_string(image_lock, "source_image_requested")
    resolved = _required_string(image_lock, "source_image_resolved")
    input_digest = _required_string(image_lock, "input_content_digest")
    child_digest = _required_string(image_lock, "child_manifest_digest")
    if (
        _IMAGE.fullmatch(requested) is None
        or resolved != source_image
        or re.fullmatch(r"sha256:[0-9a-f]{64}", child_digest) is None
        or not source_image.endswith("@" + child_digest)
        or _DIGEST.fullmatch(input_digest) is None
        or image_lock.get("platform")
        != {"os": "linux", "architecture": "amd64"}
    ):
        raise ValueError(f"R2E admitted task OCI lock mismatch for {instance_id}")
    if admission.get("source_image_requested") != requested:
        raise ValueError(f"R2E admission requested-image mismatch for {instance_id}")
    if admission.get("input_content_digest") != input_digest:
        raise ValueError(f"R2E admission input binding mismatch for {instance_id}")


def _validate_repository_admission_binding(
    manifest: dict[str, Any],
    admission: dict[str, Any],
    *,
    name: str,
) -> None:
    instance_id = _required_string(manifest, "instance_id")
    sandbox = _required_object(manifest, "sandbox")
    source_image = _required_string(sandbox, "source_image")
    expected = {
        "instance_id": instance_id,
        "source_schema": _required_string(manifest, "source_schema"),
        "task_digest": _required_string(manifest, "task_digest"),
        "locked_content_digest": _required_string(manifest, "content_digest"),
        "content_digest": _required_string(manifest, "content_digest"),
        "source_image_resolved": source_image,
        "source_image": source_image,
        "base_commit": _required_string(manifest, "base_commit"),
    }
    if any(admission.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{name} semantic admission task binding mismatch: {instance_id}")
    solution = _required_object(manifest, "solution")
    verifier = _required_object(manifest, "verifier")
    oracle_patch = _required_string(solution, "oracle_patch")
    test_patch = _required_string(verifier, "test_patch")
    if admission.get("oracle_patch_sha256") != hashlib.sha256(
        oracle_patch.encode("utf-8")
    ).hexdigest() or admission.get("test_patch_sha256") != hashlib.sha256(
        test_patch.encode("utf-8")
    ).hexdigest():
        raise ValueError(f"{name} semantic admission private-patch mismatch: {instance_id}")
    for key in (
        "base_tree",
        "model_path_policy_sha256",
        "admitted_task_tree_sha256",
    ):
        value = _required_string(admission, key)
        pattern = _COMMIT if key == "base_tree" else _DIGEST
        if pattern.fullmatch(value) is None:
            raise ValueError(f"{name} semantic admission {key} is invalid")
    image_lock = _required_object(sandbox, "image_lock")
    if image_lock.get("schema_version") != "miles-oci-image-lock-v1":
        raise ValueError(f"{name} task has no generic OCI lock: {instance_id}")
    requested = _required_string(image_lock, "source_image_requested")
    resolved = _required_string(image_lock, "source_image_resolved")
    input_digest = _required_string(image_lock, "input_content_digest")
    child_digest = _required_string(image_lock, "child_manifest_digest")
    if (
        resolved != source_image
        or _DIGEST.fullmatch(input_digest) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", child_digest) is None
        or not source_image.endswith("@" + child_digest)
        or image_lock.get("platform")
        != {"os": "linux", "architecture": "amd64"}
        or admission.get("source_image_requested") != requested
        or admission.get("input_content_digest") != input_digest
    ):
        raise ValueError(f"{name} semantic admission OCI binding mismatch: {instance_id}")


def _validate_template_evidence(value: dict[str, Any], *, name: str) -> None:
    _validate_e2b_template_roles(
        _required_object(value, "template_evidence"),
        name=name,
    )


def _validate_e2b_template_roles(
    evidence: dict[str, Any],
    *,
    name: str,
) -> None:
    roles = {"source", "agent", "empty_verifier", "oracle_verifier"}
    if set(evidence) != roles:
        raise ValueError(f"{name} semantic admission template roles are incomplete")
    required_fields = {
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
        "sandbox_id",
    }
    for role in roles:
        phase = _required_object(evidence, role)
        if set(phase) != required_fields:
            raise ValueError(f"{name} semantic admission template evidence is invalid")
        for key in ("alias_sha256", "template_identity_sha256"):
            if _DIGEST.fullmatch(_required_string(phase, key)) is None:
                raise ValueError(f"{name} semantic admission template digest is invalid")
        for key in ("template_id", "build_id", "sandbox_id"):
            if re.fullmatch(
                r"[A-Za-z0-9_-]{6,128}",
                _required_string(phase, key),
            ) is None:
                raise ValueError(f"{name} semantic admission E2B ID is invalid")
    empty = evidence["empty_verifier"]
    oracle = evidence["oracle_verifier"]
    immutable = (
        "template_id",
        "build_id",
        "alias_sha256",
        "template_identity_sha256",
    )
    if any(empty[key] != oracle[key] for key in immutable):
        raise ValueError(f"{name} verifier phases did not reuse an exact template")
    source = evidence["source"]
    if any(source[key] != empty[key] for key in immutable):
        raise ValueError(f"{name} source/verifier phases did not reuse an exact template")
    sandbox_ids = [evidence[role]["sandbox_id"] for role in sorted(roles)]
    if len(set(sandbox_ids)) != len(sandbox_ids):
        raise ValueError(f"{name} semantic admission reused an E2B sandbox")


def _semantic_admission(
    manifest: dict[str, Any],
    *,
    r2e_admissions: dict[tuple[str, str], dict[str, Any]],
    swe_rebench_admissions: dict[tuple[str, str], dict[str, Any]],
    swe_gym_admissions: dict[tuple[str, str], dict[str, Any]],
    swebench_verified_admissions: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    key = (manifest["content_digest"], manifest["source_image"])
    if manifest["source_schema"] == "r2e-gym-v1":
        admission = r2e_admissions.get(key)
    elif manifest["source_schema"] == "swe-rebench-v2":
        admission = swe_rebench_admissions.get(key)
    elif manifest["source_schema"] == "swe-gym":
        admission = swe_gym_admissions.get(key)
    elif manifest["source_schema"] == "swebench":
        admission = (swebench_verified_admissions or {}).get(key)
    else:
        admission = None
    if admission is None:
        raise ValueError(
            f"{manifest['source_schema']} has no live semantic admission; "
            "production evidence cannot be issued"
        )
    return admission


def _task_tree_sha256(task_dir: Path) -> str:
    entries: list[dict[str, Any]] = []
    for path in [task_dir, *sorted(task_dir.rglob("*"))]:
        metadata = path.lstat()
        relative = "." if path == task_dir else path.relative_to(task_dir).as_posix()
        is_directory = stat.S_ISDIR(metadata.st_mode)
        is_regular = stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
        if not is_directory and not is_regular:
            raise ValueError(
                f"materialized task contains a symlink, special file, or hardlink: {relative}"
            )
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise PermissionError(
                f"materialized task entry is not owner-only: {relative}"
            )
        private_mode = stat.S_IMODE(metadata.st_mode)
        expected_modes = {0o500} if is_directory else {0o400, 0o500}
        if private_mode not in expected_modes:
            raise PermissionError(
                f"materialized task entry is not read-only: {relative}"
            )
        if is_directory:
            entry = {
                "path": relative,
                "type": "directory",
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            }
        else:
            content = _read_stable_task_file(path, metadata)
            entry = {
                "path": relative,
                "type": "file",
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        entries.append(entry)
    rendered = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _read_stable_task_file(path: Path, before: os.stat_result) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"task file changed during evidence hashing: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read()
        after = os.fstat(descriptor)
        if (
            (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or len(content) != after.st_size
        ):
            raise ValueError(f"task file changed during evidence hashing: {path}")
        return content
    finally:
        os.close(descriptor)


def _materialization_evidence(
    task_dir: Path,
    manifest: dict[str, Any],
    admission: dict[str, Any],
) -> dict[str, Any]:
    source_image = manifest["source_image"]
    tests_dir = task_dir / "tests"
    late_marker = tests_dir / _LATE_TESTS_MARKER
    policy = tests_dir / "model_path_policy.json"
    if (
        re.search(r"@sha256:[0-9a-f]{64}$", source_image) is None
        or late_marker.is_symlink()
        or not late_marker.is_file()
        or (tests_dir / "Dockerfile").exists()
    ):
        raise ValueError(
            f"task {manifest['instance_id']} is not a production late-verifier tree"
        )
    expected_policy_sha256 = admission.get("model_path_policy_sha256")
    if expected_policy_sha256 is not None and (
        not isinstance(expected_policy_sha256, str)
        or _DIGEST.fullmatch(expected_policy_sha256) is None
        or hashlib.sha256(policy.read_bytes()).hexdigest()
        != expected_policy_sha256
    ):
        raise ValueError(
            f"task {manifest['instance_id']} path policy differs from semantic admission"
        )
    task_tree_sha256 = _task_tree_sha256(task_dir)
    expected_task_tree_sha256 = _required_string(
        admission,
        "admitted_task_tree_sha256",
    )
    if (
        _DIGEST.fullmatch(expected_task_tree_sha256) is None
        or task_tree_sha256 != expected_task_tree_sha256
    ):
        raise ValueError(
            f"task {manifest['instance_id']} tree differs from live semantic admission"
        )
    return {
        "schema_version": _MATERIALIZATION_EVIDENCE_SCHEMA,
        "instance_id": manifest["instance_id"],
        "source_schema": manifest["source_schema"],
        "task_digest": manifest["task_digest"],
        "content_digest": manifest["content_digest"],
        "source_image": source_image,
        "private_manifest_record_sha256": manifest[
            "_private_manifest_record_sha256"
        ],
        "task_tree_sha256": task_tree_sha256,
        "semantic_admission_record_sha256": _stable_digest(admission),
        "checks": {
            "production_materialization": True,
            "immutable_image": True,
            "semantic_admission": True,
            "late_verifier_upload": True,
        },
    }


def _write_admission_evidence(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace admission evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.lstat()
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
    ):
        raise PermissionError(f"admission evidence parent is unsafe: {path.parent}")
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_task(
    task_dir: Path,
    manifest: dict[str, Any],
    *,
    r2e_parser: Path | None,
    official_parsers: Path | None,
    official_constants: Path | None,
    official_rebench_eval: Path | None,
    swe_gym_harness: dict[str, Path] | None,
    swebench_harness: dict[str, Path] | None,
) -> None:
    task_dir.chmod(0o700)
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"
    for directory in (environment_dir, tests_dir, solution_dir):
        directory.mkdir(mode=0o700)

    _write_private(task_dir / "instruction.md", _instruction(manifest))
    _write_private(task_dir / "task.toml", _task_toml(manifest))
    _write_private(environment_dir / "Dockerfile", _agent_dockerfile(manifest))
    _copy_private(_ASSET_DIR / "prepare_agent.sh", environment_dir / "prepare_agent.sh", 0o700)
    _copy_private(
        _ASSET_DIR / "seal_playwright_runtime.sh",
        environment_dir / "seal_playwright_runtime.sh",
        0o500,
    )
    _copy_private(
        _ASSET_DIR / "collect_agent_patch.sh",
        environment_dir / "collect_agent_patch.sh",
        0o700,
    )
    _copy_private(
        _ASSET_DIR / "strip_agent_privileges.py",
        environment_dir / "strip_agent_privileges.py",
        0o500,
    )
    _write_solution(solution_dir / "solve.sh", manifest)
    _write_json(tests_dir / "model_path_policy.json", _model_path_policy(manifest))
    _copy_private(
        _ASSET_DIR / "model_patch_policy.py",
        tests_dir / "model_patch_policy.py",
        0o500,
    )
    _copy_private(
        _ASSET_DIR / "strip_agent_privileges.py",
        tests_dir / "strip_agent_privileges.py",
        0o500,
    )
    _write_private(
        tests_dir / "base_commit.txt",
        (manifest["base_commit"] or "") + "\n",
    )
    _write_private(
        tests_dir / _LATE_TESTS_MARKER,
        "Upload this private verifier package only after the no-network sandbox starts.\n",
    )

    schema = manifest["source_schema"]
    if schema == "r2e-gym-v1":
        if r2e_parser is None:
            raise ValueError("--r2e-execution-log-parser is required for R2E-Gym tasks; fallback parser code is forbidden")
        _write_r2e_verifier(tests_dir, manifest, r2e_parser)
    elif schema == "swe-rebench-v2":
        if (
            official_parsers is None
            or official_constants is None
            or official_rebench_eval is None
        ):
            raise ValueError(
                "--swe-rebench-log-parsers, --swe-rebench-constants, and "
                "--swe-rebench-eval are required for SWE-rebench V2 tasks; "
                "fallback verifier code is forbidden"
            )
        _write_swe_rebench_verifier(
            tests_dir,
            manifest,
            official_parsers,
            official_constants,
            official_rebench_eval,
        )
    elif schema == "swe-gym":
        if swe_gym_harness is None:
            raise ValueError("--swe-gym-harness-root is required for SWE-Gym tasks; fallback parser/test-command code is forbidden")
        _write_swebench_harness_verifier(
            tests_dir,
            manifest,
            swe_gym_harness,
            environment_name="SWE-Gym",
            enforce_oracle_path_overlap=True,
        )
    elif schema == "swebench":
        if swebench_harness is None:
            raise ValueError(
                "--swebench-harness-root is required for SWE-bench Verified "
                "tasks; fallback parser/test-command code is forbidden"
            )
        _write_swebench_verified_verifier(
            tests_dir,
            manifest,
            swebench_harness,
        )


def _instruction(manifest: dict[str, Any]) -> str:
    return manifest["problem_statement"].rstrip() + "\n\nModify only the repository implementation. Do not create, replace, or disable tests.\n"


def _agent_dockerfile(manifest: dict[str, Any]) -> str:
    base_commit = manifest["base_commit"] or ""
    gold_commit = (
        manifest["verifier"]["gold_commit"]
        if manifest["source_schema"] == "r2e-gym-v1"
        else ""
    )
    runtime_policy = _agent_runtime_policy(manifest)
    return (
        f"FROM {manifest['source_image']}\n"
        "USER root\n"
        "RUN install -d -o root -g root -m 0700 /opt/miles-swe/collected\n"
        "RUN install -d -o root -g root -m 0700 /opt/miles-swe/root-home "
        "/opt/miles-swe/root-home/xdg\n"
        f"RUN printf '%s\\n' '{_INVALID_PATCH}' "
        ">/opt/miles-swe/collected/model.patch "
        "&& chmod 0600 /opt/miles-swe/collected/model.patch\n"
        "COPY --chmod=0555 prepare_agent.sh /opt/miles-swe/prepare_agent.sh\n"
        "COPY --chmod=0500 seal_playwright_runtime.sh "
        "/opt/miles-swe/seal_playwright_runtime.sh\n"
        "COPY --chmod=0500 collect_agent_patch.sh /opt/miles-swe/collect_agent_patch.sh\n"
        "COPY --chmod=0500 strip_agent_privileges.py "
        "/opt/miles-swe/strip_agent_privileges.py\n"
        "RUN MILES_SWE_SCHEMA="
        f"{manifest['source_schema']} MILES_SWE_BASE_COMMIT={base_commit} "
        f"MILES_SWE_GOLD_COMMIT={gold_commit} "
        f"MILES_SWE_RUNTIME_POLICY={runtime_policy} "
        "/opt/miles-swe/prepare_agent.sh\n"
        "RUN python3 /opt/miles-swe/strip_agent_privileges.py /\n"
        "USER root\n"
    )


def _agent_runtime_policy(manifest: dict[str, Any]) -> str:
    """Select the narrowly supported repository-runtime preservation policy."""

    if manifest.get("source_schema") == "swe-gym":
        return _PYTHON_EDITABLE_RUNTIME_POLICY
    if manifest.get("source_schema") != "swe-rebench-v2":
        return _NO_RUNTIME_POLICY
    source_metadata = manifest.get("source_metadata")
    verifier = manifest.get("verifier")
    if not isinstance(source_metadata, dict) or not isinstance(verifier, dict):
        return _NO_RUNTIME_POLICY
    language = str(source_metadata.get("language") or "").strip().lower()
    install_config = verifier.get("install_config")
    if language not in {"js", "javascript", "ts", "typescript"} or not isinstance(
        install_config,
        dict,
    ):
        return _NO_RUNTIME_POLICY
    install = install_config.get("install")
    if isinstance(install, str):
        install = [install]
    if not isinstance(install, list) or any(
        not isinstance(command, str) for command in install
    ):
        return _NO_RUNTIME_POLICY
    if any(re.fullmatch(r"npm\s+ci(?:\s+--[A-Za-z0-9._=-]+)*", command.strip()) for command in install):
        return _NPM_RUNTIME_POLICY
    return _NO_RUNTIME_POLICY


def _write_r2e_verifier(
    tests_dir: Path,
    manifest: dict[str, Any],
    r2e_parser: Path,
) -> None:
    verifier = manifest["verifier"]
    if verifier.get("kind") != "r2e-expected-pytest-map-v1":
        raise ValueError(f"invalid R2E verifier kind for {manifest['instance_id']}")
    expected = verifier.get("expected_output")
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"R2E verifier requires a non-empty expected map: {manifest['instance_id']}")
    config = {
        "instance_id": manifest["instance_id"],
        "expected_output": expected,
        "parser_commit": _R2E_PARSER_COMMIT,
        "parser_sha256": _R2E_PARSER_SHA256,
    }
    _write_private(tests_dir / "gold_commit.txt", verifier["gold_commit"] + "\n")
    _write_json(tests_dir / "verifier_config.json", config)
    _copy_private(_ASSET_DIR / "r2e_test.sh", tests_dir / "test.sh", 0o700)
    _copy_private(_ASSET_DIR / "r2e_grader.py", tests_dir / "r2e_grader.py", 0o600)
    _copy_private(
        _ASSET_DIR / "prepare_r2e_verifier.sh",
        tests_dir / "prepare_r2e_verifier.sh",
        0o700,
    )
    _copy_private(r2e_parser, tests_dir / "r2e_execution_log_parser.py", 0o600)


def _write_swe_rebench_verifier(
    tests_dir: Path,
    manifest: dict[str, Any],
    official_parsers: Path,
    official_constants: Path,
    official_rebench_eval: Path,
) -> None:
    verifier = manifest["verifier"]
    if verifier.get("kind") != "swe-rebench-v2":
        raise ValueError(f"invalid SWE-rebench verifier kind for {manifest['instance_id']}")
    install_config = verifier.get("install_config")
    if not isinstance(install_config, dict):
        raise ValueError(f"SWE-rebench task requires install_config: {manifest['instance_id']}")
    test_commands = _test_commands(install_config.get("test_cmd"))
    log_parser = _required_string(install_config, "log_parser")
    if not re.fullmatch(r"parse_[A-Za-z0-9_]+", log_parser):
        raise ValueError(f"SWE-rebench log_parser must name an explicit function: {log_parser!r}")
    fail_to_pass = _string_list(verifier.get("fail_to_pass"), "fail_to_pass")
    pass_to_pass = _string_list(verifier.get("pass_to_pass"), "pass_to_pass")
    config = {
        "instance_id": manifest["instance_id"],
        "base_commit": manifest["base_commit"],
        "test_commands": test_commands,
        "log_parser": log_parser,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "parser_commit": _REBENCH_COMMIT,
        "log_parsers_sha256": _REBENCH_LOG_PARSERS_SHA256,
        "constants_sha256": _REBENCH_CONSTANTS_SHA256,
        "eval_sha256": _REBENCH_EVAL_SHA256,
    }
    _write_json(tests_dir / "verifier_config.json", config)
    _write_json(tests_dir / "test_commands.json", {"test_commands": test_commands})
    test_patch = _required_string(verifier, "test_patch")
    _write_private(tests_dir / "test_patch.diff", test_patch)
    _copy_private(_ASSET_DIR / "swe_rebench_test.sh", tests_dir / "test.sh", 0o700)
    _copy_private(
        _ASSET_DIR / "seal_playwright_runtime.sh",
        tests_dir / "seal_playwright_runtime.sh",
        0o500,
    )
    _copy_private(
        _ASSET_DIR / "swe_rebench_grader.py",
        tests_dir / "swe_rebench_grader.py",
        0o600,
    )
    _copy_private(_ASSET_DIR / "swe_rebench_run.py", tests_dir / "swe_rebench_run.py", 0o600)
    _copy_private(
        _ASSET_DIR / "prepare_verifier_user.sh",
        tests_dir / "prepare_verifier_user.sh",
        0o700,
    )
    agent_package = tests_dir / "lib" / "agent"
    agent_package.mkdir(parents=True, mode=0o700)
    _write_private(tests_dir / "lib" / "__init__.py", "")
    _write_private(agent_package / "__init__.py", "")
    _copy_private(official_parsers, agent_package / "log_parsers.py", 0o600)
    _copy_private(official_constants, agent_package / "swe_constants.py", 0o600)
    _copy_private(official_rebench_eval, tests_dir / "official_eval.py", 0o600)


def _write_swebench_harness_verifier(
    tests_dir: Path,
    manifest: dict[str, Any],
    harness: dict[str, Path],
    *,
    environment_name: str,
    enforce_oracle_path_overlap: bool,
) -> None:
    verifier = manifest["verifier"]
    if verifier.get("kind") != "swebench-harness-v1":
        raise ValueError(
            f"invalid {environment_name} verifier kind for {manifest['instance_id']}"
        )
    fail_to_pass = _string_list(verifier.get("fail_to_pass"), "fail_to_pass")
    pass_to_pass = _string_list(verifier.get("pass_to_pass"), "pass_to_pass")
    if not fail_to_pass:
        raise ValueError(
            f"{environment_name} task has no FAIL_TO_PASS tests: "
            f"{manifest['instance_id']}"
        )
    if len(set(fail_to_pass + pass_to_pass)) != len(fail_to_pass + pass_to_pass):
        raise ValueError(
            f"{environment_name} expected test IDs are not unique: "
            f"{manifest['instance_id']}"
        )
    source_metadata = _required_object(manifest, "source_metadata")
    config = {
        "instance_id": manifest["instance_id"],
        "repo": _required_string(manifest, "repo").lower(),
        "version": _required_string(source_metadata, "version"),
        "base_commit": manifest["base_commit"],
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "harness_commit": _SWE_GYM_HARNESS_COMMIT,
        "harness_version": _SWE_GYM_HARNESS_VERSION,
        "constants_sha256": _SWE_GYM_CONSTANTS_SHA256,
        "log_parsers_sha256": _SWE_GYM_LOG_PARSERS_SHA256,
        "grading_sha256": _SWE_GYM_GRADING_SHA256,
        "harbor_adapter_commit": _SWE_GYM_HARBOR_COMMIT,
        "harbor_adapter_sha256": _SWE_GYM_HARBOR_ADAPTER_SHA256,
        "source_image": manifest["source_image"],
    }
    _write_json(tests_dir / "verifier_config.json", config)
    test_patch = _required_string(verifier, "test_patch")
    if enforce_oracle_path_overlap:
        hidden_paths = _oracle_patch_paths(test_patch)
        model_paths = set(_model_path_policy(manifest)["allowed_paths"])
        overlap = sorted(hidden_paths & model_paths)
        if overlap:
            raise ValueError(
                f"{environment_name} oracle and hidden tests touch the same paths "
                f"for {manifest['instance_id']}: {overlap}"
            )
    _write_private(tests_dir / "test_patch.diff", test_patch)
    _copy_private(_ASSET_DIR / "swe_gym_test.sh", tests_dir / "test.sh", 0o700)
    for name in ("swe_gym_prepare.py", "swe_gym_run.py", "swe_gym_grader.py"):
        _copy_private(_ASSET_DIR / name, tests_dir / name, 0o600)
    _copy_private(_ASSET_DIR / "prepare_verifier_user.sh", tests_dir / "prepare_verifier_user.sh", 0o700)
    package = tests_dir / "lib" / "swegym" / "harness"
    package.mkdir(parents=True, mode=0o700)
    _write_private(tests_dir / "lib" / "swegym" / "__init__.py", '__version__ = "2.0.13"\n')
    _write_private(package / "__init__.py", "")
    _write_private(
        package / "test_spec.py",
        'class TestSpec:\n    """Import shim for pinned grading.py type annotations."""\n\n',
    )
    for name, source in harness.items():
        _copy_private(source, package / f"{name}.py", 0o600)


def _write_swebench_verified_verifier(
    tests_dir: Path,
    manifest: dict[str, Any],
    harness: dict[str, Path],
) -> None:
    verifier = manifest["verifier"]
    if verifier.get("kind") != "swebench-harness-v1":
        raise ValueError(
            f"invalid SWE-bench Verified verifier kind for {manifest['instance_id']}"
        )
    fail_to_pass = _string_list(verifier.get("fail_to_pass"), "fail_to_pass")
    pass_to_pass = _string_list(verifier.get("pass_to_pass"), "pass_to_pass")
    if not fail_to_pass:
        raise ValueError(
            f"SWE-bench Verified task has no FAIL_TO_PASS tests: "
            f"{manifest['instance_id']}"
        )
    if len(set(fail_to_pass + pass_to_pass)) != len(
        fail_to_pass + pass_to_pass
    ):
        raise ValueError(
            f"SWE-bench Verified expected test IDs are not unique: "
            f"{manifest['instance_id']}"
        )
    source_metadata = _required_object(manifest, "source_metadata")
    config = {
        "instance_id": manifest["instance_id"],
        "repo": _required_string(manifest, "repo").lower(),
        "version": _required_string(source_metadata, "version"),
        "base_commit": manifest["base_commit"],
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "harness_repository": _SWEBENCH_HARNESS_REPOSITORY,
        "harness_commit": _SWEBENCH_HARNESS_COMMIT,
        "harness_version": _SWEBENCH_HARNESS_VERSION,
        "constants_sha256": _SWEBENCH_CONSTANTS_SHA256,
        "log_parsers_sha256": _SWEBENCH_LOG_PARSERS_SHA256,
        "grading_sha256": _SWEBENCH_GRADING_SHA256,
        "score_semantics": _SWEBENCH_HARDENED_SCORE_SEMANTICS,
        "source_image": manifest["source_image"],
    }
    _write_json(tests_dir / "verifier_config.json", config)
    _write_private(
        tests_dir / "test_patch.diff",
        _required_string(verifier, "test_patch"),
    )
    _copy_private(_ASSET_DIR / "swe_gym_test.sh", tests_dir / "test.sh", 0o700)
    _copy_private(
        _ASSET_DIR / "swe_gym_prepare.py",
        tests_dir / "swe_gym_prepare.py",
        0o600,
    )
    _copy_private(
        _ASSET_DIR / "swebench_verified_run.py",
        tests_dir / "swe_gym_run.py",
        0o600,
    )
    _copy_private(
        _ASSET_DIR / "swebench_verified_grader.py",
        tests_dir / "swe_gym_grader.py",
        0o600,
    )
    _copy_private(
        _ASSET_DIR / "prepare_verifier_user.sh",
        tests_dir / "prepare_verifier_user.sh",
        0o700,
    )
    package = tests_dir / "lib" / "swebench" / "harness"
    package.mkdir(parents=True, mode=0o700)
    _write_private(
        tests_dir / "lib" / "swebench" / "__init__.py",
        f'__version__ = "{_SWEBENCH_HARNESS_VERSION}"\n',
    )
    _write_private(package / "__init__.py", "")
    _write_private(
        package / "test_spec.py",
        'class TestSpec:\n    """Import shim for pinned grading.py annotations."""\n\n',
    )
    for name, source in harness.items():
        _copy_private(source, package / f"{name}.py", 0o600)


def _write_solution(path: Path, manifest: dict[str, Any]) -> None:
    oracle_patch = manifest["oracle_patch"]
    if oracle_patch is None:
        content = "#!/bin/bash\nset -euo pipefail\necho 'No oracle patch was published in this task manifest.' >&2\nexit 2\n"
    else:
        encoded = base64.b64encode(oracle_patch.encode("utf-8")).decode("ascii")
        content = f"#!/bin/bash\nset -euo pipefail\nrepo=$(cat /opt/miles-swe/workdir)\nprintf '%s' '{encoded}' | base64 -d > /tmp/oracle.patch\ngit -C \"${{repo}}\" apply --binary /tmp/oracle.patch\n"
    _write_private(path, content, mode=0o700)


def _model_path_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("source_schema") == "swebench":
        return {
            "schema_version": _EVAL_MODEL_PATH_POLICY_SCHEMA,
            "policy_mode": "deny-sensitive-paths",
            "denied_basenames": sorted(_DENIED_MODEL_BASENAMES),
            "denied_components": sorted(_DENIED_MODEL_COMPONENTS),
            "deny_test_name_patterns": True,
            "rename_supported": False,
            "binary_supported": False,
            "type_change_supported": False,
        }
    oracle_patch = manifest.get("oracle_patch")
    if not isinstance(oracle_patch, str) or not oracle_patch.strip():
        raise ValueError(f"{manifest['instance_id']} has no private oracle patch for model path policy")
    paths = _oracle_patch_paths(oracle_patch)
    denied = sorted(path for path in paths if _denied_model_path(path))
    if denied:
        raise ValueError(f"{manifest['instance_id']} oracle touches denied test/config/toolchain paths: {denied}")
    return {
        "schema_version": _MODEL_PATH_POLICY_SCHEMA,
        "allowed_paths": sorted(paths),
        "oracle_patch_sha256": hashlib.sha256(oracle_patch.encode("utf-8")).hexdigest(),
        "rename_supported": False,
        "binary_supported": False,
        "type_change_supported": False,
    }


def _oracle_patch_paths(patch: str) -> set[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if line == "GIT binary patch" or line.startswith(("Binary files ", "literal ", "delta ")):
            raise ValueError("binary oracle patches are not supported by the SWE path policy")
        if line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            raise ValueError("rename/copy oracle patches are not supported by the SWE path policy")
        if line.startswith(("old mode ", "new mode ")):
            raise ValueError("mode-changing oracle patches are not supported by the SWE path policy")
        if line.startswith(("new file mode ", "deleted file mode ")) and not line.endswith((" 100644", " 100755")):
            raise ValueError("non-regular oracle patch paths are not supported")
        if not line.startswith("diff --git "):
            continue
        try:
            fields = shlex.split(line, posix=True)
        except ValueError as exc:
            raise ValueError("oracle patch has an invalid diff header") from exc
        if len(fields) != 4 or fields[:2] != ["diff", "--git"]:
            raise ValueError("oracle patch has an invalid diff header")
        old_path, new_path = fields[2:]
        if not old_path.startswith("a/") or not new_path.startswith("b/"):
            raise ValueError("oracle patch uses an unsupported absolute/nonstandard path")
        old_path = old_path[2:]
        new_path = new_path[2:]
        if old_path != new_path:
            raise ValueError("rename oracle patches are not supported by the SWE path policy")
        _validate_patch_path(new_path)
        paths.add(new_path)
    if not paths:
        raise ValueError("oracle patch contains no supported file paths")
    return paths


def _validate_patch_path(path: str) -> None:
    if not path or path.startswith("/") or "\\" in path or "\0" in path or "\n" in path or any(component in {"", ".", ".."} for component in path.split("/")):
        raise ValueError(f"unsafe patch path: {path!r}")


def _denied_model_path(path: str) -> bool:
    lowered = path.lower()
    components = lowered.split("/")
    basename = components[-1]
    return any(component in _DENIED_MODEL_COMPONENTS for component in components) or basename in _DENIED_MODEL_BASENAMES or basename.startswith("test_") or basename.endswith(("_test.py", ".ini", ".toml", ".yaml", ".yml"))


def _task_toml(manifest: dict[str, Any]) -> str:
    playwright_environment = (
        'PLAYWRIGHT_BROWSERS_PATH = "/opt/miles-swe/runtime/ms-playwright"\n'
        if _agent_runtime_policy(manifest) == _NPM_RUNTIME_POLICY
        else ""
    )
    collect = """set +e
timeout --signal=TERM --kill-after=5s 100s /opt/miles-swe/collect_agent_patch.sh
status=$?
case "$status" in
    0) exit 0 ;;
    124|137)
        grep -qx 'MILES_SWE_AGENT_STATE_INVALID' \
            /opt/miles-swe/collected/model.patch || exit 2
        exit 0
        ;;
    *) exit "$status" ;;
esac
"""
    return f"""schema_version = "1.3"

artifacts = [
  {{ source = "/logs/artifacts", exclude = ["*"] }},
  {{ source = "/opt/miles-swe/collected/model.patch", destination = "model.patch" }},
]

[task]
name = {_toml_string("miles-swe/" + manifest["instance_id"])}
description = "Repository-level SWE task with a fresh isolated verifier."
authors = []
keywords = ["swe", "repository", "isolated-verifier"]

[metadata]
difficulty = "unknown"
category = "debugging"
tags = ["swe", {_toml_string(manifest["source_schema"])}]
task_digest = {_toml_string(manifest["task_digest"])}
source_dataset = {_toml_string(manifest["source_dataset"])}
source_schema = {_toml_string(manifest["source_schema"])}

[verifier]
timeout_sec = {float(timeouts.VERIFIER_EXECUTION_TIMEOUT_SEC)}
environment_mode = "separate"
user = 0

[[verifier.collect]]
service = "main"
command = '''{collect}'''
timeout_sec = {float(timeouts.COLLECT_TIMEOUT_SEC)}
user = 0
required = true

[verifier.environment]
network_mode = "no-network"
docker_image = {_toml_string(manifest["source_image"])}
build_timeout_sec = {float(timeouts.VERIFIER_ENVIRONMENT_START_TIMEOUT_SEC)}
cpus = 4
memory_mb = 8192
storage_mb = 20480
gpus = 0
mcp_servers = []

[agent]
timeout_sec = {float(timeouts.AGENT_EXECUTION_TIMEOUT_SEC)}
user = 1000

[environment]
network_mode = "no-network"
build_timeout_sec = {float(timeouts.AGENT_ENVIRONMENT_START_TIMEOUT_SEC)}
cpus = 4
memory_mb = 8192
storage_mb = 20480
gpus = 0
mcp_servers = []

[verifier.env]

[environment.env]
HOME = "/tmp/miles-agent-home"
{playwright_environment}

[solution.env]
"""


def _required_object(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"required object {key!r} is missing")
    return result


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip() or "\0" in result:
        raise ValueError(f"required text field {key!r} is missing or invalid")
    return result.strip()


def _string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return value


def _test_commands(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("install_config.test_cmd must be text or a list")
    commands = [command for command in value if isinstance(command, str) and command.strip()]
    if len(commands) != len(value) or not commands:
        raise ValueError("install_config.test_cmd must contain only non-empty commands")
    return commands


def _stable_digest(value: dict[str, Any]) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _load_r2e_admissions(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    _validate_private_input(path)
    required_checks = {
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
    required_fields = {
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
    admissions: dict[tuple[str, str], dict[str, Any]] = {}
    for value in _read_manifest(path):
        if value.get("schema_version") != "miles-r2e-admission-v1":
            raise ValueError(f"unsupported R2E admission schema in {path}")
        if set(value) != required_fields:
            raise ValueError(f"R2E admission field set is invalid in {path}")
        instance_id = _required_string(value, "instance_id")
        task_digest = _required_string(value, "task_digest")
        input_content_digest = _required_string(value, "input_content_digest")
        locked_content_digest = _required_string(value, "locked_content_digest")
        content_digest = _required_string(value, "content_digest")
        source_image_requested = _required_string(value, "source_image_requested")
        source_image_resolved = _required_string(value, "source_image_resolved")
        source_image = _required_string(value, "source_image")
        base_commit = _required_string(value, "base_commit")
        oracle_patch_sha256 = _required_string(value, "oracle_patch_sha256")
        admitted_task_tree_sha256 = _required_string(
            value,
            "admitted_task_tree_sha256",
        )
        if (
            _INSTANCE_ID.fullmatch(instance_id) is None
            or _COMMIT.fullmatch(base_commit) is None
            or any(
                _DIGEST.fullmatch(digest) is None
                for digest in (
                    task_digest,
                    input_content_digest,
                    locked_content_digest,
                    content_digest,
                    oracle_patch_sha256,
                    admitted_task_tree_sha256,
                )
            )
            or _IMAGE.fullmatch(source_image_requested) is None
        ):
            raise ValueError(f"invalid R2E admission identity in {path}")
        if re.search(r"@sha256:[0-9a-f]{64}$", source_image) is None:
            raise ValueError(f"R2E admission image is not immutable: {source_image}")
        if source_image_resolved != source_image:
            raise ValueError(f"R2E admission resolved image mismatch in {path}")
        if value.get("image_publisher_policy") != oci_image_lock.IMAGE_PUBLISHER_POLICY:
            raise ValueError(f"R2E admission publisher policy mismatch in {path}")
        checks = _required_object(value, "checks")
        if set(checks) != set(required_checks) or any(
            checks.get(name) != expected
            for name, expected in required_checks.items()
        ):
            raise ValueError(f"R2E admission is missing a required live check: {path}")
        _validate_e2b_template_roles(
            _required_object(value, "e2b_sandbox_evidence"),
            name="R2E",
        )
        key = (content_digest, source_image)
        if key in admissions:
            raise ValueError(f"duplicate R2E admission for {content_digest}")
        admissions[key] = value
    return admissions


def _validate_repository_admission_record(
    value: dict[str, Any],
    *,
    schema: str,
    source_schema: str,
    pins: dict[str, str],
    name: str,
) -> None:
    if value.get("schema_version") != schema:
        raise ValueError(f"unsupported {name} admission schema")
    if set(value) != _REPOSITORY_ADMISSION_BASE_FIELDS | set(pins):
        raise ValueError(f"{name} admission field set is invalid")
    if value.get("source_schema") != source_schema:
        raise ValueError(f"{name} admission source schema is invalid")
    instance_id = _required_string(value, "instance_id")
    if _INSTANCE_ID.fullmatch(instance_id) is None:
        raise ValueError(f"{name} admission instance_id is invalid")
    for key in (
        "task_digest",
        "input_content_digest",
        "locked_content_digest",
        "content_digest",
        "oracle_patch_sha256",
        "test_patch_sha256",
        "model_path_policy_sha256",
        "admitted_task_tree_sha256",
    ):
        if _DIGEST.fullmatch(_required_string(value, key)) is None:
            raise ValueError(f"{name} admission {key} is invalid")
    if value["locked_content_digest"] != value["content_digest"]:
        raise ValueError(f"{name} admission locked-content binding is invalid")
    for key in ("base_commit", "base_tree"):
        if _COMMIT.fullmatch(_required_string(value, key)) is None:
            raise ValueError(f"{name} admission {key} is invalid")
    requested = _required_string(value, "source_image_requested")
    resolved = _required_string(value, "source_image_resolved")
    if (
        _IMAGE.fullmatch(requested) is None
        or value.get("source_image") != resolved
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,447}@sha256:[0-9a-f]{64}",
            resolved,
        )
        is None
    ):
        raise ValueError(f"{name} admission image binding is invalid")
    if value.get("image_publisher_policy") != oci_image_lock.IMAGE_PUBLISHER_POLICY:
        raise ValueError(f"{name} admission publisher policy is invalid")
    if any(value.get(key) != expected for key, expected in pins.items()):
        raise ValueError(f"{name} admission pinned dependency mismatch")
    if value.get("checks") != _REPOSITORY_ADMISSION_CHECKS:
        raise ValueError(f"{name} admission live checks are invalid")
    _validate_template_evidence(value, name=name)


def _load_swe_gym_admissions(
    path: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    _validate_private_input(path)
    admissions: dict[tuple[str, str], dict[str, Any]] = {}
    pins = {
        "dataset_revision": _SWE_GYM_DATASET_REVISION,
        "harness_commit": _SWE_GYM_HARNESS_COMMIT,
        "harness_version": _SWE_GYM_HARNESS_VERSION,
        "constants_sha256": _SWE_GYM_CONSTANTS_SHA256,
        "log_parsers_sha256": _SWE_GYM_LOG_PARSERS_SHA256,
        "grading_sha256": _SWE_GYM_GRADING_SHA256,
        "harbor_adapter_commit": _SWE_GYM_HARBOR_COMMIT,
        "harbor_adapter_sha256": _SWE_GYM_HARBOR_ADAPTER_SHA256,
    }
    for value in _read_manifest(path):
        _validate_repository_admission_record(
            value,
            schema="miles-swe-gym-admission-v1",
            source_schema="swe-gym",
            pins=pins,
            name="SWE-Gym",
        )
        content_digest = _required_string(value, "content_digest")
        source_image = _required_string(value, "source_image")
        key = (content_digest, source_image)
        if key in admissions:
            raise ValueError(f"duplicate SWE-Gym admission for {content_digest}")
        admissions[key] = value
    return admissions


def _load_swebench_verified_admissions(
    path: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    _validate_private_input(path)
    admissions: dict[tuple[str, str], dict[str, Any]] = {}
    pins = {
        "dataset_revision": _SWEBENCH_VERIFIED_DATASET_REVISION,
        "harness_repository": _SWEBENCH_HARNESS_REPOSITORY,
        "harness_commit": _SWEBENCH_HARNESS_COMMIT,
        "harness_version": _SWEBENCH_HARNESS_VERSION,
        "constants_sha256": _SWEBENCH_CONSTANTS_SHA256,
        "log_parsers_sha256": _SWEBENCH_LOG_PARSERS_SHA256,
        "grading_sha256": _SWEBENCH_GRADING_SHA256,
        "score_semantics": _SWEBENCH_HARDENED_SCORE_SEMANTICS,
        "harbor_adapter_commit": _SWE_GYM_HARBOR_COMMIT,
        "harbor_adapter_sha256": _SWE_GYM_HARBOR_ADAPTER_SHA256,
    }
    for value in _read_manifest(path):
        _validate_repository_admission_record(
            value,
            schema="miles-swebench-verified-hardened-local-admission-v1",
            source_schema="swebench",
            pins=pins,
            name="SWE-bench Verified",
        )
        content_digest = _required_string(value, "content_digest")
        source_image = _required_string(value, "source_image")
        key = (content_digest, source_image)
        if key in admissions:
            raise ValueError(
                f"duplicate SWE-bench Verified admission for {content_digest}"
            )
        admissions[key] = value
    return admissions


def _load_swe_rebench_admissions(
    path: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    _validate_private_input(path)
    admissions: dict[tuple[str, str], dict[str, Any]] = {}
    expected_pins = {
        "rebench_commit": _REBENCH_COMMIT,
        "log_parsers_sha256": _REBENCH_LOG_PARSERS_SHA256,
        "constants_sha256": _REBENCH_CONSTANTS_SHA256,
        "eval_sha256": _REBENCH_EVAL_SHA256,
    }
    for value in _read_manifest(path):
        _validate_repository_admission_record(
            value,
            schema="miles-swe-rebench-admission-v1",
            source_schema="swe-rebench-v2",
            pins=expected_pins,
            name="SWE-ReBench",
        )
        content_digest = _required_string(value, "content_digest")
        source_image = _required_string(value, "source_image")
        key = (content_digest, source_image)
        if key in admissions:
            raise ValueError(f"duplicate SWE-ReBench admission for {content_digest}")
        admissions[key] = value
    return admissions


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_private(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_private(path: Path, content: str, *, mode: int = 0o600) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)


def _copy_private(source: Path, destination: Path, mode: int) -> None:
    destination.write_bytes(
        _read_stable_regular_file(source, name=f"materializer asset {source.name}")
    )
    destination.chmod(mode)


def _read_stable_regular_file(path: Path, *, name: str) -> bytes:
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{name} is missing: {path}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ValueError(f"{name} must be a single-link regular file: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"{name} changed before open: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read()
        finished = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino, opened.st_size)
            != (finished.st_dev, finished.st_ino, finished.st_size)
            or opened.st_mtime_ns != finished.st_mtime_ns
            or opened.st_ctime_ns != finished.st_ctime_ns
            or len(content) != finished.st_size
        ):
            raise RuntimeError(f"{name} changed while reading: {path}")
        return content
    finally:
        os.close(descriptor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--admission-evidence",
        type=Path,
        help=(
            "owner-only production evidence JSONL; refused for dry-run or "
            "schemas without a live semantic admission"
        ),
    )
    parser.add_argument(
        "--r2e-execution-log-parser",
        type=Path,
        help=(f"official R2E-Gym execution_log_parser.py pinned at {_R2E_PARSER_COMMIT}"),
    )
    parser.add_argument(
        "--r2e-admission-manifest",
        type=Path,
        help="owner-only live admission records keyed by content_digest and image digest",
    )
    parser.add_argument(
        "--swe-rebench-log-parsers",
        type=Path,
        help="pinned canonical SWE-rebench V2 lib/agent/log_parsers.py",
    )
    parser.add_argument(
        "--swe-rebench-constants",
        type=Path,
        help="pinned canonical SWE-rebench V2 lib/agent/swe_constants.py",
    )
    parser.add_argument(
        "--swe-rebench-eval",
        type=Path,
        help="pinned canonical SWE-rebench V2 scripts/eval.py",
    )
    parser.add_argument(
        "--swe-rebench-admission-manifest",
        type=Path,
        help=(
            "owner-only live SWE-ReBench admissions keyed by exact "
            "content/image digest"
        ),
    )
    parser.add_argument(
        "--swe-gym-harness-root",
        type=Path,
        help=(f"official SWE-Gym/SWE-Bench-Package checkout pinned at {_SWE_GYM_HARNESS_COMMIT}; only checksum-pinned harness files are read"),
    )
    parser.add_argument(
        "--swebench-harness-root",
        type=Path,
        help=(
            "official SWE-bench checkout pinned at "
            f"{_SWEBENCH_HARNESS_COMMIT}; only checksum-pinned harness files are read"
        ),
    )
    parser.add_argument(
        "--swe-gym-admission-manifest",
        type=Path,
        help="owner-only live admissions keyed by instance/content/image digest",
    )
    parser.add_argument(
        "--swebench-verified-admission-manifest",
        type=Path,
        help=(
            "owner-only live SWE-bench Verified admissions keyed by exact "
            "eval instance/content/image digest"
        ),
    )
    parser.add_argument(
        "--allow-mutable-images",
        action="store_true",
        help="dry-run only: permit image tags instead of immutable OCI digests",
    )
    parser.add_argument(
        "--allow-unadmitted-r2e-dry-run",
        action="store_true",
        help="inspection only: materialize R2E before runtime/leakage golden admission",
    )
    parser.add_argument(
        "--allow-unadmitted-swe-gym-dry-run",
        action="store_true",
        help="inspection only: materialize SWE-Gym before registry/oracle admission",
    )
    parser.add_argument(
        "--allow-unadmitted-swe-rebench-dry-run",
        action="store_true",
        help=(
            "inspection only: materialize SWE-ReBench before live semantic "
            "admission"
        ),
    )
    parser.add_argument(
        "--allow-unadmitted-swebench-verified-dry-run",
        action="store_true",
        help=(
            "inspection only: materialize SWE-bench Verified before live "
            "semantic admission"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    summary = materialize(args)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
