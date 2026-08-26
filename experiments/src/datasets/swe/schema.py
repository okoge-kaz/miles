"""Normalize published SWE datasets without exposing verifier secrets to Miles.

The normalized object deliberately has two projections:

* ``to_miles_row`` contains only the model prompt and non-secret provenance.
* ``to_task_manifest`` contains the private verifier inputs needed to build a
  Harbor task.  This projection must never be used as Miles prompt data.

Supported source schemas are the published R2E-Gym V1/Subset rows, SWE-Gym and
SWE-bench rows, SWE-rebench V2 rows, and NVIDIA Nemotron SWE wrappers around
those executable tasks.  Nemotron SWE-pivot rows are intentionally rejected:
they are next-action examples rather than resettable repository environments.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "miles-swe-task-v1"
_FULL_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_R2E_STATUS = {"PASSED", "FAILED", "ERROR"}


@dataclass(frozen=True)
class NormalizedSWETask:
    """One executable repository task plus its private verifier specification."""

    instance_id: str
    source_dataset: str
    source_schema: str
    repo: str
    problem_statement: str
    base_commit: str | None
    source_image: str | None
    oracle_patch: str | None
    verifier: dict[str, Any]
    source_metadata: dict[str, Any]
    eval_only: bool
    task_binding: str

    def to_task_manifest(self) -> dict[str, Any]:
        """Return a private Harbor-materialization record."""
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "source_dataset": self.source_dataset,
            "source_schema": self.source_schema,
            "repo": self.repo,
            "problem_statement": self.problem_statement,
            "base_commit": self.base_commit,
            "sandbox": {
                "source_image": self.source_image,
                "backend_selector": "harbor",
            },
            "solution": {"oracle_patch": self.oracle_patch},
            "verifier": self.verifier,
            "source_metadata": self.source_metadata,
            "eval_only": self.eval_only,
        }
        manifest["content_digest"] = _stable_digest(manifest)
        manifest["task_digest"] = self.task_binding
        return manifest

    def to_miles_row(self, *, agent_name: str = "terminus-2") -> dict[str, Any]:
        """Return the gold-free row consumed by Miles rollout workers."""
        manifest = self.to_task_manifest()
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "source_dataset": self.source_dataset,
            "source_schema": self.source_schema,
            "task_id": self.instance_id,
            "task_digest": manifest["task_digest"],
            "eval_only": self.eval_only,
        }
        return {
            "prompt": self.problem_statement,
            "label": "",
            "metadata": {
                "instance_id": self.instance_id,
                "agent_name": agent_name,
                "source": self.source_dataset,
                "verifier": "swe_environment",
                "swe_task": provenance,
            },
        }


def normalize_swe_row(
    row: dict[str, Any],
    *,
    dataset_id: str | None = None,
    usage: str = "train",
) -> NormalizedSWETask:
    """Normalize one raw or Nemotron-wrapped SWE task, failing on ambiguity."""
    if usage not in {"train", "eval"}:
        raise ValueError(f"usage must be train or eval, got {usage!r}")
    unwrapped, wrapper_dataset = _unwrap_nemotron(row)
    source_dataset = _source_dataset(unwrapped, explicit=dataset_id, wrapper=wrapper_dataset)
    schema = _detect_schema(unwrapped, source_dataset)
    if schema == "r2e-gym-v1":
        task = _normalize_r2e(unwrapped, source_dataset)
    elif schema == "swe-rebench-v2":
        task = _normalize_swe_rebench_v2(unwrapped, source_dataset)
    elif schema == "swebench":
        task = _normalize_swebench(unwrapped, source_dataset)
    else:
        raise ValueError(f"unsupported SWE schema: {schema}")
    if usage == "train" and task.eval_only:
        raise ValueError(
            f"{task.source_dataset} is benchmark/evaluation data; refusing to emit training rows"
        )
    return task


def _unwrap_nemotron(row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    params = row.get("responses_create_params")
    if not isinstance(params, dict) or "metadata" not in params:
        return row, None
    agent_ref = row.get("agent_ref") or {}
    agent_name = str(agent_ref.get("name") or "") if isinstance(agent_ref, dict) else ""
    if agent_name != "swe_agents_train":
        raise ValueError(
            "Nemotron SWE row is not a full environment task: expected "
            "agent_ref.name='swe_agents_train'; SWE-pivot rows must use the exact-action adapter"
        )
    wrapper = params.get("metadata")
    if not isinstance(wrapper, dict):
        raise ValueError("Nemotron SWE responses_create_params.metadata must be an object")
    raw_instance = wrapper.get("instance_dict")
    if isinstance(raw_instance, str):
        try:
            instance = json.loads(raw_instance)
        except json.JSONDecodeError as exc:
            raise ValueError("Nemotron SWE metadata.instance_dict is invalid JSON") from exc
    elif isinstance(raw_instance, dict):
        instance = dict(raw_instance)
    else:
        raise ValueError("Nemotron SWE metadata.instance_dict must be JSON text or an object")
    if not isinstance(instance, dict):
        raise ValueError("Nemotron SWE metadata.instance_dict must decode to an object")
    for key in (
        "instance_id",
        "dataset_name",
        "repo",
        "base_commit",
        "problem_statement",
        "split",
    ):
        if instance.get(key) in (None, "") and wrapper.get(key) not in (None, ""):
            instance[key] = wrapper[key]
    return instance, _optional_text(wrapper.get("dataset_name"))


def _source_dataset(
    row: dict[str, Any],
    *,
    explicit: str | None,
    wrapper: str | None,
) -> str:
    source = explicit or wrapper or _optional_text(row.get("dataset_name"))
    if not source:
        raise ValueError("dataset_id is required when the source row has no dataset_name")
    return source


def _detect_schema(row: dict[str, Any], source_dataset: str) -> str:
    lowered = source_dataset.lower()
    if "r2e-gym" in lowered or {"docker_image", "commit_hash", "expected_output_json"} <= row.keys():
        return "r2e-gym-v1"
    if "rebench" in lowered or {"image_name", "install_config", "test_patch"} <= row.keys():
        return "swe-rebench-v2"
    if {"instance_id", "repo", "base_commit", "test_patch", "FAIL_TO_PASS"} <= row.keys():
        return "swebench"
    raise ValueError(
        f"cannot identify executable SWE schema for dataset {source_dataset!r}; "
        f"available keys={sorted(row)}"
    )


def _normalize_r2e(row: dict[str, Any], source_dataset: str) -> NormalizedSWETask:
    repo_name = _required_text(row, "repo_name")
    repo = _optional_text(row.get("repo")) or repo_name
    gold_commit = _required_commit(row, "commit_hash")
    published_instance_id = _optional_text(row.get("instance_id"))
    expected = _json_object(row.get("expected_output_json"), "expected_output_json")
    if not expected:
        raise ValueError("R2E-Gym row has no expected test outcomes")
    invalid_status = sorted(
        {str(status).upper() for status in expected.values()} - _R2E_STATUS
    )
    if invalid_status:
        raise ValueError(f"R2E-Gym row has unknown test statuses: {invalid_status}")
    instance_id = f"r2e-{uuid.uuid4().hex}"
    base_commit = _optional_commit(row.get("base_commit"), "base_commit")
    return NormalizedSWETask(
        instance_id=instance_id,
        source_dataset=source_dataset,
        source_schema="r2e-gym-v1",
        repo=repo,
        problem_statement=_required_text(row, "problem_statement"),
        base_commit=base_commit,
        source_image=_required_text(row, "docker_image"),
        oracle_patch=_optional_text(row.get("patch")),
        verifier={
            "kind": "r2e-expected-pytest-map-v1",
            "gold_commit": gold_commit,
            "expected_output": expected,
        },
        source_metadata={
            "split": _optional_text(row.get("split")) or "train",
            "repo_name": repo_name,
            "published_instance_id": published_instance_id,
        },
        eval_only=False,
        task_binding=secrets.token_hex(32),
    )


def _normalize_swebench(row: dict[str, Any], source_dataset: str) -> NormalizedSWETask:
    instance_id = _required_text(row, "instance_id")
    tests = _swebench_tests(row, instance_id)
    is_swe_gym = "swe-gym" in source_dataset.lower()
    source_image = _optional_text(row.get("docker_image")) or _optional_text(row.get("image_name"))
    if source_image is None:
        source_image = (
            _swe_gym_image(instance_id)
            if is_swe_gym
            else _swebench_image(instance_id, namespace="swebench")
        )
    eval_only = not is_swe_gym
    return NormalizedSWETask(
        instance_id=instance_id,
        source_dataset=source_dataset,
        source_schema="swe-gym" if is_swe_gym else "swebench",
        repo=_required_text(row, "repo"),
        problem_statement=_required_text(row, "problem_statement"),
        base_commit=_required_commit(row, "base_commit"),
        source_image=source_image,
        oracle_patch=_optional_text(row.get("patch")),
        verifier={
            "kind": "swebench-harness-v1",
            "test_patch": _required_text(row, "test_patch"),
            **tests,
        },
        source_metadata={
            "split": _optional_text(row.get("split")) or ("train" if is_swe_gym else "test"),
            "version": _optional_text(row.get("version")),
        },
        eval_only=eval_only,
        task_binding=secrets.token_hex(32),
    )


def _normalize_swe_rebench_v2(row: dict[str, Any], source_dataset: str) -> NormalizedSWETask:
    instance_id = _required_text(row, "instance_id")
    install_config = row.get("install_config")
    if not isinstance(install_config, dict):
        raise ValueError(f"SWE-rebench V2 task {instance_id} requires install_config")
    test_cmd = _test_commands(install_config.get("test_cmd"))
    log_parser = _optional_text(install_config.get("log_parser"))
    if not test_cmd or not log_parser:
        raise ValueError(
            f"SWE-rebench V2 task {instance_id} requires install_config.test_cmd and log_parser"
        )
    return NormalizedSWETask(
        instance_id=instance_id,
        source_dataset=source_dataset,
        source_schema="swe-rebench-v2",
        repo=_required_text(row, "repo"),
        problem_statement=_required_text(row, "problem_statement"),
        base_commit=_required_commit(row, "base_commit"),
        source_image=_required_text(row, "image_name"),
        oracle_patch=_optional_text(row.get("patch")),
        verifier={
            "kind": "swe-rebench-v2",
            "test_patch": _required_text(row, "test_patch"),
            **_swebench_tests(row, instance_id),
            "install_config": install_config,
        },
        source_metadata={
            "split": _optional_text(row.get("split")) or "train",
            "language": _optional_text(row.get("language"))
            or _optional_text(row.get("repo_language")),
            "interface": _optional_text(row.get("interface")),
        },
        eval_only=False,
        task_binding=secrets.token_hex(32),
    )


def _swebench_tests(row: dict[str, Any], instance_id: str) -> dict[str, list[str]]:
    fail_to_pass = _string_list(row.get("FAIL_TO_PASS"), "FAIL_TO_PASS")
    pass_to_pass = _string_list(row.get("PASS_TO_PASS"), "PASS_TO_PASS")
    if not fail_to_pass:
        raise ValueError(f"SWE task {instance_id} has no FAIL_TO_PASS tests")
    return {"fail_to_pass": fail_to_pass, "pass_to_pass": pass_to_pass}


def _swebench_image(instance_id: str, *, namespace: str) -> str:
    escaped = instance_id.lower().replace("__", "_1776_")
    return f"docker.io/{namespace}/sweb.eval.x86_64.{escaped}:latest"


def _swe_gym_image(instance_id: str) -> str:
    escaped = instance_id.lower().replace("__", "_s_")
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{escaped}:latest"


def _required_text(row: dict[str, Any], key: str) -> str:
    value = _optional_text(row.get(key))
    if value is None:
        raise ValueError(f"required SWE field {key!r} is missing or empty")
    return value


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _required_commit(row: dict[str, Any], key: str) -> str:
    value = _optional_commit(row.get(key), key)
    if value is None:
        raise ValueError(f"required SWE commit field {key!r} is missing")
    return value


def _optional_commit(value: Any, key: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    if _FULL_COMMIT.fullmatch(text) is None:
        raise ValueError(f"SWE field {key!r} must be a full 40-character hexadecimal commit")
    return text.lower()


def _json_object(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SWE field {key!r} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"SWE field {key!r} must be a JSON object")
    return value


def _string_list(value: Any, key: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SWE field {key!r} is invalid JSON") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"SWE field {key!r} must be a list of strings")
    return [item for item in value if item]


def _test_commands(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    if any(not isinstance(command, str) for command in value):
        return []
    return [command for command in value if command.strip()]


def _stable_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
