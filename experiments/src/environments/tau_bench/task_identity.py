"""Pinned Tau three task identity shared by data preparation and rollout."""

from __future__ import annotations

import hashlib
import json
from typing import Any

TAU_RELEASE = "v1.0.1"
TAU_PACKAGE_VERSION = "1.0.1"
TAU_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU_VERIFIER = "tau3_environment"
TAU_DOMAINS = ("retail", "airline", "telecom")
TAU_SPLITS = ("train", "test", "base")


def task_dict(task: Any) -> dict[str, Any]:
    """Return a stable JSON-compatible representation of an official task."""

    if hasattr(task, "model_dump"):
        return task.model_dump(mode="json")
    return task.dict()


def task_digest(task: dict[str, Any]) -> str:
    """Hash one task independently of JSON formatting."""

    payload = json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_task_identity(metadata: dict[str, Any], task: Any) -> None:
    """Fail closed unless metadata identifies the pinned official Tau three task."""

    if metadata.get("verifier") != TAU_VERIFIER:
        raise ValueError(f"Tau metadata must use verifier {TAU_VERIFIER!r}")
    if metadata.get("tau_release") != TAU_RELEASE:
        raise ValueError(f"Tau metadata is not pinned to release {TAU_RELEASE}")
    if metadata.get("tau_commit") != TAU_COMMIT:
        raise ValueError(f"Tau metadata is not pinned to commit {TAU_COMMIT}")

    domain = str(metadata.get("tau_domain") or "")
    split = str(metadata.get("tau_split") or "")
    task_id = str(metadata.get("tau_task_id") or "")
    if domain not in TAU_DOMAINS:
        raise ValueError(f"unsupported Tau three domain {domain!r}")
    if split not in TAU_SPLITS:
        raise ValueError(f"unsupported Tau three split {split!r}")
    if str(task.id) != task_id:
        raise ValueError(f"Tau three task ID mismatch: metadata={task_id!r}, runtime={task.id!r}")

    digest = task_digest(task_dict(task))
    if metadata.get("tau_task_sha256") != digest:
        raise ValueError(f"Tau three task {domain}/{task_id} digest does not match the pinned runtime")
