"""Pinned Tau task identity shared by data preparation and rollout runtime."""

from __future__ import annotations

import hashlib
import json
from typing import Any

TAU_COMMIT = "09c26a85efd1d65168cfb57865ca2ca278c8153d"


def _task_dict(task: Any) -> dict[str, Any]:
    if hasattr(task, "model_dump"):
        return task.model_dump()
    return task.dict()


def _task_digest(task: dict[str, Any]) -> str:
    payload = json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_task_identity(metadata: dict[str, Any], task: Any) -> None:
    """Fail closed unless metadata identifies the pinned official Tau task."""

    if metadata.get("tau_commit") != TAU_COMMIT:
        raise ValueError(f"Tau metadata is not pinned to commit {TAU_COMMIT}")
    task_index = int(metadata["tau_task_index"])
    task_digest = _task_digest(_task_dict(task))
    if metadata.get("tau_task_sha256") != task_digest:
        raise ValueError(f"Tau task {task_index} digest does not match the pinned environment")
