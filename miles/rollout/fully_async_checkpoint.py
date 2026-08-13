"""Durable state helpers for fully-asynchronous rollout replay.

The model checkpoint tracker remains the commit record.  A rollout state with a
newer id is harmless when model saving fails; a model checkpoint without the
matching, checksum-verified rollout state is rejected during resume.
"""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from miles.utils.types import Sample

SCHEMA_VERSION = 1
STATE_PREFIX = "fully_async_state_"
STATE_SUFFIX = ".pt"
CHECKSUM_SUFFIX = ".sha256.json"
REQUIRED_STATE_KEYS = {
    "dataset_fingerprint",
    "data_source",
    "applied_weight_version",
    "pending_prompts",
    "ready_items",
    "drain_progress",
    "prepared_batches",
    "regeneration_group_ids",
}


def encode_sample(sample: Sample) -> dict[str, Any]:
    """Return a detached, serialization-safe representation of one sample."""
    return copy.deepcopy(sample.to_dict())


def decode_sample(state: dict[str, Any]) -> Sample:
    return Sample.from_dict(copy.deepcopy(state))


def encode_group(group: list[Sample | list[Sample]]) -> list[dict[str, Any] | list[dict[str, Any]]]:
    encoded = []
    for item in group:
        if isinstance(item, list):
            encoded.append([encode_sample(sample) for sample in item])
        else:
            encoded.append(encode_sample(item))
    return encoded


def decode_group(state: list[dict[str, Any] | list[dict[str, Any]]]) -> list[Sample | list[Sample]]:
    decoded = []
    for item in state:
        if isinstance(item, list):
            decoded.append([decode_sample(sample) for sample in item])
        else:
            decoded.append(decode_sample(item))
    return decoded


def prompt_group_id(group: list[Sample]) -> int:
    ids = {sample.group_index for sample in group}
    if len(ids) != 1 or None in ids:
        raise RuntimeError(f"A prompt group must have one non-null group_index, got {sorted(ids, key=str)}")
    return int(ids.pop())


def rollout_batch_token(groups: list[list[Sample | list[Sample]]]) -> str:
    """Identify the exact prompt groups admitted into a trainer batch."""
    identities = []
    for group in groups:
        samples = [sample for item in group for sample in (item if isinstance(item, list) else [item])]
        ids = {sample.group_index for sample in samples}
        if len(ids) != 1 or None in ids:
            raise RuntimeError(f"Generated group has inconsistent group_index values: {ids}")
        identities.append(
            {
                "group_index": int(ids.pop()),
                "samples": [
                    {
                        "index": sample.index,
                        "retry_count": sample.retry_count,
                        "status": sample.status.value,
                        "response_length": sample.response_length,
                    }
                    for sample in samples
                ],
            }
        )
    payload = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def dataset_fingerprint(args, data_source) -> str:
    """Fingerprint inputs that determine prompt identity and cursor ordering."""
    prompt_path = getattr(args, "prompt_data", None)
    file_digest = _hash_prompt_file(prompt_path) if prompt_path else None
    chat_template_path = getattr(args, "chat_template_path", None)
    dataset = getattr(data_source, "dataset", None)
    config = {
        "prompt_sha256": file_digest,
        "prompt_slice": _prompt_slice(prompt_path),
        "dataset_size": len(dataset) if dataset is not None else None,
        "hf_checkpoint": getattr(args, "hf_checkpoint", None),
        "tokenizer_model": getattr(args, "tokenizer_model", None),
        "tokenizer_type": getattr(args, "tokenizer_type", None),
        "chat_template_path": chat_template_path,
        "chat_template_sha256": _hash_file(chat_template_path) if chat_template_path else None,
        "n_samples_per_prompt": getattr(args, "n_samples_per_prompt", None),
        "rollout_batch_size": getattr(args, "rollout_batch_size", None),
        "async_max_concurrent_samples": getattr(args, "async_max_concurrent_samples", None),
        "rollout_seed": getattr(args, "rollout_seed", None),
        "rollout_shuffle": getattr(args, "rollout_shuffle", None),
        "rollout_max_prompt_len": getattr(args, "rollout_max_prompt_len", None),
        "input_key": getattr(args, "input_key", None),
        "label_key": getattr(args, "label_key", None),
        "metadata_key": getattr(args, "metadata_key", None),
        "tool_key": getattr(args, "tool_key", None),
        "multimodal_keys": getattr(args, "multimodal_keys", None),
        "apply_chat_template": getattr(args, "apply_chat_template", None),
        "apply_chat_template_kwargs": getattr(args, "apply_chat_template_kwargs", None),
        "data_source_path": getattr(args, "data_source_path", None),
        "custom_generate_function_path": getattr(args, "custom_generate_function_path", None),
        "dynamic_sampling_filter_path": getattr(args, "dynamic_sampling_filter_path", None),
        "rollout_sample_filter_path": getattr(args, "rollout_sample_filter_path", None),
        "rollout_max_response_len": getattr(args, "rollout_max_response_len", None),
        "rollout_max_context_len": getattr(args, "rollout_max_context_len", None),
        "rollout_temperature": getattr(args, "rollout_temperature", None),
        "rollout_top_p": getattr(args, "rollout_top_p", None),
        "rollout_top_k": getattr(args, "rollout_top_k", None),
        "rollout_stop": getattr(args, "rollout_stop", None),
        "rollout_stop_token_ids": getattr(args, "rollout_stop_token_ids", None),
        "rollout_skip_special_tokens": getattr(args, "rollout_skip_special_tokens", None),
        "rollout_task_type": getattr(args, "rollout_task_type", None),
        "rollout_data_postprocess_path": getattr(args, "rollout_data_postprocess_path", None),
        "staleness_reference": getattr(args, "staleness_reference", None),
        "max_weight_staleness": getattr(args, "max_weight_staleness", None),
        "pause_generation_mode": getattr(args, "pause_generation_mode", None),
        "advantage_estimator": getattr(args, "advantage_estimator", None),
        "global_batch_size": getattr(args, "global_batch_size", None),
        "num_steps_per_rollout": getattr(args, "num_steps_per_rollout", None),
        "rm_type": getattr(args, "rm_type", None),
        "rm_url": getattr(args, "rm_url", None),
        "custom_rm_path": getattr(args, "custom_rm_path", None),
        "group_rm": getattr(args, "group_rm", None),
        "reward_key": getattr(args, "reward_key", None),
        "custom_reward_post_process_path": getattr(args, "custom_reward_post_process_path", None),
        "custom_convert_samples_to_train_data_path": getattr(args, "custom_convert_samples_to_train_data_path", None),
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_path(root: str | os.PathLike[str], rollout_id: int) -> Path:
    return Path(root) / "rollout" / f"{STATE_PREFIX}{rollout_id}{STATE_SUFFIX}"


def ensure_no_full_replay_sidecar(root: str | os.PathLike[str] | None, rollout_id: int | None) -> None:
    """Reject a mode downgrade that would silently discard pending prompt state."""
    if root is None or rollout_id is None or rollout_id < 0:
        return
    path = checkpoint_path(root, rollout_id)
    checksum_path = Path(f"{path}{CHECKSUM_SUFFIX}")
    if path.is_file() or checksum_path.is_file():
        raise RuntimeError(
            f"Model checkpoint {rollout_id} has a fully-async replay sidecar; resume with "
            "--fully-async-rollout-checkpoint instead of silently discarding its pending prompt ledger"
        )


def save_checkpoint(root: str | os.PathLike[str], rollout_id: int, state: dict[str, Any]) -> tuple[Path, int]:
    """Atomically publish state and its checksum, returning path and byte size."""
    path = checkpoint_path(root, rollout_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**state, "schema_version": SCHEMA_VERSION, "checkpoint_rollout_id": rollout_id}

    data_tmp = _temporary_path(path.parent, path.name)
    checksum_path = Path(f"{path}{CHECKSUM_SUFFIX}")
    checksum_tmp = _temporary_path(path.parent, checksum_path.name)
    try:
        torch.save(state, data_tmp)
        data_tmp.chmod(0o644)
        _fsync_file(data_tmp)
        digest, size = _file_digest(data_tmp)
        manifest = {"schema_version": SCHEMA_VERSION, "sha256": digest, "size": size}
        checksum_tmp.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        checksum_tmp.chmod(0o644)
        _fsync_file(checksum_tmp)

        os.replace(data_tmp, path)
        os.replace(checksum_tmp, checksum_path)
        _fsync_directory(path.parent)
    finally:
        data_tmp.unlink(missing_ok=True)
        checksum_tmp.unlink(missing_ok=True)
    return path, size


def load_checkpoint(
    root: str | os.PathLike[str],
    rollout_id: int,
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    path = checkpoint_path(root, rollout_id)
    checksum_path = Path(f"{path}{CHECKSUM_SUFFIX}")
    if not path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint {rollout_id} requires fully-async rollout state {path} and {checksum_path}"
        )

    try:
        manifest = json.loads(checksum_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read fully-async rollout checkpoint manifest: {checksum_path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported fully-async rollout checkpoint manifest: {checksum_path}")
    digest, size = _file_digest(path)
    if manifest.get("size") != size or manifest.get("sha256") != digest:
        raise RuntimeError(f"Fully-async rollout checkpoint checksum mismatch: {path}")

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"Cannot deserialize fully-async rollout checkpoint: {path}") from error
    if not isinstance(state, dict):
        raise RuntimeError(f"Fully-async rollout checkpoint is not a mapping: {path}")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported fully-async rollout checkpoint schema {state.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    if state.get("checkpoint_rollout_id") != rollout_id:
        raise RuntimeError(
            f"Rollout checkpoint id mismatch: requested={rollout_id}, " f"stored={state.get('checkpoint_rollout_id')}"
        )
    if state.get("dataset_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "Fully-async rollout checkpoint dataset/config fingerprint does not match this run: "
            f"stored={state.get('dataset_fingerprint')}, current={expected_fingerprint}"
        )
    missing_keys = REQUIRED_STATE_KEYS - state.keys()
    if missing_keys:
        raise RuntimeError(f"Fully-async rollout checkpoint is missing fields: {sorted(missing_keys)}")
    return state


def prune_checkpoints(
    root: str | os.PathLike[str],
    *,
    current_rollout_id: int,
    keep_last: int,
    archive_interval: int | None,
) -> None:
    """Retain recent states plus ids kept by the model's archival interval."""
    rollout_dir = Path(root) / "rollout"
    if not rollout_dir.is_dir():
        return
    checkpoints = sorted(
        (rollout_id, path)
        for path in rollout_dir.glob(f"{STATE_PREFIX}*{STATE_SUFFIX}")
        if (rollout_id := _rollout_id_from_path(path)) is not None and rollout_id <= current_rollout_id
    )
    recent_ids = {rollout_id for rollout_id, _ in checkpoints[-max(keep_last, 1) :]}
    for rollout_id, path in checkpoints:
        if rollout_id in recent_ids:
            continue
        if archive_interval is not None and archive_interval > 0 and (rollout_id + 1) % archive_interval == 0:
            continue
        path.unlink(missing_ok=True)
        Path(f"{path}{CHECKSUM_SUFFIX}").unlink(missing_ok=True)


def _hash_prompt_file(prompt_path: str) -> str:
    real_path = prompt_path.split("@[", maxsplit=1)[0]
    return _hash_file(real_path)


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_slice(prompt_path: str | None) -> str | None:
    if prompt_path is None or "@[" not in prompt_path:
        return None
    return "@[" + prompt_path.split("@[", maxsplit=1)[1]


def _temporary_path(directory: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=f".{prefix}.", suffix=".tmp")
    os.close(descriptor)
    return Path(name)


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rollout_id_from_path(path: Path) -> int | None:
    stem = path.name.removeprefix(STATE_PREFIX).removesuffix(STATE_SUFFIX)
    try:
        return int(stem)
    except ValueError:
        return None
