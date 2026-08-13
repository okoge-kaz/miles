"""Durable state helpers for fully-asynchronous rollout replay.

The model checkpoint tracker remains the commit record.  A rollout state with a
newer id is harmless when model saving fails; a model checkpoint without the
matching, checksum-verified rollout state is rejected during resume.
"""

import copy
import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import torch

from miles.rollout.fully_async_checkpoint_codec import SAMPLE_CODEC_VERSION, materialize_checkpoint_state
from miles.utils.types import Sample

SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, SCHEMA_VERSION})
STATE_PREFIX = "fully_async_state_"
STATE_SUFFIX = ".pt"
CHECKSUM_SUFFIX = ".sha256.json"
PARTS_MARKER = ".parts-"
MAX_PARALLEL_IO_WORKERS = 8
TENSOR_PART_BYTES = 256 * 1024 * 1024
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
    """Atomically publish state and checksum-verified tensor shards."""
    path = checkpoint_path(root, rollout_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**state, "schema_version": SCHEMA_VERSION, "checkpoint_rollout_id": rollout_id}

    data_tmp = _temporary_path(path.parent, path.name)
    checksum_path = Path(f"{path}{CHECKSUM_SUFFIX}")
    checksum_tmp = _temporary_path(path.parent, checksum_path.name)
    parts_tmp = None
    parts_path = None
    published = False
    try:
        parts_name = f"{path.name}{PARTS_MARKER}{uuid.uuid4().hex}"
        state, tensor_parts = _externalize_tensor_arrays(state, parts_name)
        part_manifests = []
        if tensor_parts:
            parts_tmp = Path(tempfile.mkdtemp(dir=path.parent, prefix=f".{parts_name}.", suffix=".tmp"))
            part_manifests = _write_tensor_parts(parts_tmp, tensor_parts)
            parts_tmp.chmod(0o755)
            _fsync_directory(parts_tmp)
            parts_path = path.parent / parts_name
            os.replace(parts_tmp, parts_path)
            parts_tmp = None
            _fsync_directory(path.parent)

        digest, main_size = _write_torch_file(data_tmp, state)
        total_size = main_size + sum(part["size"] for part in part_manifests)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "sha256": digest,
            "size": total_size,
            "main_size": main_size,
            "parts_directory": parts_name if tensor_parts else None,
            "parts": part_manifests,
        }
        checksum_tmp.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        checksum_tmp.chmod(0o644)
        _fsync_file(checksum_tmp)

        os.replace(data_tmp, path)
        os.replace(checksum_tmp, checksum_path)
        _fsync_directory(path.parent)
        published = True
        _cleanup_part_directories(path, keep=parts_name if tensor_parts else None)
    finally:
        data_tmp.unlink(missing_ok=True)
        checksum_tmp.unlink(missing_ok=True)
        if parts_tmp is not None:
            shutil.rmtree(parts_tmp, ignore_errors=True)
        if parts_path is not None and not published:
            shutil.rmtree(parts_path, ignore_errors=True)
    return path, total_size


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
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise RuntimeError(f"Unsupported fully-async rollout checkpoint manifest: {checksum_path}")
    digest, main_size = _file_digest(path)
    expected_main_size = manifest.get("main_size", manifest.get("size"))
    if expected_main_size != main_size or manifest.get("sha256") != digest:
        raise RuntimeError(f"Fully-async rollout checkpoint checksum mismatch: {path}")
    if manifest.get("schema_version") == SCHEMA_VERSION:
        parts = manifest.get("parts")
        if not isinstance(parts, list) or any(not isinstance(part, dict) for part in parts):
            raise RuntimeError(f"Malformed fully-async rollout checkpoint tensor manifest: {checksum_path}")
        part_sizes = [part.get("size") for part in parts]
        if any(type(size) is not int or size < 0 for size in part_sizes):
            raise RuntimeError(f"Malformed fully-async rollout checkpoint tensor sizes: {checksum_path}")
        if manifest.get("size") != main_size + sum(part_sizes):
            raise RuntimeError(f"Fully-async rollout checkpoint size mismatch: {path}")

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"Cannot deserialize fully-async rollout checkpoint: {path}") from error
    if not isinstance(state, dict):
        raise RuntimeError(f"Fully-async rollout checkpoint is not a mapping: {path}")
    if state.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"Unsupported fully-async rollout checkpoint schema {state.get('schema_version')}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if state.get("schema_version") != manifest.get("schema_version"):
        raise RuntimeError(
            "Fully-async rollout checkpoint schema does not match its manifest: "
            f"state={state.get('schema_version')}, manifest={manifest.get('schema_version')}"
        )
    if state.get("schema_version") == SCHEMA_VERSION:
        state = _load_external_tensor_arrays(path, state, manifest)
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
    return materialize_checkpoint_state(state)


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
        _cleanup_part_directories(path, keep=None)


def _externalize_tensor_arrays(
    state: dict[str, Any],
    parts_directory: str,
) -> tuple[dict[str, Any], list[tuple[str, list[tuple[int, torch.Tensor]]]]]:
    codec = state.get("sample_codec")
    if not isinstance(codec, dict) or codec.get("version") != SAMPLE_CODEC_VERSION:
        return state, []
    arrays = codec.get("arrays")
    array_checksums = codec.get("array_checksums")
    if not isinstance(arrays, dict) or not isinstance(array_checksums, dict):
        raise RuntimeError("Malformed fully-async Sample codec arrays")
    if arrays.keys() != array_checksums.keys():
        raise RuntimeError("Fully-async Sample codec arrays and checksums do not match")

    tensor_parts = []
    external_arrays = {}
    part_tensors = []
    part_bytes = 0
    part_name = None

    def flush_part() -> None:
        nonlocal part_tensors, part_bytes, part_name
        if part_tensors:
            tensor_parts.append((part_name, part_tensors))
            part_tensors = []
            part_bytes = 0
            part_name = None

    for field, shards in arrays.items():
        if not isinstance(shards, list):
            raise RuntimeError(f"Malformed fully-async Sample codec shards for {field}")
        checksums = array_checksums[field]
        if not isinstance(checksums, list) or len(checksums) != len(shards):
            raise RuntimeError(f"Malformed fully-async Sample codec checksums for {field}")
        references = []
        for shard_index, tensor in enumerate(shards):
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.ndim != 1:
                raise RuntimeError(f"Fully-async Sample codec shard {field}[{shard_index}] is invalid")
            padding = -part_bytes % tensor.element_size()
            if part_tensors and part_bytes + padding + tensor.nbytes > TENSOR_PART_BYTES:
                flush_part()
                padding = 0
            if part_name is None:
                part_name = f"arrays-{len(tensor_parts):05d}.bin"
            offset = part_bytes + padding
            part_tensors.append((padding, tensor))
            part_bytes = offset + tensor.nbytes
            references.append(
                {
                    "file": part_name,
                    "offset": offset,
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                    "sha256": checksums[shard_index],
                }
            )
        external_arrays[field] = references
    flush_part()

    external_codec = {
        **codec,
        "arrays": external_arrays,
        "arrays_external": True,
        "parts_directory": parts_directory if tensor_parts else None,
    }
    external_codec.pop("array_checksums")
    return {**state, "sample_codec": external_codec}, tensor_parts


def _write_tensor_parts(
    directory: Path,
    tensor_parts: list[tuple[str, list[tuple[int, torch.Tensor]]]],
) -> list[dict[str, Any]]:
    def write_part(item: tuple[str, list[tuple[int, torch.Tensor]]]) -> dict[str, Any]:
        filename, tensors = item
        size = _write_raw_tensor_file(directory / filename, tensors)
        return {"file": filename, "size": size}

    workers = min(MAX_PARALLEL_IO_WORKERS, len(tensor_parts))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        manifests = list(executor.map(write_part, tensor_parts))
    return manifests


def _load_external_tensor_arrays(
    checkpoint: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    codec = state.get("sample_codec")
    if not isinstance(codec, dict) or not codec.get("arrays_external"):
        if manifest.get("parts"):
            raise RuntimeError(f"Fully-async rollout checkpoint has unreferenced tensor parts: {checkpoint}")
        return state

    parts_directory = codec.get("parts_directory")
    manifest_directory = manifest.get("parts_directory")
    if parts_directory != manifest_directory:
        raise RuntimeError(f"Fully-async rollout checkpoint tensor-part directory mismatch: {checkpoint}")
    if parts_directory is not None and not _is_safe_part_directory(checkpoint, parts_directory):
        raise RuntimeError(f"Unsafe fully-async rollout checkpoint tensor-part directory: {parts_directory!r}")

    part_manifests = manifest.get("parts")
    if not isinstance(part_manifests, list):
        raise RuntimeError(f"Malformed fully-async rollout checkpoint tensor manifest: {checkpoint}")
    manifest_by_name = {}
    for part in part_manifests:
        if not isinstance(part, dict) or not _is_safe_part_filename(part.get("file")):
            raise RuntimeError(f"Malformed fully-async rollout checkpoint tensor part: {checkpoint}")
        filename = part["file"]
        if filename in manifest_by_name:
            raise RuntimeError(f"Duplicate fully-async rollout checkpoint tensor part {filename}")
        manifest_by_name[filename] = part

    arrays = codec.get("arrays")
    if not isinstance(arrays, dict):
        raise RuntimeError(f"Malformed fully-async rollout checkpoint external arrays: {checkpoint}")
    if any(not isinstance(field_refs, list) for field_refs in arrays.values()):
        raise RuntimeError(f"Malformed fully-async rollout checkpoint external array references: {checkpoint}")
    references = [reference for field_refs in arrays.values() for reference in field_refs]
    referenced_names = []
    referenced_tensors = []
    for reference in references:
        if not isinstance(reference, dict) or not _is_safe_part_filename(reference.get("file")):
            raise RuntimeError(f"Malformed fully-async rollout checkpoint tensor reference: {checkpoint}")
        if type(reference.get("offset")) is not int or reference["offset"] < 0:
            raise RuntimeError(f"Malformed fully-async rollout checkpoint tensor offset: {checkpoint}")
        referenced_names.append(reference["file"])
        referenced_tensors.append((reference["file"], reference.get("offset")))
    if len(referenced_tensors) != len(set(referenced_tensors)) or set(referenced_names) != set(manifest_by_name):
        raise RuntimeError(f"Fully-async rollout checkpoint tensor references do not match its manifest: {checkpoint}")

    if referenced_names and parts_directory is None:
        raise RuntimeError(f"Fully-async rollout checkpoint tensor parts have no directory: {checkpoint}")
    parts_path = checkpoint.parent / parts_directory if parts_directory is not None else None

    def load_part(filename: str) -> tuple[str, torch.Tensor]:
        part = manifest_by_name[filename]
        part_path = parts_path / filename
        try:
            size = part_path.stat().st_size
        except OSError as error:
            raise RuntimeError(f"Cannot read fully-async rollout checkpoint tensor: {part_path}") from error
        if part.get("size") != size:
            raise RuntimeError(f"Fully-async rollout checkpoint checksum mismatch: {part_path}")
        return filename, torch.from_file(str(part_path), shared=False, size=size, dtype=torch.uint8)

    loaded = {}
    if referenced_names:
        filenames = list(dict.fromkeys(referenced_names))
        workers = min(MAX_PARALLEL_IO_WORKERS, len(filenames))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            loaded.update(executor.map(load_part, filenames))

    hydrated_arrays = {}
    for field, field_refs in arrays.items():
        hydrated_arrays[field] = []
        for reference in field_refs:
            storage = loaded[reference["file"]]
            offset = reference.get("offset")
            dtype = _torch_dtype(reference.get("dtype"))
            numel = reference.get("numel")
            if (
                type(offset) is not int
                or offset < 0
                or type(numel) is not int
                or numel < 0
                or offset + numel * dtype.itemsize > storage.numel()
                or offset % dtype.itemsize != 0
            ):
                raise RuntimeError(
                    f"Fully-async rollout checkpoint tensor offset mismatch: "
                    f"{parts_path / reference['file']}"
                )
            tensor = storage[offset : offset + numel * dtype.itemsize].view(dtype)
            if _tensor_digest(tensor) != reference.get("sha256"):
                raise RuntimeError(
                    f"Fully-async rollout checkpoint checksum mismatch: "
                    f"{parts_path / reference['file']}"
                )
            hydrated_arrays[field].append(tensor)

    hydrated_codec = dict(codec)
    hydrated_codec["arrays"] = hydrated_arrays
    hydrated_codec.pop("arrays_external")
    hydrated_codec.pop("parts_directory")
    return {**state, "sample_codec": hydrated_codec}


def _is_safe_part_filename(value: Any) -> bool:
    return isinstance(value, str) and value == Path(value).name and value.endswith(".bin")


def _is_safe_part_directory(checkpoint: Path, value: Any) -> bool:
    return (
        isinstance(value, str)
        and value == Path(value).name
        and value.startswith(f"{checkpoint.name}{PARTS_MARKER}")
    )


def _cleanup_part_directories(path: Path, *, keep: str | None) -> None:
    for candidate in path.parent.glob(f"{path.name}{PARTS_MARKER}*"):
        if candidate.name != keep and candidate.is_dir():
            # The newly published manifest no longer references these paths.
            # Cleanup failure is a space leak, not a reason to report an
            # otherwise durable checkpoint as failed.
            shutil.rmtree(candidate, ignore_errors=True)


def _write_torch_file(path: Path, value: Any) -> tuple[str, int]:
    writer = _HashingWriter(path)
    try:
        torch.save(value, writer)
        writer.flush()
        os.fchmod(writer.fileno(), 0o644)
        os.fsync(writer.fileno())
        return writer.hexdigest(), writer.size
    finally:
        writer.close()


def _write_raw_tensor_file(
    path: Path,
    tensors: list[tuple[int, torch.Tensor]],
) -> int:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    buffers = []
    size = 0
    try:
        for padding, tensor in tensors:
            if padding:
                buffers.append(memoryview(bytes(padding)))
            if not tensor.is_contiguous():
                raise RuntimeError("Fully-async checkpoint tensor shards must be contiguous")
            buffers.append(memoryview(tensor.numpy()).cast("B"))
        for value in buffers:
            size += len(value)
        _writev_all(descriptor, buffers)
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        return size
    finally:
        os.close(descriptor)


def _writev_all(descriptor: int, buffers: list[memoryview]) -> None:
    max_iov = os.sysconf("SC_IOV_MAX")
    index = 0
    offset = 0
    while index < len(buffers):
        batch = buffers[index : index + max_iov]
        if offset:
            batch[0] = batch[0][offset:]
        written = os.writev(descriptor, batch)
        if written <= 0:
            raise OSError("Short write while saving fully-async checkpoint tensor shards")
        while index < len(buffers):
            remaining = len(buffers[index]) - offset
            if written < remaining:
                offset += written
                break
            written -= remaining
            index += 1
            offset = 0


def _torch_dtype(value: Any) -> torch.dtype:
    dtypes = {
        str(torch.int8): torch.int8,
        str(torch.int32): torch.int32,
        str(torch.float64): torch.float64,
        str(torch.uint8): torch.uint8,
    }
    if value not in dtypes:
        raise RuntimeError(f"Unsupported fully-async checkpoint tensor dtype: {value!r}")
    return dtypes[value]


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(memoryview(tensor.numpy()).cast("B")).hexdigest()


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


class _HashingWriter:
    """Hash bytes as torch.save writes them, avoiding a second file read."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("wb")
        self._digest = hashlib.sha256()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def write(self, value) -> int:
        written = self._handle.write(value)
        if written != len(value):
            raise OSError(f"Short write while saving fully-async checkpoint: {written} of {len(value)} bytes")
        self._digest.update(value)
        self._size += written
        return written

    def flush(self) -> None:
        self._handle.flush()

    def fileno(self) -> int:
        return self._handle.fileno()

    def tell(self) -> int:
        return self._handle.tell()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def close(self) -> None:
        self._handle.close()


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
