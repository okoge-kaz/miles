import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.utils.types import Sample

REPLAY_DATA_BUFFER_SCHEMA_VERSION = 2


def replay_data_buffer_directory(root: str | Path, rollout_id: int) -> Path:
    if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0:
        raise ValueError("ReplayDataBuffer checkpoint iteration must be a non-negative integer")
    # Schema 1 keeps the original on-disk name so existing runs remain resumable.
    return Path(root) / "rollout" / f"replay_{rollout_id:07d}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: dict) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_part(path: Path, value) -> dict:
    with path.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": path.name, "sha256": _hash_file(path), "bytes": path.stat().st_size}


def encode_group(group):
    return [[sample.to_dict() for sample in item] if isinstance(item, list) else item.to_dict() for item in group]


def decode_group(group):
    return [
        [Sample.from_dict(sample) for sample in item] if isinstance(item, list) else Sample.from_dict(item)
        for item in group
    ]


def _encode_entry(entry: DataBufferInput) -> dict:
    return {
        "prompt_group": encode_group(entry.prompt_group),
        "group": encode_group(entry.group),
    }


def _decode_entry(value: dict) -> DataBufferInput:
    return DataBufferInput(
        prompt_group=decode_group(value["prompt_group"]),
        group=decode_group(value["group"]),
    )


def save_replay_data_buffer_checkpoint(
    root: str | Path,
    rollout_id: int,
    *,
    entries: Sequence[DataBufferInput],
    pending_prompts: Sequence[list[Sample]],
    source_state: dict,
    served_version: int,
    configuration_fingerprint: str,
    groups_per_part: int,
    mode: str = "rollout",
    inflight_entries: Sequence[DataBufferInput] = (),
) -> Path:
    if mode not in ("rollout", "inflight") or (inflight_entries and mode != "inflight"):
        raise ValueError("Invalid ReplayDataBuffer mode or inflight entries in rollout mode")
    destination = replay_data_buffer_directory(root, rollout_id)
    if destination.exists():
        raise FileExistsError(f"ReplayDataBuffer checkpoint already exists: {destination}")
    if groups_per_part < 1 or isinstance(served_version, bool) or served_version < 0:
        raise ValueError("Invalid ReplayDataBuffer part size or served version")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        part_count = _write_ready_parts(temporary, entries=entries, groups_per_part=groups_per_part)
        auxiliary = _write_part(
            temporary / "auxiliary.pt",
            {
                "pending_prompts": [encode_group(group) for group in pending_prompts],
                "inflight_entries": [_encode_entry(entry) for entry in inflight_entries],
            },
        )
        source_metadata = _write_part(temporary / "source_metadata.pt", source_state.get("metadata", {}))
        state = {key: value for key, value in source_state.items() if key != "metadata"}
        header = {
            "schema": REPLAY_DATA_BUFFER_SCHEMA_VERSION,
            "mode": mode,
            "inflight_count": len(inflight_entries),
            "rollout_id": rollout_id,
            "entry_count": len(entries),
            "part_count": part_count,
            "groups_per_part": groups_per_part,
            "served_version": served_version,
            "source_state": state,
            "configuration_fingerprint": configuration_fingerprint,
            "auxiliary": auxiliary,
            "source_metadata": source_metadata,
            "index_sha256": _hash_file(temporary / "index.jsonl"),
        }
        _write_json(temporary / "header.json", header)
        _sync_directory(temporary)
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _write_ready_parts(path: Path, *, entries, groups_per_part: int) -> int:
    count = 0
    with (path / "index.jsonl").open("w") as index:
        for start in range(0, len(entries), groups_per_part):
            chunk = entries[start : start + groups_per_part]
            record = _write_part(path / f"part_{count:08d}.pt", [_encode_entry(entry) for entry in chunk])
            record["groups"] = len(chunk)
            index.write(json.dumps(record) + "\n")
            count += 1
        index.flush()
        os.fsync(index.fileno())
    return count


def commit_replay_data_buffer_checkpoint(root: str | Path, rollout_id: int) -> None:
    """Called only after the model save has finished and this rollout snapshot exists."""
    directory = replay_data_buffer_directory(root, rollout_id)
    marker = {"rollout_id": rollout_id, "header_sha256": _hash_file(directory / "header.json")}
    temporary = directory / ".COMMITTED.json.tmp"
    _write_json(temporary, marker)
    os.replace(temporary, directory / "COMMITTED.json")
    _sync_directory(directory)


def load_replay_data_buffer_header(root: str | Path, rollout_id: int) -> dict:
    directory = replay_data_buffer_directory(root, rollout_id)
    header_path = directory / "header.json"
    if not (directory / "COMMITTED.json").is_file():
        raise ValueError(f"Model iteration {rollout_id} has no committed ReplayDataBuffer checkpoint: {directory}")
    if header_path.stat().st_size > 128 * 1024:
        raise ValueError("ReplayDataBuffer header exceeds its bounded size")
    marker = json.loads((directory / "COMMITTED.json").read_text())
    if marker.get("rollout_id") != rollout_id or marker.get("header_sha256") != _hash_file(header_path):
        raise ValueError("ReplayDataBuffer checkpoint header/commit marker mismatch")
    header = json.loads(header_path.read_text())
    if header.get("schema") not in (1, REPLAY_DATA_BUFFER_SCHEMA_VERSION) or header.get("rollout_id") != rollout_id:
        raise ValueError("Unsupported ReplayDataBuffer schema or model iteration mismatch")
    if header.get("mode", "rollout") not in ("rollout", "inflight"):
        raise ValueError("Unknown ReplayDataBuffer checkpoint mode")
    for key in ("entry_count", "part_count", "served_version"):
        if isinstance(header.get(key), bool) or not isinstance(header.get(key), int) or header[key] < 0:
            raise ValueError(f"Invalid ReplayDataBuffer header {key}")
    return header


def _load_verified_part(directory: Path, record: dict, *, expected_name: str | None = None):
    name = record.get("file", "")
    if name != expected_name and not (expected_name is None and re.fullmatch(r"part_\d{8}\.pt", name)):
        raise ValueError("Unsafe or out-of-order ReplayDataBuffer part filename")
    path = directory / name
    if path.is_symlink() or path.stat().st_size != record.get("bytes") or _hash_file(path) != record.get("sha256"):
        raise ValueError(f"ReplayDataBuffer part checksum/size mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def iter_replay_data_buffer_parts(root: str | Path, rollout_id: int, header: dict) -> Iterator[list[DataBufferInput]]:
    directory = replay_data_buffer_directory(root, rollout_id)
    count = total = 0
    digest = hashlib.sha256()
    with (directory / "index.jsonl").open("rb") as index:
        for line in index:
            if len(line) > 4096:
                raise ValueError("Oversized ReplayDataBuffer index record")
            digest.update(line)
            record = json.loads(line)
            if record.get("file") != f"part_{count:08d}.pt":
                raise ValueError("ReplayDataBuffer parts must have a contiguous stream order")
            values = _load_verified_part(directory, record)
            if len(values) != record.get("groups") or not 0 < len(values) <= header["groups_per_part"]:
                raise ValueError("ReplayDataBuffer part group count mismatch")
            count += 1
            total += len(values)
            if count > header["part_count"] or total > header["entry_count"]:
                raise ValueError("ReplayDataBuffer index exceeds committed counts")
            yield [_decode_entry(value) for value in values]
    if count != header["part_count"] or total != header["entry_count"] or digest.hexdigest() != header["index_sha256"]:
        raise ValueError("ReplayDataBuffer index is incomplete or its checksum does not match")


def load_replay_data_buffer_auxiliary(root: str | Path, rollout_id: int, header: dict) -> dict:
    value = _load_verified_part(
        replay_data_buffer_directory(root, rollout_id), header["auxiliary"], expected_name="auxiliary.pt"
    )
    inflight = [_decode_entry(entry) for entry in value.get("inflight_entries", [])]
    if len(inflight) != header.get("inflight_count", 0) or (inflight and header.get("mode") != "inflight"):
        raise ValueError("ReplayDataBuffer inflight count/mode mismatch")
    return {
        "pending_prompts": [decode_group(group) for group in value["pending_prompts"]],
        "inflight_entries": inflight,
    }


def load_replay_data_buffer_source_state(root: str | Path, rollout_id: int, header: dict) -> dict:
    metadata = _load_verified_part(
        replay_data_buffer_directory(root, rollout_id), header["source_metadata"], expected_name="source_metadata.pt"
    )
    return {**header["source_state"], "metadata": metadata}


def optional_replay_data_buffer_header(root: str | Path, rollout_id: int) -> dict | None:
    rollout = Path(root) / "rollout"
    has_rollout_state = rollout.is_dir() and (
        replay_data_buffer_directory(root, rollout_id).exists()
        or next(rollout.glob("global_dataset_state_dict_*.pt"), None) is not None
        or next(rollout.glob("replay_*"), None) is not None
    )
    # Converted initial weights have no rollout history. Once training history
    # exists, a missing selected snapshot is a partial checkpoint, not a fresh run.
    return load_replay_data_buffer_header(root, rollout_id) if has_rollout_state else None


def resume_weight_version(args) -> int:
    if not getattr(args, "use_replay_data_buffer", False) or args.load is None:
        return 0
    tracker = Path(args.load) / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or getattr(args, "finetune", False):
        return 0
    iteration = getattr(args, "ckpt_step", None)
    if iteration is None:
        raw = tracker.read_text().strip()
        if raw == "release":
            return 0
        iteration = int(raw)
    header = optional_replay_data_buffer_header(args.load, iteration)
    return header["served_version"] if header is not None else 0
