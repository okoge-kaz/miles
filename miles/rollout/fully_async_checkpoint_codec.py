"""Lossless packed Sample storage for fully-async rollout checkpoints."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from miles.utils.types import Sample

SAMPLE_CODEC_STATE_KEY = "sample_codec"
SAMPLE_CODEC_VERSION = 1


@dataclass(frozen=True)
class _PackedFieldSpec:
    numpy_dtype: np.dtype
    torch_dtype: torch.dtype
    element_type: type
    minimum: int | None = None
    maximum: int | None = None


# Use compact integer representations only after validating every value. Python
# floats are binary64, so float64 preserves their exact value, including -0 and
# NaN payloads supported by NumPy conversion.
_PACKED_FIELDS = {
    "tokens": _PackedFieldSpec(np.dtype(np.int32), torch.int32, int, -(2**31), 2**31 - 1),
    "loss_mask": _PackedFieldSpec(np.dtype(np.int8), torch.int8, int, -(2**7), 2**7 - 1),
    "rollout_log_probs": _PackedFieldSpec(np.dtype(np.float64), torch.float64, float),
    "teacher_log_probs": _PackedFieldSpec(np.dtype(np.float64), torch.float64, float),
    "opd_reverse_kl": _PackedFieldSpec(np.dtype(np.float64), torch.float64, float),
}


class CheckpointSampleEncoder:
    """Build one identity-deduplicated Sample table for a coherent snapshot."""

    def __init__(self) -> None:
        self._record_ids: dict[tuple[int, tuple[tuple[str, int], ...]], int] = {}
        self._records: list[dict[str, Any]] = []
        # Identity is the deduplication contract. Retain every source object so
        # CPython cannot recycle an id while this table is being constructed.
        self._source_samples: list[Sample] = []
        self._chunks: dict[str, list[list[Any]]] = {field: [] for field in _PACKED_FIELDS}
        self._lengths = {field: 0 for field in _PACKED_FIELDS}
        self._finished = False

    def encode_group(
        self,
        group: list[Sample | list[Sample]],
        *,
        metadata_updates: dict[str, int] | None = None,
    ) -> list[int | list[int]]:
        """Encode group shape as Sample-table references without mutating it."""
        encoded: list[int | list[int]] = []
        for item in group:
            if isinstance(item, list):
                encoded.append(
                    [self.encode_sample(sample, metadata_updates=metadata_updates) for sample in item]
                )
            else:
                encoded.append(self.encode_sample(item, metadata_updates=metadata_updates))
        return encoded

    def encode_sample(self, sample: Sample, *, metadata_updates: dict[str, int] | None = None) -> int:
        """Return an identity reference, optionally overlaying serialized metadata."""
        if self._finished:
            raise RuntimeError("Cannot add Samples after the checkpoint codec is finalized")
        update_key = tuple(sorted((metadata_updates or {}).items()))
        identity = (id(sample), update_key)
        if identity in self._record_ids:
            return self._record_ids[identity]

        sample_state = sample.to_dict()
        if metadata_updates:
            metadata = sample_state.get("metadata")
            if not isinstance(metadata, dict):
                raise TypeError(f"Sample metadata must be a mapping, got {type(metadata).__name__}")
            sample_state["metadata"] = {**metadata, **metadata_updates}

        slices = {}
        for field, spec in _PACKED_FIELDS.items():
            value = sample_state.get(field)
            if not _can_pack(value, spec):
                continue
            start = self._lengths[field]
            length = len(value)
            slices[field] = (start, length)
            self._chunks[field].append(value)
            self._lengths[field] += length
            del sample_state[field]

        record_id = len(self._records)
        self._record_ids[identity] = record_id
        self._source_samples.append(sample)
        self._records.append({"state": copy.deepcopy(sample_state), "slices": slices})
        return record_id

    def finish(self) -> dict[str, Any]:
        """Detach packed values into contiguous CPU tensors."""
        if self._finished:
            raise RuntimeError("Checkpoint Sample codec was already finalized")
        self._finished = True
        arrays = {
            field: torch.from_numpy(_concatenate_chunks(self._chunks[field], self._lengths[field], spec))
            for field, spec in _PACKED_FIELDS.items()
        }
        self._chunks.clear()
        self._source_samples.clear()
        self._record_ids.clear()
        return {
            "version": SAMPLE_CODEC_VERSION,
            "records": self._records,
            "arrays": arrays,
        }


class _CheckpointSampleDecoder:
    def __init__(self, codec: dict[str, Any]) -> None:
        version = codec.get("version") if isinstance(codec, dict) else None
        if version != SAMPLE_CODEC_VERSION:
            raise RuntimeError(f"Unsupported fully-async Sample codec: {version!r}")
        self._records = codec.get("records")
        self._arrays = codec.get("arrays")
        if not isinstance(self._records, list) or not isinstance(self._arrays, dict):
            raise RuntimeError("Malformed fully-async Sample codec table")
        self._validate_arrays()

    def decode_group(self, group: list[int | list[int]]) -> list[dict[str, Any] | list[dict[str, Any]]]:
        decoded = []
        for item in group:
            if isinstance(item, list):
                decoded.append([self.decode_sample(reference) for reference in item])
            else:
                decoded.append(self.decode_sample(item))
        return decoded

    def decode_sample(self, reference: int) -> dict[str, Any]:
        if type(reference) is not int or reference < 0 or reference >= len(self._records):
            raise RuntimeError(f"Invalid fully-async Sample table reference: {reference!r}")
        record = self._records[reference]
        if not isinstance(record, dict) or not isinstance(record.get("state"), dict):
            raise RuntimeError(f"Malformed fully-async Sample table record {reference}")
        slices = record.get("slices")
        if not isinstance(slices, dict) or slices.keys() - _PACKED_FIELDS.keys():
            raise RuntimeError(f"Malformed packed fields in fully-async Sample table record {reference}")

        sample_state = copy.deepcopy(record["state"])
        for field, location in slices.items():
            if (
                not isinstance(location, (tuple, list))
                or len(location) != 2
                or type(location[0]) is not int
                or type(location[1]) is not int
            ):
                raise RuntimeError(f"Malformed {field} slice in fully-async Sample table record {reference}")
            start, length = location
            array = self._arrays[field]
            if start < 0 or length < 0 or start + length > array.numel():
                raise RuntimeError(f"Out-of-bounds {field} slice in fully-async Sample table record {reference}")
            sample_state[field] = array[start : start + length].tolist()
        return sample_state

    def _validate_arrays(self) -> None:
        if self._arrays.keys() != _PACKED_FIELDS.keys():
            raise RuntimeError("Fully-async Sample codec has an unexpected packed-array set")
        for field, spec in _PACKED_FIELDS.items():
            array = self._arrays[field]
            if not isinstance(array, torch.Tensor) or array.device.type != "cpu":
                raise RuntimeError(f"Fully-async Sample codec {field} array must be a CPU tensor")
            if array.dtype != spec.torch_dtype or array.ndim != 1:
                raise RuntimeError(
                    f"Fully-async Sample codec {field} array must be rank-one {spec.torch_dtype}, "
                    f"got {array.dtype} rank {array.ndim}"
                )


def materialize_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    """Expand a packed checkpoint into the legacy detached-dictionary contract."""
    codec = state.get(SAMPLE_CODEC_STATE_KEY)
    if codec is None:
        return state
    decoder = _CheckpointSampleDecoder(codec)
    materialized = {key: value for key, value in state.items() if key != SAMPLE_CODEC_STATE_KEY}
    materialized["pending_prompts"] = [decoder.decode_group(group) for group in state["pending_prompts"]]
    materialized["ready_items"] = [
        {
            **item,
            "prompt_group": decoder.decode_group(item["prompt_group"]),
            "result": decoder.decode_group(item["result"]),
        }
        for item in state["ready_items"]
    ]
    materialized["drain_progress"] = [
        {**progress, "data": [decoder.decode_group(group) for group in progress["data"]]}
        for progress in state["drain_progress"]
    ]
    materialized["prepared_batches"] = [
        {**prepared, "samples": [decoder.decode_group(group) for group in prepared["samples"]]}
        for prepared in state["prepared_batches"]
    ]
    return materialized


def _can_pack(value: Any, spec: _PackedFieldSpec) -> bool:
    if type(value) is not list:
        return False
    for item in value:
        if type(item) is not spec.element_type:
            return False
        if spec.minimum is not None and (item < spec.minimum or item > spec.maximum):
            return False
    return True


def _concatenate_chunks(chunks: list[list[Any]], length: int, spec: _PackedFieldSpec) -> np.ndarray:
    output = np.empty(length, dtype=spec.numpy_dtype)
    offset = 0
    for chunk in chunks:
        next_offset = offset + len(chunk)
        output[offset:next_offset] = chunk
        offset = next_offset
    return output
