"""Lossless packed Sample storage for fully-async rollout checkpoints."""

import copy
import hashlib
import time
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from miles.utils.types import Sample

SAMPLE_CODEC_STATE_KEY = "sample_codec"
SAMPLE_CODEC_VERSION = 3
SUPPORTED_SAMPLE_CODEC_VERSIONS = frozenset({1, 2, SAMPLE_CODEC_VERSION})
ARRAY_SHARD_BYTES = 256 * 1024 * 1024
_PACKED_RESPONSE_FIELD = "response"


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
_PACKED_RESPONSE_SPEC = _PackedFieldSpec(np.dtype(np.uint8), torch.uint8, str)
_PACKED_ARRAY_SPECS = {**_PACKED_FIELDS, _PACKED_RESPONSE_FIELD: _PACKED_RESPONSE_SPEC}


@dataclass(frozen=True)
class _CachedPackedField:
    source: Any
    tensor: torch.Tensor
    owner: torch.Tensor
    owner_sha256: str


@dataclass(frozen=True)
class _CachedSampleFields:
    sample_ref: weakref.ReferenceType[Sample]
    fields: dict[str, _CachedPackedField]
    static_state: dict[str, Any]


@dataclass(frozen=True)
class _PackedTensorChunk:
    tensor: torch.Tensor
    owner: torch.Tensor
    owner_sha256: str


class CheckpointPackedFieldCache:
    """Prepack immutable completed-sample fields while generation is active.

    Fully-async checkpoint mode rejects custom sample filters, and Miles only
    changes lifecycle metadata after generation completes. The source-list
    identity check still makes reassignment safe: a replaced field falls back to
    the ordinary lossless conversion at snapshot time.
    """

    def __init__(self) -> None:
        self._samples: dict[int, _CachedSampleFields] = {}
        self.total_pack_seconds = 0.0
        self.total_packed_bytes = 0
        self.live_packed_bytes = 0
        self.pack_operations = 0

    def cache_group(self, group: list[Sample | list[Sample]]) -> None:
        samples = [sample for item in group for sample in (item if isinstance(item, list) else [item])]
        started = time.perf_counter()
        fields_by_sample: list[dict[str, _CachedPackedField]] = [{} for _ in samples]
        for field, spec in _PACKED_FIELDS.items():
            packable = [
                (index, value)
                for index, sample in enumerate(samples)
                if _can_pack(value := getattr(sample, field, None), spec)
            ]
            if not packable:
                continue
            owner = torch.empty(sum(len(value) for _, value in packable), dtype=spec.torch_dtype)
            owner_numpy = owner.numpy()
            offset = 0
            slices = []
            for sample_index, value in packable:
                end = offset + len(value)
                owner_numpy[offset:end] = value
                slices.append((sample_index, value, owner[offset:end]))
                offset = end
            owner_sha256 = _tensor_digest(owner)
            for sample_index, source, tensor in slices:
                fields_by_sample[sample_index][field] = _CachedPackedField(
                    source=source,
                    tensor=tensor,
                    owner=owner,
                    owner_sha256=owner_sha256,
                )

        encoded_responses = [
            (index, value, _encode_response(value))
            for index, sample in enumerate(samples)
            if _can_pack_response(value := getattr(sample, _PACKED_RESPONSE_FIELD, None))
        ]
        if encoded_responses:
            owner = torch.empty(sum(len(encoded) for _, _, encoded in encoded_responses), dtype=torch.uint8)
            owner_numpy = owner.numpy()
            offset = 0
            slices = []
            for sample_index, source, encoded in encoded_responses:
                end = offset + len(encoded)
                owner_numpy[offset:end] = np.frombuffer(encoded, dtype=np.uint8)
                slices.append((sample_index, source, owner[offset:end]))
                offset = end
            owner_sha256 = _tensor_digest(owner)
            for sample_index, source, tensor in slices:
                fields_by_sample[sample_index][_PACKED_RESPONSE_FIELD] = _CachedPackedField(
                    source=source,
                    tensor=tensor,
                    owner=owner,
                    owner_sha256=owner_sha256,
                )

        for sample, fields in zip(samples, fields_by_sample, strict=True):
            self._replace_sample(sample, fields)
        packed_bytes = sum(field.tensor.nbytes for fields in fields_by_sample for field in fields.values())
        self.total_pack_seconds += time.perf_counter() - started
        self.total_packed_bytes += packed_bytes
        self.live_packed_bytes += packed_bytes
        self.pack_operations += len(samples)

    def cache_sample(self, sample: Sample) -> None:
        self.cache_group([sample])

    def _replace_sample(self, sample: Sample, fields: dict[str, _CachedPackedField]) -> None:
        key = id(sample)
        existing = self._samples.get(key)
        if existing is not None:
            self.live_packed_bytes -= sum(field.tensor.nbytes for field in existing.fields.values())
        static_state = sample.to_dict()
        static_state.pop("metadata", None)
        for field in _PACKED_ARRAY_SPECS:
            static_state.pop(field, None)

        sample_ref = weakref.ref(sample, lambda ref, sample_id=key: self._drop(sample_id, ref))
        self._samples[key] = _CachedSampleFields(
            sample_ref=sample_ref,
            fields=fields,
            static_state=copy.deepcopy(static_state),
        )

    def get_chunk(self, sample: Sample, field: str, source: Any) -> _PackedTensorChunk | None:
        cached = self._samples.get(id(sample))
        if cached is None or cached.sample_ref() is not sample:
            return None
        packed = cached.fields.get(field)
        if packed is None or packed.source is not source:
            return None
        return _PackedTensorChunk(
            tensor=packed.tensor,
            owner=packed.owner,
            owner_sha256=packed.owner_sha256,
        )

    def get_static_state(self, sample: Sample) -> dict[str, Any] | None:
        cached = self._samples.get(id(sample))
        if cached is None or cached.sample_ref() is not sample:
            return None
        return cached.static_state

    def _drop(self, sample_id: int, sample_ref: weakref.ReferenceType[Sample]) -> None:
        cached = self._samples.get(sample_id)
        if cached is not None and cached.sample_ref is sample_ref:
            self.live_packed_bytes -= sum(field.tensor.nbytes for field in cached.fields.values())
            del self._samples[sample_id]

    def stats(self) -> dict[str, float | int]:
        """Return lightweight observability without serializing cache payloads."""
        return {
            "live_samples": len(self._samples),
            "live_packed_bytes": self.live_packed_bytes,
            "total_packed_bytes": self.total_packed_bytes,
            "pack_operations": self.pack_operations,
            "total_pack_seconds": self.total_pack_seconds,
        }


class CheckpointSampleEncoder:
    """Build one identity-deduplicated Sample table for a coherent snapshot."""

    def __init__(self, packed_cache: CheckpointPackedFieldCache | None = None) -> None:
        self._packed_cache = packed_cache
        self._record_ids: dict[tuple[int, tuple[tuple[str, int], ...]], int] = {}
        self._records: list[dict[str, Any]] = []
        # Identity is the deduplication contract. Retain every source object so
        # CPython cannot recycle an id while this table is being constructed.
        self._source_samples: list[Sample] = []
        self._chunks: dict[str, list[list[Any] | _PackedTensorChunk]] = {field: [] for field in _PACKED_ARRAY_SPECS}
        self._lengths = {field: 0 for field in _PACKED_ARRAY_SPECS}
        self._finished = False

    def encode_group(
        self,
        group: list[Sample | list[Sample]],
        *,
        metadata_updates: dict[str, int] | None = None,
        use_packed_cache: bool = False,
    ) -> list[int | list[int]]:
        """Encode group shape as Sample-table references without mutating it."""
        encoded: list[int | list[int]] = []
        for item in group:
            if isinstance(item, list):
                encoded.append(
                    [
                        self.encode_sample(
                            sample,
                            metadata_updates=metadata_updates,
                            use_packed_cache=use_packed_cache,
                        )
                        for sample in item
                    ]
                )
            else:
                encoded.append(
                    self.encode_sample(
                        item,
                        metadata_updates=metadata_updates,
                        use_packed_cache=use_packed_cache,
                    )
                )
        return encoded

    def encode_sample(
        self,
        sample: Sample,
        *,
        metadata_updates: dict[str, int] | None = None,
        use_packed_cache: bool = False,
    ) -> int:
        """Return an identity reference, optionally overlaying serialized metadata."""
        if self._finished:
            raise RuntimeError("Cannot add Samples after the checkpoint codec is finalized")
        update_key = tuple(sorted((metadata_updates or {}).items()))
        identity = (id(sample), update_key)
        if identity in self._record_ids:
            return self._record_ids[identity]

        cached_state = (
            self._packed_cache.get_static_state(sample)
            if use_packed_cache and self._packed_cache is not None
            else None
        )
        sample_state = (
            {**cached_state, "metadata": copy.deepcopy(sample.metadata)}
            if cached_state is not None
            else sample.to_dict()
        )
        if metadata_updates:
            metadata = sample_state.get("metadata")
            if not isinstance(metadata, dict):
                raise TypeError(f"Sample metadata must be a mapping, got {type(metadata).__name__}")
            sample_state["metadata"] = {**metadata, **metadata_updates}

        slices = {}
        for field, spec in _PACKED_FIELDS.items():
            value = getattr(sample, field, None)
            cached = (
                self._packed_cache.get_chunk(sample, field, value)
                if use_packed_cache and self._packed_cache is not None
                else None
            )
            if cached is None and not _can_pack(value, spec):
                # The prepack cache deliberately omits these large fields from
                # its static state. Re-read the live value so reassignment (and
                # distinctions such as None versus []) remain lossless.
                sample_state[field] = copy.deepcopy(value)
                continue
            start = self._lengths[field]
            length = len(value)
            slices[field] = (start, length)
            self._chunks[field].append(cached if cached is not None else value)
            self._lengths[field] += length
            sample_state.pop(field, None)

        response = getattr(sample, _PACKED_RESPONSE_FIELD, None)
        cached_response = (
            self._packed_cache.get_chunk(sample, _PACKED_RESPONSE_FIELD, response)
            if use_packed_cache and self._packed_cache is not None
            else None
        )
        if cached_response is None and not _can_pack_response(response):
            sample_state[_PACKED_RESPONSE_FIELD] = copy.deepcopy(response)
        else:
            response_chunk = cached_response or _pack_response(response)
            start = self._lengths[_PACKED_RESPONSE_FIELD]
            length = response_chunk.tensor.numel()
            slices[_PACKED_RESPONSE_FIELD] = (start, length)
            self._chunks[_PACKED_RESPONSE_FIELD].append(response_chunk)
            self._lengths[_PACKED_RESPONSE_FIELD] += length
            sample_state.pop(_PACKED_RESPONSE_FIELD, None)

        record_id = len(self._records)
        self._record_ids[identity] = record_id
        self._source_samples.append(sample)
        # A cached static state is already a private deep copy. Only lifecycle
        # metadata and fallback packed fields were copied above, so another full
        # deepcopy would add checkpoint-boundary CPU without increasing isolation.
        record_state = sample_state if cached_state is not None else copy.deepcopy(sample_state)
        self._records.append({"state": record_state, "slices": slices})
        return record_id

    def finish(self) -> dict[str, Any]:
        """Detach packed values into contiguous CPU tensors."""
        if self._finished:
            raise RuntimeError("Checkpoint Sample codec was already finalized")
        self._finished = True
        packed = {
            field: _detach_chunks_into_shards(self._chunks[field], spec) for field, spec in _PACKED_ARRAY_SPECS.items()
        }
        arrays = {field: value[0] for field, value in packed.items()}
        array_checksums = {field: value[1] for field, value in packed.items()}
        self._chunks.clear()
        self._source_samples.clear()
        self._record_ids.clear()
        return {
            "version": SAMPLE_CODEC_VERSION,
            "records": self._records,
            "arrays": arrays,
            "array_checksums": array_checksums,
        }


class _CheckpointSampleDecoder:
    def __init__(self, codec: dict[str, Any]) -> None:
        version = codec.get("version") if isinstance(codec, dict) else None
        if version not in SUPPORTED_SAMPLE_CODEC_VERSIONS:
            raise RuntimeError(f"Unsupported fully-async Sample codec: {version!r}")
        self._version = version
        self._records = codec.get("records")
        self._arrays = codec.get("arrays")
        self._array_specs = _PACKED_FIELDS if version < 3 else _PACKED_ARRAY_SPECS
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
        if not isinstance(slices, dict) or slices.keys() - self._array_specs.keys():
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
            if start < 0 or length < 0 or start + length > self._array_lengths[field]:
                raise RuntimeError(f"Out-of-bounds {field} slice in fully-async Sample table record {reference}")
            if field == _PACKED_RESPONSE_FIELD:
                sample_state[field] = self._slice_to_response(start, length)
            else:
                sample_state[field] = self._slice_to_list(field, start, length)
        return sample_state

    def _validate_arrays(self) -> None:
        if self._arrays.keys() != self._array_specs.keys():
            raise RuntimeError("Fully-async Sample codec has an unexpected packed-array set")
        self._array_lengths = {}
        for field, spec in self._array_specs.items():
            value = self._arrays[field]
            arrays = [value] if self._version == 1 else value
            if not isinstance(arrays, list):
                raise RuntimeError(f"Fully-async Sample codec {field} shards must be a list")
            for array in arrays:
                if not isinstance(array, torch.Tensor) or array.device.type != "cpu":
                    raise RuntimeError(f"Fully-async Sample codec {field} array must be a CPU tensor")
                if array.dtype != spec.torch_dtype or array.ndim != 1:
                    raise RuntimeError(
                        f"Fully-async Sample codec {field} array must be rank-one {spec.torch_dtype}, "
                        f"got {array.dtype} rank {array.ndim}"
                    )
            self._arrays[field] = arrays
            self._array_lengths[field] = sum(array.numel() for array in arrays)

    def _slice_to_list(self, field: str, start: int, length: int) -> list[Any]:
        if length == 0:
            return []
        values = []
        offset = 0
        end = start + length
        for array in self._arrays[field]:
            shard_end = offset + array.numel()
            if shard_end > start and offset < end:
                local_start = max(start - offset, 0)
                local_end = min(end - offset, array.numel())
                values.extend(array[local_start:local_end].tolist())
            if shard_end >= end:
                break
            offset = shard_end
        return values

    def _slice_to_response(self, start: int, length: int) -> str:
        values = bytearray()
        offset = 0
        end = start + length
        for array in self._arrays[_PACKED_RESPONSE_FIELD]:
            shard_end = offset + array.numel()
            if shard_end > start and offset < end:
                local_start = max(start - offset, 0)
                local_end = min(end - offset, array.numel())
                values.extend(memoryview(array[local_start:local_end].numpy()).cast("B"))
            if shard_end >= end:
                break
            offset = shard_end
        return values.decode("utf-8", errors="surrogatepass")


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
    # Keep empty lists in the ordinary Sample state. They have no payload to
    # compact, and emitting multiple zero-length slices at the same file offset
    # would make external tensor references ambiguous during validation.
    if type(value) is not list or not value:
        return False
    for item in value:
        if type(item) is not spec.element_type:
            return False
        if spec.minimum is not None and (item < spec.minimum or item > spec.maximum):
            return False
    return True


def _can_pack_response(value: Any) -> bool:
    return type(value) is str and bool(value)


def _encode_response(value: str) -> bytes:
    return value.encode("utf-8", errors="surrogatepass")


def _pack_response(value: str) -> _PackedTensorChunk:
    encoded = _encode_response(value)
    tensor = torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy())
    return _PackedTensorChunk(
        tensor=tensor,
        owner=tensor,
        owner_sha256=_tensor_digest(tensor),
    )


def _detach_chunks_into_shards(
    chunks: list[list[Any] | _PackedTensorChunk],
    spec: _PackedFieldSpec,
) -> tuple[list[torch.Tensor], list[str]]:
    """Keep prepacked tensors and convert only uncached list runs."""
    arrays = []
    checksums = []
    pending_lists = []
    pending_length = 0
    chunk_index = 0
    while chunk_index < len(chunks):
        chunk = chunks[chunk_index]
        if isinstance(chunk, _PackedTensorChunk):
            packed_arrays = _pack_list_chunks(pending_lists, pending_length, spec)
            arrays.extend(packed_arrays)
            checksums.extend(_tensor_digest(tensor) for tensor in packed_arrays)
            pending_lists = []
            pending_length = 0

            owner = chunk.owner
            owner_chunks = []
            while chunk_index < len(chunks):
                candidate = chunks[chunk_index]
                if not isinstance(candidate, _PackedTensorChunk) or candidate.owner is not owner:
                    break
                owner_chunks.append(candidate.tensor)
                chunk_index += 1
            if _chunks_cover_owner(owner_chunks, owner):
                split = _split_large_tensor(owner, spec)
                split_checksums = (
                    [chunk.owner_sha256] if len(split) == 1 else [_tensor_digest(tensor) for tensor in split]
                )
            else:
                split = owner_chunks
                split_checksums = [_tensor_digest(tensor) for tensor in split]
            arrays.extend(split)
            checksums.extend(split_checksums)
        else:
            pending_lists.append(chunk)
            pending_length += len(chunk)
            chunk_index += 1
    packed_arrays = _pack_list_chunks(pending_lists, pending_length, spec)
    arrays.extend(packed_arrays)
    checksums.extend(_tensor_digest(tensor) for tensor in packed_arrays)
    return arrays, checksums


def _chunks_cover_owner(chunks: list[torch.Tensor], owner: torch.Tensor) -> bool:
    address = owner.data_ptr()
    for chunk in chunks:
        if chunk.data_ptr() != address:
            return False
        address += chunk.nbytes
    return address == owner.data_ptr() + owner.nbytes


def _pack_list_chunks(
    chunks: list[list[Any]],
    length: int,
    spec: _PackedFieldSpec,
) -> list[torch.Tensor]:
    if not chunks:
        return []
    elements_per_shard = max(1, ARRAY_SHARD_BYTES // spec.numpy_dtype.itemsize)
    arrays = []
    chunk_index = 0
    chunk_offset = 0
    remaining = length
    while remaining:
        shard_length = min(elements_per_shard, remaining)
        output = torch.empty(shard_length, dtype=spec.torch_dtype)
        output_numpy = output.numpy()
        output_offset = 0
        while output_offset < shard_length:
            chunk = chunks[chunk_index]
            available = len(chunk) - chunk_offset
            take = min(available, shard_length - output_offset)
            source = chunk[chunk_offset : chunk_offset + take]
            if isinstance(source, torch.Tensor):
                output[output_offset : output_offset + take].copy_(source)
            else:
                output_numpy[output_offset : output_offset + take] = source
            output_offset += take
            chunk_offset += take
            if chunk_offset == len(chunk):
                chunk_index += 1
                chunk_offset = 0
        arrays.append(output)
        remaining -= shard_length
    return arrays


def _split_large_tensor(tensor: torch.Tensor, spec: _PackedFieldSpec) -> list[torch.Tensor]:
    elements_per_shard = max(1, ARRAY_SHARD_BYTES // spec.numpy_dtype.itemsize)
    if tensor.numel() <= elements_per_shard:
        return [tensor]
    return [chunk.clone() for chunk in tensor.split(elements_per_shard)]


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(memoryview(tensor.numpy()).cast("B")).hexdigest()
