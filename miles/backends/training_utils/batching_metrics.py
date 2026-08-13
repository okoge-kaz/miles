from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from miles.utils.ft_utils.process_group_utils import GeneralPGUtil, GroupInfo


@dataclass(frozen=True)
class LocalBatchingMetrics:
    """Token scheduling statistics for one local optimizer step.

    ``microbatch_tokens`` contains padding-inclusive scheduled token slots, so
    its sum equals ``scheduled_tokens``.
    """

    useful_tokens: int
    scheduled_tokens: int
    microbatch_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.useful_tokens < 0 or self.scheduled_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if sum(self.microbatch_tokens) != self.scheduled_tokens:
            raise ValueError("microbatch token counts must sum to scheduled_tokens")


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return (value + multiple - 1) // multiple * multiple


def _step_microbatch_indices(
    *,
    micro_batch_size: int | None,
    micro_batch_indices: Sequence[Sequence[int]] | None,
    num_microbatches_by_step: Sequence[int],
    step_id: int,
) -> tuple[tuple[int, ...], ...]:
    microbatch_offset = sum(num_microbatches_by_step[:step_id])
    num_microbatches = num_microbatches_by_step[step_id]

    if micro_batch_indices is not None:
        end = microbatch_offset + num_microbatches
        return tuple(tuple(indices) for indices in micro_batch_indices[microbatch_offset:end])

    if micro_batch_size is None:
        raise ValueError("fixed batching requires micro_batch_size")

    sample_offset = microbatch_offset * micro_batch_size
    return tuple(tuple(range(sample_offset + i * micro_batch_size, sample_offset + (i + 1) * micro_batch_size)) for i in range(num_microbatches))


def _thd_scheduled_tokens(
    lengths: Sequence[int],
    *,
    tp_size: int,
    cp_size: int,
    pad_multiplier: int,
    allgather_cp: bool,
) -> int:
    pad_size = tp_size * pad_multiplier
    if allgather_cp:
        return _round_up(sum(lengths), cp_size * pad_size)

    if cp_size == 1:
        local_tokens = sum(lengths)
    else:
        local_tokens = sum(2 * _round_up(length, 2 * cp_size) // (2 * cp_size) for length in lengths)
    return _round_up(local_tokens, pad_size) * cp_size


def _bshd_scheduled_tokens(
    indices: Sequence[int],
    *,
    total_lengths: Sequence[int],
    max_seq_lens: Sequence[int] | None,
    cp_size: int,
    allgather_cp: bool,
) -> int:
    if max_seq_lens is None:
        raise ValueError("bshd batching metrics require max_seq_lens")
    if not indices:
        return 0

    max_seq_len = int(max_seq_lens[indices[0]])
    if any(int(total_lengths[index]) > max_seq_len for index in indices):
        raise ValueError("max_seq_lens is smaller than a sequence in its microbatch")

    if allgather_cp or cp_size == 1:
        scheduled_per_sample = max_seq_len
    else:
        local_tokens = 2 * _round_up(max_seq_len, 2 * cp_size) // (2 * cp_size)
        scheduled_per_sample = local_tokens * cp_size
    return len(indices) * scheduled_per_sample


def compute_local_batching_metrics(
    *,
    rollout_data: Mapping[str, Any],
    micro_batch_size: int | None,
    micro_batch_indices: Sequence[Sequence[int]] | None,
    num_microbatches_by_step: Sequence[int],
    step_id: int,
    qkv_format: str,
    tp_size: int,
    cp_size: int,
    pad_multiplier: int,
    allgather_cp: bool,
) -> LocalBatchingMetrics:
    """Compute padding-aware metrics without consuming or changing the data iterator."""

    total_lengths = rollout_data["total_lengths"]
    max_seq_lens = rollout_data.get("max_seq_lens")
    index_groups = _step_microbatch_indices(
        micro_batch_size=micro_batch_size,
        micro_batch_indices=micro_batch_indices,
        num_microbatches_by_step=num_microbatches_by_step,
        step_id=step_id,
    )

    useful_by_microbatch = []
    scheduled_by_microbatch = []
    for indices in index_groups:
        lengths = tuple(int(total_lengths[index]) for index in indices)
        useful_by_microbatch.append(sum(lengths))
        if qkv_format == "thd":
            scheduled_by_microbatch.append(
                _thd_scheduled_tokens(
                    lengths,
                    tp_size=tp_size,
                    cp_size=cp_size,
                    pad_multiplier=pad_multiplier,
                    allgather_cp=allgather_cp,
                )
            )
        elif qkv_format == "bshd":
            scheduled_by_microbatch.append(
                _bshd_scheduled_tokens(
                    indices,
                    total_lengths=total_lengths,
                    max_seq_lens=max_seq_lens,
                    cp_size=cp_size,
                    allgather_cp=allgather_cp,
                )
            )
        else:
            raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    useful_tokens = sum(useful_by_microbatch)
    scheduled_tokens = sum(scheduled_by_microbatch)
    if scheduled_tokens < useful_tokens:
        raise AssertionError(f"scheduled tokens ({scheduled_tokens}) must cover useful tokens ({useful_tokens})")
    return LocalBatchingMetrics(
        useful_tokens=useful_tokens,
        scheduled_tokens=scheduled_tokens,
        microbatch_tokens=tuple(scheduled_by_microbatch),
    )


def _summarize_batching_metrics(rows: Sequence[LocalBatchingMetrics]) -> dict[str, float]:
    """Summarize padding-inclusive load across the DP replicas of one step.

    ``dp_token_imbalance`` is the bottleneck DP rank's scheduled load divided
    by mean scheduled load, minus one. Zero therefore means perfect balance.
    """

    useful_tokens = sum(row.useful_tokens for row in rows)
    scheduled_tokens = sum(row.scheduled_tokens for row in rows)
    microbatch_tokens = [value for row in rows for value in row.microbatch_tokens]
    dp_tokens = [row.scheduled_tokens for row in rows]

    packing_efficiency = useful_tokens / scheduled_tokens if scheduled_tokens else 1.0
    mean_dp_tokens = statistics.fmean(dp_tokens) if dp_tokens else 0.0
    dp_token_imbalance = max(dp_tokens) / mean_dp_tokens - 1.0 if mean_dp_tokens else 0.0

    return {
        "useful_tokens": float(useful_tokens),
        "scheduled_tokens": float(scheduled_tokens),
        "padding_or_unused_token_frac": 1.0 - packing_efficiency,
        "microbatch_token_min": float(min(microbatch_tokens, default=0)),
        "microbatch_token_max": float(max(microbatch_tokens, default=0)),
        "microbatch_token_p50": float(statistics.median(microbatch_tokens)) if microbatch_tokens else 0.0,
        "dp_token_imbalance": dp_token_imbalance,
        "packing_efficiency": packing_efficiency,
    }


def aggregate_batching_metrics(
    local_metrics: LocalBatchingMetrics,
    dp: GroupInfo,
) -> dict[str, float]:
    """Gather one small fixed-size row per DP rank and summarize it on every rank."""

    if dp.size == 1:
        return _summarize_batching_metrics([local_metrics])

    encoded = torch.tensor(
        [local_metrics.useful_tokens, local_metrics.scheduled_tokens, *local_metrics.microbatch_tokens],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    gathered = [torch.empty_like(encoded) for _ in range(dp.size)]
    GeneralPGUtil.create(dp.group).all_gather(gathered, encoded, dp.group)
    rows = [
        LocalBatchingMetrics(
            useful_tokens=int(row[0].item()),
            scheduled_tokens=int(row[1].item()),
            microbatch_tokens=tuple(int(value) for value in row[2:].tolist()),
        )
        for row in gathered
    ]
    return _summarize_batching_metrics(rows)
