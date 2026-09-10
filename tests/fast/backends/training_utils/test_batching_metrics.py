from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.cp_utils as cp_utils_mod
import miles.backends.training_utils.data as data_mod
from miles.backends.training_utils.batching_metrics import (
    LocalBatchingMetrics,
    _summarize_batching_metrics,
    compute_local_batching_metrics,
)
from miles.backends.training_utils.data import DataIterator, get_batch


def _compute(
    total_lengths: list[int],
    *,
    micro_batch_size: int | None = None,
    micro_batch_indices: list[list[int]] | None = None,
    num_microbatches_by_step: list[int] | None = None,
    step_id: int = 0,
    qkv_format: str = "thd",
    tp_size: int = 1,
    cp_size: int = 1,
    pad_multiplier: int = 128,
    allgather_cp: bool = False,
    max_seq_lens: list[int] | None = None,
) -> LocalBatchingMetrics:
    rollout_data = {"total_lengths": total_lengths}
    if max_seq_lens is not None:
        rollout_data["max_seq_lens"] = max_seq_lens
    return compute_local_batching_metrics(
        rollout_data=rollout_data,
        micro_batch_size=micro_batch_size,
        micro_batch_indices=micro_batch_indices,
        num_microbatches_by_step=num_microbatches_by_step or [1],
        step_id=step_id,
        qkv_format=qkv_format,
        tp_size=tp_size,
        cp_size=cp_size,
        pad_multiplier=pad_multiplier,
        allgather_cp=allgather_cp,
    )


def test_fixed_thd_metrics_include_per_microbatch_padding() -> None:
    metrics = _compute(
        [100, 200, 300, 400],
        micro_batch_size=2,
        num_microbatches_by_step=[2],
        tp_size=2,
    )

    assert metrics == LocalBatchingMetrics(
        useful_tokens=1_000,
        scheduled_tokens=1_280,
        microbatch_tokens=(512, 768),
    )


def test_dynamic_schedule_selects_the_requested_optimizer_step() -> None:
    metrics = _compute(
        [10, 20, 30, 40, 50, 60],
        micro_batch_indices=[[0, 2], [1], [3, 5], [4]],
        num_microbatches_by_step=[2, 2],
        step_id=1,
    )

    assert metrics == LocalBatchingMetrics(
        useful_tokens=150,
        scheduled_tokens=256,
        microbatch_tokens=(128, 128),
    )


@pytest.mark.parametrize("allgather_cp", [False, True])
def test_thd_context_parallel_padding_matches_forward_layout(allgather_cp: bool) -> None:
    metrics = _compute(
        [5, 8],
        micro_batch_size=2,
        cp_size=2,
        pad_multiplier=4,
        allgather_cp=allgather_cp,
    )

    assert metrics.useful_tokens == 13
    assert metrics.scheduled_tokens == 16


@pytest.mark.parametrize(
    ("cp_size", "allgather_cp", "expected_scheduled"),
    [(1, False, 32), (2, False, 24), (2, True, 20)],
)
def test_bshd_metrics_use_forward_max_sequence_length(
    cp_size: int,
    allgather_cp: bool,
    expected_scheduled: int,
) -> None:
    metrics = _compute(
        [5, 8],
        micro_batch_size=2,
        qkv_format="bshd",
        cp_size=cp_size,
        allgather_cp=allgather_cp,
        max_seq_lens=[10 if cp_size == 2 else 16] * 2,
    )

    assert metrics.scheduled_tokens == expected_scheduled


def test_global_summary_reports_packing_and_dp_balance() -> None:
    summary = _summarize_batching_metrics(
        [
            LocalBatchingMetrics(useful_tokens=100, scheduled_tokens=128, microbatch_tokens=(64, 64)),
            LocalBatchingMetrics(useful_tokens=200, scheduled_tokens=256, microbatch_tokens=(128, 128)),
        ]
    )

    assert summary == pytest.approx(
        {
            "useful_tokens": 300.0,
            "scheduled_tokens": 384.0,
            "padding_or_unused_token_frac": 0.21875,
            "microbatch_token_min": 64.0,
            "microbatch_token_max": 128.0,
            "microbatch_token_p50": 96.0,
            "dp_token_imbalance": 1.0 / 3.0,
            "packing_efficiency": 0.78125,
        }
    )


def test_bshd_requires_precomputed_max_sequence_lengths() -> None:
    with pytest.raises(ValueError, match="max_seq_lens"):
        _compute([5, 8], micro_batch_size=2, qkv_format="bshd")


@pytest.mark.parametrize(
    ("qkv_format", "cp_size", "allgather_cp", "max_seq_len"),
    [
        ("thd", 1, False, None),
        ("thd", 2, False, None),
        ("thd", 2, True, None),
        ("bshd", 1, False, 16),
        ("bshd", 2, False, 16),
        ("bshd", 2, True, 16),
    ],
)
def test_scheduled_tokens_match_get_batch_output(
    monkeypatch,
    qkv_format: str,
    cp_size: int,
    allgather_cp: bool,
    max_seq_len: int | None,
) -> None:
    state = SimpleNamespace(
        cp=SimpleNamespace(rank=0, size=cp_size),
        tp=SimpleNamespace(size=1),
    )
    monkeypatch.setattr(data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    lengths = [5, 8]
    rollout_data = {
        "tokens": [torch.arange(length) for length in lengths],
        "loss_masks": [torch.ones(length - 2, dtype=torch.int) for length in lengths],
        "total_lengths": lengths,
        "response_lengths": [length - 2 for length in lengths],
    }
    if max_seq_len is not None:
        rollout_data["max_seq_lens"] = [max_seq_len] * len(lengths)

    metrics = compute_local_batching_metrics(
        rollout_data=rollout_data,
        micro_batch_size=2,
        micro_batch_indices=None,
        num_microbatches_by_step=[1],
        step_id=0,
        qkv_format=qkv_format,
        tp_size=1,
        cp_size=cp_size,
        pad_multiplier=4,
        allgather_cp=allgather_cp,
    )
    batch = get_batch(
        DataIterator(rollout_data, micro_batch_size=2),
        ["tokens", "loss_masks", "total_lengths", "response_lengths", "max_seq_lens"],
        pad_multiplier=4,
        qkv_format=qkv_format,
        allgather_cp=allgather_cp,
    )

    assert metrics.scheduled_tokens == batch["tokens"].numel() * cp_size
