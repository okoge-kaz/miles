from argparse import Namespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

from miles.backends.training_utils.log_utils import aggregate_train_losses
from miles.backends.training_utils.loss import _pack_logging_values
from miles.backends.training_utils.loss_hub.staleness_aware_loss import (
    STALENESS_AWARE_LOSS_PART_PREFIX,
    apply_staleness_aware_loss,
    compute_staleness_decay_weights,
)

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def _args(**overrides) -> Namespace:
    values = {
        "use_staleness_aware_loss": True,
        "log_staleness_aware_loss_details": False,
        "use_tis": True,
        "zero_reward_on_truncated": True,
        "safe_training_staleness": 2,
        "calculate_per_token_loss": False,
        "qkv_format": "thd",
    }
    values.update(overrides)
    return Namespace(**values)


def test_decay_is_one_through_safe_staleness_and_reciprocal_afterward() -> None:
    weights = compute_staleness_decay_weights(
        [0, 2, 3, 5],
        safe_training_staleness=2,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(weights, torch.tensor([1.0, 1.0, 0.5, 0.25]))


def test_only_truncated_zero_reward_pg_loss_is_attenuated() -> None:
    make_parallel_state()
    pg_loss = torch.tensor([2.0, 4.0, 1.0, 3.0, 5.0], requires_grad=True)
    final_masks = [torch.ones(2), torch.tensor([1.0, 1.0, 0.0])]

    weighted_loss, parts = apply_staleness_aware_loss(
        args=_args(),
        batch={
            "total_lengths": [3, 4],
            "response_lengths": [2, 3],
            "sample_staleness": [5, 20],
            "truncated": [1, 0],
        },
        pg_loss_tokens=pg_loss,
        final_masks=final_masks,
    )

    torch.testing.assert_close(weighted_loss, torch.tensor([0.5, 1.0, 1.0, 3.0, 5.0]))
    weighted_loss.sum().backward()
    torch.testing.assert_close(pg_loss.grad, torch.tensor([0.25, 0.25, 1.0, 1.0, 1.0]))
    assert all(not value.requires_grad for value in parts.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the interactive-node smoke")
def test_cuda_gradient_smoke() -> None:
    make_parallel_state()
    device = torch.device("cuda")
    pg_loss = torch.tensor([2.0, 4.0, 1.0], device=device, requires_grad=True)

    weighted_loss, _ = apply_staleness_aware_loss(
        args=_args(),
        batch={
            "total_lengths": [3, 2],
            "response_lengths": [2, 1],
            "sample_staleness": [5, 20],
            "truncated": [1, 0],
        },
        pg_loss_tokens=pg_loss,
        final_masks=[torch.ones(2, device=device), torch.ones(1, device=device)],
    )

    weighted_loss.sum().backward()
    torch.testing.assert_close(pg_loss.grad, torch.tensor([0.25, 0.25, 1.0], device=device))


def test_additive_parts_report_sample_count_and_loss_token_fraction(monkeypatch) -> None:
    make_parallel_state()
    monkeypatch.setattr(
        "miles.backends.training_utils.log_utils.MultiPGUtil.all_reduce",
        lambda *_args, **_kwargs: None,
    )
    _, parts = apply_staleness_aware_loss(
        args=_args(),
        batch={
            "total_lengths": [3, 4],
            "response_lengths": [2, 3],
            "sample_staleness": [5, 20],
            "truncated": [1, 0],
        },
        pg_loss_tokens=torch.tensor([2.0, 4.0, 1.0, 3.0, 5.0]),
        final_masks=[torch.ones(2), torch.tensor([1.0, 1.0, 0.0])],
    )
    packed = _pack_logging_values(2, {"loss": torch.tensor(2.0), **parts}, device=torch.device("cpu"))

    metrics = aggregate_train_losses([packed])

    assert metrics["staleness_aware_loss/truncated_zero_reward_sample_count"] == 1.0
    assert metrics["staleness_aware_loss/truncated_zero_reward_loss_token_fraction"] == 0.5
    assert metrics["staleness_aware_loss/truncated_zero_reward_mean_gradient_scale"] == 0.25
    assert not any("post_tis" in key for key in metrics)
    assert not any(key.startswith(STALENESS_AWARE_LOSS_PART_PREFIX) for key in metrics)


def test_opt_in_details_report_post_tis_objective_before_and_after_scaling(monkeypatch) -> None:
    make_parallel_state()
    monkeypatch.setattr(
        "miles.backends.training_utils.log_utils.MultiPGUtil.all_reduce",
        lambda *_args, **_kwargs: None,
    )
    _, parts = apply_staleness_aware_loss(
        args=_args(log_staleness_aware_loss_details=True),
        batch={
            "total_lengths": [3, 4],
            "response_lengths": [2, 3],
            "sample_staleness": [5, 20],
            "truncated": [1, 0],
        },
        pg_loss_tokens=torch.tensor([2.0, -4.0, 1.0, -3.0, 5.0]),
        final_masks=[torch.ones(2), torch.tensor([1.0, 1.0, 0.0])],
    )
    packed = _pack_logging_values(2, {"loss": torch.tensor(2.0), **parts}, device=torch.device("cpu"))

    metrics = aggregate_train_losses([packed])

    assert metrics["staleness_aware_loss/post_tis_pre_scaling_abs_pg_objective_per_loss_token"] == 2.5
    assert (
        metrics["staleness_aware_loss/truncated_zero_reward_post_tis_pre_scaling_abs_pg_objective_per_all_loss_token"]
        == 1.5
    )
    assert metrics["staleness_aware_loss/truncated_zero_reward_post_tis_pre_scaling_abs_pg_objective_fraction"] == 0.6
    assert metrics["staleness_aware_loss/post_tis_post_scaling_abs_pg_objective_per_loss_token"] == 1.375
    assert (
        metrics["staleness_aware_loss/truncated_zero_reward_post_tis_post_scaling_abs_pg_objective_per_all_loss_token"]
        == 0.375
    )
    assert metrics[
        "staleness_aware_loss/truncated_zero_reward_post_tis_post_scaling_abs_pg_objective_fraction"
    ] == pytest.approx(3 / 11)


def test_disabled_feature_preserves_loss_without_batch_metadata() -> None:
    pg_loss = torch.ones(2, requires_grad=True)

    unchanged, parts = apply_staleness_aware_loss(
        args=_args(use_staleness_aware_loss=False),
        batch={},
        pg_loss_tokens=pg_loss,
        final_masks=[],
    )

    assert unchanged is pg_loss
    assert parts == {}


def test_enabled_feature_rejects_missing_training_staleness() -> None:
    make_parallel_state()

    with pytest.raises(RuntimeError, match="training-staleness provenance"):
        apply_staleness_aware_loss(
            args=_args(),
            batch={
                "total_lengths": [2],
                "response_lengths": [1],
                "truncated": [1],
            },
            pg_loss_tokens=torch.ones(1),
            final_masks=[torch.ones(1)],
        )


def test_detail_logging_requires_tis() -> None:
    with pytest.raises(RuntimeError, match="requires --use-tis"):
        apply_staleness_aware_loss(
            args=_args(log_staleness_aware_loss_details=True, use_tis=False),
            batch={},
            pg_loss_tokens=torch.ones(1),
            final_masks=[],
        )
