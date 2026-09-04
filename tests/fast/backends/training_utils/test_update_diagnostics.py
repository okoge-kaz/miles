from argparse import Namespace
import math

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

from miles.backends.training_utils.log_utils import aggregate_train_losses
from miles.backends.training_utils.loss import _pack_logging_values
from miles.backends.training_utils.update_diagnostics import (
    UPDATE_PART_PREFIX,
    compute_update_diagnostic_parts,
    finalize_update_diagnostic_parts,
    optimizer_step_diagnostics,
)

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def test_advantage_diagnostics_use_only_final_loss_tokens() -> None:
    make_parallel_state()
    parts = compute_update_diagnostic_parts(
        Namespace(log_update_diagnostics=True, qkv_format="thd"),
        {
            "total_lengths": [3, 3],
            "response_lengths": [2, 2],
        },
        torch.tensor([-1.0, 1.0, 3.0, 5.0]),
        [torch.ones(2), torch.tensor([1.0, 0.0])],
    )
    metrics = finalize_update_diagnostic_parts({key: float(value) for key, value in parts.items()})

    assert metrics["final_loss_tokens"] == 3
    assert metrics["advantage_abs_mean"] == pytest.approx(5 / 3)
    assert metrics["advantage_rms"] == pytest.approx(math.sqrt(11 / 3))
    assert metrics["advantage_std"] == pytest.approx(math.sqrt(8 / 3))


def test_update_parts_piggyback_on_existing_loss_reduction(monkeypatch) -> None:
    make_parallel_state()
    monkeypatch.setattr(
        "miles.backends.training_utils.log_utils.MultiPGUtil.all_reduce",
        lambda *_args, **_kwargs: None,
    )
    packed = _pack_logging_values(
        4,
        {
            "loss": torch.tensor(4.0),
            f"{UPDATE_PART_PREFIX}final_loss_tokens": torch.tensor(3.0),
            f"{UPDATE_PART_PREFIX}advantage_sum": torch.tensor(3.0),
            f"{UPDATE_PART_PREFIX}advantage_square_sum": torch.tensor(11.0),
            f"{UPDATE_PART_PREFIX}advantage_abs_sum": torch.tensor(5.0),
        },
        device=torch.device("cpu"),
    )

    metrics = aggregate_train_losses([packed])

    assert metrics["loss"] == 1.0
    assert metrics["final_loss_tokens"] == 3.0
    assert metrics["advantage_std"] == pytest.approx(math.sqrt(8 / 3))
    assert not any(key.startswith(UPDATE_PART_PREFIX) for key in metrics)


def test_optimizer_diagnostics_reuse_preclip_norm() -> None:
    metrics = optimizer_step_diagnostics(
        Namespace(log_update_diagnostics=True, clip_grad=1.0),
        optimizer_step_applied=True,
        grad_norm=torch.tensor(2.0),
        num_zeros_in_grad=None,
    )

    assert metrics["optimizer_step_applied"] == 1.0
    assert metrics["grad_norm_pre_clip"] == 2.0
    assert metrics["grad_clip_coefficient"] == pytest.approx(1.0 / (2.0 + 1.0e-6))
    assert "num_zeros_in_grad" not in metrics


def test_num_zeros_is_logged_only_when_optimizer_already_computed_it() -> None:
    metrics = optimizer_step_diagnostics(
        Namespace(log_update_diagnostics=True, clip_grad=1.0),
        optimizer_step_applied=True,
        grad_norm=0.5,
        num_zeros_in_grad=17,
    )

    assert metrics["grad_clip_coefficient"] == 1.0
    assert metrics["num_zeros_in_grad"] == 17.0


def test_update_diagnostics_are_feature_gated() -> None:
    assert (
        optimizer_step_diagnostics(
            Namespace(log_update_diagnostics=False, clip_grad=1.0),
            optimizer_step_applied=True,
            grad_norm=2.0,
            num_zeros_in_grad=0,
        )
        == {}
    )
