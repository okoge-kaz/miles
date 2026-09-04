from argparse import Namespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

from miles.backends.training_utils.log_utils import log_train_step
from miles.backends.training_utils.loss_hub.staleness_metrics import (
    compute_sample_staleness_parts,
    finalize_sample_staleness_metrics,
)

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])


def _args(*, histogram: bool = False) -> Namespace:
    return Namespace(
        log_sample_staleness_metrics=True,
        log_sample_staleness_ratio_histogram=histogram,
        sample_staleness_max_bin=2,
        calculate_per_token_loss=False,
        qkv_format="thd",
    )


def _computed_metrics(*, histogram: bool = False) -> tuple[dict[str, float], torch.Tensor]:
    make_parallel_state()
    pg_loss = torch.tensor([1.0, 3.0, 4.0, 8.0], requires_grad=True)
    original_masks = [torch.ones(2), torch.ones(2)]
    final_masks = [torch.ones(2), torch.tensor([1.0, 0.0])]
    parts = compute_sample_staleness_parts(
        args=_args(histogram=histogram),
        batch={
            "sample_staleness": [0, 2],
            "total_lengths": [3, 3],
            "response_lengths": [2, 2],
        },
        original_local_masks=original_masks,
        final_masks=final_masks,
        pg_loss_tokens=pg_loss,
        ppo_clipfrac=torch.tensor([0.0, 1.0, 1.0, 0.0]),
        policy_log_ratio=torch.zeros(4),
        objective_log_ratio=torch.zeros(4),
        tis_metrics={"tis_clipfrac": torch.tensor([0.0, 0.0, 1.0, 0.0])},
    )
    assert all(not value.requires_grad for value in parts.values())
    metrics = {key: float(value) for key, value in parts.items()}
    finalize_sample_staleness_metrics(metrics)
    return metrics, pg_loss


def test_effective_distribution_uses_final_mask_and_sample_reducer() -> None:
    metrics, pg_loss = _computed_metrics()

    # Sample 0 contributes |1|/2 + |3|/2 = 2. Sample 1 contributes
    # |4|/1 + masked(|8|) = 4, so effective mass is 1/3 vs 2/3 even
    # though consumed sequence and raw-token mass are both 1/2.
    assert metrics["sample_staleness/s_0/consumed_sequence_mass"] == pytest.approx(0.5)
    assert metrics["sample_staleness/s_2/consumed_response_token_mass"] == pytest.approx(0.5)
    assert metrics["sample_staleness/s_0/effective_contribution_mass"] == pytest.approx(1 / 3)
    assert metrics["sample_staleness/s_2/effective_contribution_mass"] == pytest.approx(2 / 3)
    assert metrics["sample_staleness/s_2/correction_mask_fraction"] == pytest.approx(0.5)
    assert metrics["sample_staleness/s_2/importance_clip_fraction"] == pytest.approx(0.5)
    assert metrics["sample_staleness/s_0/ppo_clip_fraction"] == pytest.approx(0.5)
    assert metrics["sample_staleness/s_2/policy_rollout_ratio_token_ess"] == pytest.approx(1.0)
    assert metrics["sample_staleness/s_0/mean_abs_ppo_objective_log_ratio"] == 0.0

    # Telemetry is detached from the objective graph.
    pg_loss.sum().backward()
    torch.testing.assert_close(pg_loss.grad, torch.ones_like(pg_loss))


def test_fixed_ratio_histogram_is_normalized_per_staleness_bin() -> None:
    metrics, _ = _computed_metrics(histogram=True)

    for staleness in ("s_0", "s_2"):
        histogram = {
            key: value
            for key, value in metrics.items()
            if key.startswith(f"sample_staleness/{staleness}/policy_rollout_log_ratio_hist/")
        }
        assert len(histogram) == 15
        assert sum(histogram.values()) == pytest.approx(1.0)
        assert metrics[f"sample_staleness/{staleness}/approx_p95_abs_policy_rollout_log_ratio_capped_1"] == 0.001


def test_overflow_bin_is_stable() -> None:
    make_parallel_state()
    parts = compute_sample_staleness_parts(
        args=_args(),
        batch={"sample_staleness": [99], "total_lengths": [2], "response_lengths": [1]},
        original_local_masks=[torch.ones(1)],
        final_masks=[torch.ones(1)],
        pg_loss_tokens=torch.ones(1),
        ppo_clipfrac=torch.zeros(1),
        policy_log_ratio=torch.zeros(1),
        objective_log_ratio=torch.zeros(1),
        tis_metrics=None,
    )
    metrics = {key: float(value) for key, value in parts.items()}
    finalize_sample_staleness_metrics(metrics)
    assert metrics["sample_staleness/s_ge_3/consumed_sequence_mass"] == 1.0


def test_sequence_ess_is_invariant_to_microbatch_partitioning() -> None:
    make_parallel_state()

    def compute_parts(log_ratios: list[float]) -> dict[str, torch.Tensor]:
        sample_count = len(log_ratios)
        return compute_sample_staleness_parts(
            args=_args(),
            batch={
                "sample_staleness": [0] * sample_count,
                "total_lengths": [2] * sample_count,
                "response_lengths": [1] * sample_count,
            },
            original_local_masks=[torch.ones(1) for _ in log_ratios],
            final_masks=[torch.ones(1) for _ in log_ratios],
            pg_loss_tokens=torch.ones(sample_count),
            ppo_clipfrac=torch.zeros(sample_count),
            policy_log_ratio=torch.tensor(log_ratios),
            objective_log_ratio=torch.zeros(sample_count),
            tis_metrics=None,
        )

    full_parts = compute_parts([2.0, 0.0, -2.0])
    split_parts = compute_parts([2.0])
    for key, value in compute_parts([0.0, -2.0]).items():
        split_parts[key] = split_parts.get(key, torch.zeros_like(value)) + value

    assert full_parts.keys() == split_parts.keys()
    full_metrics = {key: float(value) for key, value in full_parts.items()}
    split_metrics = {key: float(value) for key, value in split_parts.items()}
    finalize_sample_staleness_metrics(full_metrics)
    finalize_sample_staleness_metrics(split_metrics)

    metric = "sample_staleness/s_0/policy_rollout_ratio_sequence_ess"
    assert split_metrics[metric] == pytest.approx(full_metrics[metric], rel=1e-12)


def test_deterministic_algorithm_setting_is_restored() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        _computed_metrics()
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)


def test_actor_sample_staleness_metrics_use_a_root_namespace() -> None:
    metrics = log_train_step(
        args=Namespace(),
        loss_dict={
            "loss": torch.tensor(1.5),
            "sample_staleness/s_0/consumed_sequence_mass": torch.tensor(1.0),
        },
        grad_norm=2.0,
        rollout_id=3,
        step_id=1,
        num_steps_per_rollout=2,
        should_log=False,
    )

    assert metrics["train/loss"] == 1.5
    assert metrics["sample_staleness/s_0/consumed_sequence_mass"] == 1.0
    assert "train/sample_staleness/s_0/consumed_sequence_mass" not in metrics
    assert metrics["train/step"] == 7


def test_non_actor_sample_staleness_metrics_remain_role_scoped() -> None:
    metrics = log_train_step(
        args=Namespace(),
        loss_dict={"sample_staleness/s_0/consumed_sequence_mass": 1.0},
        grad_norm=2.0,
        rollout_id=0,
        step_id=0,
        num_steps_per_rollout=1,
        role="critic",
        should_log=False,
    )

    assert metrics["train/critic-sample_staleness/s_0/consumed_sequence_mass"] == 1.0
