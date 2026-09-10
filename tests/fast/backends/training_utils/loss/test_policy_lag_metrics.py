import math
from argparse import Namespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

from miles.backends.training_utils.log_utils import aggregate_train_losses, log_train_step
from miles.backends.training_utils.loss import _pack_logging_values
from miles.backends.training_utils.loss_hub import losses as loss_utils
from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss
from miles.backends.training_utils.loss_hub.policy_lag_metrics import (
    POLICY_LAG_PART_PREFIX,
    compute_policy_lag_parts,
    compute_ppo_loss_sensitivity,
    finalize_policy_lag_parts,
)

register_cpu_ci(est_time=8, suite="stage-a-cpu", labels=[])


def _args(*, per_token: bool = True) -> Namespace:
    return Namespace(calculate_per_token_loss=per_token, qkv_format="thd")


def _batch(lengths: list[int], truncated: list[int]) -> dict:
    return {
        "total_lengths": [length + 1 for length in lengths],
        "response_lengths": lengths,
        "truncated": truncated,
    }


def _finalize(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    return finalize_policy_lag_parts({key: float(value) for key, value in parts.items()})


def _collect(
    *,
    delta: torch.Tensor,
    sensitivity: torch.Tensor | None,
    lengths: list[int],
    truncated: list[int],
    target: torch.Tensor | None = None,
    initial_masks: list[torch.Tensor] | None = None,
    per_token: bool = True,
    actual_masks: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    make_parallel_state()
    if initial_masks is None:
        initial_masks = [torch.ones(length) for length in lengths]
    if target is None:
        target = torch.ones_like(delta)
    parts = compute_policy_lag_parts(
        args=_args(per_token=per_token),
        batch=_batch(lengths, truncated),
        delta=delta,
        initial_loss_masks=initial_masks,
        target_weights=target,
        loss_sensitivity_base=sensitivity,
        actual_post_loss_masks=actual_masks,
    )
    return _finalize(parts)


@pytest.mark.parametrize(
    ("target", "mass", "delta_sq", "weighted_rms", "mass_retained", "delta_sq_retained"),
    [
        (torch.ones(100), 180.0, 90.9, math.sqrt(0.505), 1.0, 1.0),
        (torch.full((100,), 0.5), 90.0, 45.45, math.sqrt(0.505), 0.5, 0.5),
        (
            torch.cat([torch.ones(90), torch.zeros(10)]),
            90.0,
            0.9,
            0.1,
            0.5,
            1 / 101,
        ),
        (
            torch.cat([torch.ones(90), torch.full((10,), 1 / 19)]),
            1800 / 19,
            1071 / 190,
            math.sqrt(0.0595),
            10 / 19,
            119 / 1919,
        ),
    ],
)
def test_acceptance_fixture(
    target: torch.Tensor,
    mass: float,
    delta_sq: float,
    weighted_rms: float,
    mass_retained: float,
    delta_sq_retained: float,
) -> None:
    delta = torch.cat([torch.full((90,), 0.1), torch.ones(10)])
    # The fixture specifies lambda itself. The collector applies the reference
    # token normalizer (1 / 100), so scale its pre-reducer input by 100.
    sensitivity = torch.cat([torch.full((90,), 100.0), torch.full((10,), 900.0)])

    metrics = _collect(
        delta=delta,
        sensitivity=sensitivity,
        lengths=[90, 10],
        truncated=[0, 1],
        target=target,
    )
    prefix = "policy_lag/all_response/"

    assert metrics[f"{prefix}delta_rms"] == pytest.approx(math.sqrt(0.109))
    assert metrics[f"{prefix}loss_sensitivity_sum_pre"] == pytest.approx(180.0)
    assert metrics[f"{prefix}loss_sensitivity_delta_sq_sum_pre"] == pytest.approx(90.9)
    assert metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_pre"] == pytest.approx(math.sqrt(0.505))
    assert metrics[f"{prefix}loss_sensitivity_sum_post"] == pytest.approx(mass)
    assert metrics[f"{prefix}loss_sensitivity_delta_sq_sum_post"] == pytest.approx(delta_sq)
    assert metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_post"] == pytest.approx(weighted_rms)
    assert metrics[f"{prefix}loss_sensitivity_retained_fraction"] == pytest.approx(mass_retained)
    assert metrics[f"{prefix}loss_sensitivity_delta_sq_retained_fraction"] == pytest.approx(delta_sq_retained)


def test_zero_loss_truncated_population_is_undefined_post() -> None:
    delta = torch.tensor([0.1, 0.2, 1.0, 2.0])
    metrics = _collect(
        delta=delta,
        sensitivity=torch.full((4,), 4.0),
        lengths=[2, 2],
        truncated=[0, 1],
        target=torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
    prefix = "policy_lag/truncated/"

    assert metrics[f"{prefix}loss_sensitivity_sum_post"] == 0.0
    assert metrics[f"{prefix}loss_sensitivity_delta_sq_sum_post"] == 0.0
    assert math.isnan(metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_post"])
    assert metrics[f"{prefix}loss_sensitivity_weighted_metrics_valid_post"] == 0.0
    assert metrics[f"{prefix}loss_sensitivity_retained_fraction"] == 0.0
    assert metrics[f"{prefix}loss_sensitivity_delta_sq_retained_fraction"] == 0.0
    assert metrics[f"{prefix}final_loss_mask_nonzero_fraction"] == 0.0


def test_zero_loss_target_covers_inactive_tokens_in_truncated_response() -> None:
    make_parallel_state()
    args = Namespace(qkv_format="thd", zero_loss_on_truncated=True)
    initial_masks = [torch.tensor([1, 0, 1]), torch.tensor([1, 0, 1])]
    batch = {
        "total_lengths": [4, 4],
        "response_lengths": [3, 3],
        "loss_masks": [initial_masks[0], torch.zeros(3)],
        "policy_lag_initial_loss_masks": initial_masks,
        "truncated": [0, 1],
    }

    returned_masks, targets = loss_utils._policy_lag_initial_masks_and_targets(
        args,
        batch,
        template=torch.zeros(6),
        staleness_weights=torch.ones(6),
    )

    assert returned_masks is initial_masks
    torch.testing.assert_close(targets, torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32))


def test_empty_populations_keep_counts_and_sums_distinct_from_means() -> None:
    metrics = _collect(
        delta=torch.empty(0),
        sensitivity=torch.empty(0),
        lengths=[],
        truncated=[],
    )

    for population in ("all_response", "truncated", "non_truncated"):
        prefix = f"policy_lag/{population}/"
        assert metrics[f"{prefix}response_token_count"] == 0.0
        assert metrics[f"{prefix}loss_sensitivity_sum_pre"] == 0.0
        assert math.isnan(metrics[f"{prefix}delta_abs_mean"])
        assert math.isnan(metrics[f"{prefix}delta_rms"])
        assert math.isnan(metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_pre"])


def test_nonfinite_response_inputs_are_counted_and_invalidate_dependencies() -> None:
    metrics = _collect(
        delta=torch.tensor([0.5, torch.inf]),
        sensitivity=torch.tensor([2.0, torch.nan]),
        lengths=[1, 1],
        truncated=[0, 1],
    )

    all_prefix = "policy_lag/all_response/"
    nontruncated_prefix = "policy_lag/non_truncated/"
    assert metrics[f"{all_prefix}nonfinite_delta_count"] == 1.0
    assert metrics[f"{all_prefix}nonfinite_loss_sensitivity_count_pre"] == 1.0
    assert math.isnan(metrics[f"{all_prefix}delta_rms"])
    assert math.isnan(metrics[f"{all_prefix}loss_sensitivity_sum_pre"])
    assert metrics[f"{all_prefix}loss_sensitivity_weighted_metrics_valid_pre"] == 0.0
    assert metrics[f"{nontruncated_prefix}delta_rms"] == pytest.approx(0.5)
    assert metrics[f"{nontruncated_prefix}loss_sensitivity_weighted_metrics_valid_pre"] == 1.0


def test_finite_fp32_delta_does_not_overflow_during_square() -> None:
    metrics = _collect(
        delta=torch.tensor([1.0e30], dtype=torch.float32),
        sensitivity=torch.ones(1),
        lengths=[1],
        truncated=[0],
    )

    assert math.isfinite(metrics["policy_lag/all_response/delta_rms"])
    assert metrics["policy_lag/all_response/delta_rms"] == pytest.approx(1.0e30)
    assert math.isfinite(metrics["policy_lag/all_response/loss_sensitivity_delta_sq_sum_pre"])


def test_unsupported_surrogate_emits_delta_only() -> None:
    metrics = _collect(
        delta=torch.tensor([0.25, -0.5]),
        sensitivity=None,
        lengths=[1, 1],
        truncated=[1, 0],
    )

    assert metrics["policy_lag/loss_sensitivity_supported"] == 0.0
    assert metrics["policy_lag/all_response/delta_rms"] == pytest.approx(math.sqrt((0.25**2 + 0.5**2) / 2))
    assert not any("loss_sensitivity_sum" in key for key in metrics)


def test_population_statistics_are_additive_before_ratios() -> None:
    metrics = _collect(
        delta=torch.tensor([0.1, 0.2, 0.8, 1.0]),
        sensitivity=torch.tensor([4.0, 8.0, 12.0, 16.0]),
        lengths=[2, 2],
        truncated=[0, 1],
        target=torch.tensor([1.0, 0.5, 0.25, 0.0]),
    )

    for suffix in (
        "response_token_count",
        "loss_sensitivity_sum_pre",
        "loss_sensitivity_sum_post",
        "loss_sensitivity_delta_sq_sum_pre",
        "loss_sensitivity_delta_sq_sum_post",
    ):
        all_value = metrics[f"policy_lag/all_response/{suffix}"]
        split_value = metrics[f"policy_lag/truncated/{suffix}"] + metrics[f"policy_lag/non_truncated/{suffix}"]
        assert all_value == pytest.approx(split_value)


def test_uniform_attenuation_preserves_weighted_rms() -> None:
    metrics = _collect(
        delta=torch.tensor([0.1, 0.5, 2.0]),
        sensitivity=torch.tensor([3.0, 6.0, 9.0]),
        lengths=[1, 2],
        truncated=[0, 1],
        target=torch.full((3,), 0.3),
    )
    prefix = "policy_lag/all_response/"

    assert metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_post"] == pytest.approx(
        metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_pre"]
    )
    assert metrics[f"{prefix}loss_sensitivity_retained_fraction"] == pytest.approx(0.3)
    assert metrics[f"{prefix}loss_sensitivity_delta_sq_retained_fraction"] == pytest.approx(0.3)


def test_removing_low_delta_tokens_can_increase_weighted_rms() -> None:
    metrics = _collect(
        delta=torch.tensor([0.01, 2.0]),
        sensitivity=torch.tensor([2.0, 2.0]),
        lengths=[1, 1],
        truncated=[1, 0],
        target=torch.tensor([0.0, 1.0]),
    )
    prefix = "policy_lag/all_response/"

    assert (
        metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_post"]
        > metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_pre"]
    )


def test_sequence_reducer_keeps_per_sequence_coefficients() -> None:
    metrics = _collect(
        delta=torch.tensor([1.0, 1.0, 1.0]),
        sensitivity=torch.ones(3),
        lengths=[2, 1],
        truncated=[0, 1],
        per_token=False,
    )

    assert metrics["policy_lag/all_response/loss_sensitivity_sum_pre"] == pytest.approx(1.0)
    assert metrics["policy_lag/non_truncated/loss_sensitivity_sum_pre"] == pytest.approx(0.5)
    assert metrics["policy_lag/truncated/loss_sensitivity_sum_pre"] == pytest.approx(0.5)


def test_actual_post_series_uses_training_token_normalizer() -> None:
    initial = [torch.ones(2), torch.ones(2)]
    actual = [torch.zeros(2), torch.ones(2)]
    metrics = _collect(
        delta=torch.ones(4),
        sensitivity=torch.full((4,), 4.0),
        lengths=[2, 2],
        truncated=[1, 0],
        target=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        initial_masks=initial,
        actual_masks=actual,
    )
    prefix = "policy_lag/all_response/"

    assert metrics[f"{prefix}loss_sensitivity_sum_post"] == pytest.approx(2.0)
    assert metrics[f"{prefix}loss_sensitivity_sum_actual"] == pytest.approx(8 / 3)
    assert metrics[f"{prefix}loss_sensitivity_weighted_delta_rms_actual"] == pytest.approx(1.0)


@pytest.mark.parametrize("dual_clip", [None, 1.5])
def test_ppo_adapter_matches_autograd_including_boundaries(dual_clip: float | None) -> None:
    ratios = torch.tensor(
        [math.exp(-20), 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.5, math.exp(20)],
        dtype=torch.float64,
    )
    advantages = torch.tensor([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0])
    selected_logprobs = ratios.log().float().requires_grad_(True)
    losses, _ = compute_policy_loss(
        -selected_logprobs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=dual_clip,
    )
    expected = torch.autograd.grad(losses.sum(), selected_logprobs)[0].abs()

    actual = compute_ppo_loss_sensitivity(
        ppo_log_ratio=selected_logprobs,
        advantages=advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=dual_clip,
    )

    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_microbatch_parts_are_combined_before_finalizing(monkeypatch) -> None:
    make_parallel_state()
    monkeypatch.setattr(
        "miles.backends.training_utils.log_utils.MultiPGUtil.all_reduce",
        lambda *_args, **_kwargs: None,
    )

    def parts(delta: float, truncated: int) -> dict[str, torch.Tensor]:
        return compute_policy_lag_parts(
            args=_args(),
            batch=_batch([1], [truncated]),
            delta=torch.tensor([delta]),
            initial_loss_masks=[torch.ones(1)],
            target_weights=torch.ones(1),
            loss_sensitivity_base=torch.tensor([2.0]),
        )

    packed = [
        _pack_logging_values(1, {"loss": torch.tensor(0.0), **parts(3.0, 1)}, device=torch.device("cpu")),
        _pack_logging_values(1, {"loss": torch.tensor(0.0), **parts(4.0, 0)}, device=torch.device("cpu")),
    ]
    metrics = aggregate_train_losses(packed)

    assert metrics["policy_lag/all_response/delta_rms"] == pytest.approx(math.sqrt(12.5))
    assert metrics["policy_lag/truncated/delta_rms"] == pytest.approx(3.0)
    assert metrics["policy_lag/non_truncated/delta_rms"] == pytest.approx(4.0)
    assert all(key.startswith(POLICY_LAG_PART_PREFIX) for key in packed[0]["diagnostic_keys"])
    assert not any(key.startswith(POLICY_LAG_PART_PREFIX) for key in metrics)


def test_actor_policy_lag_metrics_use_root_namespace() -> None:
    metrics = log_train_step(
        args=Namespace(),
        loss_dict={"policy_lag/all_response/delta_rms": torch.tensor(0.25)},
        grad_norm=2.0,
        rollout_id=3,
        step_id=1,
        num_steps_per_rollout=2,
        should_log=False,
    )

    assert metrics["policy_lag/all_response/delta_rms"] == pytest.approx(0.25)
    assert "train/policy_lag/all_response/delta_rms" not in metrics
    assert metrics["train/step"] == 7


def test_logging_does_not_change_loss_gradient_or_rng(monkeypatch) -> None:
    make_parallel_state()

    def run(log_policy_lag_metrics: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_log_probs = torch.tensor([0.1, -0.2], requires_grad=True)
        monkeypatch.setattr(
            loss_utils,
            "get_log_probs_and_entropy",
            lambda *unused_args, **unused_kwargs: {"log_probs": [selected_log_probs]},
        )
        args = Namespace(
            advantage_estimator="grpo",
            calculate_per_token_loss=False,
            custom_pg_loss_reducer_function_path=None,
            custom_tis_function_path=None,
            entropy_coef=0.0,
            eps_clip=0.2,
            eps_clip_c=None,
            eps_clip_high=0.28,
            fuse_one_step_actor_logprobs=False,
            get_mismatch_metrics=False,
            kl_loss_coef=0.0,
            log_policy_lag_metrics=log_policy_lag_metrics,
            observe_training_entropy=False,
            qkv_format="thd",
            use_kl_loss=False,
            use_m2po=False,
            use_opsm=False,
            use_rollout_logprobs=False,
            skip_actor_forward_only=False,
            use_staleness_aware_loss=False,
            use_tis=False,
            zero_loss_on_truncated=False,
        )
        rng_before = torch.random.get_rng_state().clone()
        loss, _ = loss_utils.policy_loss_function(
            args,
            {
                "advantages": [torch.tensor([1.0, -0.5])],
                "log_probs": [torch.zeros(2)],
                "rollout_log_probs": [torch.tensor([-0.3, -0.4])],
                "unconcat_tokens": [torch.tensor([1, 2, 3])],
                "response_lengths": [2],
                "total_lengths": [3],
                "loss_masks": [torch.ones(2)],
                "truncated": [0],
            },
            logits=torch.zeros((1, 3, 8)),
            sum_of_sample_mean=lambda value: value.sum(),
        )
        loss.backward()
        return loss.detach(), selected_log_probs.grad.detach(), rng_before

    disabled_loss, disabled_grad, disabled_rng = run(False)
    enabled_loss, enabled_grad, enabled_rng = run(True)

    torch.testing.assert_close(enabled_loss, disabled_loss)
    torch.testing.assert_close(enabled_grad, disabled_grad)
    torch.testing.assert_close(enabled_rng, disabled_rng)
    torch.testing.assert_close(torch.random.get_rng_state(), enabled_rng)
