from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import pytest
import torch

from examples.infra_features.train_infer_mismatch_helper.mis import compute_mis_weights_with_cp
from miles.backends.training_utils.loss import compute_advantages_and_returns
from miles.backends.training_utils.loss_hub import losses as loss_utils
from miles.backends.training_utils.loss_hub.corrections import icepop_function, vanilla_tis_function

from .loss_test_utils import make_args, make_inputs, make_parallel_state, make_rollout_data


def _fused_args(**overrides) -> Namespace:
    defaults = dict(
        advantage_estimator="grpo",
        kl_coef=0.0,
        entropy_coef=0.0,
        observe_training_entropy=False,
        calculate_per_token_loss=True,
        use_tis=True,
        get_mismatch_metrics=False,
        tis_clip=2.0,
        tis_clip_low=0.0,
        use_rollout_logprobs=False,
        fuse_one_step_actor_logprobs=True,
        verify_fused_one_step_actor_logprobs=False,
    )
    defaults.update(overrides)
    return make_args(**defaults)


def _batch(*, rollout_log_probs: torch.Tensor, old_log_probs: torch.Tensor | None = None) -> dict:
    batch = {
        "advantages": [torch.tensor([1.0, -0.5, 0.25], dtype=torch.float32)],
        "rollout_log_probs": [rollout_log_probs],
        "unconcat_tokens": [torch.tensor([3, 5, 7, 11], dtype=torch.long)],
        "response_lengths": [3],
        "total_lengths": [4],
        "loss_masks": [torch.ones(3, dtype=torch.float32)],
    }
    if old_log_probs is not None:
        batch["log_probs"] = [old_log_probs]
    return batch


def _run_loss(
    monkeypatch,
    *,
    args: Namespace,
    current_values: torch.Tensor,
    batch: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    current = current_values.clone().requires_grad_(True)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *unused_args, **unused_kwargs: {"log_probs": [current]},
    )
    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 4, 16), dtype=torch.float32),
        sum_of_sample_mean=lambda value: value.float().mean(),
    )
    loss.backward()
    assert current.grad is not None
    return loss.detach(), metrics, current.grad.detach().clone()


def test_fused_proximal_ratio_is_one_with_nonzero_gradient(monkeypatch):
    make_parallel_state()
    current = torch.tensor([-1.2, -0.7, -2.1], dtype=torch.float32)
    rollout = torch.tensor([-1.4, -0.6, -2.4], dtype=torch.float32)

    _, metrics, gradient = _run_loss(
        monkeypatch,
        args=_fused_args(),
        current_values=current,
        batch=_batch(rollout_log_probs=rollout),
    )

    assert float(metrics["ppo_kl"]) == 0.0
    assert float(metrics["ois"]) == 1.0
    assert float(metrics["pg_clipfrac"]) == 0.0
    # calculate_per_token_loss reports an additive contribution that becomes 1
    # after the standard train-metric token normalizer (three valid tokens here).
    assert float(metrics["ess_ratio"]) / 3 == 1.0
    assert float(metrics["fused_one_step_logprobs_enabled"]) == 1.0
    assert float(metrics["fused_prox_ratio_max_abs_from_one"]) <= 1e-7
    assert torch.count_nonzero(gradient).item() > 0


def test_fused_and_legacy_loss_and_gradient_match_when_anchors_match(monkeypatch):
    make_parallel_state()
    current = torch.tensor([-1.2, -0.7, -2.1], dtype=torch.float32)
    rollout = torch.tensor([-1.4, -0.6, -2.4], dtype=torch.float32)

    legacy_loss, legacy_metrics, legacy_gradient = _run_loss(
        monkeypatch,
        args=_fused_args(fuse_one_step_actor_logprobs=False),
        current_values=current,
        batch=_batch(rollout_log_probs=rollout, old_log_probs=current.clone()),
    )
    fused_loss, fused_metrics, fused_gradient = _run_loss(
        monkeypatch,
        args=_fused_args(),
        current_values=current,
        batch=_batch(rollout_log_probs=rollout),
    )

    torch.testing.assert_close(fused_loss, legacy_loss, rtol=0, atol=0)
    torch.testing.assert_close(fused_gradient, legacy_gradient, rtol=0, atol=0)
    torch.testing.assert_close(fused_metrics["tis"], legacy_metrics["tis"], rtol=0, atol=0)
    torch.testing.assert_close(fused_metrics["pg_loss"], legacy_metrics["pg_loss"], rtol=0, atol=0)


def test_fused_path_supports_no_behavior_correction(monkeypatch):
    make_parallel_state()
    current = torch.tensor([-1.2, -0.7, -2.1], dtype=torch.float32)
    rollout = torch.tensor([-1.4, -0.6, -2.4], dtype=torch.float32)

    loss, metrics, gradient = _run_loss(
        monkeypatch,
        args=_fused_args(use_tis=False),
        current_values=current,
        batch=_batch(rollout_log_probs=rollout),
    )

    assert torch.isfinite(loss)
    assert torch.count_nonzero(gradient).item() > 0
    assert "tis" not in metrics
    assert float(metrics["fused_prox_ratio_max_abs_from_one"]) == 0.0
    assert float(metrics["policy_rollout_abs_diff"]) > 0.0


def test_fused_path_preserves_reference_policy_kl_loss(monkeypatch):
    make_parallel_state()
    current = torch.tensor([-1.2, -0.7, -2.1], dtype=torch.float32)
    rollout = torch.tensor([-1.4, -0.6, -2.4], dtype=torch.float32)
    batch = _batch(rollout_log_probs=rollout)
    batch["ref_log_probs"] = [torch.tensor([-1.3, -0.8, -2.0], dtype=torch.float32)]

    loss, metrics, gradient = _run_loss(
        monkeypatch,
        args=_fused_args(
            use_kl_loss=True,
            kl_loss_coef=0.1,
            kl_loss_type="low_var_kl",
            use_unbiased_kl=False,
        ),
        current_values=current,
        batch=batch,
    )

    assert torch.isfinite(loss)
    assert torch.count_nonzero(gradient).item() > 0
    assert "kl_loss" in metrics


def test_tis_weights_and_anchor_do_not_receive_gradients():
    train = torch.tensor([-1.0, -2.0], requires_grad=True)
    rollout = torch.tensor([-1.2, -1.7], requires_grad=True)
    pg_loss = torch.tensor([0.4, -0.3], requires_grad=True)
    args = Namespace(tis_clip_low=0.0, tis_clip=2.0)

    weighted, _, _ = vanilla_tis_function(
        args,
        pg_loss=pg_loss,
        train_log_probs=[train],
        rollout_log_probs=[rollout],
        loss_masks=[torch.ones(2)],
    )
    weighted.sum().backward()

    assert pg_loss.grad is not None
    assert train.grad is None
    assert rollout.grad is None


@pytest.mark.parametrize("correction", ["tis", "icepop", "mis"])
def test_fused_corrections_receive_detached_anchor_and_rollout_pair(monkeypatch, correction: str):
    make_parallel_state()
    current = torch.tensor([-1.2, -0.7, -2.1], dtype=torch.float32)
    rollout = torch.tensor([-1.4, -0.6, -2.4], dtype=torch.float32)
    captured: dict[str, list[torch.Tensor]] = {}

    if correction == "tis":
        target: Callable = vanilla_tis_function
        custom_path = None
    elif correction == "icepop":
        target = icepop_function
        custom_path = "test.icepop"
    else:
        target = compute_mis_weights_with_cp
        custom_path = "test.mis"

    def capture_pair(*args, **kwargs):
        captured["train"] = kwargs["train_log_probs"]
        captured["rollout"] = kwargs["rollout_log_probs"]
        return target(*args, **kwargs)

    if custom_path is None:
        monkeypatch.setattr(loss_utils, "vanilla_tis_function", capture_pair)
    else:
        monkeypatch.setattr(loss_utils, "load_function", lambda unused_path: capture_pair)

    args = _fused_args(custom_tis_function_path=custom_path)
    if correction == "mis":
        for key, value in dict(
            use_rs=False,
            tis_level="token",
            rs_level="token",
            tis_mode="truncate",
            tis_lower_bound=0.0,
            tis_upper_bound=2.0,
            rs_lower_bound=None,
            rs_upper_bound=None,
            rs_veto_threshold=None,
            tis_batch_normalize=False,
        ).items():
            setattr(args, key, value)

    _run_loss(
        monkeypatch,
        args=args,
        current_values=current,
        batch=_batch(rollout_log_probs=rollout),
    )

    assert len(captured["train"]) == len(captured["rollout"]) == 1
    torch.testing.assert_close(captured["train"][0], current)
    torch.testing.assert_close(captured["rollout"][0], rollout)
    assert not captured["train"][0].requires_grad
    assert not captured["rollout"][0].requires_grad


def test_shadow_validation_never_changes_the_fused_loss_or_gradient(monkeypatch):
    make_parallel_state()
    current = torch.tensor([-1.2, -0.7, -2.1], dtype=torch.float32)
    rollout = torch.tensor([-1.4, -0.6, -2.4], dtype=torch.float32)
    args = _fused_args(verify_fused_one_step_actor_logprobs=True)

    exact_batch = _batch(rollout_log_probs=rollout)
    exact_batch["legacy_actor_log_probs"] = [current.clone()]
    exact_loss, exact_metrics, exact_gradient = _run_loss(
        monkeypatch,
        args=args,
        current_values=current,
        batch=exact_batch,
    )

    shifted_batch = _batch(rollout_log_probs=rollout)
    shifted_batch["legacy_actor_log_probs"] = [current + torch.tensor([0.01, -0.02, 0.03])]
    shifted_loss, shifted_metrics, shifted_gradient = _run_loss(
        monkeypatch,
        args=args,
        current_values=current,
        batch=shifted_batch,
    )

    torch.testing.assert_close(shifted_loss, exact_loss, rtol=0, atol=0)
    torch.testing.assert_close(shifted_gradient, exact_gradient, rtol=0, atol=0)
    assert float(exact_metrics["verify_anchor_logprob_abs_max"]) == 0.0
    assert float(shifted_metrics["verify_anchor_logprob_abs_mean"]) > 0.0
    for key in (
        "verify_anchor_logprob_abs_p99",
        "verify_anchor_logprob_abs_max",
        "verify_tis_weight_abs_mean",
        "verify_tis_weight_abs_p99",
        "verify_tis_clip_decision_disagreement",
    ):
        assert key in shifted_metrics


def test_fused_grpo_advantages_match_legacy_without_actor_logprobs():
    make_parallel_state()
    legacy_args = make_args(advantage_estimator="grpo", kl_coef=0.0)
    fused_args = make_args(
        advantage_estimator="grpo",
        kl_coef=0.0,
        fuse_one_step_actor_logprobs=True,
    )
    inputs = make_inputs(
        seed=17,
        batch_size=4,
        prompt_lens=[4, 5, 6, 7],
        response_lens=[2, 3, 4, 5],
        vocab_size=32,
        args=legacy_args,
    )
    legacy = make_rollout_data(inputs)
    fused = make_rollout_data(inputs)
    fused.pop("log_probs")

    compute_advantages_and_returns(legacy_args, legacy)
    compute_advantages_and_returns(fused_args, fused)

    for fused_value, legacy_value in zip(fused["advantages"], legacy["advantages"], strict=True):
        torch.testing.assert_close(fused_value, legacy_value, rtol=0, atol=0)
    for fused_value, legacy_value in zip(fused["returns"], legacy["returns"], strict=True):
        torch.testing.assert_close(fused_value, legacy_value, rtol=0, atol=0)
