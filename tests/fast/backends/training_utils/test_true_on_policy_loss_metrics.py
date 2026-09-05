from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss_hub import losses as loss_utils


def _make_args(*, use_rollout_logprobs: bool) -> Namespace:
    return Namespace(
        use_rollout_logprobs=use_rollout_logprobs,
        skip_actor_forward_only=False,
        fuse_one_step_actor_logprobs=False,
        verify_fused_one_step_actor_logprobs=False,
        use_opsm=False,
        advantage_estimator="ppo",
        get_mismatch_metrics=False,
        use_tis=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        entropy_coef=0.0,
        use_kl_loss=False,
        use_unbiased_kl=False,
        kl_loss_type="k1",
        kl_loss_coef=0.0,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        allgather_cp=False,
        observe_training_entropy=False,
    )


def _make_batch(*, old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor) -> dict:
    return {
        "advantages": [torch.tensor([1.0, -0.5], dtype=torch.float32)],
        "log_probs": [old_log_probs],
        "rollout_log_probs": [rollout_log_probs],
        "unconcat_tokens": [torch.tensor([7, 8, 9], dtype=torch.long)],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.tensor([1.0, 1.0], dtype=torch.float32)],
    }


def _patch_single_rank_loss_helpers(monkeypatch):
    monkeypatch.setattr(
        loss_utils,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format="thd", max_seq_lens=None: loss_masks,
    )
    # Both ESS helpers reach for the global ParallelState to size the CP group,
    # which these tests deliberately run without.
    monkeypatch.setattr(
        loss_utils,
        "compute_ess_ratio_contribution",
        lambda *, ppo_kl, **kwargs: ppo_kl.new_tensor(1.0),
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_sequence_level_ess_parts",
        lambda *, log_ratio, loss_masks, **kwargs: (
            log_ratio.new_tensor(1.0, dtype=torch.float64),
            log_ratio.new_tensor(1.0, dtype=torch.float64),
            log_ratio.new_tensor(float(len(loss_masks)), dtype=torch.float64),
        ),
    )


@pytest.mark.parametrize(
    ("use_rollout_logprobs", "train_log_probs", "old_log_probs", "rollout_log_probs", "expected_abs_diff"),
    [
        (
            False,
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            0.45,
        ),
        (
            True,
            torch.tensor([0.50, 1.00], dtype=torch.float32),
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            0.0,
        ),
    ],
)
def test_train_rollout_logprob_abs_diff_uses_policy_loss_reference_logprobs(
    monkeypatch,
    use_rollout_logprobs: bool,
    train_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    expected_abs_diff: float,
):
    args = _make_args(use_rollout_logprobs=use_rollout_logprobs)
    batch = _make_batch(old_log_probs=old_log_probs, rollout_log_probs=rollout_log_probs)

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [train_log_probs.clone()],
            "entropy": [torch.zeros_like(train_log_probs)],
        },
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff"], torch.tensor(expected_abs_diff))


@pytest.mark.parametrize("use_rollout_logprobs", [False, True])
def test_policy_loss_only_backpropagates_through_current_policy(monkeypatch, use_rollout_logprobs: bool):
    args = _make_args(use_rollout_logprobs=use_rollout_logprobs)
    old_source = torch.tensor([0.10, 0.20], requires_grad=True)
    rollout_source = torch.tensor([0.12, 0.22], requires_grad=True)
    advantage_source = torch.tensor([1.0, 2.0], requires_grad=True)
    current_logits = torch.tensor([[[0.15], [0.25], [0.0]]], requires_grad=True)
    batch = _make_batch(
        old_log_probs=old_source.sin(),
        rollout_log_probs=rollout_source.sin(),
    )
    if use_rollout_logprobs:
        batch.pop("log_probs")
    batch["advantages"] = [advantage_source.square()]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)

    def fake_get_log_probs_and_entropy(logits, *args, **kwargs):
        return {"log_probs": [logits.flatten()[:2].sin()]}

    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        fake_get_log_probs_and_entropy,
    )

    loss, _ = loss_utils.policy_loss_function(
        args,
        batch,
        logits=current_logits,
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    loss.backward()

    assert current_logits.grad is not None
    assert torch.count_nonzero(current_logits.grad) > 0
    assert old_source.grad is None
    assert rollout_source.grad is None
    assert advantage_source.grad is None


def test_kl_loss_does_not_backpropagate_through_reference_scores(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    args.use_kl_loss = True
    args.kl_loss_coef = 0.5
    reference_source = torch.tensor([0.30, 0.40], requires_grad=True)
    current_logits = torch.tensor([[[0.15], [0.25], [0.0]]], requires_grad=True)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20]),
        rollout_log_probs=torch.tensor([0.12, 0.22]),
    )
    batch["ref_log_probs"] = [reference_source.sin()]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda logits, *args, **kwargs: {"log_probs": [logits.flatten()[:2].sin()]},
    )

    loss, _ = loss_utils.policy_loss_function(
        args,
        batch,
        logits=current_logits,
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    loss.backward()

    assert current_logits.grad is not None
    assert torch.count_nonzero(current_logits.grad) > 0
    assert reference_source.grad is None


def test_custom_tis_can_ignore_missing_trainer_scored_log_probs(monkeypatch):
    args = _make_args(use_rollout_logprobs=True)
    args.use_tis = True
    args.custom_tis_function_path = "tests.custom_tis"
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20]),
        rollout_log_probs=torch.tensor([0.12, 0.22]),
    )
    batch.pop("log_probs")

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda logits, *args, **kwargs: {"log_probs": [logits.flatten()[:2].sin()]},
    )

    def custom_tis(**kwargs):
        assert kwargs["train_log_probs"] is None
        assert all(not tensor.requires_grad for tensor in kwargs["rollout_log_probs"])
        return kwargs["pg_loss"], kwargs["loss_masks"], {}

    monkeypatch.setattr(loss_utils, "load_function", lambda path: custom_tis)
    monkeypatch.setattr(
        loss_utils,
        "get_sum_of_sample_mean",
        lambda *args, **kwargs: lambda tensor: tensor.float().mean(),
    )

    logits = torch.tensor([[[0.15], [0.25], [0.0]]], requires_grad=True)
    loss, _ = loss_utils.policy_loss_function(
        args,
        batch,
        logits=logits,
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0


def test_zero_weighted_entropy_nan_does_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        assert kwargs["with_entropy"] is False
        return {"log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)]}

    monkeypatch.setattr(loss_utils, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["entropy_loss"], torch.tensor(0.0))


def test_zero_weighted_kl_nan_does_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    args.use_kl_loss = True
    args.kl_loss_coef = 0.0
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )
    batch["ref_log_probs"] = [torch.tensor([float("nan"), float("nan")], dtype=torch.float32)]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)],
        },
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])


def test_masked_nonfinite_ppo_terms_do_not_poison_policy_loss(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, float("nan")], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, float("nan")], dtype=torch.float32),
    )
    batch["loss_masks"] = [torch.tensor([1.0, 0.0], dtype=torch.float32)]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [torch.tensor([0.10, float("nan")], dtype=torch.float32)],
        },
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: (tensor * batch["loss_masks"][0]).sum(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["ppo_kl"])


def test_m2po_metrics_survive_the_token_normaliser(monkeypatch):
    """M2PO's clip bounds are per-step scalars, but every reported_loss entry is
    divided by the token normaliser downstream.

    Emitting `torch.tensor(value)` looks right locally and arrives scaled by
    1/num_tokens -- observed as a clip epsilon of 5.5e-05 where the floor is 0.3,
    a factor of 5445. Broadcasting to a per-token constant is what makes the
    sample mean return the scalar itself.
    """
    args = _make_args(use_rollout_logprobs=True)
    args.use_m2po = True
    args.m2po_budget = 0.04
    args.m2po_miniclip_low = 0.3
    args.m2po_miniclip_high = 0.5

    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )

    monkeypatch.setattr(
        loss_utils, "get_parallel_state", lambda: SimpleNamespace(tp=SimpleNamespace(group=None))
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *a, **k: {
            "log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)],
            "entropy": [torch.zeros(2, dtype=torch.float32)],
        },
    )

    def strict_reducer(tensor):
        # The real reducer splits by response length, so a 0-d input is a bug --
        # and `ppo_kl` is already reduced by the time reported_loss is built.
        assert tensor.dim() >= 1, "reducer needs a per-token tensor"
        return tensor.float().mean()

    _, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=strict_reducer,
    )

    # This batch is on-policy, so the budget cannot bind and the floors stand.
    assert float(metrics["m2po_eps_clip"]) == pytest.approx(0.3)
    assert float(metrics["m2po_eps_clip_high"]) == pytest.approx(0.5)


def test_policy_rollout_metrics_survive_use_rollout_logprobs(monkeypatch):
    """`train_rollout_*` compares a tensor with itself under --use-rollout-logprobs.

    `old_log_probs` becomes `batch["rollout_log_probs"]`, and the mismatch metrics
    are referenced to it, so the difference is identically zero: abs_diff and kl
    pin at 0.0 and both ESS ratios at 1.0. That is indistinguishable from a
    healthy run, which is the dangerous part -- 1.0 is also what "no mismatch"
    looks like.

    The `policy_rollout_*` family references the forward pass instead, so it
    still measures something. This is the whole point of that family; if it ever
    reads 0.0/1.0 here again, it has been re-pointed at the denominator.
    """
    args = _make_args(use_rollout_logprobs=True)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )
    # The trainer's forward disagrees with the rollout engine -- the thing to measure.
    train_log_probs = torch.tensor([0.40, 0.80], dtype=torch.float32)

    monkeypatch.setattr(
        loss_utils, "get_parallel_state", lambda: SimpleNamespace(tp=SimpleNamespace(group=None))
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *a, **k: {"log_probs": [train_log_probs.clone()], "entropy": [torch.zeros(2)]},
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    _, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    # The existing family is degenerate here, and stays that way: it is upstream's
    # true-on-policy check and other users read it.
    assert float(metrics["train_rollout_logprob_abs_diff"]) == 0.0
    assert float(metrics["train_rollout_kl"]) == 0.0

    # The new family sees the real gap: |0.10-0.40| and |0.20-0.80| average to 0.45.
    assert float(metrics["policy_rollout_abs_diff"]) == pytest.approx(0.45)
    assert float(metrics["policy_rollout_kl"]) > 0.0
    # Sequence-level parts are emitted for aggregate_train_losses to combine.
    assert "_policy_seq_ess_sum_w" in metrics
