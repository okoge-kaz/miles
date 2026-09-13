from dataclasses import replace

import pytest
import torch

from miles.backends.training_utils import loss as dispatcher
from miles.backends.training_utils.cp_utils import get_local_response_loss_masks
from miles.backends.training_utils.loss_hub import losses
from miles.backends.training_utils.loss_hub import sequence_filter as filtering
from miles.backends.training_utils.parallel import GroupInfo, set_parallel_state

from .loss_test_utils import make_args, make_parallel_state


def _args(**overrides):
    return make_args(
        **{
            "fuse_one_step_actor_logprobs": True,
            "use_train_rollout_logprob_sequence_filter": True,
            "train_rollout_logprob_sequence_filter_threshold": 2.0,
            "calculate_per_token_loss": True,
            "use_tis": True,
            "tis_clip_low": 0.2,
            "tis_clip": 5.0,
            "entropy_coef": 0,
            "observe_training_entropy": False,
            **overrides,
        }
    )


def _batch(ratios, masks=None):
    rollout = [torch.full((len(row),), -8.0) for row in ratios]
    train = [old + torch.tensor(row).log() for old, row in zip(rollout, ratios, strict=True)]
    lengths = [len(row) for row in ratios]
    return train, {
        "rollout_log_probs": rollout,
        "advantages": [torch.ones(n) for n in lengths],
        "loss_masks": masks or [torch.ones(n) for n in lengths],
        "response_lengths": lengths,
        "total_lengths": [n + 3 for n in lengths],
        "unconcat_tokens": [torch.ones(n + 3, dtype=torch.long) for n in lengths],
    }


def test_symmetric_arithmetic_mean_and_whole_response_mask():
    make_parallel_state()
    train, batch = _batch(
        [
            [0.4, 0.4],  # Inverse ratio=2.5, reject despite rho < 2.
            [3.0, 1, 1, 1, 1, 1, 1, 1],  # Mean=1.25, retain the token with rho=3.
            [2.0, 2.0],  # Threshold is inclusive.
            [4.0, 0.25],  # Geometric mean=1, symmetric arithmetic mean=4.
        ]
    )
    filtered, metrics = filtering.filter_train_rollout_logprob_sequences(_args(), batch, train)
    assert [int(mask.sum()) for mask in filtered["loss_masks"]] == [0, 8, 2, 0]
    assert [int(mask.sum()) for mask in batch["loss_masks"]] == [2, 8, 2, 2]
    assert metrics[filtering.SEQUENCE_FILTER_TOKEN_COUNT] == 10
    assert filtered["advantages"] is batch["advantages"]


def test_padding_empty_masks_and_nonfinite_error():
    make_parallel_state()
    train, batch = _batch([[1, 1e10], [1, 1], [1, 1]])
    batch["loss_masks"] = [torch.tensor([1.0, 0]), torch.zeros(2), torch.ones(2)]
    train[0][1] = float("nan")  # Masked padding must not poison a valid sequence.
    train[2][1] = float("nan")  # Invalid active error rejects the response.
    filtered, metrics = filtering.filter_train_rollout_logprob_sequences(_args(), batch, train)
    assert [int(mask.sum()) for mask in filtered["loss_masks"]] == [1, 0, 0]
    assert metrics[filtering.SEQUENCE_FILTER_PART_PREFIX + "sequences"] == 2
    assert metrics[filtering.SEQUENCE_FILTER_PART_PREFIX + "rejected_sequences"] == 1


@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_cp_decision_uses_all_response_chunks(monkeypatch, qkv_format):
    state = make_parallel_state()
    args = _args(qkv_format=qkv_format)
    train, batch = _batch([[6.0, 1, 1, 1, 1, 1, 1], [0.1, 1, 1, 1, 1], [3.0]])
    batch["max_seq_lens"] = [16, 12, 8]
    expected, _ = filtering.filter_train_rollout_logprob_sequences(args, batch, train)
    totals = torch.tensor([[12.0, 7], [14.0, 5], [3.0, 1]])
    local_parts = []
    for rank in range(2):
        set_parallel_state(replace(state, cp=GroupInfo(rank=rank, size=2, group=None)))
        local_train = get_local_response_loss_masks(
            batch["total_lengths"], batch["response_lengths"], train, qkv_format, batch["max_seq_lens"]
        )
        local_rollout = get_local_response_loss_masks(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["rollout_log_probs"],
            qkv_format,
            batch["max_seq_lens"],
        )

        def reduce(parts, group):
            local_parts.append(parts.clone())
            parts.copy_(totals)

        monkeypatch.setattr(filtering.dist, "all_reduce", reduce)
        actual, _ = filtering.filter_train_rollout_logprob_sequences(
            args, {**batch, "rollout_log_probs": local_rollout}, local_train
        )
        for left, right in zip(actual["loss_masks"], expected["loss_masks"], strict=True):
            torch.testing.assert_close(left, right)
    torch.testing.assert_close(sum(local_parts), totals)
    set_parallel_state(state)


def _loss(monkeypatch, ratios, **overrides):
    make_parallel_state()
    train, batch = _batch(ratios)
    train = [values.requires_grad_() for values in train]
    forwards = []

    def log_probs(*args, **kwargs):
        forwards.append(True)
        return {"log_probs": train}

    monkeypatch.setattr(losses, "get_log_probs_and_entropy", log_probs)
    loss, count, logging = dispatcher.loss_function(
        _args(**overrides),
        batch,
        num_microbatches=1,
        logits=torch.zeros(1, sum(batch["total_lengths"]), 16),
        apply_megatron_loss_scaling=True,
    )
    assert len(forwards) == 1
    return loss, count, logging, train


def test_filter_and_tis_gradient_and_global_token_normalizer(monkeypatch):
    # The accepted response has outliers on both sides of the TIS band, but its
    # response mean is below two. The second response is entirely rejected.
    ratios = [[6.0, 0.1] + [1.0] * 14, [0.4] * 5]
    loss, count, _, train = _loss(monkeypatch, ratios)
    assert count == 16  # Not 21, not one mean per microbatch/response.
    (loss / count).backward()
    weights = torch.tensor([5.0, 0.2] + [1.0] * 14)
    torch.testing.assert_close(train[0].grad, -weights / 16)
    torch.testing.assert_close(train[1].grad, torch.zeros(5))
    torch.testing.assert_close(loss.detach(), -weights.sum())


@pytest.mark.parametrize("use_tis", [True, False])
def test_fully_filtered_microbatch_has_zero_loss_gradient_and_tokens(monkeypatch, use_tis):
    loss, count, _, train = _loss(monkeypatch, [[0.25] * 4], use_tis=use_tis)
    assert count == 0
    assert loss == 0
    loss.backward()
    torch.testing.assert_close(train[0].grad, torch.zeros(4))


def test_loss_and_filter_logging_aggregate_over_unequal_microbatches(monkeypatch):
    from miles.backends.training_utils.log_utils import MultiPGUtil, aggregate_train_losses

    monkeypatch.setattr(MultiPGUtil, "all_reduce", lambda *args, **kwargs: None)
    first, n1, log1, _ = _loss(monkeypatch, [[1.5, 1.5], [0.25] * 5])
    second, n2, log2, _ = _loss(monkeypatch, [[1.0] * 4])
    metrics = aggregate_train_losses([log1, log2])
    assert metrics["loss"] == pytest.approx(float((first + second).detach() / (n1 + n2)))
    prefix = "train_rollout_logprob_sequence_filter/"
    assert metrics[prefix + "rejected_sequence_fraction"] == pytest.approx(1 / 3)
    assert metrics[prefix + "rejected_token_fraction"] == pytest.approx(5 / 11)
    assert metrics[prefix + "kept_tokens"] == 6
    _, _, empty, _ = _loss(monkeypatch, [[0.25] * 5])
    metrics = aggregate_train_losses([empty])
    assert metrics["loss"] == 0
    assert metrics[prefix + "rejected_sequence_fraction"] == 1
