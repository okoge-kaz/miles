import pytest
import torch

from miles.backends.training_utils import loss

from .loss_test_utils import make_args, make_parallel_state


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (None, None, [-30.0, 3.0, 40.0]),
        (-20.0, 20.0, [-20.0, 3.0, 20.0]),
        (None, 20.0, [-30.0, 3.0, 20.0]),
        (-20.0, None, [-20.0, 3.0, 40.0]),
    ],
)
def test_clipping_preserves_grpo_returns_and_skipped_logprob_path(low, high, expected):
    make_parallel_state()
    args = make_args(
        kl_coef=0.0,
        skip_actor_forward_only=True,
        advantage_clip_low=low,
        advantage_clip_high=high,
    )
    data = {
        "rewards": [-30.0, 3.0, 40.0],
        "response_lengths": [2, 1, 3],
        "total_lengths": [4, 3, 5],
        "loss_masks": [torch.ones(2), torch.ones(1), torch.ones(3)],
    }
    loss.compute_advantages_and_returns(args, data)
    for advantage, target, reward, clipped in zip(
        data["advantages"], data["returns"], data["rewards"], expected, strict=True
    ):
        torch.testing.assert_close(advantage, torch.full_like(advantage, clipped))
        torch.testing.assert_close(target, torch.full_like(target, reward))
    assert "log_probs" not in data


def test_clipping_happens_after_optional_normalization(monkeypatch):
    make_parallel_state()
    args = make_args(
        kl_coef=0.0,
        normalize_advantages=True,
        advantage_clip_low=-20.0,
        advantage_clip_high=20.0,
    )
    data = {
        "rewards": [2.0],
        "response_lengths": [3],
        "total_lengths": [5],
        "loss_masks": [torch.ones(3)],
        "log_probs": [torch.zeros(3)],
    }
    monkeypatch.setattr(loss, "normalize_advantages", lambda *args: [torch.tensor([-50.0, 1.0, 50.0])])
    loss.compute_advantages_and_returns(args, data)
    torch.testing.assert_close(data["advantages"][0], torch.tensor([-20.0, 1.0, 20.0]))
    torch.testing.assert_close(data["returns"][0], torch.full((3,), 2.0))
