from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import pytest

from experiments.src.training import truncation


@dataclass
class RewardSample:
    class Status(Enum):
        COMPLETED = 1
        TRUNCATED = 2

    status: Status = Status.COMPLETED
    response_length: int = 3
    loss_mask: list[int] | None = None
    reward: float | None = None


@pytest.fixture
def scorer(monkeypatch):
    calls = []

    async def score(args, sample, **kwargs):
        calls.append(sample)
        return 1.0

    monkeypatch.setattr(truncation, "_score_builtin", score)
    return calls


async def test_zero_reward_preserves_loss_mask_and_skips_scorer(scorer):
    sample = RewardSample(status=RewardSample.Status.TRUNCATED, loss_mask=[1, 0, 1])
    assert await truncation.zero_reward(SimpleNamespace(), sample) == 0.0
    assert sample.loss_mask == [1, 0, 1]
    assert scorer == []


async def test_zero_loss_preserves_normal_reward_and_zeros_response_mask(scorer):
    sample = RewardSample(status=RewardSample.Status.TRUNCATED, loss_mask=[1, 0, 1])
    assert await truncation.zero_loss(SimpleNamespace(), sample) == 1.0
    assert sample.loss_mask == [0, 0, 0]
    assert scorer == [sample]


@pytest.mark.parametrize("adapter", [truncation.zero_reward, truncation.zero_loss])
async def test_completed_response_keeps_reward_and_existing_mask(scorer, adapter):
    sample = RewardSample(loss_mask=[0, 1, 1])
    assert await adapter(SimpleNamespace(), sample) == 1.0
    assert sample.loss_mask == [0, 1, 1]


async def test_batched_zero_loss_keeps_order_and_grpo_cohort(scorer):
    samples = [RewardSample(status=RewardSample.Status.TRUNCATED), RewardSample()]
    rewards = await truncation.zero_loss(SimpleNamespace(), samples)
    assert rewards == [1.0, 1.0]
    assert len(samples) == 2 and samples[0].loss_mask == [0, 0, 0]
    assert samples[1].loss_mask is None


async def test_batched_zero_reward_keeps_order_and_scores_only_completed(scorer):
    samples = [RewardSample(), RewardSample(status=RewardSample.Status.TRUNCATED)]
    assert await truncation.zero_reward(SimpleNamespace(), samples) == [1.0, 0.0]
    assert scorer == [samples[0]]
