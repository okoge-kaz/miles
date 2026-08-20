from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

from miles.rollout.queue_policy import should_prefetch_rollout_batches, validate_fully_async_queue_args

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def make_args(**overrides):
    values = dict(
        fully_async=True,
        fully_async_queue_type="queue-recycle",
        fully_async_queue_factor=1,
        max_weight_staleness=2,
        staleness_reference="prefill",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("queue-recycle", True),
        ("queue-max", False),
        ("queue-drop", False),
    ],
)
def test_batch_prefetch_contract(policy, expected):
    assert should_prefetch_rollout_batches(make_args(fully_async_queue_type=policy)) is expected


def test_non_fully_async_training_keeps_its_existing_prefetch():
    args = make_args(fully_async=False, fully_async_queue_type="queue-recycle")
    assert should_prefetch_rollout_batches(args) is True


def test_queue_max_requires_prefill_age_bound():
    validate_fully_async_queue_args(make_args(fully_async_queue_type="queue-max"))

    with pytest.raises(ValueError, match="requires --max-weight-staleness"):
        validate_fully_async_queue_args(make_args(fully_async_queue_type="queue-max", max_weight_staleness=None))
    with pytest.raises(ValueError, match="requires --staleness-reference prefill"):
        validate_fully_async_queue_args(
            make_args(fully_async_queue_type="queue-max", staleness_reference="completion")
        )


def test_queue_drop_uses_capacity_instead_of_age_bound():
    validate_fully_async_queue_args(
        make_args(
            fully_async_queue_type="queue-drop",
            fully_async_queue_factor=2,
            max_weight_staleness=None,
        )
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_fully_async_queue_args(make_args(fully_async_queue_type="queue-drop"))
    with pytest.raises(ValueError, match="must be at least 1"):
        validate_fully_async_queue_args(
            make_args(
                fully_async_queue_type="queue-drop",
                fully_async_queue_factor=0,
                max_weight_staleness=None,
            )
        )


@pytest.mark.parametrize("policy", ["queue-recycle", "queue-max"])
def test_queue_factor_is_rejected_when_it_cannot_affect_selection(policy):
    with pytest.raises(ValueError, match="only used by queue-drop"):
        validate_fully_async_queue_args(make_args(fully_async_queue_type=policy, fully_async_queue_factor=2))


def test_negative_age_bound_is_rejected():
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_fully_async_queue_args(make_args(max_weight_staleness=-1))


def test_queue_recycle_requires_positive_age_bound():
    with pytest.raises(ValueError, match="strict gap < bound rule admits no group"):
        validate_fully_async_queue_args(make_args(max_weight_staleness=0))

    validate_fully_async_queue_args(make_args(fully_async_queue_type="queue-max", max_weight_staleness=0))


def test_named_policy_requires_fully_async():
    with pytest.raises(ValueError, match="requires --fully-async"):
        validate_fully_async_queue_args(
            make_args(
                fully_async=False,
                fully_async_queue_type="queue-drop",
                max_weight_staleness=None,
            )
        )
