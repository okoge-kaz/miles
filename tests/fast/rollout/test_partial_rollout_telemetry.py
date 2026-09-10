import pytest

from miles.rollout.partial_rollout_telemetry import (
    START_ROLLOUT_ID_KEY,
    TOKEN_BOUNDARY_LEDGER_KEY,
    collect_partial_rollout_staleness_metrics,
    collect_partial_rollout_work_metrics,
    stamp_partial_rollout_start,
)
from miles.rollout.recycle_compute_metrics import SAMPLE_REFERENCE_VERSION_KEY, TRAIN_VERSION_KEY
from miles.utils.types import Sample


def test_stamp_partial_rollout_start_uses_token_length_and_keeps_origin() -> None:
    sample = Sample(index=7, response="", response_length=1)

    stamp_partial_rollout_start([sample], rollout_id=3)
    stamp_partial_rollout_start([sample], rollout_id=4)

    assert sample.metadata[START_ROLLOUT_ID_KEY] == 3
    assert sample.metadata[TOKEN_BOUNDARY_LEDGER_KEY] == [[1, 3]]


def test_token_ledger_separates_carried_prefix_from_current_suffix() -> None:
    sample = Sample(index=8, response_length=3)
    stamp_partial_rollout_start([sample], rollout_id=2)
    sample.response_length = 5
    stamp_partial_rollout_start([sample], rollout_id=3)
    sample.response_length = 7
    sample.loss_mask = [1, 0, 1, 1, 1, 1, 1]

    metrics = collect_partial_rollout_staleness_metrics([[sample]], rollout_id=4)

    assert sample.metadata[TOKEN_BOUNDARY_LEDGER_KEY] == [[3, 2], [5, 3]]
    assert metrics["staleness/token_lag/exact/num_tokens"] == 7.0
    assert metrics["staleness/token_lag/exact/mean"] == pytest.approx(8 / 7)
    assert metrics["staleness/token_lag/exact/loss_token/num_tokens"] == 6.0
    assert metrics["staleness/token_lag/exact/loss_token/mean"] == pytest.approx(1.0)
    assert metrics["staleness/partial_rollout/carried_prefix_tokens"] == 5.0
    assert metrics["staleness/partial_rollout/current_suffix_tokens"] == 2.0
    assert metrics["staleness/partial_rollout/carried_prefix_token_frac"] == pytest.approx(5 / 7)


def test_collect_partial_rollout_staleness_matches_fully_async_group_namespace() -> None:
    fresh = Sample(index=0, group_index=0)
    retained = Sample(
        index=1,
        group_index=1,
        response_length=4,
        metadata={START_ROLLOUT_ID_KEY: 2},
    )
    retained_sibling = Sample(index=2, group_index=1)

    metrics = collect_partial_rollout_staleness_metrics(
        [[fresh], [retained, retained_sibling]],
        rollout_id=4,
    )

    assert metrics["staleness/total/mean"] == 1.0
    assert metrics["staleness/total/count_0"] == 1.0
    assert metrics["staleness/total/count_2"] == 1.0
    assert "staleness/rollout/mean" not in metrics
    assert metrics["staleness/pre_queue/mean"] == 1.0
    assert metrics["staleness/in_queue/max"] == 0.0
    assert metrics["staleness/partial_rollout/resumed_group_frac"] == 0.5
    assert metrics["staleness/partial_rollout/resumed_sample_frac"] == pytest.approx(1 / 3)
    assert metrics["staleness/partial_rollout/sample_total/num_samples"] == 3.0

    assert fresh.metadata[SAMPLE_REFERENCE_VERSION_KEY] == 4
    assert retained.metadata[SAMPLE_REFERENCE_VERSION_KEY] == 2
    assert retained_sibling.metadata[SAMPLE_REFERENCE_VERSION_KEY] == 4
    assert all(sample.metadata[TRAIN_VERSION_KEY] == 4 for sample in (fresh, retained, retained_sibling))


def test_collect_partial_rollout_staleness_rejects_future_origin() -> None:
    sample = Sample(index=9, metadata={START_ROLLOUT_ID_KEY: 5})

    with pytest.raises(RuntimeError, match="Invalid partial-rollout origin"):
        collect_partial_rollout_staleness_metrics([[sample]], rollout_id=4)


def test_partial_rollout_work_accounting_separates_dispositions() -> None:
    accepted = Sample(index=1, response_length=4)
    carried = Sample(index=2, response_length=2)
    stamp_partial_rollout_start([carried], rollout_id=3)
    carried.response_length = 3
    stamp_partial_rollout_start([carried], rollout_id=4)
    surplus = Sample(index=3, response_length=1)
    stamp_partial_rollout_start([surplus], rollout_id=2)
    surplus.response_length = 3
    filtered = Sample(index=4, response_length=2)

    metrics = collect_partial_rollout_work_metrics(
        launched_groups=5,
        launched_trajectories=10,
        launched_existing_response_tokens=6,
        accepted=[[accepted]],
        carried=[[carried]],
        dynamic_filter_discarded=[[filtered]],
        completed_surplus_discarded=[[surplus]],
        generation_failed_groups=1,
        rollout_id=4,
    )

    assert metrics["rollout/partial_rollout/launched_groups"] == 5.0
    assert metrics["rollout/partial_rollout/launched_trajectories"] == 10.0
    assert metrics["rollout/partial_rollout/launched_existing_response_tokens"] == 6.0
    assert metrics["rollout/partial_rollout/accepted/response_tokens"] == 4.0
    assert metrics["rollout/partial_rollout/accepted/current_rollout_response_tokens"] == 4.0
    assert metrics["rollout/partial_rollout/carried/response_tokens"] == 3.0
    assert metrics["rollout/partial_rollout/carried/current_rollout_response_tokens"] == 1.0
    assert metrics["rollout/partial_rollout/completed_surplus_discarded/groups"] == 1.0
    assert metrics["rollout/partial_rollout/completed_surplus_discarded/current_rollout_response_tokens"] == 2.0
    assert metrics["rollout/partial_rollout/dynamic_filter_discarded/groups"] == 1.0
    assert metrics["rollout/partial_rollout/generation_failed_groups"] == 1.0
    assert metrics["rollout/partial_rollout/accounting_error_groups"] == 0.0
