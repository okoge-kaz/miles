from argparse import Namespace

import pytest

from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def _entry(index, versions, provenance):
    sample = Sample(
        index=index,
        group_index=index,
        reward=1.0,
        status=Sample.Status.COMPLETED,
        metadata={"policy_provenance": provenance},
        weight_versions=[WeightVersionsPerCall([WeightVersionSpan(str(v), a, b) for v, a, b in versions])],
    )
    return DataBufferInput(prompt_group=[sample], group=[sample])


def _buffer(**overrides):
    args = Namespace(
        **{
            "rollout_batch_size": 1,
            "async_data_buffer_capacity_factor": 8,
            "dynamic_sampling_filter_path": None,
            "reward_key": None,
            "max_weight_staleness": 1,
            **overrides,
        }
    )
    rejected = []
    return DefaultDataBuffer(DataBufferConstructorInput(args, rejected.append)), rejected


@pytest.mark.parametrize(
    "definition,selected",
    [("oldest", 2), ("newest", 1), ("token-mean", 2), ("submission", 2), ("first-prefill", 2), ("ready", 1)],
)
async def test_selected_staleness_definition_controls_group_admission(definition, selected):
    buffer, rejected = _buffer(async_staleness_metric=definition)
    keys = ("submitted_weight_version", "first_prefill_weight_version", "ready_weight_version")
    await buffer.put(_entry(1, [(1, 0, 1), (5, 1, 4)], dict(zip(keys, (3, 4, 5), strict=True))))
    await buffer.put(_entry(2, [(6, 0, 4)], dict.fromkeys(keys, 6)))

    assert (await buffer.get(current_version=6)).group[0].index == selected
    assert [group[0].index for group in rejected] == ([1] if selected == 2 else [])


async def test_bounded_first_prefill_rejects_missing_provenance():
    buffer, _ = _buffer(async_staleness_metric="first-prefill")
    await buffer.put(_entry(1, [(6, 0, 4)], {}))
    with pytest.raises(ValueError, match="Missing provenance"):
        await buffer.get(current_version=6)


async def test_priority_order_keeps_ties_stable_and_unknown_versions_last():
    buffer, _ = _buffer(async_data_buffer_order="oldest-policy-first", max_weight_staleness=None)
    for index, version in enumerate((None, 3, 1, 1, 5)):
        await buffer.put(_entry(index, [] if version is None else [(version, 0, 4)], {}))
    assert [(await buffer.get(current_version=6)).group[0].index for _ in range(5)] == [2, 3, 1, 4, 0]
