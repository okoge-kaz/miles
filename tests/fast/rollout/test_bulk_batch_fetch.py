import asyncio

import pytest

from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DefaultDataBuffer
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group, put_group


@pytest.fixture
async def consumer(monkeypatch):
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=3, max_weight_staleness=2), FakeDataSource())
    fn._ensure_buffer()
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    yield fn
    await fn._close()


@pytest.mark.parametrize("unused_policy", ["drop", "retry"])
async def test_batch_refills_stale_groups_without_overconsuming(consumer, unused_policy):
    fn = consumer
    fn._output = DefaultDataBuffer(
        DataBufferConstructorInput(
            args=fn.args,
            unused_handler_fn=fn._recycle if unused_policy == "retry" else lambda group: None,
        )
    )
    stale = make_group(9, weight_versions=["7"])
    groups = [make_group(3, weight_versions=["8"]), stale, make_group(1, weight_versions=["9"])]
    groups += [make_group(2, weight_versions=["10"]), make_group(4, weight_versions=["10"])]
    for group in groups:
        await put_group(fn._output, group)

    output = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))

    assert [group[0].group_index for group in output.samples] == [1, 2, 3]
    assert fn.data_source.recycled == ([stale] if unused_policy == "retry" else [])
    assert output.metrics["rollout/fully_async/stale_groups_filtered"] == 1
    assert output.metrics["rollout/fully_async/max_staleness"] == 2
    assert len(fn._output) == 1
    assert (await fn._output.get(current_version=10)).group == groups[-1]


async def test_partial_batch_waits_only_for_missing_groups_at_fixed_version(consumer):
    fn = consumer
    empty = asyncio.Event()
    versions = []

    class ObservedBuffer(DefaultDataBuffer):
        async def get(self, *, current_version=None, **kwargs):
            versions.append(current_version)
            if not len(self):
                empty.set()
            return await super().get(current_version=current_version, **kwargs)

    fn._output = ObservedBuffer(DataBufferConstructorInput(args=fn.args, unused_handler_fn=fn._recycle))
    first = make_group(1, weight_versions=["8"])
    await put_group(fn._output, first)
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0, weight_version=10)))
    await asyncio.wait_for(empty.wait(), 2)
    assert not drain.done()
    assert len(fn._output) == 0
    # Even a later notification cannot change this request's selection clock.
    fn.on_weight_version(11)
    for index in (2, 3):
        await put_group(fn._output, make_group(index, weight_versions=["8"]))
    output = await asyncio.wait_for(drain, 2)

    assert output.samples[0] is first
    assert versions == [10, 10, 10]
    assert fn.data_source.recycled == []
    assert output.metrics["perf/buffer_empty_wait_time"] > 0


async def test_batch_larger_than_buffer_capacity_makes_progress(monkeypatch):
    args = make_args(rollout_batch_size=5, async_data_buffer_capacity_factor=0.2)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    try:
        output = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=0)), 2)
        assert len(output.samples) == 5
    finally:
        await fn._close()


@pytest.mark.parametrize("failure", ["caller_cancel", "producer_error", "producer_cancel", "producer_return"])
async def test_interrupted_partial_batch_stops_consuming_before_return(consumer, failure):
    fn = consumer
    waiting = asyncio.Event()
    cleaned_up = asyncio.Event()
    fail = asyncio.Event()

    class WaitingBuffer(DefaultDataBuffer):
        async def get(self, **kwargs):
            if not len(self):
                waiting.set()
                try:
                    return await super().get(**kwargs)
                finally:
                    await asyncio.sleep(0)
                    cleaned_up.set()
            return await super().get(**kwargs)

    async def producer():
        await fail.wait()
        if failure == "producer_error":
            raise RuntimeError("producer failed")

    await fn._close()
    fn._worker = asyncio.create_task(producer())
    fn._output = WaitingBuffer(DataBufferConstructorInput(args=fn.args, unused_handler_fn=fn._recycle))
    await put_group(fn._output, make_group(1, weight_versions=["10"]))
    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0, weight_version=10)))
    await asyncio.wait_for(waiting.wait(), 2)
    assert fn._draining
    if failure == "caller_cancel":
        drain.cancel()
    elif failure == "producer_cancel":
        fn._worker.cancel()
    else:
        fail.set()
    expected = asyncio.CancelledError if failure.endswith("cancel") else RuntimeError
    with pytest.raises(expected):
        await asyncio.wait_for(drain, 2)
    assert cleaned_up.is_set()
    assert not fn._draining
    await put_group(fn._output, make_group(2, weight_versions=["10"]))
    await asyncio.sleep(0)
    assert len(fn._output) == 1


async def test_restore_failure_beats_completed_batch(consumer):
    fn = consumer

    class FailedRestore:
        checks = 0

        def check_restore(self):
            self.checks += 1
            if self.checks > 1:
                raise RuntimeError("restore corruption")

        async def close(self):
            pass

    fn._replay_data_buffer = FailedRestore()
    for index in range(3):
        await put_group(fn._output, make_group(index, weight_versions=["10"]))
    with pytest.raises(RuntimeError, match="restore corruption"):
        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=10))
    assert not fn._draining
