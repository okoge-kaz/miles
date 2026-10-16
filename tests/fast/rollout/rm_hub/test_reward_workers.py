import asyncio
import sys
from copy import deepcopy
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from experiments.src.training.truncation import zero_loss, zero_reward
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.rollout.rm_hub import worker as reward_worker
from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from miles.rollout.rm_hub.math_dapo_utils import compute_score as compute_score_dapo
from miles.utils.types import Sample
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group


class GatedWorker:
    def __init__(self):
        self.started = asyncio.Queue()
        self.release = asyncio.Event()
        self.score = SimpleNamespace(remote=self._score)

    async def _score(self, rm_type, response, label, metadata):
        self.started.put_nowait(response)
        await self.release.wait()
        return int(response == label), 0.01, {}


@pytest.fixture
async def executor(monkeypatch):
    remote = GatedWorker()
    executor = reward_worker.RewardExecutor([remote])
    monkeypatch.setattr(reward_worker, "_executor", executor)
    yield executor, remote
    remote.release.set()
    await asyncio.gather(*executor._pending, return_exceptions=True)


@pytest.mark.parametrize("rm_type", ["deepscaler", "gemma_math", "dapo", "math", "f1", "gpqa", "ifbench"])
async def test_ready_group_is_fetched_while_another_reward_is_pending(monkeypatch, executor, rm_type):
    _, remote = executor
    args = make_args(rollout_batch_size=1, rm_type=rm_type, custom_rm_path=None)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    fn._ensure_buffer()
    ready = make_group(5)
    await fn._output.put(DataBufferInput(prompt_group=ready, group=ready))
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    unfinished = Sample(prompt="p", response="answer", label="answer")
    task = asyncio.create_task(async_rm(args, unfinished))
    try:
        await asyncio.wait_for(remote.started.get(), 3)
        output = await asyncio.wait_for(fn(RolloutFnTrainInput(rollout_id=0, weight_version=0)), 3)
        assert output.samples == [ready]
        assert not task.done()
        assert unfinished.reward is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await fn._close()


async def test_cancelled_score_keeps_capacity_until_worker_finishes(executor):
    pool, remote = executor
    first = asyncio.create_task(pool.score("f1", "first", "first"))
    assert await asyncio.wait_for(remote.started.get(), 3) == "first"
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(pool.score("math", "second", "second"))
    await asyncio.sleep(0)
    assert remote.started.empty()
    remote.release.set()
    assert await asyncio.wait_for(second, 3) == 1
    assert remote.started.get_nowait() == "second"


async def test_worker_exception_propagates_and_releases_capacity(executor):
    pool, remote = executor

    async def fail(rm_type, response, label, metadata):
        raise RuntimeError("scoring worker failed")

    remote.score.remote = fail
    with pytest.raises(RuntimeError, match="scoring worker failed"):
        await pool.score("math", "x", "x")
    remote.score.remote = remote._score
    remote.release.set()
    assert await asyncio.wait_for(pool.score("f1", "y", "y"), 3) == 1


async def test_unscored_group_is_not_buffered(monkeypatch, executor):
    _, remote = executor
    args = make_args(rollout_batch_size=1, rm_type="deepscaler", custom_rm_path=None)

    async def generate(state, group, **kwargs):
        for sample in group:
            sample.reward = None
            sample.reward = await async_rm(args, sample)
        return group

    fn = make_fn(monkeypatch, args, FakeDataSource(), generate=generate)
    fn._ensure_buffer()
    producing = asyncio.create_task(fn._worker_loop())
    try:
        await asyncio.wait_for(remote.started.get(), 3)
        assert len(fn._output) == 0
    finally:
        producing.cancel()
        await asyncio.gather(producing, return_exceptions=True)
        await fn._close()


async def test_truncation_wrapper_uses_same_async_scorer(executor):
    _, remote = executor
    args = SimpleNamespace(custom_rm_path="experiments.src.training.truncation.zero_loss", rm_type="deepscaler")
    sample = Sample(prompt="", response="a", label="a", response_length=2, status=Sample.Status.TRUNCATED)
    assert await zero_reward(args, sample) == 0
    assert remote.started.empty()
    task = asyncio.create_task(zero_loss(args, sample))
    assert await asyncio.wait_for(remote.started.get(), 3) == "a"
    assert sample.loss_mask == [0, 0]
    remote.release.set()
    assert await task == 1


async def test_real_workers_preserve_rewards_and_can_be_reinitialized(ray_local_mode):
    examples = [
        (r"</think>\boxed{42}", "42"),
        (r"</think>\boxed{41}", "42"),
        (r"</think>\boxed{\frac{1}{2}}", "0.5"),
        (r"</think>\boxed{x+x}", "2*x"),
        (r"\boxed{42}", "42"),
    ]
    expected = [get_deepscaler_rule_based_reward(response, label) for response, label in examples]
    for _ in range(2):
        await reward_worker.initialize_reward_workers(2)
        try:
            actual = await asyncio.wait_for(
                asyncio.gather(*(reward_worker.score_reward("deepscaler", *example) for example in examples)), 60
            )
            assert actual == expected
            assert reward_worker.collect_reward_metrics()["rollout/reward/completed_samples"] == len(examples)
        finally:
            await reward_worker.close_reward_workers()


async def test_real_workers_preserve_mixed_rewards_metadata_and_order(ray_local_mode):
    args = SimpleNamespace(custom_rm_path=None, rm_type=None)
    samples = [
        Sample(response=r"</think>\boxed{42}", label="42", metadata={"rm_type": "deepscaler"}),
        Sample(response=r"<channel|>\boxed{42}", label="42", metadata={"rm_type": "gemma_math"}),
        Sample(response="Answer: 42", label="42", metadata={"rm_type": "dapo"}),
        Sample(response=r"\boxed{41}", label="42", metadata={"rm_type": "math"}),
        Sample(response="hello", label="hello world", metadata={"rm_type": "f1"}),
        Sample(response="Answer: B", label="Paris", metadata={"rm_type": "gpqa", "choices": ["London", "Paris"]}),
        Sample(response=r"\boxed{hello world}", label="hello world", metadata={"rm_type": "boxed_f1"}),
    ]
    inline_samples = deepcopy(samples)
    expected = await batched_async_rm(args, inline_samples)
    assert expected == [1, 1, compute_score_dapo("Answer: 42", "42"), 0, 2 / 3, 1.0, 1.0]
    for worker_count, log_event_loop in [(0, False), (0, True), (2, False)]:
        batch = deepcopy(samples)
        await reward_worker.initialize_reward_workers(worker_count, log_event_loop=log_event_loop)
        try:
            assert await asyncio.wait_for(batched_async_rm(args, batch), 60) == expected
            assert [sample.metadata for sample in batch] == [sample.metadata for sample in inline_samples]
        finally:
            await reward_worker.close_reward_workers()


async def test_cancelled_score_does_not_apply_late_metadata(executor):
    pool, remote = executor

    async def infer(rm_type, response, label, metadata):
        await remote._score(rm_type, response, label, metadata)
        return 1, 0.01, {"correct_answer": "Paris"}

    remote.score.remote = infer
    metadata = {"choices": ["Paris", "London"]}
    task = asyncio.create_task(pool.score("gpqa", "Paris", "Paris", metadata=metadata))
    await asyncio.wait_for(remote.started.get(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    remote.release.set()
    await asyncio.gather(*pool._pending)
    assert metadata == {"choices": ["Paris", "London"]}


async def test_ifbench_dispatch_preserves_metadata_and_strict_result(monkeypatch):
    seen = []

    def strict_reward(response, label, *, metadata):
        seen.append((response, label, deepcopy(metadata)))
        return float(response == "expected")

    # The optional checker package is not installed in the fast-test environment.
    monkeypatch.setitem(
        sys.modules, "miles.rollout.rm_hub.ifbench", SimpleNamespace(compute_ifbench_reward=strict_reward)
    )
    args = SimpleNamespace(custom_rm_path=None, rm_type="ifbench")
    metadata = {"prompt_text": "p", "instruction_id_list": ["id"], "kwargs": [{"count": 2}]}
    for response, expected in [("expected", 1.0), ("wrong", 0.0)]:
        sample = Sample(response=response, label="label", metadata=deepcopy(metadata))
        assert await async_rm(args, sample) == expected
        assert seen[-1] == (response, "label", metadata)
        assert sample.metadata == metadata


async def test_ifbench_worker_receives_metadata(executor):
    _, remote = executor
    original_score = remote._score
    received = []

    async def score(rm_type, response, label, metadata):
        received.append((rm_type, response, label, deepcopy(metadata)))
        return await original_score(rm_type, response, label, metadata)

    remote.score.remote = score
    metadata = {"prompt_text": "p", "instruction_id_list": ["id"], "kwargs": [{"count": 2}]}
    args = SimpleNamespace(custom_rm_path=None, rm_type="ifbench")
    task = asyncio.create_task(async_rm(args, Sample(response="answer", label="answer", metadata=metadata)))
    await asyncio.wait_for(remote.started.get(), 3)
    assert received == [("ifbench", "answer", "answer", metadata)]
    remote.release.set()
    assert await task == 1


async def test_ifbench_initialization_is_serialized_even_after_cancellation():
    first, second = GatedWorker(), GatedWorker()
    pool = reward_worker.RewardExecutor([first, second])
    tasks = []
    try:
        tasks.append(asyncio.create_task(pool.score("ifbench", "first", "first")))
        assert await asyncio.wait_for(first.started.get(), 3) == "first"
        tasks[0].cancel()
        await asyncio.gather(tasks[0], return_exceptions=True)
        tasks.append(asyncio.create_task(pool.score("ifbench", "second", "second")))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert second.started.empty()
        first.release.set()
        assert await asyncio.wait_for(second.started.get(), 3) == "second"
        # Initialization is complete: a third score can run while the second waits.
        tasks.append(asyncio.create_task(pool.score("ifbench", "third", "third")))
        assert await asyncio.wait_for(first.started.get(), 3) == "third"
        assert await asyncio.wait_for(tasks[2], 3) == 1
        assert not tasks[1].done()
    finally:
        first.release.set()
        second.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*pool._pending, return_exceptions=True)


async def test_event_loop_metrics_are_collected_reset_and_stopped_without_workers():
    pool = reward_worker.RewardExecutor([], log_event_loop=True)
    monitor = pool._monitor

    async def wait_for_measurement():
        while not pool._loop_lags:
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(wait_for_measurement(), 3)
        metrics = pool.collect_metrics()
        assert metrics["perf/rollout_event_loop_lag_mean"] >= 0
        assert 0 <= metrics["perf/rollout_event_loop_lag_p95"] <= metrics["perf/rollout_event_loop_lag_max"]
        assert not any("event_loop_lag" in name for name in pool.collect_metrics())
    finally:
        await pool.close()
    assert monitor.cancelled()


@pytest.mark.parametrize("rm_type", ["remote_rm", "custom", "random", "deterministic_random"])
async def test_external_and_random_rewards_bypass_cpu_pool(monkeypatch, executor, rm_type):
    pool, _ = executor
    score = AsyncMock(side_effect=AssertionError("CPU pool must not handle this reward"))
    monkeypatch.setattr(pool, "score", score)
    external = AsyncMock(return_value=0.75)
    monkeypatch.setattr("miles.rollout.rm_hub.remote_rm", external)
    monkeypatch.setattr("miles.rollout.rm_hub.load_function", lambda path: external)
    args = SimpleNamespace(custom_rm_path="test.custom" if rm_type == "custom" else None, rm_type=rm_type)
    result = await async_rm(args, Sample(response="answer", label="answer"))
    if rm_type in {"remote_rm", "custom"}:
        assert result == 0.75
        external.assert_awaited_once()
    else:
        assert result in {0, 1}
    score.assert_not_awaited()
