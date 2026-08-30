from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
import copy
import gc
import hashlib
import json
import struct
from argparse import Namespace
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

import miles.rollout.fully_async_rollout as fully_async
import miles.rollout.replay_buffer as replay_buffer_module
import miles.rollout.replay_buffer_codec as codec_module
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.replay_buffer import (
    dataset_fingerprint,
    decode_group,
    ensure_no_replay_buffer,
    load_replay_buffer,
    prune_replay_buffers,
    replay_buffer_path,
    save_replay_buffer,
)
from miles.rollout.replay_buffer_codec import (
    SAMPLE_CODEC_STATE_KEY,
    ReplayBufferPackedFieldCache,
    ReplayBufferSampleEncoder,
    materialize_replay_buffer_state,
)
from miles.utils.types import Sample


class _GenerateState:
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}
        self.aborted = False

    def reset(self) -> None:
        self.aborted = False


class _DataSource:
    def __init__(self, groups: list[list[Sample]] | None = None):
        self.groups = deque(groups or [])
        self.buffer: list[list[Sample]] = []
        self.next_group_id = 100
        self.cursor = 0

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        assert num_samples == 1
        self.cursor += 1
        if self.buffer:
            return [self.buffer.pop(0)]
        if self.groups:
            return [self.groups.popleft()]
        self.next_group_id += 1
        return [_prompt_group(self.next_group_id)]

    def add_samples(self, groups: list[list[Sample]]) -> None:
        self.buffer.extend(groups)

    def checkpoint_state(self) -> dict:
        return {"cursor": self.cursor, "next_group_id": self.next_group_id}

    def checkpoint_retry_buffer_groups(self) -> list[list[Sample]]:
        return list(self.buffer)

    def restore_checkpoint_state(self, state: dict) -> None:
        self.cursor = state["cursor"]
        self.next_group_id = state["next_group_id"]
        self.buffer.clear()


class _Version:
    def __init__(self, value: int):
        self.value = value

    async def get(self, args) -> int:
        return self.value


def _args(**overrides) -> Namespace:
    values = {
        "rollout_global_dataset": True,
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 2,
        "max_weight_staleness": None,
        "async_max_concurrent_samples": 2,
        "dynamic_sampling_filter_path": None,
        "rollout_sample_filter_path": None,
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 30000,
        "staleness_reference": "completion",
        "use_replay_buffer": True,
        "replay_buffer_type": "rollout",
        "prompt_data": None,
        "data_source_path": "miles.rollout.data_source.RolloutDataSourceWithBuffer",
    }
    values.update(overrides)
    return Namespace(**values)


def _prompt_group(group_id: int) -> list[Sample]:
    return [
        Sample(group_index=group_id, index=group_id * 10 + index, prompt=f"prompt-{group_id}") for index in range(2)
    ]


def _complete(group: list[Sample], version: int) -> list[Sample]:
    for sample in group:
        sample.response = "answer"
        sample.response_length = 1
        sample.reward = 1.0
        sample.status = Sample.Status.COMPLETED
        sample.weight_versions = [str(version)]
    return group


def _make_fn(monkeypatch, args, source, generate):
    monkeypatch.setattr(fully_async, "GenerateState", _GenerateState)
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate)
    fn = fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=source))
    fn._weight_version = _Version(5)
    fn.commit_applied_weight_version(5)
    return fn


async def _stop(fn: fully_async.FullyAsyncRolloutFn) -> None:
    tasks = [task for task in [fn._worker, *fn._active] if task is not None]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def _queue_args(policy: str, **overrides) -> Namespace:
    policy_args = {
        "fully_async_queue_type": policy,
        "fully_async_queue_factor": 1,
    }
    if policy == "queue-max":
        policy_args |= {"max_weight_staleness": 0, "staleness_reference": "prefill"}
    elif policy == "queue-drop":
        policy_args["max_weight_staleness"] = None
    policy_args.update(overrides)
    return _args(**policy_args)


def _ready_item(group_id: int, version: int = 5) -> tuple[list[Sample], list[Sample]]:
    prompt_group = _prompt_group(group_id)
    return prompt_group, _complete(copy.deepcopy(prompt_group), version)


def _partial_item(group_id: int) -> fully_async._InflightReplayItem:
    prompt_group = _prompt_group(group_id)
    generation_group = copy.deepcopy(prompt_group)
    for sample in generation_group:
        sample.tokens = [100, sample.index]
        sample.response = "partial"
        sample.response_length = 1
        sample.rollout_log_probs = [-0.25]
        sample.status = Sample.Status.ABORTED
    return fully_async._InflightReplayItem(prompt_group, generation_group)


@pytest.mark.parametrize("policy", ["queue-recycle", "queue-max", "queue-drop"])
async def test_ready_queue_restores_in_policy_storage(monkeypatch, policy):
    args = _queue_args(policy, rollout_batch_size=1, fully_async_queue_factor=2 if policy == "queue-drop" else 1)
    original = _make_fn(monkeypatch, args, _DataSource(), lambda *args, **kwargs: None)
    items = [_ready_item(group_id) for group_id in (1, 2)]
    original._pending_prompts = {group_id: _prompt_group(group_id) for group_id in (1, 2)}
    if policy == "queue-recycle":
        original._output = asyncio.Queue()
        for item in items:
            original._output.put_nowait(item)
    else:
        original._policy_output = deque(items)
        original._policy_output_ready = asyncio.Event()
        original._policy_output_ready.set()

    state = await original.replay_buffer_state(0)
    restored = _make_fn(monkeypatch, args, _DataSource(), lambda *args, **kwargs: None)
    await restored.restore_replay_buffer_state(state)

    if policy == "queue-recycle":
        restored_items = list(restored._output._queue)
        assert restored._policy_output is None
    else:
        restored_items = list(restored._policy_output)
        assert restored._output is None
        assert restored._policy_output_ready.is_set()
    assert [item[0][0].group_index for item in restored_items] == [1, 2]
    assert restored._resume_metrics["resume/replay_buffer/ready_groups_restored"] == 2


async def test_replay_buffer_rejects_cross_policy_and_capacity_restore(monkeypatch):
    recycle = _make_fn(
        monkeypatch,
        _queue_args("queue-recycle"),
        _DataSource(),
        lambda *args, **kwargs: None,
    )
    recycle_state = await recycle.replay_buffer_state(0)

    queue_max = _make_fn(monkeypatch, _queue_args("queue-max"), _DataSource(), lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="queue configuration"):
        await queue_max.restore_replay_buffer_state(recycle_state)

    state_without_queue_config = copy.deepcopy(recycle_state)
    del state_without_queue_config["queue_config"]
    restored_recycle = _make_fn(
        monkeypatch,
        _queue_args("queue-recycle"),
        _DataSource(),
        lambda *args, **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="missing required queue_config"):
        await restored_recycle.restore_replay_buffer_state(state_without_queue_config)

    queue_drop = _make_fn(
        monkeypatch,
        _queue_args("queue-drop", fully_async_queue_factor=1),
        _DataSource(),
        lambda *args, **kwargs: None,
    )
    drop_state = await queue_drop.replay_buffer_state(0)
    different_capacity = _make_fn(
        monkeypatch,
        _queue_args("queue-drop", fully_async_queue_factor=2),
        _DataSource(),
        lambda *args, **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="queue configuration"):
        await different_capacity.restore_replay_buffer_state(drop_state)


async def test_queue_drop_snapshot_applies_waiting_completion_evictions(monkeypatch):
    args = _queue_args(
        "queue-drop",
        rollout_batch_size=1,
        fully_async_queue_factor=2,
        save_debug_rollout_data="dump-{rollout_id}.pt",
    )
    original = _make_fn(monkeypatch, args, _DataSource(), lambda *args, **kwargs: None)
    items = [_ready_item(group_id) for group_id in (1, 2, 3, 4)]
    original._pending_prompts = {group_id: _prompt_group(group_id) for group_id in (1, 2, 3, 4)}
    original._policy_output = deque(items[:2])
    original._policy_output_ready = asyncio.Event()
    original._policy_output_ready.set()
    original._completed_waiting = {group_id: items[group_id - 1] for group_id in (3, 4)}

    for depth_before, (_, group) in enumerate(items):
        record = original._queue_lifecycle.begin_attempt(group, submission_version=5)
        original._queue_lifecycle.group_ready(record, group, ready_version=5, reward_values=[1.0, 1.0])
        original._producer_response_lengths.record("generated", group)
        fully_async.add_selection_population(
            original._producer_selection_populations,
            population_name="generated",
            samples=fully_async._iter_samples(group),
        )
        if depth_before < 2:
            fully_async.stamp_group_weight_version(group, fully_async.QUEUE_PUT_VERSION_KEY, 5)
            original._queue_lifecycle.enqueued(
                group,
                queue_put_version=5,
                depth_before=depth_before,
                depth_after=depth_before + 1,
            )

    original._pipeline_telemetry.add_trained_batch(accepted_tokens=7, optimizer_updates=2)
    state = await original.replay_buffer_state(0)
    materialized = materialize_replay_buffer_state(state)
    ready_items = [fully_async._decode_ready_item(item) for item in materialized["ready_items"]]
    pending_ids = [fully_async.prompt_group_id(decode_group(group)) for group in materialized["pending_prompts"]]
    telemetry = state["queue_telemetry"]

    assert [item[0][0].group_index for item in ready_items] == [3, 4]
    assert pending_ids == [3, 4]
    assert state["snapshot_counts"]["queue_evicted_groups"] == 2
    assert state["pipeline_telemetry"]["accepted_loss_tokens"] == 7
    assert telemetry["queue_evicted_groups"] == 2
    assert telemetry["queue_evicted_tokens"] == 4
    assert telemetry["producer_response_lengths"]["sample_lengths"]["queue_evicted"] == [1, 1, 1, 1]
    assert len(telemetry["producer_selection_populations"]["generated"]["_sample_count"]) == 8
    assert [record["group_index"] for record in telemetry["lifecycle"]["terminal_records"]] == [1, 2]
    assert {record["group_index"] for record in telemetry["lifecycle"]["live_records"].values()} == {3, 4}

    restored = _make_fn(monkeypatch, args, _DataSource(), lambda *args, **kwargs: None)
    await restored.restore_replay_buffer_state(state)
    assert [item[0][0].group_index for item in restored._policy_output] == [3, 4]
    assert set(restored._pending_prompts) == {3, 4}
    assert restored._queue_evicted_groups == 2
    assert restored._queue_evicted_tokens == 4
    assert restored._pipeline_telemetry.checkpoint_state()["accepted_loss_tokens"] == 7
    assert len(restored._producer_selection_populations["generated"]["_sample_count"]) == 8
    assert restored.data_source.buffer == []


async def test_queue_drop_live_eviction_finishes_pending_prompt(monkeypatch):
    args = _queue_args("queue-drop", rollout_batch_size=1)
    fn = _make_fn(monkeypatch, args, _DataSource(), lambda *args, **kwargs: None)
    fn._policy_output = deque()
    fn._policy_output_ready = asyncio.Event()
    fn._pending_prompts = {group_id: _prompt_group(group_id) for group_id in (1, 2)}

    await fn._enqueue_completed_group(_ready_item(1))
    await fn._enqueue_completed_group(_ready_item(2))

    assert set(fn._pending_prompts) == {2}
    assert [item[0][0].group_index for item in fn._policy_output] == [2]
    assert fn._queue_evicted_groups == 1


async def test_prepared_batch_is_a_warm_resume_hit_and_active_prompt_is_regenerated(monkeypatch):
    gate = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 1:
            await gate.wait()
        return _complete(group, 5)

    args = _args(save_debug_rollout_data="dump-{rollout_id}.pt")
    source = _DataSource([_prompt_group(1), _prompt_group(2)])
    original = _make_fn(monkeypatch, args, source, generate)
    output = await original(RolloutFnTrainInput(rollout_id=0))
    await _wait_until(lambda: any(group_id == 2 for group_id in original._pending_prompts))
    state = await original.replay_buffer_state(0)
    await _stop(original)

    invalid = _make_fn(monkeypatch, args, _DataSource(), generate)
    invalid_state = copy.deepcopy(state)
    invalid_state["prepared_batches"][0]["group_ids"] = [999]
    with pytest.raises(RuntimeError, match="does not match stored group"):
        await invalid.restore_replay_buffer_state(invalid_state)

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, generate)
    await restored.restore_replay_buffer_state(state)

    warm_output = await restored(RolloutFnTrainInput(rollout_id=0))
    assert warm_output.samples[0][0].index == output.samples[0][0].index
    assert warm_output.debug_metadata == output.debug_metadata
    assert warm_output.metrics["resume/replay_buffer/warm_prepared_batch_hit"] == 1
    assert restored._worker is None
    assert [group[0].group_index for group in restored_source.buffer] == [2]

    token = fully_async.rollout_batch_token(warm_output.samples)
    with pytest.raises(RuntimeError, match="token mismatch"):
        await restored.acknowledge_trained_batch(0, "wrong-token")
    assert 1 in restored._pending_prompts
    await restored.acknowledge_trained_batch(0, token)
    await restored.acknowledge_trained_batch(0, token)
    with pytest.raises(RuntimeError, match="unknown"):
        await restored.acknowledge_trained_batch(0, "wrong-after-commit")
    assert 1 not in restored._pending_prompts
    assert 2 in restored._pending_prompts


async def test_inflight_buffer_restores_partial_tokens_and_continues_generation(monkeypatch, tmp_path):
    interrupted = asyncio.Event()
    first_generation_started = asyncio.Event()

    async def abort_requests(_args):
        interrupted.set()

    async def generate(state, group, sampling_params, evaluation=False):
        if any(sample.response_length > 0 for sample in group):
            for sample in group:
                if sample.status == Sample.Status.COMPLETED:
                    continue
                assert sample.tokens == [100, sample.index]
                assert sample.response == "partial"
                assert sample.rollout_log_probs == [-0.25]
                sample.tokens.append(200 + sample.index)
                sample.response += "-done"
                sample.response_length += 1
                sample.rollout_log_probs.append(-0.5)
                sample.reward = 1.0
                sample.status = Sample.Status.COMPLETED
            return group

        first_generation_started.set()
        await interrupted.wait()
        for sample in group:
            sample.tokens = [100, sample.index]
            sample.response = "partial"
            sample.response_length = 1
            sample.rollout_log_probs = [-0.25]
            sample.status = Sample.Status.ABORTED
        return group

    monkeypatch.setattr(fully_async, "_abort_inflight_requests", abort_requests)
    args = _args(replay_buffer_type="inflight")
    original = _make_fn(monkeypatch, args, _DataSource([_prompt_group(1)]), generate)
    original._ensure_worker()
    await first_generation_started.wait()

    state = await original.replay_buffer_state(0)
    path, _ = save_replay_buffer(tmp_path, 0, state)
    assert path.name == "replay_buffer_0.pt"
    state = load_replay_buffer(
        tmp_path,
        0,
        expected_fingerprint=original.replay_buffer_dataset_fingerprint(),
    )
    materialized = materialize_replay_buffer_state(state)
    [inflight] = materialized["inflight_items"]
    assert [sample["tokens"] for sample in inflight["generation_group"]] == [[100, 10], [100, 11]]
    assert [sample["response"] for sample in inflight["generation_group"]] == ["partial", "partial"]
    assert materialized["snapshot_counts"]["inflight_groups"] == 1
    assert materialized["snapshot_counts"]["inflight_response_tokens"] == 2
    assert all("kv_cache" not in str(key) for key in materialized)
    await _stop(original)

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, generate)
    await restored.restore_replay_buffer_state(state)
    restored._weight_version = _Version(9)
    restored.commit_applied_weight_version(9)
    output = await restored(RolloutFnTrainInput(rollout_id=0))

    samples = output.samples[0]
    assert [sample.tokens for sample in samples] == [[100, 10, 210], [100, 11, 211]]
    assert [sample.response for sample in samples] == ["partial-done", "partial-done"]
    assert [sample.rollout_log_probs for sample in samples] == [[-0.25, -0.5], [-0.25, -0.5]]
    assert restored_source.buffer == []
    assert {sample.metadata[fully_async.SUBMISSION_VERSION_KEY] for sample in samples} == {5}
    assert {sample.metadata[fully_async.TRAJECTORY_START_VERSION_KEY] for sample in samples} == {5}
    assert output.metrics["resume/replay_buffer/inflight_groups_restored"] == 1
    assert output.metrics["resume/replay_buffer/inflight_tokens_restored"] == 2
    await _stop(restored)


async def test_inflight_capture_treats_unstarted_pending_group_as_unfinished(monkeypatch):
    interrupted = asyncio.Event()
    generation_started = asyncio.Event()

    async def abort_requests(_args):
        interrupted.set()

    async def generate(state, group, sampling_params, evaluation=False):
        generation_started.set()
        await interrupted.wait()
        return group

    monkeypatch.setattr(fully_async, "_abort_inflight_requests", abort_requests)
    fn = _make_fn(
        monkeypatch,
        _args(replay_buffer_type="inflight"),
        _DataSource([_prompt_group(1)]),
        generate,
    )
    fn._ensure_worker()
    await generation_started.wait()

    state = materialize_replay_buffer_state(await fn.replay_buffer_state(0))

    [inflight] = state["inflight_items"]
    assert {sample["status"] for sample in inflight["generation_group"]} == {"pending"}
    assert state["snapshot_counts"]["inflight_groups"] == 1
    await _stop(fn)


async def test_inflight_recapture_keeps_original_prompt_lease_order(monkeypatch):
    fn = _make_fn(
        monkeypatch,
        _args(replay_buffer_type="inflight"),
        _DataSource(),
        lambda *args, **kwargs: None,
    )
    first = _partial_item(1)
    second = _partial_item(2)
    fn._pending_prompts = {1: first.prompt_group, 2: second.prompt_group}
    fn._inflight_replay = deque([second])

    async def completed_first():
        return first.prompt_group, first.generation_group

    task = asyncio.create_task(completed_first())
    await task
    fn._active.add(task)

    await fn._materialize_active_inflight_requests()

    assert [fully_async.prompt_group_id(item.prompt_group) for item in fn._inflight_replay] == [1, 2]


async def test_inflight_capture_preserves_terminal_group_awaiting_reward(monkeypatch):
    fn = _make_fn(
        monkeypatch,
        _args(replay_buffer_type="inflight"),
        _DataSource(),
        lambda *args, **kwargs: None,
    )
    prompt_group = _prompt_group(1)
    generation_group = copy.deepcopy(prompt_group)
    for sample in generation_group:
        sample.response = "complete response"
        sample.response_length = 1
        sample.status = Sample.Status.COMPLETED
        sample.reward = None
    fn._pending_prompts = {1: prompt_group}

    async def completed_without_group_reward():
        return prompt_group, generation_group

    task = asyncio.create_task(completed_without_group_reward())
    await task
    fn._active.add(task)

    await fn._materialize_active_inflight_requests()

    [item] = fn._inflight_replay
    assert fully_async.prompt_group_id(item.prompt_group) == 1
    assert all(sample.status == Sample.Status.COMPLETED for sample in item.generation_group)
    assert all(sample.reward is None for sample in item.generation_group)


def test_inflight_continuation_preserves_completed_sibling_timing():
    completed, partial = _prompt_group(1)
    completed.status = Sample.Status.COMPLETED
    completed.reward = 1.0
    completed.response_length = 1
    completed.metadata = {
        fully_async.SUBMISSION_VERSION_KEY: 3,
        fully_async.TRAJECTORY_START_VERSION_KEY: 3,
        fully_async.TRAJECTORY_START_TIME_KEY: 10.0,
        fully_async.SAMPLE_GENERATION_COMPLETE_VERSION_KEY: 4,
        fully_async.SAMPLE_GENERATION_COMPLETE_TIME_KEY: 12.0,
        fully_async.ATTEMPT_WALL_SECONDS_KEY: 2.0,
        fully_async.REWARD_SECONDS_KEY: 0.25,
    }
    partial.status = Sample.Status.ABORTED
    partial.response_length = 1
    partial.metadata = {
        fully_async.SUBMISSION_VERSION_KEY: 3,
        fully_async.TRAJECTORY_START_VERSION_KEY: 3,
        fully_async.TRAJECTORY_START_TIME_KEY: 10.0,
    }

    preserved = fully_async._prepare_generation_attempt(
        [completed, partial],
        continuation=True,
    )
    fully_async.stamp_attempt_wall_seconds([completed, partial], 5.0)
    fully_async._restore_continuation_metadata([completed, partial], preserved)

    assert completed.metadata[fully_async.ATTEMPT_WALL_SECONDS_KEY] == 2.0
    assert completed.metadata[fully_async.REWARD_SECONDS_KEY] == 0.25
    assert partial.metadata[fully_async.ATTEMPT_WALL_SECONDS_KEY] == 5.0


async def test_failed_inflight_capture_restarts_the_worker(monkeypatch):
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False):
        generation_started.set()
        await release_generation.wait()
        return _complete(group, 5)

    async def fail_abort(_args):
        raise RuntimeError("abort failed")

    monkeypatch.setattr(fully_async, "_abort_inflight_requests", fail_abort)
    fn = _make_fn(
        monkeypatch,
        _args(replay_buffer_type="inflight"),
        _DataSource([_prompt_group(1)]),
        generate,
    )
    fn._ensure_worker()
    await generation_started.wait()

    with pytest.raises(RuntimeError, match="abort failed"):
        await fn.replay_buffer_state(0)

    assert fn._worker is not None
    assert not fn.state.aborted
    release_generation.set()
    await _wait_until(lambda: fn._queue_size() == 1)
    await _stop(fn)


async def test_restore_rejects_a_different_replay_buffer_type(monkeypatch):
    rollout = _make_fn(monkeypatch, _args(), _DataSource(), lambda *args, **kwargs: None)
    state = await rollout.replay_buffer_state(0)
    inflight = _make_fn(
        monkeypatch,
        _args(replay_buffer_type="inflight"),
        _DataSource(),
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="type does not match"):
        await inflight.restore_replay_buffer_state(state)


async def test_replay_buffer_state_requires_opt_in(monkeypatch):
    fn = _make_fn(
        monkeypatch,
        _args(use_replay_buffer=False),
        _DataSource(),
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="disabled"):
        await fn.replay_buffer_state(0)


async def test_regenerated_prompt_does_not_alias_pending_lease(monkeypatch):
    gate = asyncio.Event()
    mutated = asyncio.Event()
    generated_groups = []

    async def generate(state, group, sampling_params, evaluation=False):
        generated_groups.append(group)
        group[0].response = "partial mutation"
        group[0].tokens = [1]
        mutated.set()
        await gate.wait()
        return _complete(group, 5)

    args = _args()
    source = _DataSource([_prompt_group(1)])
    original = _make_fn(monkeypatch, args, source, generate)
    original._ensure_worker()
    await mutated.wait()
    state = await original.replay_buffer_state(0)
    await _stop(original)
    mutated.clear()
    generated_groups.clear()

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, generate)
    await restored.restore_replay_buffer_state(state)
    restored._ensure_worker()
    await mutated.wait()

    lease = restored._pending_prompts[1]
    assert generated_groups[-1][0].response == "partial mutation"
    assert generated_groups[-1] is not lease
    assert lease[0].response == ""
    assert lease[0].tokens == []
    await _stop(restored)


def test_regeneration_prioritizes_active_leases_and_keeps_retry_buffer_order(monkeypatch):
    source = _DataSource()
    source.buffer = [_prompt_group(3), _prompt_group(1)]
    fn = _make_fn(monkeypatch, _args(), source, lambda *args, **kwargs: None)
    fn._pending_prompts = {group_id: _prompt_group(group_id) for group_id in (1, 2, 3, 4)}

    assert fn._regeneration_group_ids(materialized={2}) == [4, 3, 1]


def test_restored_overfull_queue_does_not_release_capacity_early(monkeypatch):
    async def generate(state, group, sampling_params, evaluation=False):
        return _complete(group, 5)

    fn = _make_fn(monkeypatch, _args(), _DataSource(), generate)
    fn._output = asyncio.Queue()
    for group_id in range(fully_async.OUTPUT_QUEUE_MAX_GROUPS + 2):
        prompt = _prompt_group(group_id)
        fn._output.put_nowait((prompt, _complete(prompt, 5)))
    fn._output_slots = asyncio.Semaphore(0)

    fn._output.get_nowait()
    fn._release_output_slot()
    assert fn._output_slots._value == 0
    fn._output.get_nowait()
    fn._release_output_slot()
    assert fn._output_slots._value == 0
    fn._output.get_nowait()
    fn._release_output_slot()
    assert fn._output_slots._value == 1


async def test_restored_overfull_queue_max_does_not_release_capacity_early(monkeypatch):
    fn = _make_fn(
        monkeypatch,
        _queue_args("queue-max"),
        _DataSource(),
        lambda *args, **kwargs: None,
    )
    fn._policy_output = deque(_ready_item(group_id) for group_id in range(fully_async.OUTPUT_QUEUE_MAX_GROUPS + 2))
    fn._output_slots = asyncio.Semaphore(0)

    await fn._take_policy_groups(1)
    assert fn._output_slots._value == 0
    await fn._take_policy_groups(1)
    assert fn._output_slots._value == 0
    await fn._take_policy_groups(1)
    assert fn._output_slots._value == 1


async def test_ready_group_rechecks_staleness_after_resume(monkeypatch):
    gate = asyncio.Event()

    async def first_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 1:
            await gate.wait()
        return _complete(group, 5)

    args = _args(max_weight_staleness=1)
    original = _make_fn(monkeypatch, args, _DataSource([_prompt_group(1), _prompt_group(2)]), first_generate)
    original._ensure_worker()
    await _wait_until(lambda: original._output.qsize() == 1 and len(original._active) == 1)
    state = await original.replay_buffer_state(0)
    await _stop(original)

    async def resumed_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 2:
            await gate.wait()
        return _complete(group, 6)

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, resumed_generate)
    await restored.restore_replay_buffer_state(state)
    restored._weight_version = _Version(6)
    restored.commit_applied_weight_version(6)

    output = await restored(RolloutFnTrainInput(rollout_id=0))
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 1
    assert output.metrics["resume/replay_buffer/ready_groups_restored"] == 1
    await _stop(restored)


async def test_queue_max_ready_group_rechecks_prefill_staleness_after_resume(monkeypatch):
    gate = asyncio.Event()

    async def first_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 1:
            await gate.wait()
        result = _complete(group, 5)
        for sample in result:
            sample.first_prefill_weight_versions = [5]
            sample.min_forward_weight_versions = [5]
            sample.max_forward_weight_versions = [5]
            sample.last_forward_weight_versions = [5]
        return result

    args = _queue_args("queue-max")
    original = _make_fn(monkeypatch, args, _DataSource([_prompt_group(1), _prompt_group(2)]), first_generate)
    original._ensure_worker()
    await _wait_until(lambda: len(original._policy_output) == 1 and len(original._active) == 1)
    state = await original.replay_buffer_state(0)
    await _stop(original)

    async def resumed_generate(state, group, sampling_params, evaluation=False):
        result = _complete(group, 6)
        for sample in result:
            sample.first_prefill_weight_versions = [6]
            sample.min_forward_weight_versions = [6]
            sample.max_forward_weight_versions = [6]
            sample.last_forward_weight_versions = [6]
        return result

    restored = _make_fn(monkeypatch, args, _DataSource(), resumed_generate)
    await restored.restore_replay_buffer_state(state)
    restored._weight_version = _Version(6)
    restored.commit_applied_weight_version(6)

    output = await restored(RolloutFnTrainInput(rollout_id=0))
    assert output.samples[0][0].group_index == 2
    assert output.metrics["rollout/fully_async/stale_groups_dropped"] == 1
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    assert output.metrics["resume/replay_buffer/ready_groups_restored"] == 1
    await _stop(restored)


async def test_partial_drain_keeps_already_admitted_group_after_resume(monkeypatch):
    gate = asyncio.Event()

    async def first_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 1:
            await gate.wait()
        return _complete(group, 5)

    args = _args(rollout_batch_size=2, async_max_concurrent_samples=4, max_weight_staleness=1)
    original = _make_fn(monkeypatch, args, _DataSource([_prompt_group(1), _prompt_group(2)]), first_generate)
    drain = asyncio.create_task(original(RolloutFnTrainInput(rollout_id=0)))
    await _wait_until(lambda: len(original._drain_progress.get(0, fully_async._DrainProgress(0, 0)).data) == 1)
    state = await original.replay_buffer_state(0)
    drain.cancel()
    await asyncio.gather(drain, return_exceptions=True)
    await _stop(original)

    async def resumed_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 2:
            await gate.wait()
        return _complete(group, 5)

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, resumed_generate)
    await restored.restore_replay_buffer_state(state)
    # A partial drain resumes with the model checkpoint that scheduled its fixed
    # train version. Advancing to version 6 here would change T_b halfway through
    # the same batch, which is not a valid resume sequence.
    restored._weight_version = _Version(5)
    restored.commit_applied_weight_version(5)

    output = await restored(RolloutFnTrainInput(rollout_id=0))
    assert {group[0].group_index for group in output.samples} == {1, 2}
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    assert output.metrics["resume/replay_buffer/partial_drains_restored"] == 1
    await _stop(restored)


async def test_queue_handoff_and_finished_active_task_are_not_downgraded_to_prompt_only(monkeypatch):
    async def generate(state, group, sampling_params, evaluation=False):
        return _complete(group, 5)

    fn = _make_fn(monkeypatch, _args(), _DataSource(), generate)
    prompt_1 = _prompt_group(1)
    prompt_2 = _prompt_group(2)
    fn._pending_prompts = {1: _prompt_group(1), 2: _prompt_group(2)}
    fn._output = asyncio.Queue()
    fn._output.put_nowait((prompt_1, _complete(prompt_1, 5)))

    claimed = asyncio.create_task(fn._output.get())
    fn._queue_gets.add(claimed)
    await claimed

    async def finished_item():
        return prompt_2, _complete(prompt_2, 5)

    finished = asyncio.create_task(finished_item())
    await finished
    fn._active.add(finished)

    state = await fn.replay_buffer_state(0)
    assert len(state["ready_items"]) == 2
    assert state["snapshot_counts"]["claimed_groups"] == 1
    assert state["snapshot_counts"]["finished_active_groups"] == 1
    for ready_state in materialize_replay_buffer_state(state)["ready_items"]:
        _, result = fully_async._decode_ready_item(ready_state)
        assert fully_async.group_lifecycle_weight_version(result, fully_async.QUEUE_PUT_VERSION_KEY) == 5


def test_packed_sample_codec_is_lossless_and_restores_independent_samples():
    nan_with_payload = struct.unpack("=d", struct.pack("=Q", 0x7FF8000000001234))[0]
    sample = Sample(
        group_index=1,
        index=2,
        prompt="prompt",
        tokens=[-(2**31), 0, 1, 2**31 - 1],
        response="response",
        response_length=4,
        loss_mask=[-1, 0, 1, 1],
        rollout_log_probs=[-0.0, float("nan"), nan_with_payload, 1.25],
        teacher_log_probs=[2.5, -3.5, 4.5, 5.5],
        opd_reverse_kl=[0.25, 0.5, 0.75, 1.0],
        metadata={"nested": {"value": [1]}},
    )
    sample.rollout_routed_experts = np.arange(9, dtype=np.int32).reshape(3, 3)
    sample.dynamic_attribute = {"preserved": True}
    sample.validate()

    encoder = ReplayBufferSampleEncoder()
    group_state = encoder.encode_group([sample, [sample]])
    state = _codec_state(group_state, encoder.finish())
    materialized = materialize_replay_buffer_state(state)
    flat_state, nested_states = materialized["pending_prompts"][0]

    assert len(state[SAMPLE_CODEC_STATE_KEY]["records"]) == 1
    assert flat_state is not nested_states[0]
    assert flat_state["tokens"] == sample.tokens
    assert flat_state["loss_mask"] == sample.loss_mask
    assert (
        np.asarray(flat_state["rollout_log_probs"], dtype=np.float64).tobytes()
        == np.asarray(sample.rollout_log_probs, dtype=np.float64).tobytes()
    )
    assert flat_state["teacher_log_probs"] == sample.teacher_log_probs
    assert flat_state["opd_reverse_kl"] == sample.opd_reverse_kl
    assert np.array_equal(flat_state["rollout_routed_experts"], sample.rollout_routed_experts)
    assert flat_state["rollout_routed_experts"] is not sample.rollout_routed_experts
    assert flat_state["dynamic_attribute"] == sample.dynamic_attribute

    decoded_flat, decoded_nested = decode_group(materialized["pending_prompts"][0])
    decoded_nested = decoded_nested[0]
    decoded_flat.tokens.append(7)
    decoded_flat.metadata["nested"]["value"].append(2)
    assert decoded_nested.tokens == sample.tokens
    assert decoded_nested.metadata == sample.metadata


def test_packed_sample_codec_falls_back_without_changing_unsupported_element_types():
    sample = Sample(tokens=[2**40], loss_mask=[True], rollout_log_probs=[np.float32(1.5)])
    encoder = ReplayBufferSampleEncoder()
    group_state = encoder.encode_group([sample])
    materialized = materialize_replay_buffer_state(_codec_state(group_state, encoder.finish()))
    [sample_state] = materialized["pending_prompts"][0]

    assert sample_state["tokens"] == [2**40]
    assert type(sample_state["tokens"][0]) is int
    assert sample_state["loss_mask"] == [True]
    assert type(sample_state["loss_mask"][0]) is bool
    assert type(sample_state["rollout_log_probs"][0]) is np.float32


def test_packed_sample_codec_applies_queue_overlay_without_mutating_or_aliasing_source():
    sample = Sample(group_index=1, index=2, metadata={"source": 1})
    encoder = ReplayBufferSampleEncoder()
    prompt_state = encoder.encode_group([sample])
    result_state = encoder.encode_group([sample], metadata_updates={fully_async.QUEUE_PUT_VERSION_KEY: 9})
    state = _codec_state(prompt_state, encoder.finish())
    state["ready_items"] = [{"prompt_group": prompt_state, "result": result_state}]
    materialized = materialize_replay_buffer_state(state)
    [prompt] = materialized["ready_items"][0]["prompt_group"]
    [result] = materialized["ready_items"][0]["result"]

    assert len(state[SAMPLE_CODEC_STATE_KEY]["records"]) == 2
    assert fully_async.QUEUE_PUT_VERSION_KEY not in sample.metadata
    assert fully_async.QUEUE_PUT_VERSION_KEY not in prompt["metadata"]
    assert result["metadata"][fully_async.QUEUE_PUT_VERSION_KEY] == 9
    assert prompt is not result


def test_prepacked_sample_cache_is_lossless_and_reassigned_fields_fall_back():
    sample = Sample(
        tokens=[1, 2, 3],
        response="first line\n雪🙂",
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.0, -1.25, float("nan")],
    )
    cache = ReplayBufferPackedFieldCache()
    cache.cache_group([sample])
    assert cache.stats()["live_packed_bytes"] == 3 * (4 + 1 + 8) + len(sample.response.encode("utf-8"))

    sample.tokens = [4, 5, 6, 7]
    sample.loss_mask = []
    sample.teacher_log_probs = (-2.0, -3.0)
    encoder = ReplayBufferSampleEncoder(cache)
    group_state = encoder.encode_group([sample], use_packed_cache=True)
    materialized = materialize_replay_buffer_state(_codec_state(group_state, encoder.finish()))
    [sample_state] = materialized["pending_prompts"][0]

    assert sample_state["tokens"] == sample.tokens
    assert sample_state["response"] == sample.response
    assert sample_state["loss_mask"] == sample.loss_mask
    assert sample_state["teacher_log_probs"] == sample.teacher_log_probs
    assert (
        np.asarray(sample_state["rollout_log_probs"], dtype=np.float64).tobytes()
        == np.asarray(sample.rollout_log_probs, dtype=np.float64).tobytes()
    )


def test_prepacked_group_uses_one_tensor_and_current_lifecycle_metadata():
    samples = [
        Sample(tokens=[1, 2], rollout_log_probs=[-1.0, -2.0], metadata={"version": 1}),
        Sample(tokens=[3, 4, 5], rollout_log_probs=[-3.0, -4.0, -5.0], metadata={"version": 1}),
    ]
    cache = ReplayBufferPackedFieldCache()
    cache.cache_group(samples)
    samples[0].metadata["version"] = 2

    encoder = ReplayBufferSampleEncoder(cache)
    group_state = encoder.encode_group(samples, use_packed_cache=True)
    codec = encoder.finish()

    assert len(codec["arrays"]["tokens"]) == 1
    assert codec["arrays"]["tokens"][0].tolist() == [1, 2, 3, 4, 5]
    materialized = materialize_replay_buffer_state(_codec_state(group_state, codec))
    first, second = materialized["pending_prompts"][0]
    assert first["tokens"] == samples[0].tokens
    assert second["tokens"] == samples[1].tokens
    assert first["metadata"] == {"version": 2}
    assert second["metadata"] == {"version": 1}


async def test_replay_buffer_capture_restores_gc_after_failure(monkeypatch):
    fn = _make_fn(monkeypatch, _args(), _DataSource(), lambda *args, **kwargs: None)

    def fail(_rollout_id):
        assert not gc.isenabled()
        raise RuntimeError("capture failed")

    monkeypatch.setattr(fn, "_capture_replay_buffer_state", fail)
    was_enabled = gc.isenabled()
    with pytest.raises(RuntimeError, match="capture failed"):
        await fn.replay_buffer_state(1)
    assert gc.isenabled() == was_enabled


def test_packed_sample_codec_spans_multiple_tensor_shards(monkeypatch):
    monkeypatch.setattr(codec_module, "ARRAY_SHARD_BYTES", 16)
    sample = Sample(tokens=list(range(11)), rollout_log_probs=[-float(index) for index in range(11)])
    encoder = ReplayBufferSampleEncoder()
    group_state = encoder.encode_group([sample])
    codec = encoder.finish()

    assert [tensor.numel() for tensor in codec["arrays"]["tokens"]] == [4, 4, 3]
    assert [tensor.numel() for tensor in codec["arrays"]["rollout_log_probs"]] == [2, 2, 2, 2, 2, 1]
    materialized = materialize_replay_buffer_state(_codec_state(group_state, codec))
    [sample_state] = materialized["pending_prompts"][0]
    assert sample_state["tokens"] == sample.tokens
    assert sample_state["rollout_log_probs"] == sample.rollout_log_probs


def _codec_state(group_state, codec):
    return {
        "dataset_fingerprint": "dataset-a",
        "data_source": {},
        "applied_weight_version": 0,
        "pending_prompts": [group_state],
        "ready_items": [],
        "drain_progress": [],
        "prepared_batches": [],
        "regeneration_group_ids": [],
        SAMPLE_CODEC_STATE_KEY: codec,
    }


def test_replay_buffer_checksum_fingerprint_and_retention(tmp_path: Path):
    state = {
        "dataset_fingerprint": "dataset-a",
        "data_source": {},
        "applied_weight_version": 0,
        "pending_prompts": [],
        "ready_items": [],
        "drain_progress": [],
        "prepared_batches": [],
        "regeneration_group_ids": [],
        "value": [1, 2, 3],
    }
    with pytest.raises(FileNotFoundError, match="requires replay-buffer state"):
        load_replay_buffer(tmp_path, 3, expected_fingerprint="dataset-a")

    path, size = save_replay_buffer(tmp_path, 4, state)
    assert path.stat().st_size == size
    assert path.stat().st_mode & 0o777 == 0o644
    assert Path(f"{path}.sha256.json").stat().st_mode & 0o777 == 0o644
    manifest = json.loads(Path(f"{path}.sha256.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_replay_buffer(tmp_path, 4, expected_fingerprint="dataset-a")
    assert loaded["value"] == [1, 2, 3]
    assert loaded["rollout_id"] == 4
    assert "checkpoint_rollout_id" not in loaded
    with pytest.raises(RuntimeError, match="resume with --use-replay-buffer"):
        ensure_no_replay_buffer(tmp_path, 4)

    with pytest.raises(RuntimeError, match="fingerprint"):
        load_replay_buffer(tmp_path, 4, expected_fingerprint="dataset-b")

    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_replay_buffer(tmp_path, 4, expected_fingerprint="dataset-a")

    for rollout_id in range(5, 9):
        save_replay_buffer(tmp_path, rollout_id, {**state, "value": rollout_id})
    prune_replay_buffers(tmp_path, current_rollout_id=8, keep_last=2, archive_interval=5)
    assert (tmp_path / "rollout" / "replay_buffer_4.pt").is_file()
    assert not (tmp_path / "rollout" / "replay_buffer_5.pt").exists()
    assert not (tmp_path / "rollout" / "replay_buffer_6.pt").exists()
    assert (tmp_path / "rollout" / "replay_buffer_7.pt").is_file()
    assert (tmp_path / "rollout" / "replay_buffer_8.pt").is_file()


def test_rollout_replay_buffer_rejects_inflight_payload(tmp_path: Path):
    state = {
        "dataset_fingerprint": "dataset-a",
        "data_source": {},
        "applied_weight_version": 0,
        "pending_prompts": [],
        "ready_items": [],
        "drain_progress": [],
        "prepared_batches": [],
        "regeneration_group_ids": [],
        "replay_buffer_type": "rollout",
        "inflight_items": [{}],
    }

    with pytest.raises(RuntimeError, match="cannot contain inflight"):
        save_replay_buffer(tmp_path, 0, state)


def test_replay_buffer_external_tensor_shards_are_verified_and_pruned(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(codec_module, "ARRAY_SHARD_BYTES", 16)
    monkeypatch.setattr(replay_buffer_module, "TENSOR_PART_BYTES", 16)
    sample = Sample(
        tokens=list(range(20)),
        response="external response 雪🙂" * 20,
        rollout_log_probs=[-float(index) for index in range(20)],
    )
    encoder = ReplayBufferSampleEncoder()
    group_state = encoder.encode_group([sample])
    state = _codec_state(group_state, encoder.finish())

    path, size = save_replay_buffer(tmp_path, 12, state)
    checksum_path = Path(f"{path}.sha256.json")
    manifest = json.loads(checksum_path.read_text(encoding="utf-8"))
    parts_path = path.parent / manifest["parts_directory"]
    assert parts_path.stat().st_mode & 0o777 == 0o755
    assert len(manifest["parts"]) > 1
    assert size == path.stat().st_size + sum(part["size"] for part in manifest["parts"])
    for part in manifest["parts"]:
        part_path = parts_path / part["file"]
        assert part_path.stat().st_mode & 0o777 == 0o644

    loaded = load_replay_buffer(tmp_path, 12, expected_fingerprint="dataset-a")
    [sample_state] = loaded["pending_prompts"][0]
    assert sample_state["tokens"] == sample.tokens
    assert sample_state["response"] == sample.response
    assert sample_state["rollout_log_probs"] == sample.rollout_log_probs

    first_part = parts_path / manifest["parts"][0]["file"]
    first_part.write_bytes(first_part.read_bytes() + b"corruption")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_replay_buffer(tmp_path, 12, expected_fingerprint="dataset-a")

    prune_replay_buffers(tmp_path, current_rollout_id=12, keep_last=1, archive_interval=None)
    save_replay_buffer(tmp_path, 13, state)
    prune_replay_buffers(tmp_path, current_rollout_id=13, keep_last=1, archive_interval=None)
    assert not path.exists()
    assert not checksum_path.exists()
    assert not parts_path.exists()


def test_replay_buffer_preserves_empty_packable_lists(tmp_path: Path):
    sample = Sample(tokens=[], loss_mask=[], rollout_log_probs=[])
    encoder = ReplayBufferSampleEncoder()
    group_state = encoder.encode_group([sample])
    state = _codec_state(group_state, encoder.finish())

    save_replay_buffer(tmp_path, 13, state)
    loaded = load_replay_buffer(tmp_path, 13, expected_fingerprint="dataset-a")
    [sample_state] = loaded["pending_prompts"][0]

    assert sample_state["tokens"] == []
    assert sample_state["loss_mask"] == []
    assert sample_state["rollout_log_probs"] == []


def test_replay_buffer_tensor_write_failure_publishes_nothing(tmp_path: Path, monkeypatch):
    sample = Sample(tokens=[1, 2, 3], rollout_log_probs=[-1.0, -2.0, -3.0])
    encoder = ReplayBufferSampleEncoder()
    group_state = encoder.encode_group([sample])
    state = _codec_state(group_state, encoder.finish())

    def fail(*_args, **_kwargs):
        raise OSError("injected part failure")

    monkeypatch.setattr(replay_buffer_module, "_write_tensor_parts", fail)
    with pytest.raises(OSError, match="injected part failure"):
        save_replay_buffer(tmp_path, 14, state)

    rollout_dir = tmp_path / "rollout"
    assert list(rollout_dir.iterdir()) == []


def test_replay_buffer_manifest_write_failure_removes_published_tensor_parts(tmp_path: Path, monkeypatch):
    sample = Sample(tokens=[1, 2, 3], response="answer", rollout_log_probs=[-1.0, -2.0, -3.0])
    encoder = ReplayBufferSampleEncoder()
    group_state = encoder.encode_group([sample])
    state = _codec_state(group_state, encoder.finish())

    def fail(*_args, **_kwargs):
        raise OSError("injected manifest failure")

    monkeypatch.setattr(replay_buffer_module, "_write_torch_file", fail)
    with pytest.raises(OSError, match="injected manifest failure"):
        save_replay_buffer(tmp_path, 15, state)

    assert list((tmp_path / "rollout").iterdir()) == []


def test_save_replay_buffer_does_not_reread_payload_for_checksum(tmp_path: Path, monkeypatch):
    state = _codec_state([], ReplayBufferSampleEncoder().finish())

    def fail_if_called(path):
        raise AssertionError(f"save reread payload at {path}")

    monkeypatch.setattr(replay_buffer_module, "_file_digest", fail_if_called)
    path, size = save_replay_buffer(tmp_path, 2, state)
    assert path.stat().st_size == size


def test_load_replay_buffer_does_not_search_old_filename(tmp_path: Path):
    state = {
        **_codec_state([], ReplayBufferSampleEncoder().finish()),
        "schema_version": 1,
        "checkpoint_rollout_id": 7,
    }
    state.pop(SAMPLE_CODEC_STATE_KEY)
    path = tmp_path / "rollout" / "fully_async_state_7.pt"
    path.parent.mkdir(parents=True)
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256.json").write_text(
        json.dumps({"schema_version": 1, "sha256": digest, "size": path.stat().st_size}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="replay_buffer_7.pt"):
        load_replay_buffer(tmp_path, 7, expected_fingerprint="dataset-a")


def test_load_replay_buffer_rejects_old_schema(tmp_path: Path):
    state = {
        **_codec_state([], ReplayBufferSampleEncoder().finish()),
        "schema_version": 2,
        "checkpoint_rollout_id": 8,
    }
    path = replay_buffer_path(tmp_path, 8)
    path.parent.mkdir(parents=True)
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256.json").write_text(
        json.dumps({"schema_version": 2, "sha256": digest, "size": path.stat().st_size}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Unsupported replay-buffer manifest"):
        load_replay_buffer(tmp_path, 8, expected_fingerprint="dataset-a")


def test_dataset_fingerprint_includes_model_tokenizer_and_chat_template(tmp_path: Path):
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text("template-a", encoding="utf-8")
    source = Namespace(dataset=[object()])
    args = Namespace(
        prompt_data=str(prompt_path),
        hf_checkpoint="/models/model-a",
        tokenizer_model="/models/tokenizer-a",
        tokenizer_type="HuggingFaceTokenizer",
        chat_template_path=str(template_path),
    )

    initial = dataset_fingerprint(args, source)
    template_path.write_text("template-b", encoding="utf-8")
    assert dataset_fingerprint(args, source) != initial

    template_path.write_text("template-a", encoding="utf-8")
    args.hf_checkpoint = "/models/model-b"
    assert dataset_fingerprint(args, source) != initial

    args.hf_checkpoint = "/models/model-a"
    args.tokenizer_model = "/models/tokenizer-b"
    assert dataset_fingerprint(args, source) != initial

    args.tokenizer_model = "/models/tokenizer-a"
    args.rollout_stop = ["<END>"]
    assert dataset_fingerprint(args, source) != initial

    del args.rollout_stop
    args.search_r1_format_score = 0.2
    assert dataset_fingerprint(args, source) != initial


@pytest.mark.parametrize(
    ("argument", "changed_value"),
    (
        ("tau_max_turns", 12),
        ("tau_max_steps", 80),
        ("tau_user_provider", "gemini"),
        ("tau_user_model", "gemini-2.5-flash-lite"),
        ("tau_user_max_tokens", 256),
        ("tau_user_temperature", 0.5),
        ("tau_user_top_p", 0.9),
        ("tau_user_request_timeout", 30.0),
        ("tau_user_max_retries", 2),
        ("tau_tool_call_parser", "qwen25"),
        ("tau_overlap_db_restore_with_prefill", True),
    ),
)
def test_dataset_fingerprint_includes_tau_environment_configuration(
    tmp_path: Path,
    argument: str,
    changed_value,
):
    prompt_path = tmp_path / "tau-prompts.jsonl"
    prompt_path.write_text('{"prompt": "tau"}\n', encoding="utf-8")
    source = Namespace(dataset=[object()])
    args = Namespace(prompt_data=str(prompt_path))
    initial = dataset_fingerprint(args, source)

    setattr(args, argument, changed_value)

    assert dataset_fingerprint(args, source) != initial


def test_replay_buffer_retention_counts_existing_sparse_buffers(tmp_path: Path):
    state = {
        "dataset_fingerprint": "dataset-a",
        "data_source": {},
        "applied_weight_version": 0,
        "pending_prompts": [],
        "ready_items": [],
        "drain_progress": [],
        "prepared_batches": [],
        "regeneration_group_ids": [],
    }
    for rollout_id in (9, 19, 29):
        save_replay_buffer(tmp_path, rollout_id, state)

    prune_replay_buffers(tmp_path, current_rollout_id=29, keep_last=2, archive_interval=100)

    assert not (tmp_path / "rollout" / "replay_buffer_9.pt").exists()
    assert (tmp_path / "rollout" / "replay_buffer_19.pt").is_file()
    assert (tmp_path / "rollout" / "replay_buffer_29.pt").is_file()
