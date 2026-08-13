from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import asyncio
import copy
from argparse import Namespace
from collections import deque
from pathlib import Path

import pytest

import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.fully_async_checkpoint import (
    dataset_fingerprint,
    ensure_no_full_replay_sidecar,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from miles.utils.types import Sample


class _GenerateState:
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}


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
        "fully_async_rollout_checkpoint": True,
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


async def test_prepared_batch_is_a_warm_resume_hit_and_active_prompt_is_regenerated(monkeypatch):
    gate = asyncio.Event()

    async def generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 1:
            await gate.wait()
        return _complete(group, 5)

    args = _args()
    source = _DataSource([_prompt_group(1), _prompt_group(2)])
    original = _make_fn(monkeypatch, args, source, generate)
    output = await original(RolloutFnTrainInput(rollout_id=0))
    await _wait_until(lambda: any(group_id == 2 for group_id in original._pending_prompts))
    state = await original.checkpoint_state(0)
    await _stop(original)

    invalid = _make_fn(monkeypatch, args, _DataSource(), generate)
    invalid_state = copy.deepcopy(state)
    invalid_state["prepared_batches"][0]["group_ids"] = [999]
    with pytest.raises(RuntimeError, match="does not match stored group"):
        await invalid.restore_checkpoint_state(invalid_state)

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, generate)
    await restored.restore_checkpoint_state(state)

    warm_output = await restored(RolloutFnTrainInput(rollout_id=0))
    assert warm_output.samples[0][0].index == output.samples[0][0].index
    assert warm_output.metrics["resume/fully_async/warm_prepared_batch_hit"] == 1
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
    state = await original.checkpoint_state(0)
    await _stop(original)
    mutated.clear()
    generated_groups.clear()

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, generate)
    await restored.restore_checkpoint_state(state)
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


async def test_ready_group_rechecks_staleness_after_resume(monkeypatch):
    gate = asyncio.Event()

    async def first_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 1:
            await gate.wait()
        return _complete(group, 5)

    args = _args(max_weight_staleness=0)
    original = _make_fn(monkeypatch, args, _DataSource([_prompt_group(1), _prompt_group(2)]), first_generate)
    original._ensure_worker()
    await _wait_until(lambda: original._output.qsize() == 1 and len(original._active) == 1)
    state = await original.checkpoint_state(0)
    await _stop(original)

    async def resumed_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 2:
            await gate.wait()
        return _complete(group, 6)

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, resumed_generate)
    await restored.restore_checkpoint_state(state)
    restored._weight_version = _Version(6)
    restored.commit_applied_weight_version(6)

    output = await restored(RolloutFnTrainInput(rollout_id=0))
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 1
    assert output.metrics["resume/fully_async/ready_groups_restored"] == 1
    await _stop(restored)


async def test_partial_drain_keeps_already_admitted_group_after_resume(monkeypatch):
    gate = asyncio.Event()

    async def first_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 1:
            await gate.wait()
        return _complete(group, 5)

    args = _args(rollout_batch_size=2, async_max_concurrent_samples=4, max_weight_staleness=0)
    original = _make_fn(monkeypatch, args, _DataSource([_prompt_group(1), _prompt_group(2)]), first_generate)
    drain = asyncio.create_task(original(RolloutFnTrainInput(rollout_id=0)))
    await _wait_until(lambda: len(original._drain_progress.get(0, fully_async._DrainProgress(0)).data) == 1)
    state = await original.checkpoint_state(0)
    drain.cancel()
    await asyncio.gather(drain, return_exceptions=True)
    await _stop(original)

    async def resumed_generate(state, group, sampling_params, evaluation=False):
        if group[0].group_index != 2:
            await gate.wait()
        return _complete(group, 6)

    restored_source = _DataSource()
    restored = _make_fn(monkeypatch, args, restored_source, resumed_generate)
    await restored.restore_checkpoint_state(state)
    restored._weight_version = _Version(6)
    restored.commit_applied_weight_version(6)

    output = await restored(RolloutFnTrainInput(rollout_id=0))
    assert {group[0].group_index for group in output.samples} == {1, 2}
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    assert output.metrics["resume/fully_async/partial_drains_restored"] == 1
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

    state = await fn.checkpoint_state(0)
    assert len(state["ready_items"]) == 2
    assert state["snapshot_counts"]["claimed_groups"] == 1
    assert state["snapshot_counts"]["finished_active_groups"] == 1
    for ready_state in state["ready_items"]:
        _, result = fully_async._decode_ready_item(ready_state)
        assert fully_async.group_lifecycle_weight_version(result, fully_async.QUEUE_PUT_VERSION_KEY) == 5


def test_checkpoint_checksum_fingerprint_and_retention(tmp_path: Path):
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
    with pytest.raises(FileNotFoundError, match="requires fully-async rollout state"):
        load_checkpoint(tmp_path, 3, expected_fingerprint="dataset-a")

    path, size = save_checkpoint(tmp_path, 4, state)
    assert path.stat().st_size == size
    assert path.stat().st_mode & 0o777 == 0o644
    assert Path(f"{path}.sha256.json").stat().st_mode & 0o777 == 0o644
    assert load_checkpoint(tmp_path, 4, expected_fingerprint="dataset-a")["value"] == [1, 2, 3]
    with pytest.raises(RuntimeError, match="resume with --fully-async-rollout-checkpoint"):
        ensure_no_full_replay_sidecar(tmp_path, 4)

    with pytest.raises(RuntimeError, match="fingerprint"):
        load_checkpoint(tmp_path, 4, expected_fingerprint="dataset-b")

    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_checkpoint(tmp_path, 4, expected_fingerprint="dataset-a")

    for rollout_id in range(5, 9):
        save_checkpoint(tmp_path, rollout_id, {**state, "value": rollout_id})
    prune_checkpoints(tmp_path, current_rollout_id=8, keep_last=2, archive_interval=5)
    assert (tmp_path / "rollout" / "fully_async_state_4.pt").is_file()
    assert not (tmp_path / "rollout" / "fully_async_state_5.pt").exists()
    assert not (tmp_path / "rollout" / "fully_async_state_6.pt").exists()
    assert (tmp_path / "rollout" / "fully_async_state_7.pt").is_file()
    assert (tmp_path / "rollout" / "fully_async_state_8.pt").is_file()


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


def test_checkpoint_retention_counts_existing_sparse_sidecars(tmp_path: Path):
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
        save_checkpoint(tmp_path, rollout_id, state)

    prune_checkpoints(tmp_path, current_rollout_id=29, keep_last=2, archive_interval=100)

    assert not (tmp_path / "rollout" / "fully_async_state_9.pt").exists()
    assert (tmp_path / "rollout" / "fully_async_state_19.pt").is_file()
    assert (tmp_path / "rollout" / "fully_async_state_29.pt").is_file()
