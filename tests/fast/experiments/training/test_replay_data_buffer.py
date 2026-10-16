import asyncio
import json
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest
import torch

import miles.rollout.fully_async_data_buffer as data_buffer_module
import miles.rollout.replay_data_buffer_checkpoint as persistence
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.rollout.replay_data_buffer import ReplayDataBuffer, _configuration_fingerprint, _snapshot_entries
from miles.rollout.replay_data_buffer_checkpoint import (
    commit_replay_data_buffer_checkpoint,
    iter_replay_data_buffer_parts,
    load_replay_data_buffer_auxiliary,
    load_replay_data_buffer_header,
    load_replay_data_buffer_source_state,
    resume_weight_version,
    save_replay_data_buffer_checkpoint,
)
from miles.utils.async_buffer_arguments import add_buffer_arguments, validate_buffer_arguments
from miles.utils.types import RewardSpec, Sample, WeightVersionSpan, WeightVersionsPerCall


def _entry(i, version=1):
    sample = Sample(
        index=i,
        group_index=i,
        prompt=f"p{i}",
        tokens=[1, 2, 3],
        response_length=2,
        response="answer",
        reward=0.5,
        status=Sample.Status.COMPLETED,
        loss_mask=[1, 0],
        rollout_log_probs=[-1.0, -2.0],
        metadata={"category": "math"},
        reward_spec=RewardSpec(custom_rm_path="example.reward"),
        weight_versions=[WeightVersionsPerCall([WeightVersionSpan(str(version), 1, 3)])],
    )
    return DataBufferInput([Sample(index=i, group_index=i, prompt=f"p{i}")], [sample])


def _args(tmp_path, **overrides):
    settings = dict(
        load=str(tmp_path / "old"),
        save=str(tmp_path / "new"),
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        async_data_buffer_capacity_factor=1.0,
        dynamic_sampling_filter_path=None,
        async_data_buffer_order="fifo",
        max_weight_staleness=None,
        reward_key=None,
        use_replay_data_buffer=True,
        replay_data_buffer_async_restore=True,
    )
    settings.update(overrides)
    return Namespace(**settings)


def _save(root, entries, *, fingerprint="configuration", iteration=0):
    directory = save_replay_data_buffer_checkpoint(
        root,
        iteration,
        entries=entries,
        pending_prompts=[[Sample(index=99, group_index=99, prompt="unfinished")]],
        source_state={
            "sample_offset": 8,
            "sample_index": 8,
            "metadata": {"epoch_note": "kept"},
            "prompt_fingerprint": "dataset",
        },
        served_version=5,
        configuration_fingerprint=fingerprint,
        groups_per_part=2,
    )
    commit_replay_data_buffer_checkpoint(root, iteration)
    return directory


def test_iteration_zero_roundtrip_real_tokens_masks_reward_versions_and_auxiliary(tmp_path):
    directory = _save(tmp_path, [_entry(0), _entry(1)])
    assert directory == tmp_path / "rollout/replay_0000000"
    header = load_replay_data_buffer_header(tmp_path, 0)
    restored = list(iter_replay_data_buffer_parts(tmp_path, 0, header))[0]
    assert restored[0].group[0].to_dict() == _entry(0).group[0].to_dict()
    assert restored[0].group[0].reward_spec.custom_rm_path == "example.reward"
    assert load_replay_data_buffer_source_state(tmp_path, 0, header)["metadata"] == {"epoch_note": "kept"}
    assert load_replay_data_buffer_auxiliary(tmp_path, 0, header)["pending_prompts"][0][0].index == 99
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("0\n")
    torch.save({}, tmp_path / "rollout/global_dataset_state_dict_0.pt")
    assert resume_weight_version(Namespace(load=str(tmp_path), use_replay_data_buffer=True, ckpt_step=0)) == 5
    with pytest.raises(FileExistsError):
        _save(tmp_path, [_entry(0)])
    marker = json.loads((directory / "COMMITTED.json").read_text())
    marker["rollout_id"] = 1
    (directory / "COMMITTED.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="mismatch"):
        load_replay_data_buffer_header(tmp_path, 0)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_cli_names_and_legacy_aliases_resume_schema_one(tmp_path, legacy, asynchronous):
    _save(tmp_path, [_entry(0)])
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("0\n")
    parser = ArgumentParser()
    add_buffer_arguments(parser)
    stem = "replay-buffer" if legacy else "replay-data-buffer"
    command = [f"--use-{stem}"]
    if asynchronous:
        command.append(f"--{stem}-async-restore")
    args = parser.parse_args(command)
    args.load = str(tmp_path)
    assert args.use_replay_data_buffer is True
    assert args.replay_data_buffer_async_restore is asynchronous
    assert resume_weight_version(args) == 5


def test_replay_data_buffer_cli_defaults_remain_disabled():
    parser = ArgumentParser()
    add_buffer_arguments(parser)
    args = parser.parse_args([])
    assert args.use_replay_data_buffer is False
    assert args.replay_data_buffer_async_restore is False


@pytest.mark.parametrize("backend,multi_policy", [("kubernetes", False), ("ray", True)])
def test_replay_rejects_unsupported_orchestration(backend, multi_policy):
    from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

    args = Namespace(
        use_replay_data_buffer=True,
        cluster_backend=backend,
        megatron_config=encode_megatron_config("solver", "verifier") if multi_policy else None,
    )
    with pytest.raises(ValueError, match="single-policy" if multi_policy else "only Ray"):
        validate_buffer_arguments(args)


def test_first_part_is_available_before_corrupt_tail_fails(tmp_path):
    directory = _save(tmp_path, [_entry(i) for i in range(20)])
    header = load_replay_data_buffer_header(tmp_path, 0)
    (directory / "part_00000001.pt").write_bytes(b"corrupt")
    parts = iter_replay_data_buffer_parts(tmp_path, 0, header)
    assert len(next(parts)) == 2
    with pytest.raises(ValueError, match="checksum"):
        next(parts)


def test_resume_uses_committed_header_without_legacy_cursor_and_rejects_partial_training_save(tmp_path):
    _save(tmp_path, [_entry(0)])
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("0\n")
    args = Namespace(load=str(tmp_path), use_replay_data_buffer=True, ckpt_step=None)
    assert resume_weight_version(args) == 5
    tracker.write_text("1\n")
    with pytest.raises(ValueError, match="no committed"):
        resume_weight_version(args)


def test_converted_checkpoint_is_fresh_without_rollout_history(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("0\n")
    assert resume_weight_version(Namespace(load=str(tmp_path), use_replay_data_buffer=True, ckpt_step=None)) == 0


@pytest.mark.parametrize("order,expected", [("fifo", [3, 1, 2]), ("oldest-policy-first", [1, 2, 3])])
async def test_buffer_order_is_stable(tmp_path, order, expected):
    args = _args(tmp_path, async_data_buffer_order=order, async_data_buffer_capacity_factor=3)
    buffer = DefaultDataBuffer(DataBufferConstructorInput(args, lambda group: None))
    for version in (3, 1, 2):
        await buffer.put(_entry(version, version))
    assert [item.group[0].index for item in buffer.snapshot_entries()] == expected
    assert [(await buffer.get(current_version=5)).group[0].index for _ in range(3)] == expected


async def test_priority_snapshot_includes_older_producer_blocked_on_capacity(tmp_path):
    args = _args(tmp_path, async_data_buffer_order="oldest-policy-first")
    owner = _owner(args)
    await owner._output.put(_entry(3, version=3))
    await owner._output.put(_entry(4, version=4))
    owner._pending_ready = {object(): _entry(1, version=1)}
    assert [entry.group[0].index for entry in _snapshot_entries(owner)] == [1, 3, 4]


@pytest.mark.parametrize("lag,message", [(None, "Missing provenance"), (float("nan"), "finite"), (-1, "non-negative")])
async def test_custom_staleness_missing_or_invalid_does_not_admit_bounded_groups(tmp_path, monkeypatch, lag, message):
    args = _args(tmp_path, custom_staleness_function_path="example.clock", max_weight_staleness=3)
    monkeypatch.setattr(data_buffer_module, "load_function", lambda path: (lambda *a: lag) if path else None)
    buffer = DefaultDataBuffer(DataBufferConstructorInput(args, lambda group: None))
    await buffer.put(_entry(0))
    with pytest.raises(ValueError, match=message):
        await buffer.get(current_version=5)


def _owner(args):
    class Source:
        def __init__(self):
            self.buffer = []

        def state_dict(self):
            return {"sample_offset": 8, "metadata": {}, "prompt_fingerprint": "dataset"}

        def load_state_dict(self, value):
            self.state = value

        def add_samples(self, groups):
            self.buffer.extend(groups)

        def save(self, rollout_id):
            path = persistence.replay_data_buffer_directory(args.save, rollout_id).parent
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / f"global_dataset_state_dict_{rollout_id}.pt")

    buffer = DefaultDataBuffer(DataBufferConstructorInput(args, lambda group: None))
    return SimpleNamespace(
        args=args,
        data_source=Source(),
        _ensure_buffer=lambda: None,
        _output=buffer,
        _weight_version=6,
        _draining=False,
        _producer_resumed=asyncio.Event(),
        _pending_ready={},
        _active={},
        _original_prompts={},
    )


async def test_async_restore_first_batch_before_tail_and_checkpoint_without_deadlock(tmp_path, monkeypatch):
    args = _args(tmp_path)
    owner = _owner(args)
    replay = ReplayDataBuffer(owner)
    _save(args.load, [_entry(i) for i in range(12)], fingerprint=_configuration_fingerprint(args))
    header = load_replay_data_buffer_header(args.load, 0)
    await replay._start_restore(0, header)
    assert len(owner._output) == 2 and not replay._restore_task.done()
    first = [(await owner._output.get(current_version=6)).group[0].index for _ in range(2)]
    assert first == [0, 1]
    await asyncio.wait_for(replay._save(1), timeout=5)
    restored_header = load_replay_data_buffer_header(args.save, 1)
    remaining = [
        entry.group[0].index for part in iter_replay_data_buffer_parts(args.save, 1, restored_header) for entry in part
    ]
    assert remaining == list(range(2, 12))
    assert owner.data_source.buffer[0][0].index == 99


async def test_async_restore_corruption_surfaces_after_first_batch(tmp_path):
    args = _args(tmp_path)
    owner = _owner(args)
    replay = ReplayDataBuffer(owner)
    directory = _save(args.load, [_entry(i) for i in range(4)])
    header = load_replay_data_buffer_header(args.load, 0)
    (directory / "part_00000001.pt").write_bytes(b"bad-tail")
    await replay._start_restore(0, header)
    for _ in range(2):
        await owner._output.get(current_version=6)
    with pytest.raises(ValueError, match="checksum"):
        await replay.before_production()
    with pytest.raises(ValueError, match="checksum"):
        await replay._save(1)
    assert not persistence.replay_data_buffer_directory(args.save, 1).exists()


def test_schema_one_remains_readable_and_inflight_mode_mismatch_fails(tmp_path):
    args = _args(tmp_path)
    directory = _save(args.load, [_entry(0)], fingerprint=_configuration_fingerprint(args))
    header_path = directory / "header.json"
    header = json.loads(header_path.read_text())
    header["schema"] = 1
    header.pop("mode")
    header.pop("inflight_count")
    header_path.write_text(json.dumps(header))
    commit_replay_data_buffer_checkpoint(args.load, 0)
    legacy = load_replay_data_buffer_header(args.load, 0)
    assert legacy["schema"] == 1
    assert len(next(iter_replay_data_buffer_parts(args.load, 0, legacy))) == 1
    replay = ReplayDataBuffer(_owner(args))
    replay.mode = "inflight"
    with pytest.raises(ValueError, match="mode differs"):
        replay.load(0)


@pytest.mark.parametrize("metric", ["submission", "ready"])
def test_multi_policy_rejects_single_policy_lifecycle_clock(metric):
    from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

    args = Namespace(megatron_config=encode_megatron_config("solver", "verifier"), async_staleness_metric=metric)
    with pytest.raises(ValueError, match="Submission/ready staleness supports only single-policy"):
        validate_buffer_arguments(args)
