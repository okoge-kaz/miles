import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import miles.rollout.fully_async_rollout as fully_async
import miles.rollout.replay_data_buffer_inflight as inflight_module
from miles.rollout.replay_data_buffer import ReplayDataBuffer
from miles.rollout.replay_data_buffer_checkpoint import (
    iter_replay_data_buffer_parts,
    load_replay_data_buffer_auxiliary,
    load_replay_data_buffer_header,
)
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.types import Sample
from tests.fast.experiments.training.test_replay_data_buffer import _args, _entry, _owner


def make_owner(args):
    owner = object.__new__(fully_async.FullyAsyncRolloutFn)
    owner.__dict__.update(_owner(args).__dict__)
    owner._worker = None
    owner.state = SimpleNamespace(aborted=False, sampling_params={})
    owner.state.reset = lambda: setattr(owner.state, "aborted", False)
    owner._scheduler = make_submission_scheduler(args, default="sample")
    owner._track_lifecycle = True
    owner._replay_data_buffer = ReplayDataBuffer(owner)
    return owner


async def test_capture_keeps_blocked_ready_once_and_resumes_prefix_without_new_prompt(tmp_path, monkeypatch):
    args = _args(
        tmp_path,
        replay_data_buffer_mode="inflight",
        replay_data_buffer_async_restore=False,
        rollout_submission_granularity=None,
        group_rm=False,
    )
    owner = make_owner(args)
    ready, blocked, partial, no_reward = [_entry(i) for i in range(4)]
    partial.group[0].status = Sample.Status.ABORTED
    partial.group[0].reward = None
    partial.group[0].metadata["policy_provenance"] = {"submitted_weight_version": 1, "first_prefill_weight_version": 2}
    saved_prefix = copy.deepcopy(partial.group[0].to_dict())
    no_reward.group[0].reward = None
    await owner._output.restore_entry(ready)
    owner._output._capacity = 1
    aborted = asyncio.Event()

    async def generation(entry, gate=None):
        if gate is not None:
            await gate.wait()
        return entry

    for entry, gate in [(blocked, None), (partial, aborted), (no_reward, None)]:
        task = asyncio.create_task(generation(entry, gate))
        owner._active[task] = entry.group
        owner._original_prompts[task] = entry.prompt_group
        if entry is blocked:
            owner._pending_ready[task] = entry
    owner._worker = asyncio.create_task(owner._output.restore_entry(blocked))
    await asyncio.sleep(0)
    assert not owner._worker.done()

    async def abort_requests(_args):
        assert owner.state.aborted
        aborted.set()

    async def park_worker():
        await asyncio.Event().wait()

    monkeypatch.setattr(inflight_module, "_abort_requests", AsyncMock(side_effect=abort_requests))
    monkeypatch.setattr(owner, "_worker_loop", park_worker)
    await asyncio.wait_for(owner._replay_data_buffer._save(0), 10)
    header = load_replay_data_buffer_header(args.save, 0)
    restored = [entry for part in iter_replay_data_buffer_parts(args.save, 0, header) for entry in part]
    assert [entry.group[0].index for entry in restored] == [0, 1]
    assert header["mode"] == "inflight" and header["inflight_count"] == 1
    auxiliary = load_replay_data_buffer_auxiliary(args.save, 0, header)
    assert auxiliary["inflight_entries"][0].group[0].to_dict() == saved_prefix
    assert auxiliary["pending_prompts"] == []
    assert not owner._active and not owner._pending_ready and not owner.state.aborted
    inflight_module._abort_requests.assert_awaited_once()
    await owner._close()

    args.load = args.save
    resumed = make_owner(args)
    await resumed._replay_data_buffer._start_restore(0, header)
    assert len(resumed._output) == 2
    resumed.data_source.get_samples = lambda _: pytest.fail("prefix continuation consumed a fresh prompt")

    async def continue_group(state, group, **kwargs):
        assert group[0].to_dict() == saved_prefix
        group[0].status = Sample.Status.COMPLETED
        group[0].reward = 1.0
        kwargs["sample_done_callback"]()
        return group

    monkeypatch.setattr(fully_async, "generate_and_rm_group", continue_group)
    task, group = resumed._submit_one_group()
    await task
    entry = resumed._collect_group_result(task, group)
    assert entry.group[0].tokens == saved_prefix["tokens"]
    assert entry.group[0].metadata["policy_provenance"]["submitted_weight_version"] == 1
    assert entry.group[0].metadata["policy_provenance"]["first_prefill_weight_version"] == 2
    assert resumed._scheduler.samples_in_flight == 0
    assert not resumed._replay_data_buffer.inflight.pending
    await resumed._close()


async def test_capture_propagates_abort_failure_and_releases_producer(tmp_path, monkeypatch):
    args = _args(tmp_path, replay_data_buffer_mode="inflight", rollout_submission_granularity=None, group_rm=False)
    owner = make_owner(args)
    task = asyncio.create_task(asyncio.Event().wait())
    owner._active[task] = _entry(0).group
    owner._original_prompts[task] = _entry(0).prompt_group
    monkeypatch.setattr(inflight_module, "_abort_requests", AsyncMock(side_effect=RuntimeError("abort failed")))
    with pytest.raises(RuntimeError, match="abort failed"):
        await owner._replay_data_buffer._save(0)
    assert not owner.state.aborted and owner._producer_resumed.is_set()
    assert not (tmp_path / "new/rollout/replay_0000000").exists()
    await owner._close()


@pytest.mark.parametrize("change", ["custom", "multimodal", "logprobs", "mask"])
def test_unsupported_or_misaligned_prefix_is_rejected(change):
    sample = _entry(0).group[0]
    if change == "custom":
        sample.generate_function_path = "example.generator"
    elif change == "multimodal":
        sample.multimodal_inputs = {"image": "data"}
    elif change == "logprobs":
        sample.rollout_log_probs = None
    else:
        sample.loss_mask = [1]
    with pytest.raises(ValueError):
        inflight_module.validate_inflight_group([sample])


async def test_single_turn_continuation_appends_tokens_logprobs_masks_and_versions(monkeypatch):
    from miles.rollout.base_types import GenerateFnInput
    from miles.rollout.generate_hub import single_turn

    sample = _entry(0).group[0]
    sample.status = Sample.Status.ABORTED
    sample.response = ""  # Token prefixes can decode to an empty string.
    sample.reward = None
    args = SimpleNamespace(sglang_router_ip="router", sglang_router_port=1, sglang_speculative_algorithm=None)
    monkeypatch.setattr(single_turn, "compute_prompt_ids_from_sample", lambda *_: [1])
    monkeypatch.setattr(single_turn, "compute_routing_headers", lambda *_: None)
    monkeypatch.setattr(single_turn, "compute_request_payload", lambda _, **kw: (kw, None))
    response = {
        "text": "continued",
        "meta_info": {
            "output_token_logprobs": [(-0.25, 4, None)],
            "weight_version": "6",
            "finish_reason": {"type": "stop"},
        },
    }
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(single_turn, "post", post)
    state = SimpleNamespace(args=args)
    await single_turn.generate(
        GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 5}, evaluation=False)
    )
    payload = post.call_args.args[1]
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["sampling_params"]["max_new_tokens"] == 3
    assert sample.tokens == [1, 2, 3, 4] and sample.response_length == 3
    assert sample.loss_mask == [1, 0, 1]
    assert sample.rollout_log_probs == [-1.0, -2.0, -0.25]
    assert [span.version for span in sample.all_weight_version_spans] == ["1", "6"]
    assert sample.status == Sample.Status.COMPLETED
