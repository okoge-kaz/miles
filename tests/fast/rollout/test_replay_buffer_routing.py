from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import asyncio
import copy
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from tests.fast.rollout.test_replay_buffer import _args, _DataSource, _make_fn, _prompt_group, _stop, _Version

import miles.rollout.experimental_fully_async_rollout as fully_async
from miles.rollout.base_types import GenerateFnInput, RolloutFnTrainInput
from miles.rollout.generate_hub import single_turn
from miles.rollout.generate_utils.generate_endpoint_utils import update_sample_from_response
from miles.rollout.generate_utils.routing_replay import merge_routing_prefix
from miles.rollout.replay_buffer import dataset_fingerprint, load_replay_buffer, save_replay_buffer
from miles.rollout.replay_buffer_codec import materialize_replay_buffer_state
from miles.utils.types import Sample


def _routing_args(**overrides):
    return _args(
        use_rollout_routing_replay=True,
        num_layers=2,
        moe_router_topk=2,
        num_experts=16,
        sglang_speculative_algorithm=None,
        **overrides,
    )


def _output(tokens, value, total_tokens, *, finish="abort"):
    routes = np.full((total_tokens - 1, 2, 2), value, dtype=np.int32)
    return {
        "text": "x" * len(tokens),
        "meta_info": {
            "output_token_logprobs": [(-0.5, token, None) for token in tokens],
            "routed_experts": pybase64.b64encode(routes.tobytes()).decode("ascii"),
            "finish_reason": {"type": finish},
        },
    }


async def _update(sample, output):
    await update_sample_from_response(
        _routing_args(),
        sample,
        payload={"input_ids": sample.tokens or [100]},
        output=output,
        preserve_routing_prefix=True,
    )


async def test_repeated_abort_preserves_old_routing_and_logprobs():
    sample = Sample(tokens=[100, 101])
    await _update(sample, _output([10, 11], 1, 4))
    original_logprobs = list(sample.rollout_log_probs)
    await _update(sample, _output([12], 9, 5))
    await _update(sample, _output([13, 14], 7, 7, finish="stop"))
    assert sample.tokens == [100, 101, 10, 11, 12, 13, 14]
    assert sample.rollout_log_probs[:2] == original_logprobs
    # The last old token gets its first forward in the next request.
    np.testing.assert_array_equal(sample.rollout_routed_experts[:, 0, 0], [1, 1, 1, 9, 7, 7])
    sample.validate()


async def test_abort_before_new_token_keeps_saved_routes():
    sample = Sample(tokens=[100])
    await _update(sample, _output([10], 1, 2))
    before = copy.deepcopy(sample)
    await _update(sample, {"text": "", "meta_info": {"finish_reason": {"type": "abort"}}})
    assert sample.tokens == before.tokens
    assert sample.rollout_log_probs == before.rollout_log_probs
    np.testing.assert_array_equal(sample.rollout_routed_experts, before.rollout_routed_experts)


@pytest.mark.parametrize("max_new_tokens", [1, 3])
async def test_single_turn_resume_uses_saved_prefix_routing(monkeypatch, max_new_tokens):
    sample = Sample(tokens=[100])
    await _update(sample, _output([10], 1, 2))
    args = _routing_args(
        sglang_router_policy="round_robin",
        rollout_max_response_len=3,
        rollout_max_context_len=10,
        use_rollout_indexer_replay=False,
    )
    requests = []

    async def post(url, payload, headers):
        requests.append(payload)
        return _output([11, 12], 9, 4, finish="stop")

    monkeypatch.setattr(single_turn, "post", post)
    monkeypatch.setattr(single_turn, "compute_prompt_ids_from_sample", lambda state, sample: [100])
    result = await single_turn.generate(
        GenerateFnInput(
            state=SimpleNamespace(args=args),
            sample=sample,
            sampling_params={"max_new_tokens": max_new_tokens},
            evaluation=False,
        )
    )
    if max_new_tokens == 1:
        assert not requests
        assert result.samples.status == Sample.Status.TRUNCATED
        np.testing.assert_array_equal(result.samples.rollout_routed_experts[:, 0, 0], [1])
    else:
        assert requests[0]["input_ids"] == [100, 10]
        assert requests[0]["return_routed_experts"]
        np.testing.assert_array_equal(result.samples.rollout_routed_experts[:, 0, 0], [1, 9, 9])


def test_unstarted_abort_needs_no_routing():
    assert (
        merge_routing_prefix(None, None, prefix_tokens=2, prefix_response_tokens=0, new_tokens=0, num_layers=2, topk=2)
        is None
    )


@pytest.mark.parametrize("bad", [None, np.zeros((2, 2, 2), np.int32), np.zeros((3, 2, 2), np.int64)])
def test_missing_or_invalid_prefix_routing_is_rejected(bad):
    with pytest.raises(ValueError, match="persisted prefix"):
        merge_routing_prefix(
            bad,
            np.zeros((4, 2, 2), np.int32),
            prefix_tokens=4,
            prefix_response_tokens=1,
            new_tokens=1,
            num_layers=2,
            topk=2,
        )


def test_missing_new_routing_is_rejected():
    with pytest.raises(ValueError, match="generation response"):
        merge_routing_prefix(
            None,
            None,
            prefix_tokens=2,
            prefix_response_tokens=0,
            new_tokens=1,
            num_layers=2,
            topk=2,
        )


def test_fingerprint_binds_routing_mode_and_layout():
    source = SimpleNamespace(dataset=[])
    args = _routing_args()
    original = dataset_fingerprint(args, source)
    for name, value in [
        ("use_rollout_routing_replay", False),
        ("num_layers", 3),
        ("moe_router_topk", 4),
        ("num_experts", 32),
    ]:
        changed = copy.copy(args)
        setattr(changed, name, value)
        assert dataset_fingerprint(changed, source) != original


@pytest.mark.parametrize("buffer_type", ["rollout", "inflight"])
async def test_completed_cached_routes_survive_disk_snapshot(monkeypatch, tmp_path, buffer_type):
    args = _routing_args(replay_buffer_type=buffer_type)
    original = _make_fn(monkeypatch, args, _DataSource(), lambda *args, **kwargs: None)
    prompt = _prompt_group(1)
    result = copy.deepcopy(prompt)
    for sample in result:
        await _update(sample, _output([sample.index], 3, 2, finish="stop"))
        sample.reward = 1.0
    original._pending_prompts = {1: prompt}
    original._output = asyncio.Queue()
    original._output.put_nowait((prompt, result))
    original._replay_buffer_packed_fields.cache_group(result)
    save_replay_buffer(tmp_path, 0, await original.replay_buffer_state(0))
    restored_state = load_replay_buffer(tmp_path, 0, expected_fingerprint=dataset_fingerprint(args, _DataSource()))
    restored = _make_fn(monkeypatch, args, _DataSource(), lambda *args, **kwargs: None)
    await restored.restore_replay_buffer_state(restored_state)
    [(_, restored_group)] = list(restored._output._queue)
    for sample in restored_group:
        np.testing.assert_array_equal(sample.rollout_routed_experts, np.full((1, 2, 2), 3, np.int32))
        assert sample.rollout_routed_experts is not result[0].rollout_routed_experts
        sample.validate()


async def test_inflight_disk_resume_retains_routes_through_trainer_admission(monkeypatch, tmp_path):
    interrupted, started = asyncio.Event(), asyncio.Event()

    async def abort_requests(_args):
        interrupted.set()

    async def generate(state, group, sampling_params, evaluation=False):
        if any(sample.response_length for sample in group):
            for sample in group:
                np.testing.assert_array_equal(sample.rollout_routed_experts, np.full((1, 2, 2), 1, np.int32))
                await _update(sample, _output([200 + sample.index], 9, 3, finish="stop"))
                sample.reward = float(sample.index % 2)
            return group
        started.set()
        await interrupted.wait()
        for sample in group:
            await _update(sample, _output([sample.index], 1, 2))
        return group

    monkeypatch.setattr(fully_async, "_abort_inflight_requests", abort_requests)
    args = _routing_args(replay_buffer_type="inflight")
    original = _make_fn(monkeypatch, args, _DataSource([_prompt_group(1)]), generate)
    original._ensure_worker()
    await started.wait()
    state = await original.replay_buffer_state(0)
    save_replay_buffer(tmp_path, 0, state)
    await _stop(original)
    restored_state = load_replay_buffer(tmp_path, 0, expected_fingerprint=dataset_fingerprint(args, _DataSource()))
    [inflight] = materialize_replay_buffer_state(restored_state)["inflight_items"]
    assert all(sample["rollout_routed_experts"].shape == (1, 2, 2) for sample in inflight["generation_group"])
    restored = _make_fn(monkeypatch, args, _DataSource(), generate)
    await restored.restore_replay_buffer_state(restored_state)
    restored._weight_version = _Version(9)
    restored.commit_applied_weight_version(9)
    output = await restored(RolloutFnTrainInput(rollout_id=0))
    for sample in output.samples[0]:
        np.testing.assert_array_equal(sample.rollout_routed_experts[:, 0, 0], [1, 9])
        assert sample.rollout_log_probs == [-0.5, -0.5]
        sample.validate()
    assert output.metrics["resume/replay_buffer/inflight_tokens_restored"] == 2
    await _stop(restored)
