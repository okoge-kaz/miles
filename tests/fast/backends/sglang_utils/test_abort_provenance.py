import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.sglang_utils.abort_provenance import _patch_tokenizer_manager


def _scheduler_stats():
    metadata = {
        "first_prefill_weight_version": 10,
        "min_forward_weight_version": 10,
        "max_forward_weight_version": 11,
        "last_forward_weight_version": 11,
        "response_weight_version_segments": [[0, 2, 10], [2, 5, 11]],
    }
    return SimpleNamespace(convert_to_policy_version_meta_info=lambda _: deepcopy(metadata))


def _manager(abort_handler=None):
    class Manager:
        async def _handle_batch_output(self, output):
            # A non-streaming client consumes intermediate events too. Keeping
            # only state.out_list[-1] would lose provenance before the abort.
            self.rid_to_state["test-request"].out_list.clear()

        def _handle_abort_req(self, request):
            state = self.rid_to_state.pop(request.rid, None)
            if state is not None:
                state.out_list.append(
                    {
                        "meta_info": {
                            "finish_reason": {"type": "abort"},
                            "completion_tokens": len(state.output_ids),
                        }
                    }
                )

    if abort_handler is not None:
        Manager._handle_abort_req = abort_handler
    _patch_tokenizer_manager(Manager)
    state = SimpleNamespace(
        out_list=[],
        output_ids=list(range(5)),
        obj=SimpleNamespace(stream=False, return_logprob=False),
        time_stats=SimpleNamespace(set_finished_time=Mock(), get_e2e_latency=lambda: 1.0),
        get_text=lambda: "partial response",
        event=asyncio.Event(),
    )
    manager = Manager()
    manager.rid_to_state = {"test-request": state}
    manager.server_args = SimpleNamespace(weight_version="99")
    return manager, state


def _cache_batch(manager, stats=None):
    output = SimpleNamespace(rids=["test-request"], completion_tokens=[5], time_stats=stats)
    asyncio.run(manager._handle_batch_output(output))


@pytest.mark.parametrize("retained_tokens", [0, 3, 5])
def test_abort_preserves_only_received_scheduler_provenance(retained_tokens):
    manager, state = _manager()
    _cache_batch(manager, [_scheduler_stats()])
    state.output_ids = state.output_ids[:retained_tokens]
    manager._handle_abort_req(SimpleNamespace(rid="test-request"))

    metadata = state.out_list[-1]["meta_info"]
    assert metadata["first_prefill_weight_version"] == 10
    assert metadata["last_forward_weight_version"] == 11
    segments = metadata["response_weight_version_segments"]
    assert sum(end - start for start, end, _ in segments) == retained_tokens
    assert all(version in (10, 11) for _, _, version in segments)


def test_abort_without_scheduler_output_does_not_invent_versions():
    manager, state = _manager()
    _cache_batch(manager)
    manager._handle_abort_req(SimpleNamespace(rid="test-request"))
    assert "first_prefill_weight_version" not in state.out_list[-1]["meta_info"]
    assert "response_weight_version_segments" not in state.out_list[-1]["meta_info"]


def test_abort_after_normal_completion_and_repeated_install_are_safe():
    manager, state = _manager()
    handler = type(manager)._handle_batch_output
    _patch_tokenizer_manager(type(manager))
    assert type(manager)._handle_batch_output is handler
    manager.rid_to_state.clear()
    manager._handle_abort_req(SimpleNamespace(rid="test-request"))
    assert state.out_list == []


def test_existing_abort_provenance_takes_precedence():
    def handle_abort(self, request):
        state = self.rid_to_state.pop(request.rid)
        state.out_list.append(
            {
                "meta_info": {
                    "finish_reason": {"type": "abort"},
                    "completion_tokens": 5,
                    "first_prefill_weight_version": 12,
                    "response_weight_version_segments": [[0, 5, 12]],
                }
            }
        )

    manager, state = _manager(handle_abort)
    _cache_batch(manager, [_scheduler_stats()])
    manager._handle_abort_req(SimpleNamespace(rid="test-request"))
    metadata = state.out_list[-1]["meta_info"]
    assert metadata["first_prefill_weight_version"] == 12
    assert metadata["response_weight_version_segments"] == [[0, 5, 12]]
    assert "last_forward_weight_version" not in metadata


def test_installed_sglang_abort_echo_preserves_cached_prefix():
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    _, state = _manager()
    state.finished = False
    state.prompt_token_ids = None
    state.policy_version_meta_info = _scheduler_stats().convert_to_policy_version_meta_info(5)
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.rid_to_state = {"test-request": state}
    manager.config_value = lambda key: "99"
    manager.incremental_streaming_output = False
    manager._finalize_lora_lease = lambda state: None
    manager._handle_abort_req(AbortReq(rid="test-request"))
    metadata = state.out_list[-1]["meta_info"]
    assert metadata["weight_version"] == "99"
    assert metadata["first_prefill_weight_version"] == 10
    assert metadata["last_forward_weight_version"] == 11
    assert metadata["response_weight_version_segments"] == [[0, 2, 10], [2, 5, 11]]
