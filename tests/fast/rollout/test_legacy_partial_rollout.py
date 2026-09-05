from types import SimpleNamespace

import pytest

from miles.rollout import sglang_rollout
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils.types import Sample


@pytest.mark.asyncio
async def test_legacy_resume_uses_persisted_tokens_when_response_text_is_empty(monkeypatch) -> None:
    class _Tokenizer:
        def encode(self, _prompt, *, add_special_tokens):
            assert not add_special_tokens
            return [10]

    state = SimpleNamespace(processor=None, tokenizer=_Tokenizer())
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    request = {}

    async def _post(_url, payload, *, headers):
        request.update(payload)
        assert headers is None
        return {
            "text": "continued",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.5, 12]],
            },
        }

    monkeypatch.setattr(sglang_rollout, "post", _post)
    args = SimpleNamespace(
        ci_test=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_opd=False,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=False,
        sglang_router_policy="random",
        sglang_speculative_algorithm=None,
    )
    sample = Sample(
        prompt="prompt",
        response="",
        tokens=[10, 11],
        response_length=1,
        status=Sample.Status.ABORTED,
    )

    result = await sglang_rollout.generate(args, sample, {"max_new_tokens": 3})

    assert request["input_ids"] == [10, 11]
    assert request["sampling_params"]["max_new_tokens"] == 2
    assert result.tokens == [10, 11, 12]
    assert result.response_length == 2
    assert result.status == Sample.Status.COMPLETED


def test_legacy_wrapper_reports_buffer_depth_after_abort(monkeypatch) -> None:
    output = RolloutFnTrainOutput(
        samples=[[Sample(index=1)]],
        metrics={"staleness/total/mean": 1.0},
    )
    aborted = [[Sample(index=2)]]
    sentinel = object()
    monkeypatch.setattr(sglang_rollout, "generate_rollout_async", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(sglang_rollout, "run", lambda value: (output, aborted) if value is sentinel else None)

    class _DataSource:
        def __init__(self):
            self.buffer = []

        def get_samples(self, _num_samples):
            raise AssertionError("The mocked generator must not request samples")

        def add_samples(self, samples):
            self.buffer.extend(samples)

        def get_buffer_length(self):
            return len(self.buffer)

    data_source = _DataSource()
    args = SimpleNamespace(rollout_global_dataset=True, partial_rollout=True)

    result = sglang_rollout.generate_rollout(args, rollout_id=7, data_source=data_source)

    assert result is output
    assert result.metrics["rollout/partial_rollout/buffer_depth_after_abort"] == 1.0
