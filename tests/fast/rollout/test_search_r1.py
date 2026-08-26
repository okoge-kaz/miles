import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

if "numpy" not in sys.modules:
    try:
        __import__("numpy")
    except ModuleNotFoundError:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.ndarray = object
        sys.modules["numpy"] = numpy_stub

if "torch" not in sys.modules:
    try:
        __import__("torch")
    except ModuleNotFoundError:

        class _Tensor:
            def __init__(self, values):
                self.values = list(values)

            def std(self):
                mean = sum(self.values) / len(self.values)
                return (sum((value - mean) ** 2 for value in self.values) / (len(self.values) - 1)) ** 0.5

        torch_stub = types.ModuleType("torch")
        torch_stub.dtype = object
        torch_stub.Size = tuple
        torch_stub.Tensor = _Tensor
        torch_stub.float64 = object()
        torch_stub.tensor = lambda values, dtype=None: _Tensor(values)
        sys.modules["torch"] = torch_stub

from miles.rollout.filter_hub.dynamic_sampling_filters import check_no_aborted_and_reward_nonzero_std
from miles.utils.types import Sample
from experiments.src.environments.search_r1 import retrieval_server

SEARCH_R1_DIR = Path(__file__).resolve().parents[3] / "examples" / "experimental" / "search-r1"
sys.path.insert(0, str(SEARCH_R1_DIR))

# The rollout itself is tested with a fake generation state and HTTP response;
# avoid importing SGLang/Ray just to replace those two boundaries immediately.
sglang_rollout_stub = types.ModuleType("miles.rollout.sglang_rollout")
sglang_rollout_stub.GenerateState = object
http_utils_stub = types.ModuleType("miles.utils.http_utils")
http_utils_stub.post = None
aiohttp_stub = types.ModuleType("aiohttp")
aiohttp_stub.ClientError = RuntimeError
aiohttp_stub.ClientTimeout = object
aiohttp_stub.ClientSession = object
saved_sglang_rollout = sys.modules.get("miles.rollout.sglang_rollout")
saved_http_utils = sys.modules.get("miles.utils.http_utils")
saved_aiohttp = sys.modules.get("aiohttp")
sys.modules["miles.rollout.sglang_rollout"] = sglang_rollout_stub
sys.modules["miles.utils.http_utils"] = http_utils_stub
sys.modules["aiohttp"] = aiohttp_stub
try:
    import generate_with_search as search_r1  # noqa: E402
    import local_search_server  # noqa: E402
finally:
    if saved_sglang_rollout is None:
        del sys.modules["miles.rollout.sglang_rollout"]
    else:
        sys.modules["miles.rollout.sglang_rollout"] = saved_sglang_rollout
    if saved_http_utils is None:
        del sys.modules["miles.utils.http_utils"]
    else:
        sys.modules["miles.utils.http_utils"] = saved_http_utils
    if saved_aiohttp is None:
        del sys.modules["aiohttp"]
    else:
        sys.modules["aiohttp"] = saved_aiohttp


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [1000 + index for index, _ in enumerate(text)]}


class _GenerateState:
    tokenizer = _Tokenizer()

    def __init__(self, args):
        self.args = args


def _args():
    return SimpleNamespace(
        partial_rollout=False,
        sglang_router_ip="router",
        sglang_router_port=30000,
        sglang_speculative_algorithm=None,
    )


def _sampling_params():
    return {
        "stop": ["</search>", "</answer>"],
        "no_stop_trim": True,
        "max_new_tokens": 512,
    }


def _output(text, token_ids, version):
    return {
        "text": text,
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [[-0.1 * (index + 1), token_id] for index, token_id in enumerate(token_ids)],
            "weight_version": str(version),
            "first_prefill_weight_version": version,
            "min_forward_weight_version": version,
            "max_forward_weight_version": version,
            "last_forward_weight_version": version,
            "response_weight_version": f"response-{version}",
            "prompt_tokens": 10,
            "cached_tokens": 2,
        },
    }


def test_local_retriever_contract_reads_nested_and_flattened_documents():
    payload = {
        "result": [
            [
                {"document": {"contents": '"Nested title"\nNested passage'}},
                {"contents": '"Flat title"\nFlat passage'},
            ]
        ]
    }

    assert local_search_server._extract_contexts(payload) == [
        {"document": {"contents": '"Nested title"\nNested passage'}},
        {"document": {"contents": '"Flat title"\nFlat passage'}},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"result": []},
        {"result": [[{"document": {}}]]},
        {"result": [[{"document": {"contents": ""}}]]},
    ],
)
def test_local_retriever_contract_rejects_malformed_payloads(payload):
    with pytest.raises(local_search_server.LocalSearchError):
        local_search_server._extract_contexts(payload)


def test_retrieval_server_combines_requests_into_one_native_search():
    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def __call__(self, queries):
            self.calls.append(queries)
            return queries

    class FakeIndex:
        def __init__(self):
            self.calls = []

        def search(self, embeddings, topk):
            self.calls.append((embeddings, topk))
            return [[0.9, 0.8], [0.7, 0.6]], [[0, 1], [2, 3]]

    encoder = FakeEncoder()
    index = FakeIndex()
    requests = [
        retrieval_server.SearchRequest(queries=("first",), topk=1, return_scores=False),
        retrieval_server.SearchRequest(queries=("second",), topk=2, return_scores=True),
    ]

    responses = retrieval_server.search_batch(
        index,
        ["title 0", "title 1", "title 2", "title 3"],
        ["text 0", "text 1", "text 2", "text 3"],
        encoder,
        requests,
    )

    assert encoder.calls == [["first", "second"]]
    assert index.calls == [(["first", "second"], 2)]
    assert responses == [
        {"result": [[{"document": {"contents": '"title 0"\ntext 0'}}]]},
        {
            "result": [
                [
                    {"document": {"contents": '"title 2"\ntext 2'}, "score": 0.7},
                    {"document": {"contents": '"title 3"\ntext 3'}, "score": 0.6},
                ]
            ]
        },
    ]


async def test_retrieval_server_serializes_concurrent_requests_as_one_batch():
    batches = []

    def fake_search(requests):
        batches.append(requests)
        return [{"result": [[request.queries[0]]]} for request in requests]

    batcher = retrieval_server.RequestBatcher(fake_search, max_requests=8, wait_ms=50)
    await batcher.start()
    try:
        requests = [
            retrieval_server.SearchRequest(queries=(f"query-{index}",), topk=3, return_scores=False)
            for index in range(4)
        ]
        responses = await asyncio.gather(*(batcher.submit(request) for request in requests))
    finally:
        await batcher.stop()

    assert len(batches) == 1
    assert [request.queries[0] for request in batches[0]] == [f"query-{index}" for index in range(4)]
    assert responses == [{"result": [[f"query-{index}"]]} for index in range(4)]


def test_search_r1_requires_exact_action_stops():
    with pytest.raises(ValueError, match="</search>"):
        search_r1._validate_sampling_params({"stop": ["</answer>"], "no_stop_trim": True})
    with pytest.raises(ValueError, match="no_stop_trim"):
        search_r1._validate_sampling_params({"stop": ["</search>", "</answer>"], "no_stop_trim": False})


async def test_generate_masks_observations_and_records_every_policy_version(monkeypatch):
    first_text = "<think>lookup</think><search>alpha</search>"
    second_text = "<think>done</think><answer>beta</answer>"
    outputs = [
        _output(first_text, [11, 12], version=4),
        _output(second_text, [21, 22], version=5),
    ]

    async def fake_post(url, payload):
        assert url == "http://router:30000/generate"
        assert payload["sampling_params"] == _sampling_params()
        return outputs.pop(0)

    async def fake_search(query):
        assert query == "alpha"
        return "Doc 1(Title: Alpha) Retrieved passage"

    monkeypatch.setattr(search_r1, "GenerateState", _GenerateState)
    monkeypatch.setattr(search_r1, "post", fake_post)
    monkeypatch.setattr(search_r1, "search", fake_search)

    sample = Sample(prompt="prompt", label={"ground_truth": {"target": ["beta"]}})
    result = await search_r1.generate(_args(), sample, _sampling_params())

    observation = "\n\n<information>Doc 1(Title: Alpha) Retrieved passage</information>\n\n"
    observation_length = len(_Tokenizer()(observation)["input_ids"])
    assert result.status == Sample.Status.COMPLETED
    assert result.response == first_text + observation + second_text
    assert result.loss_mask == [1, 1] + [0] * observation_length + [1, 1]
    assert result.rollout_log_probs == [-0.1, -0.2] + [0.0] * observation_length + [-0.1, -0.2]
    assert result.weight_versions == ["4", "5"]
    assert result.first_prefill_weight_versions == [4, 5]
    assert result.min_forward_weight_versions == [4, 5]
    assert result.max_forward_weight_versions == [4, 5]
    assert result.last_forward_weight_versions == [4, 5]
    assert result.response_weight_versions == ["response-4", "response-5"]
    assert result.metadata["search_r1_turns"] == 2
    assert result.metadata["search_r1_search_calls"] == 1
    assert result.prefix_cache_info.cached_tokens == 4
    assert result.prefix_cache_info.total_prompt_tokens == 20
    result.validate()
    assert outputs == []


async def test_generate_aborts_instead_of_training_on_retriever_failure(monkeypatch):
    async def fake_post(url, payload):
        return _output("<think>lookup</think><search>alpha</search>", [31, 32], version=7)

    async def failed_search(query):
        raise search_r1.SearchBackendError("retriever unavailable")

    monkeypatch.setattr(search_r1, "GenerateState", _GenerateState)
    monkeypatch.setattr(search_r1, "post", fake_post)
    monkeypatch.setattr(search_r1, "search", failed_search)

    sample = Sample(prompt="prompt", label={"ground_truth": {"target": ["beta"]}})
    result = await search_r1.generate(_args(), sample, _sampling_params())

    assert result.status == Sample.Status.ABORTED
    assert result.response == "<think>lookup</think><search>alpha</search>"
    assert result.weight_versions == ["7"]
    assert result.metadata["search_r1_search_calls"] == 1
    assert result.metadata["search_r1_error"] == "retriever unavailable"
    assert result.reward is None
    result.validate()


async def test_generate_can_skip_logprobs_for_inference_only_measurement(monkeypatch):
    output = _output("<answer>beta</answer>", [41, 42], version=8)
    del output["meta_info"]["output_token_logprobs"]

    async def fake_post(url, payload):
        assert "return_logprob" not in payload
        return output

    monkeypatch.setitem(search_r1.SEARCH_R1_CONFIGS, "return_logprob", False)
    monkeypatch.setattr(search_r1, "GenerateState", _GenerateState)
    monkeypatch.setattr(search_r1, "post", fake_post)

    sample = Sample(prompt="prompt", label={"ground_truth": {"target": ["beta"]}})
    result = await search_r1.generate(_args(), sample, _sampling_params())

    expected_response_tokens = _Tokenizer()(result.response)["input_ids"]
    assert result.status == Sample.Status.COMPLETED
    assert result.response == "<answer>beta</answer>"
    assert result.rollout_log_probs is None
    assert result.loss_mask == [1] * len(expected_response_tokens)
    result.validate()


def test_combined_filter_rejects_aborted_before_reading_reward():
    args = SimpleNamespace(reward_key=None)
    aborted = Sample(status=Sample.Status.ABORTED, reward=None)
    completed = Sample(status=Sample.Status.COMPLETED, reward=1.0)

    result = check_no_aborted_and_reward_nonzero_std(args, [aborted, completed])

    assert result.keep is False
    assert result.reason == "group_has_aborted"


def test_combined_filter_keeps_only_nonconstant_valid_reward_groups():
    args = SimpleNamespace(reward_key=None)
    varying = [
        Sample(status=Sample.Status.COMPLETED, reward=0.0),
        Sample(status=Sample.Status.COMPLETED, reward=1.0),
    ]
    constant = [
        Sample(status=Sample.Status.COMPLETED, reward=0.0),
        Sample(status=Sample.Status.COMPLETED, reward=0.0),
    ]

    assert bool(check_no_aborted_and_reward_nonzero_std(args, varying).keep)
    constant_result = check_no_aborted_and_reward_nonzero_std(args, constant)
    assert not bool(constant_result.keep)
    assert constant_result.reason == "zero_std_0.0"
