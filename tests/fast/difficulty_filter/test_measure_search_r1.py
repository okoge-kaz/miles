import json
from types import SimpleNamespace

from experiments.tools.difficulty_filter import measure_search_r1
from experiments.tools.difficulty_filter.measure_search_r1 import (
    EpisodeResult,
    _configure_search_environment,
    _done_indices,
    _load_rows,
    build_record,
    protocol_tokenization,
)


def _row():
    return {
        "label": {"ground_truth": {"target": ["Paris"]}},
        "metadata": {"source": "q-1", "question": "Capital?"},
    }


def test_record_keeps_accuracy_and_agentic_cost_separate():
    episodes = [
        EpisodeResult(
            reward=1.0,
            status="completed",
            response="<think>x</think><search>France</search><information>...</information>"
            "<think>y</think><answer>Paris</answer>",
            generated_tokens=20,
            observation_tokens=30,
            turns=2,
            search_calls=1,
        ),
        EpisodeResult(
            reward=0.0,
            status="truncated",
            response="<think>x</think><search>France</search>",
            generated_tokens=10,
            observation_tokens=0,
            turns=1,
            search_calls=1,
        ),
    ]

    record = build_record(4, _row(), episodes, correct_threshold=0.5)

    assert record["index"] == 4
    assert record["label"] == ["Paris"]
    assert record["pass_rate"] == 0.5
    assert record["response_len_mean"] == 15
    assert record["observation_len_mean"] == 15
    assert record["search_calls_mean"] == 1
    assert record["turns_mean"] == 1.5
    assert record["searched_frac"] == 1
    assert record["answered_frac"] == 0.5
    assert record["truncated_frac"] == 0.5


def test_protocol_tags_may_be_multiple_non_special_tokens():
    class Tokenizer:
        all_special_ids = [99]

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(character) for character in text]

        def decode(self, token_ids, **kwargs):
            return "".join(chr(token_id) for token_id in token_ids)

    result = protocol_tokenization(Tokenizer())

    assert all(not item["atomic_special_token"] for item in result.values())
    assert set(result) == {
        "<think>",
        "</think>",
        "<search>",
        "</search>",
        "<information>",
        "</information>",
        "<answer>",
        "</answer>",
    }
    assert result["<search>"]["token_ids"] == [ord(character) for character in "<search>"]


def test_resume_ignores_a_partial_trailing_record(tmp_path):
    output = tmp_path / "nq.jsonl"
    output.write_text(json.dumps({"index": 2}) + "\n{" + "\n")

    assert _done_indices(str(output)) == {2}


def test_parquet_training_rows_keep_stable_indices(monkeypatch):
    raw_rows = [
        {
            "prompt": [{"role": "user", "content": "first"}],
            "reward_model": {"ground_truth": {"target": ["one"]}},
            "metadata": {"source": "nq"},
        },
        {
            "prompt": [{"role": "user", "content": "second"}],
            "reward_model": {"ground_truth": {"target": ["two"]}},
            "metadata": {"source": "hotpotqa"},
        },
    ]

    class FakeBatch:
        def to_pylist(self):
            return raw_rows

    class FakeParquetFile:
        def __init__(self, path):
            assert path == "train.parquet"

        def iter_batches(self):
            return iter([FakeBatch()])

    monkeypatch.setattr(measure_search_r1, "pq", SimpleNamespace(ParquetFile=FakeParquetFile))

    rows = _load_rows(
        "train.parquet",
        input_key="prompt",
        label_key="reward_model",
        metadata_key="metadata",
        limit=None,
    )

    assert [row["index"] for row in rows] == [0, 1]
    assert rows[1]["label"]["ground_truth"]["target"] == ["two"]


def test_offline_measurement_disables_unused_rollout_logprobs(monkeypatch):
    args = SimpleNamespace(
        search_url="http://127.0.0.1:8000/retrieve",
        max_turns=3,
        search_topk=3,
        search_concurrency=16,
        search_timeout=60,
        search_max_attempts=3,
        format_score=0.0,
    )
    monkeypatch.delenv("SEARCH_R1_RETURN_LOGPROB", raising=False)

    _configure_search_environment(args)

    assert measure_search_r1.os.environ["SEARCH_R1_RETURN_LOGPROB"] == "0"
