import json
import math

from experiments.src.offline_eval.measure_search_r1 import EpisodeResult
from experiments.src.offline_eval.measure_search_r1 import _done_indices
from experiments.src.offline_eval.measure_search_r1 import build_record
from experiments.src.offline_eval.measure_search_r1 import protocol_tokenization
from experiments.src.offline_eval.report_search_r1 import summarise


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


def test_report_weights_operational_metrics_by_trajectory_count():
    first = build_record(
        0,
        _row(),
        [EpisodeResult(1, "completed", "<answer>Paris</answer>", 4, 0, 1, 0)],
        correct_threshold=0.5,
    )
    second = build_record(
        1,
        _row(),
        [
            EpisodeResult(0, "completed", "<search>a</search>", 8, 10, 3, 3),
            EpisodeResult(0, "completed", "<search>b</search>", 8, 10, 3, 3),
            EpisodeResult(0, "completed", "<search>c</search>", 8, 10, 3, 3),
        ],
        correct_threshold=0.5,
    )

    summary = summarise([first, second])

    assert summary["accuracy"] == 0.5
    assert summary["episodes"] == 4
    assert math.isclose(summary["search_calls"], 2.25)
    assert math.isclose(summary["turns"], 2.5)


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
    assert result["<search>"]["token_ids"] == [ord(character) for character in "<search>"]


def test_resume_ignores_a_partial_trailing_record(tmp_path):
    output = tmp_path / "nq.jsonl"
    output.write_text(json.dumps({"index": 2}) + "\n{" + "\n")

    assert _done_indices(str(output)) == {2}
