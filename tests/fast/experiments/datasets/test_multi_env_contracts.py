import json
import pytest

from experiments.src.datasets.instruction_following.prepare import adapt_if_training
from experiments.src.nemo_gym.runtime.serve import model_config


def test_if_training_conversion_preserves_all_constraints_and_prompt():
    constraints = {
        "instruction_id": ["keywords:existence", "length_constraints:number_words"],
        "kwargs": [{"keywords": ["orchid"]}, {"num_words": 10, "relation": "at least"}],
    }
    messages = [{"role": "user", "content": "Write at least ten words including orchid."}]
    converted = adapt_if_training({"ground_truth": repr([constraints]), "messages": messages, "key": "train-1"})
    assert converted["prompt"] == messages
    assert json.loads(converted["label"]) == constraints
    assert converted["metadata"]["instruction_id_list"] == constraints["instruction_id"]
    assert converted["metadata"]["kwargs"] == constraints["kwargs"]
    assert converted["metadata"]["prompt_text"] == messages[-1]["content"]


@pytest.mark.parametrize(
    "label,options",
    [("B", [{"A": "first\\nB: second", "B": None}]), ("A", [{"A": "only one choice"}]), ("Z", [{"A": "a", "B": "b"}])],
)
def test_mcqa_conversion_excludes_inconsistent_source_labels(label, options):
    from experiments.src.datasets.source_adapters import adapt_knowledge_mcqa

    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "Choose an answer."}]},
        "expected_answer": label,
        "options": options,
    }
    assert adapt_knowledge_mcqa(row) is None
    valid = adapt_knowledge_mcqa({**row, "expected_answer": "B", "options": [{"A": "a", "B": "b"}]})
    assert valid["label"] == "B" and valid["metadata"]["valid_letters"] == ["A", "B"]


def test_fixed_model_connection_uses_native_config_and_keeps_keys_private(tmp_path):
    path = tmp_path / "user-model.json"
    path.write_text(
        json.dumps(
            {
                "openai_model": "qwen3-4b-sft-user-iter4000",
                "openai_base_url": "http://localhost:30180/v1",
                "openai_api_key": "test-only-secret",
            }
        )
    )
    path.chmod(0o600)
    parameters = model_config(str(path))
    assert parameters["openai_model"] == "qwen3-4b-sft-user-iter4000"
    assert parameters["openai_base_url"] == "http://localhost:30180/v1"
    assert parameters["openai_api_key"] == "test-only-secret"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        model_config(str(path))


@pytest.mark.parametrize(
    "tests",
    [
        {"entry_point": "solve", "test_code": "assert solve() == 1"},
        {"inputs": ["1"], "outputs": []},
        {"inputs": [[1]], "outputs": ["1"]},
    ],
)
def test_code_conversion_rejects_tests_that_nemo_cannot_execute(tests):
    from experiments.src.datasets.source_adapters import adapt_competitive_coding

    prompt = [{"role": "user", "content": "Solve this problem."}]
    native = {"responses_create_params": {"input": prompt}, "verifier_metadata": {"unit_tests": tests}}
    with pytest.raises(ValueError):
        adapt_competitive_coding(native)


@pytest.mark.parametrize("num_proc", [1, 2])
def test_failed_if_conversion_does_not_publish_partial_training_data(tmp_path, num_proc):
    from experiments.src.datasets.common.prepare import convert

    source, destination = tmp_path / "source.jsonl", tmp_path / "training.jsonl"
    row = {
        "key": "train-1",
        "messages": [{"role": "user", "content": "Write orchid."}],
        "ground_truth": repr([{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["orchid"]}]}]),
    }
    source.write_text(json.dumps(row) + "\n" + json.dumps({**row, "messages": []}) + "\n")
    with pytest.raises(ValueError, match="must end with a user message"):
        convert(source, destination, adapt_if_training, num_proc=num_proc, chunk_size=1)
    assert not destination.exists()
    source.write_text(json.dumps(row) + "\n")
    manifest = convert(source, destination, adapt_if_training)
    assert manifest["rows"] == 1 and manifest == convert(source, destination, adapt_if_training)
    # Already prepared IF data predates the optional adapter identity.
    legacy_manifest = {key: value for key, value in manifest.items() if key != "adapter"}
    destination.with_suffix(".manifest.json").write_text(json.dumps(legacy_manifest))
    assert convert(source, destination, adapt_if_training) == legacy_manifest
    destination.write_text("corrupt\n")
    with pytest.raises(ValueError, match="does not match"):
        convert(source, destination, adapt_if_training)
