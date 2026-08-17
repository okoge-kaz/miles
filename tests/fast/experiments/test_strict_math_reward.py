import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.rm_hub import batched_async_rm
from miles.rollout.recycle_compute_metrics import SELECTION_METRICS_KEY
from miles.utils.types import Sample

from experiments.src.staleness_ratio.strict_math_reward import (
    eos_tokens_from_config,
    has_strict_terminal_answer,
    resolve_eos_tokens,
    score_strict_math_sample,
    strict_math_reward,
    strict_terminal_answer_error,
)


def _sample(response, label="42", status="completed"):
    return SimpleNamespace(response=response, label=label, status=status)


@pytest.mark.parametrize(
    "response",
    [
        "Reasoning.\n\nAnswer: \\boxed{42}",
        "Reasoning.\nAnswer:\\boxed{\\frac{1}{2}}   \n",
        r"Answer: \boxed{42}",
        r"Answer: $\boxed{42}$",
        r"Answer: $ \boxed{42} $",
        r"Answer: \boxed{42} ✅",
        "Answer: \\boxed{42}.\n",
    ],
)
def test_accepts_one_terminal_answer(response):
    assert has_strict_terminal_answer(response)


@pytest.mark.parametrize(
    "response",
    [
        "Derive an intermediate value: \\boxed{40}.\nAnswer: \\boxed{42}",
        "The calculation gives \\boxed{42}.\nAnswer: \\boxed{42}",
        "Use \\boxed{x=1} and \\boxed{y=2}.\nAnswer: \\boxed{3}",
    ],
)
def test_accepts_reasoning_boxes_before_one_terminal_answer(response):
    assert has_strict_terminal_answer(response)


@pytest.mark.parametrize(
    "response",
    [
        "Answer: \\boxed{42}\nAnswer: \\boxed{42}",
        "Answer: \\boxed{42}\nTherefore 42.",
        r"Answer: \boxed{42},,,,,,,,,",
        r"Answer: \boxed{42} ✅ ✅",
        r"Answer: \boxed{42}<model_eos>more",
        "Answer: \\boxed{42}\nAnswer:",
        r"The Answer: \boxed{42}",
        "Answer:\n\\boxed{42}",
        r"Answer: \boxed{42",
    ],
)
def test_rejects_non_terminal_or_repeated_answers(response):
    assert not has_strict_terminal_answer(response)


def test_accepts_only_configured_terminal_eos_tokens():
    response = r"Answer: \boxed{42}<model_eos>"
    assert not has_strict_terminal_answer(response)
    assert has_strict_terminal_answer(response, ("<model_eos>",))
    assert has_strict_terminal_answer(
        r"Answer: \boxed{42}<first_eos><model_eos>",
        ("<first_eos>", "<model_eos>"),
    )


def test_reads_multiple_eos_tokens_and_ids_from_configs(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        '{"eos_token":"<eos_a>","added_tokens_decoder":'
        '{"10":{"content":"<eos_a>"},"11":{"content":"<eos_b>"}}}'
    )
    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id":[10,11]}'
    )
    assert eos_tokens_from_config(tmp_path) == ("<eos_a>", "<eos_b>")
    assert has_strict_terminal_answer(
        r"Answer: \boxed{42}<eos_b>", eos_tokens_from_config(tmp_path)
    )


def test_resolves_multiple_eos_tokens_from_tokenizer_object():
    tokenizer = SimpleNamespace(
        eos_token=("<eos_a>", "<eos_b>"),
        eos_token_id=(10, 11),
        convert_ids_to_tokens=lambda ids: ["<eos_a>", "<eos_b>"],
    )
    assert resolve_eos_tokens(tokenizer=tokenizer) == ("<eos_a>", "<eos_b>")


def test_prompt_0005_shape_is_rejected_for_two_answer_markers():
    response = (
        "Answer: $a+b+c=1$\n\n"
        "But let me verify that result.\n\n"
        "Answer: \\boxed{1}"
    )
    assert strict_terminal_answer_error(response) == "multiple_answer_markers"


def test_scores_format_and_correctness_together():
    assert score_strict_math_sample(_sample("Work.\nAnswer: \\boxed{42}")) == 1.0
    assert score_strict_math_sample(_sample("Work.\nAnswer: \\boxed{41}")) == 0.0
    assert score_strict_math_sample(_sample("Answer: \\boxed{42}\nAnswer: \\boxed{42}")) == 0.0
    assert (
        score_strict_math_sample(_sample("Work: \\boxed{40}.\nAnswer: \\boxed{42}"))
        == 1.0
    )


def test_records_math_reward_and_format_diagnostics():
    repeated = _sample("Answer: \\boxed{42}\nAnswer: \\boxed{42}")

    assert score_strict_math_sample(repeated) == 0.0
    assert repeated.metadata[SELECTION_METRICS_KEY] == {
        "strict_math/math_reward": 1.0,
        "strict_math/strict_reward": 0.0,
        "strict_math/format_valid": 0.0,
        "strict_math/multiple_answer_markers": 1.0,
        "strict_math/reward_disagreement": 1.0,
    }


def test_truncated_math_correct_response_is_logged_but_not_rewarded():
    sample = _sample(r"Answer: \boxed{42}", status="truncated")

    assert score_strict_math_sample(sample) == 0.0
    assert sample.metadata[SELECTION_METRICS_KEY]["strict_math/math_reward"] == 1.0
    assert sample.metadata[SELECTION_METRICS_KEY]["strict_math/format_valid"] == 0.0
    assert sample.metadata[SELECTION_METRICS_KEY]["strict_math/reward_disagreement"] == 1.0


@pytest.mark.parametrize("status", ["truncated", "aborted", "failed"])
def test_rejects_incomplete_sample_status(status):
    assert score_strict_math_sample(_sample(r"Answer: \boxed{42}", status=status)) == 0.0


def test_accepts_enum_like_completed_status():
    status = SimpleNamespace(value="completed")
    assert score_strict_math_sample(_sample(r"Answer: \boxed{42}", status=status)) == 1.0


def test_custom_rm_supports_single_and_batch_contracts(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text('{"eos_token":"<eos>"}')
    args = SimpleNamespace(hf_checkpoint=str(tmp_path))
    good = _sample(r"Answer: \boxed{42}")
    bad = _sample(r"Answer: \boxed{41}")
    good_with_eos = _sample(r"Answer: \boxed{42}<eos>")
    assert asyncio.run(strict_math_reward(args, good)) == 1.0
    assert asyncio.run(strict_math_reward(args, good_with_eos)) == 1.0
    assert asyncio.run(strict_math_reward(args, [good, bad])) == [1.0, 0.0]


def test_custom_rm_path_dispatches_through_miles():
    args = SimpleNamespace(
        custom_rm_path="experiments.src.staleness_ratio.strict_math_reward.strict_math_reward",
        rm_type="math",
        multi_lora=False,
    )
    samples = [
        Sample(response=r"Answer: \boxed{42}", label="42", status=Sample.Status.COMPLETED),
        Sample(
            response="Answer: \\boxed{42}\nAnswer: \\boxed{42}",
            label="42",
            status=Sample.Status.COMPLETED,
        ),
    ]
    assert asyncio.run(batched_async_rm(args, samples)) == [1.0, 0.0]
