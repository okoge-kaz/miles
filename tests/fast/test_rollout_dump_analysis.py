import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DECODE = _load_module(
    "reasoning_eval_decode_rollout_dump_test_module",
    "experiments/tools/reasoning_eval/decode_rollout_dump.py",
)
TRACE = _load_module(
    "reasoning_eval_trace_rollout_text_test_module",
    "experiments/tools/reasoning_eval/trace_rollout_text.py",
)


def _sample(
    *,
    response: str,
    response_tokens: list[int],
    status: str,
    reward: float,
    index: int,
) -> dict:
    return {
        "index": index,
        "group_index": index,
        "prompt": "Solve the problem.",
        "label": "42",
        "tokens": [101, 102, *response_tokens],
        "response_length": len(response_tokens),
        "response": response,
        "status": status,
        "reward": reward,
        "first_prefill_weight_versions": [4],
    }


def test_decode_rollout_dump_splits_zero_reward_outcomes_and_zero_length(tmp_path: Path):
    source = tmp_path / "7.pt"
    torch.save(
        {
            "rollout_id": 7,
            "metadata": {"dynamic_global_batch_size": 3},
            "samples": [
                _sample(response="correct", response_tokens=[11], status="completed", reward=1.0, index=0),
                _sample(response="cut off", response_tokens=[21, 22], status="truncated", reward=0.0, index=1),
                _sample(response="", response_tokens=[], status="completed", reward=0.0, index=2),
            ],
        },
        source,
    )
    output_dir = tmp_path / "decoded"
    output_dir.mkdir()
    args = SimpleNamespace(
        output_dir=output_dir,
        tag="s20",
        include_prompt_token_ids=False,
        include_logprobs=False,
        examples_per_category=4,
        markdown_response_chars=12_000,
    )

    summary = DECODE.decode_file(source, args=args, tokenizer=None)

    assert summary["training_step"] == 8
    assert summary["metadata"] == {"dynamic_global_batch_size": 3}
    assert summary["outcome_counts"] == {
        "reward positive": 1,
        "reward zero (truncated)": 1,
        "reward zero (not truncated)": 1,
    }
    rows = [
        json.loads(line)
        for line in (output_dir / "s20-rollout-7.samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[2]["response_token_ids"] == []
    assert rows[2]["prompt_length"] == 2
    assert rows[2]["rollout_id"] == 7
    markdown = (output_dir / "s20-rollout-7.examples.md").read_text(encoding="utf-8")
    assert "## reward zero (truncated)" in markdown
    assert "## reward zero (not truncated)" in markdown


def test_trace_rollout_text_separates_strict_and_audit_guess_candidates(tmp_path: Path):
    source = tmp_path / "12.pt"
    torch.save(
        {
            "rollout_id": 12,
            "samples": [
                _sample(
                    response="I will guess 42.",
                    response_tokens=[1, 2, 3],
                    status="completed",
                    reward=1.0,
                    index=0,
                ),
                _sample(
                    response="We can guess a value after deriving the bound.",
                    response_tokens=[4, 5, 6],
                    status="completed",
                    reward=0.0,
                    index=1,
                ),
                _sample(
                    response="The answer is 7.",
                    response_tokens=[7, 8],
                    status="completed",
                    reward=0.0,
                    index=2,
                ),
                _sample(
                    response="Given time low. I choose 3.",
                    response_tokens=[9, 10],
                    status="completed",
                    reward=0.0,
                    index=3,
                ),
            ],
        },
        source,
    )

    row, examples = TRACE._scan_file(source, examples_per_category=0)

    assert row["training_step"] == 13
    assert row["explicit_guess"] == 1
    assert row["explicit_guess_reward_positive"] == 1
    assert row["self_guess_mention"] == 2
    assert row["guess_audit_candidate"] == 3
    assert row["guess_audit_candidate_reward_positive"] == 1
    assert row["short_nonmatch_review"] == 1
    example_categories = {example["category"] for example in examples}
    assert "guess_audit_candidate" in example_categories
    assert "short_nonmatch_review" in example_categories
