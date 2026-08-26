from __future__ import annotations

import subprocess
import sys

from experiments.src.datasets.search_r1.build_eval import PROMPT_TEMPLATE, SPLITS, convert_row


def test_search_conversion_matches_training_prompt_and_reward_schema():
    row = {
        "id": "nq-1",
        "question": "Who won?",
        "golden_answers": ["Alice", "A. Example"],
    }
    first = convert_row(row)
    second = convert_row(row)

    assert first == second
    assert first == {
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question="Who won?")}],
        "reward_model": {
            "ground_truth": {"target": ["Alice", "A. Example"]},
            "style": "rule",
        },
        "metadata": {"source": "nq-1", "question": "Who won?"},
    }


def test_search_conversion_normalizes_single_answer_and_rejects_missing_fields():
    converted = convert_row({"question": "Where?", "golden_answers": "Paris"})
    assert converted is not None
    assert converted["reward_model"]["ground_truth"]["target"] == ["Paris"]
    assert convert_row({"question": "", "golden_answers": ["Paris"]}) is None
    assert convert_row({"question": "Where?", "golden_answers": []}) is None


def test_search_converter_covers_all_reported_benchmarks_without_rl_imports():
    assert set(SPLITS) == {
        "nq",
        "hotpotqa",
        "triviaqa",
        "popqa",
        "2wikimultihopqa",
        "musique",
        "bamboogle",
    }
    program = """
import sys
import experiments.src.datasets.search_r1.build_eval
assert not any(name == 'miles' or name.startswith('miles.') for name in sys.modules)
assert 'experiments.src.environments.search_r1.retrieval_server' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True)
