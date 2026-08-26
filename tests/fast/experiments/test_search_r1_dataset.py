from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.src.datasets.search_r1.build_eval import (
    MANIFEST_NAME,
    PROMPT_TEMPLATE,
    SPLITS,
    build_datasets,
    convert_row,
    validate_datasets,
)


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


def _write_source_rows(root: Path) -> None:
    for index, relative in enumerate(SPLITS.values()):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps(
                {
                    "id": f"row-{index}",
                    "question": f"question {index}",
                    "golden_answers": [f"answer {index}"],
                }
            )
            + "\n",
            encoding="utf-8",
        )


def test_search_converter_publishes_a_deterministic_validated_manifest(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_source_rows(source_root)

    first = build_datasets(source_root, output_root, limit=500)
    first_manifest = (output_root / MANIFEST_NAME).read_bytes()
    second = build_datasets(source_root, output_root, limit=500)

    assert first == second == validate_datasets(output_root)
    assert (output_root / MANIFEST_NAME).read_bytes() == first_manifest


def test_search_manifest_rejects_a_stale_or_modified_output(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_source_rows(source_root)
    build_datasets(source_root, output_root, limit=500)
    (output_root / "nq-miles.jsonl").write_text("stale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_datasets(output_root)


def test_search_converter_checks_every_source_before_publication(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_source_rows(source_root)
    build_datasets(source_root, output_root, limit=500)
    manifest_before = (output_root / MANIFEST_NAME).read_bytes()
    (source_root / SPLITS["bamboogle"]).unlink()

    with pytest.raises(FileNotFoundError, match="bamboogle"):
        build_datasets(source_root, output_root, limit=500)

    assert (output_root / MANIFEST_NAME).read_bytes() == manifest_before
    validate_datasets(output_root)
