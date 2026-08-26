from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.tools.reasoning_eval.export_miles_aime import (
    BENCHMARK_OUTPUTS,
    EXPECTED_RECORDS,
    export_all,
    export_benchmark,
)
from miles.utils.arguments import _resolve_eval_datasets


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_aime_eval_config_uses_canonical_artifacts_and_math_reward() -> None:
    args = SimpleNamespace(
        eval_config=str(REPO_ROOT / "experiments/configs/eval_aime.yaml"),
        eval_prompt_data=None,
    )
    datasets = _resolve_eval_datasets(args)

    assert [dataset.name for dataset in datasets] == ["aime24", "aime25", "aime26"]
    assert all(dataset.n_samples_per_eval_prompt == 16 for dataset in datasets)
    assert all(dataset.max_response_len == 16384 for dataset in datasets)
    assert all(dataset.rm_type == "math" for dataset in datasets)
    assert [dataset.path for dataset in datasets] == [
        "/data/aime-2024/aime-2024.jsonl",
        "/data/aime-2025/aime-2025.jsonl",
        "/data/aime-2026/aime-2026.jsonl",
    ]


def _write_source(source_root: Path, benchmark: str, *, duplicate_last_id: bool = False) -> None:
    source = source_root / benchmark / "test.jsonl"
    source.parent.mkdir(parents=True)
    rows = []
    for index in range(EXPECTED_RECORDS):
        record_id = f"{benchmark}-{0 if duplicate_last_id and index == EXPECTED_RECORDS - 1 else index}"
        rows.append(
            {
                "id": record_id,
                "problem": f"problem {index}",
                "expected_answer": str(index),
                "reference_solution": f"private solution {index}",
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_export_all_writes_canonical_miles_rows_and_provenance(tmp_path: Path) -> None:
    source_root = tmp_path / "nemo-skills"
    output_root = tmp_path / "datasets"
    for benchmark in BENCHMARK_OUTPUTS:
        _write_source(source_root, benchmark)

    exports = export_all(source_root=source_root, output_root=output_root)

    assert [info.benchmark for info in exports] == list(BENCHMARK_OUTPUTS)
    for info in exports:
        rows = _read_jsonl(info.output)
        assert len(rows) == EXPECTED_RECORDS
        assert rows[0]["prompt"][0]["role"] == "user"
        assert "problem 0" in rows[0]["prompt"][0]["content"]
        assert "\\boxed" in rows[0]["prompt"][0]["content"]
        assert rows[0]["label"] == "0"
        assert rows[0]["metadata"] == {
            "benchmark": info.benchmark,
            "eval_only": True,
            "record_id": f"{info.benchmark}-0",
            "rm_type": "math",
            "source": "nemo-skills-26.03",
            "verifier": "math",
        }
        assert "private solution" not in info.output.read_text(encoding="utf-8")

        provenance_path = info.output.with_name(f"{info.output.name}.provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert provenance["records"] == EXPECTED_RECORDS
        assert provenance["source_sha256"] == info.source_sha256
        assert provenance["output_sha256"] == info.output_sha256


def test_invalid_source_does_not_replace_an_existing_export(tmp_path: Path) -> None:
    source_root = tmp_path / "nemo-skills"
    output_root = tmp_path / "datasets"
    benchmark = "aime24"
    _write_source(source_root, benchmark, duplicate_last_id=True)
    output = output_root / BENCHMARK_OUTPUTS[benchmark]
    output.parent.mkdir(parents=True)
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id"):
        export_benchmark(source_root=source_root, output_root=output_root, benchmark=benchmark)

    assert output.read_text(encoding="utf-8") == "existing\n"
    assert list(output.parent.glob("*.partial-*")) == []


def test_unknown_benchmark_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported benchmark"):
        export_benchmark(
            source_root=tmp_path / "source",
            output_root=tmp_path / "output",
            benchmark="aime27",
        )


def test_prepare_job_exports_cached_and_new_prepared_data() -> None:
    prepare_job = (
        REPO_ROOT / "experiments/scripts/reasoning_eval/prepare-aime-data.sbatch"
    ).read_text(encoding="utf-8")

    assert "experiments/tools/reasoning_eval/export_miles_aime.py" in prepare_job
    assert 'MILES_AIME_DATA_ROOT="${MILES_AIME_DATA_ROOT:-${DATASET_DIR}}"' in prepare_job
    assert prepare_job.count("export_miles_aime_data\n") == 2
