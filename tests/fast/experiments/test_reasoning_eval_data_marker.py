from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tools.reasoning_eval.prepared_data import (
    format_dataset_environment,
    validate_prepared_data,
)


BENCHMARKS = ("aime24", "aime25", "aime26")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_dataset(data_root: Path, benchmark: str) -> None:
    path = data_root / benchmark / "test.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {
            "id": f"{benchmark}-{index}",
            "problem": f"problem {index}",
            "expected_answer": str(index),
        }
        for index in range(30)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_all_datasets(data_root: Path) -> None:
    for benchmark in BENCHMARKS:
        _write_dataset(data_root, benchmark)


def _write_marker(path: Path, *, image: Path, benchmarks: str, extra: str = "") -> None:
    path.write_text(
        f"nemo_skills_image={image}\nbenchmarks={benchmarks}\n{extra}",
        encoding="utf-8",
    )


def test_superset_marker_is_accepted_after_required_data_validation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image = tmp_path / "nemo-skills.sif"
    marker = data_root / "_PREPARED"
    _write_all_datasets(data_root)
    _write_marker(
        marker,
        image=image,
        benchmarks="aime24 aime25 aime26 gpqa livecodebench",
    )

    datasets = validate_prepared_data(
        data_root=data_root,
        image=image,
        benchmarks=BENCHMARKS,
        marker=marker,
    )

    assert [dataset.benchmark for dataset in datasets] == list(BENCHMARKS)
    assert all(dataset.records == 30 for dataset in datasets)
    assert all(len(dataset.sha256) == 64 for dataset in datasets)


def test_marker_must_contain_every_required_benchmark(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image = tmp_path / "nemo-skills.sif"
    marker = data_root / "_PREPARED"
    _write_all_datasets(data_root)
    _write_marker(marker, image=image, benchmarks="aime24 aime25")

    with pytest.raises(ValueError, match="missing benchmarks: aime26"):
        validate_prepared_data(
            data_root=data_root,
            image=image,
            benchmarks=BENCHMARKS,
            marker=marker,
        )


def test_recorded_dataset_checksum_detects_mutation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image = tmp_path / "nemo-skills.sif"
    marker = data_root / "_PREPARED"
    _write_all_datasets(data_root)
    datasets = validate_prepared_data(
        data_root=data_root,
        image=image,
        benchmarks=BENCHMARKS,
    )
    _write_marker(
        marker,
        image=image,
        benchmarks=" ".join(BENCHMARKS),
        extra=format_dataset_environment(datasets) + "\n",
    )
    aime24_path = data_root / "aime24" / "test.jsonl"
    aime24_path.write_text(
        aime24_path.read_text(encoding="utf-8").replace("problem 0", "mutated problem", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum mismatch for aime24"):
        validate_prepared_data(
            data_root=data_root,
            image=image,
            benchmarks=BENCHMARKS,
            marker=marker,
        )


def test_marker_rejects_different_evaluator_image(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    marker = data_root / "_PREPARED"
    _write_all_datasets(data_root)
    _write_marker(
        marker,
        image=tmp_path / "old-image.sif",
        benchmarks=" ".join(BENCHMARKS),
    )

    with pytest.raises(ValueError, match="different NeMo Skills image"):
        validate_prepared_data(
            data_root=data_root,
            image=tmp_path / "new-image.sif",
            benchmarks=BENCHMARKS,
            marker=marker,
        )


def test_aime_setup_jobs_use_pbs_cpu_queue_without_project() -> None:
    scripts = REPO_ROOT / "experiments/scripts/reasoning_eval"
    for name in ("import-evaluator-images.sbatch", "prepare-aime-data.sbatch"):
        text = (scripts / name).read_text(encoding="utf-8")
        assert "#PBS -q R9920261300" in text
        assert "#PBS -P" not in text
