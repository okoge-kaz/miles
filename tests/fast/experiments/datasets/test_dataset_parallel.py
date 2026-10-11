import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from experiments.src.datasets.common.io import write_jsonl
from experiments.src.datasets.common.parallel import map_rows
from experiments.src.datasets.common.prepare import convert
from experiments.src.datasets.instruction_following.prepare import adapt_if_training


def _if_row(index):
    return {
        "key": f"train-{index}",
        "messages": [{"role": "user", "content": f"Question 日本語 {index}"}],
        "ground_truth": repr([{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["orchid"]}]}]),
    }


def _worker_identity(index):
    time.sleep(0.1 if index == 0 else 0.005)
    return index, os.getpid()


def _run(module, *args, check=True):
    return subprocess.run(
        [sys.executable, "-m", f"experiments.src.datasets.{module}", *map(str, args)],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=60,
        check=check,
    )


def test_parallel_map_runs_in_workers_and_preserves_order():
    before = {child.pid for child in multiprocessing.active_children()}
    with map_rows(range(12), _worker_identity, num_proc=2, chunk_size=1) as rows:
        results = list(rows)
    assert [index for index, _ in results] == list(range(12))
    assert all(pid != os.getpid() for _, pid in results)
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_parallel_map_bounds_prefetch_and_closes_workers_on_early_exit():
    consumed = []

    def source():
        for index in range(10_000):
            consumed.append(index)
            yield index

    before = {child.pid for child in multiprocessing.active_children()}
    with map_rows(source(), int, num_proc=2, chunk_size=3) as rows:
        assert next(rows) == 0
        assert len(consumed) <= 12
    assert {child.pid for child in multiprocessing.active_children()} == before


@pytest.mark.parametrize("num_proc", [1, 2])
@pytest.mark.parametrize("failure", ["reader", "adapter"])
def test_errors_follow_the_valid_prefix(num_proc, failure):
    def source():
        yield "1"
        yield "2"
        if failure == "reader":
            raise ValueError("invalid source row")
        yield "invalid integer"

    with map_rows(source(), int, num_proc=num_proc, chunk_size=3) as rows:
        assert next(rows) == 1
        assert next(rows) == 2
        with pytest.raises(ValueError, match="invalid"):
            next(rows)


def test_parallel_if_output_and_manifest_match_serial(tmp_path):
    source = tmp_path / "input.jsonl"
    write_jsonl(source, (_if_row(index) for index in range(37)))
    serial, parallel = tmp_path / "serial.jsonl", tmp_path / "parallel.jsonl"
    expected = convert(source, serial, adapt_if_training)
    actual = convert(source, parallel, adapt_if_training, num_proc=2, chunk_size=3)
    assert serial.read_bytes() == parallel.read_bytes()
    assert actual == expected
    assert convert(source, parallel, adapt_if_training, num_proc=1) == expected


@pytest.mark.parametrize("existing", [False, True])
def test_failed_parallel_write_does_not_publish_or_overwrite_output(tmp_path, existing):
    output = tmp_path / "output.jsonl"
    if existing:
        output.write_text("previous valid output\n")
    with pytest.raises(ValueError):
        with map_rows(["1", "bad"], int, num_proc=2, chunk_size=1) as rows:
            write_jsonl(output, rows)
    if existing:
        assert output.read_text() == "previous valid output\n"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob("output.jsonl.*"))


@pytest.mark.parametrize(
    "module,dataset",
    [("instruction_following.prepare", None), ("convert", "competitive-coding"), ("convert", "knowledge-mcqa")],
)
def test_conversion_clis_match_serial_output(tmp_path, module, dataset):
    if dataset is None:
        row = _if_row(0)
    else:
        row = {"responses_create_params": {"input": [{"role": "user", "content": "Question 日本語"}]}}
        if dataset == "competitive-coding":
            row["verifier_metadata"] = {"unit_tests": {"inputs": ["1"], "outputs": ["1"]}}
        else:
            row["expected_answer"] = "A"
            row["options"] = [{"A": "correct", "B": "incorrect"}]
    source = tmp_path / "source.jsonl"
    write_jsonl(source, ({**row, "uuid": str(index)} for index in range(9)))
    outputs = []
    for num_proc in [1, 2]:
        destination = tmp_path / f"out-{num_proc}.jsonl"
        extra = ["--dataset", dataset] if dataset else []
        _run(module, "--input", source, "--output", destination, *extra, "--num-proc", num_proc, "--chunk-size", 2)
        outputs.append(destination.read_bytes())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "bad_row",
    [
        "not json\n",
        json.dumps(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "bad"}]},
                "verifier_metadata": {"unit_tests": {"inputs": ["1"], "outputs": []}},
            }
        )
        + "\n",
    ],
)
def test_code_limit_ignores_errors_after_the_requested_prefix(tmp_path, bad_row):
    source = tmp_path / "source.jsonl"
    valid = {
        "responses_create_params": {"input": [{"role": "user", "content": "good"}]},
        "verifier_metadata": {"unit_tests": {"inputs": ["1"], "outputs": ["1"]}},
    }
    # The first row is skipped; --limit counts converted rows, not source rows.
    source.write_text("{}\n" + json.dumps(valid) + "\n" + bad_row)
    outputs = []
    for num_proc in [1, 2]:
        destination = tmp_path / f"out-{num_proc}.jsonl"
        summary = json.loads(
            _run(
                "convert",
                "--dataset",
                "competitive-coding",
                "--input",
                source,
                "--output",
                destination,
                "--num-proc",
                num_proc,
                "--chunk-size",
                8,
                "--limit",
                1,
            ).stdout
        )
        assert summary["kept"] == 1 and summary["skipped"] == 1
        outputs.append(destination.read_bytes())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("num_proc", [0, -1])
def test_cli_rejects_invalid_worker_count_before_creating_output(tmp_path, num_proc):
    output = tmp_path / "output.jsonl"
    result = _run(
        "instruction_following.prepare",
        "--input",
        tmp_path / "missing",
        "--output",
        output,
        "--num-proc",
        num_proc,
        check=False,
    )
    assert result.returncode != 0 and "positive integer" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("num_proc", [1, 2])
def test_tau_split_validation_is_preserved_across_workers(num_proc):
    from experiments.src.datasets.tau2.prepare import _adapt_split_task, _assign_splits

    native = {
        "config": {"domain": "airline"},
        "responses_create_params": {"input": [{"role": "user", "content": "Book a flight"}]},
    }
    source = [{**native, "task": {"id": value}} for value in ("train-1", "test-1", "train-1")]
    splits = {"airline": {"train": {"train-1"}, "test": {"test-1"}}}
    with map_rows(_assign_splits(source, splits), _adapt_split_task, num_proc=num_proc, chunk_size=2) as rows:
        split, row = next(rows)
        assert split == "train" and row["metadata"]["eval_only"] is False
        split, row = next(rows)
        assert split == "test" and row["metadata"]["eval_only"] is True
        with pytest.raises(ValueError, match="Duplicate task"):
            next(rows)
