import json
from types import SimpleNamespace

from experiments.tools.difficulty_filter import apply_filter


def test_load_records_accepts_search_r1_operational_fields(tmp_path):
    path = tmp_path / "search.passrate.jsonl"
    path.write_text(
        json.dumps(
            {
                "index": 7,
                "pass_rate": 0.375,
                "n_correct": 3,
                "n_samples": 8,
                "response_len_mean": 42,
                "truncated_frac": 0.0,
                "label": ["Paris"],
                "rewards": [1, 1, 1, 0, 0, 0, 0, 0],
                "observation_len_mean": 128,
                "search_calls_mean": 1.5,
                "statuses": ["completed"] * 8,
            }
        )
        + "\n"
    )

    records = apply_filter.load_records(path)

    assert records[7].pass_rate == 0.375
    assert records[7].n_samples == 8


def test_iter_prompt_rows_reads_parquet_batches(monkeypatch):
    rows = [{"prompt": "a"}, {"prompt": "b"}]

    class FakeBatch:
        def to_pylist(self):
            return rows

    class FakeParquetFile:
        def __init__(self, path):
            assert path == "train.parquet"

        def iter_batches(self):
            return iter([FakeBatch()])

    monkeypatch.setattr(apply_filter, "pq", SimpleNamespace(ParquetFile=FakeParquetFile))

    assert list(apply_filter.iter_prompt_rows("train.parquet")) == rows
