import json

import pytest

from experiments.tools.difficulty_filter.reconcile_pass_rate_fragments import meta_path, reconcile


def record(index, reward):
    return {
        "index": index,
        "pass_rate": float(reward),
        "n_correct": int(reward),
        "n_samples": 1,
        "response_len_mean": 1.0,
        "truncated_frac": 0.0,
        "label": str(index),
        "rewards": [float(reward)],
    }


def write_fragment(path, rows, start, end, policy="test"):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metadata = {
        "policy": policy,
        "n_samples": 1,
        "start_index": start,
        "end_index": end,
    }
    with open(meta_path(path), "w") as destination:
        json.dump(metadata, destination)


def indices(path):
    return [json.loads(line)["index"] for line in path.read_text().splitlines()]


def test_reconcile_prefers_primary_and_only_marks_exact_coverage(tmp_path):
    primary = tmp_path / "primary.jsonl"
    tail = tmp_path / "tail.jsonl"
    gap = tmp_path / "gap.jsonl"
    write_fragment(primary, [record(3, 0), record(0, 1), record(1, 1)], 0, 5)
    write_fragment(tail, [record(4, 1), record(3, 1)], 3, 5)

    assert not reconcile(primary, [tail], mark_complete=True)
    assert indices(primary) == [0, 1, 3, 4]
    assert json.loads(primary.read_text().splitlines()[2])["rewards"] == [0.0]
    assert not (tmp_path / "primary.jsonl.complete").exists()

    write_fragment(gap, [record(2, 1)], 2, 3)
    assert reconcile(primary, [tail, gap], mark_complete=True)
    assert indices(primary) == [0, 1, 2, 3, 4]
    assert (tmp_path / "primary.jsonl.measured").exists()
    assert (tmp_path / "primary.jsonl.complete").exists()


def test_reconcile_rejects_different_measurement_metadata(tmp_path):
    primary = tmp_path / "primary.jsonl"
    fragment = tmp_path / "fragment.jsonl"
    write_fragment(primary, [record(0, 1)], 0, 2)
    write_fragment(fragment, [record(1, 1)], 1, 2, policy="other")

    with pytest.raises(SystemExit, match="measurement metadata differs"):
        reconcile(primary, [fragment])


def test_reconcile_rejects_malformed_non_trailing_record(tmp_path):
    primary = tmp_path / "primary.jsonl"
    fragment = tmp_path / "fragment.jsonl"
    write_fragment(primary, [record(0, 1)], 0, 2)
    fragment.write_text("{bad json\n" + json.dumps(record(1, 1)) + "\n")
    with open(meta_path(fragment), "w") as destination:
        json.dump({"policy": "test", "n_samples": 1, "start_index": 1, "end_index": 2}, destination)

    with pytest.raises(SystemExit, match="malformed non-trailing"):
        reconcile(primary, [fragment])
