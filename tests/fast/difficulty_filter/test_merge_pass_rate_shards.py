import json

import pytest

from experiments.tools.difficulty_filter.merge_pass_rate_shards import merge, meta_path


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_shard(path, indices, start, end):
    write_jsonl(path, [{"index": index} for index in indices])
    metadata = {"policy": "test", "n_samples": 16, "start_index": start, "end_index": end}
    with open(meta_path(path), "w") as destination:
        json.dump(metadata, destination)


def test_merge_validates_coverage_and_sorts_global_indices(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    write_jsonl(prompts, [{"prompt": str(index)} for index in range(4)])
    shard_0 = tmp_path / "shard0.jsonl"
    shard_1 = tmp_path / "shard1.jsonl"
    write_shard(shard_0, [1, 0], 0, 2)
    write_shard(shard_1, [3, 2], 2, 4)
    output = tmp_path / "merged.jsonl"

    merge([shard_0, shard_1], output, prompts)

    assert [json.loads(line)["index"] for line in output.read_text().splitlines()] == [0, 1, 2, 3]
    with open(meta_path(output)) as source:
        assert json.load(source)["end_index"] == 4


def test_merge_rejects_missing_measurement(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    write_jsonl(prompts, [{"prompt": str(index)} for index in range(4)])
    shard_0 = tmp_path / "shard0.jsonl"
    shard_1 = tmp_path / "shard1.jsonl"
    write_shard(shard_0, [0], 0, 2)
    write_shard(shard_1, [2, 3], 2, 4)

    with pytest.raises(SystemExit, match="incomplete shard merge"):
        merge([shard_0, shard_1], tmp_path / "merged.jsonl", prompts)
