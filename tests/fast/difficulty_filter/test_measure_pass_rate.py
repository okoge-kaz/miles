import json

from experiments.src.difficulty_filter.measure_pass_rate import (
    cap_max_new_tokens,
    load_prompts,
    repair_trailing_record,
    sample_request_sizes,
)


def test_sample_request_sizes_defaults_to_one_request():
    assert sample_request_sizes(16) == [16]


def test_sample_request_sizes_splits_group_and_preserves_remainder():
    assert sample_request_sizes(16, 8) == [8, 8]
    assert sample_request_sizes(18, 8) == [8, 8, 2]


def test_cap_max_new_tokens_reserves_context_for_prompt():
    assert cap_max_new_tokens(32768, 32768, 195) == 32573
    assert cap_max_new_tokens(16384, 32768, 195) == 16384
    assert cap_max_new_tokens(32768, None, 195) == 32768


def test_load_prompts_preserves_source_indices_for_contiguous_range(tmp_path):
    prompt_data = tmp_path / "prompts.jsonl"
    prompt_data.write_text(
        "".join(json.dumps({"prompt": f"p{i}", "label": str(i)}) + "\n" for i in range(6))
    )

    rows = load_prompts(prompt_data, "prompt", "label", start_index=2, end_index=5)

    assert [row["index"] for row in rows] == [2, 3, 4]
    assert [row["prompt"] for row in rows] == ["p2", "p3", "p4"]


def test_repair_trailing_record_adds_newline_to_valid_json(tmp_path):
    output = tmp_path / "measurements.jsonl"
    output.write_text('{"index": 1}')

    assert repair_trailing_record(output) == "added missing newline"
    assert output.read_text() == '{"index": 1}\n'


def test_repair_trailing_record_discards_only_malformed_tail(tmp_path):
    output = tmp_path / "measurements.jsonl"
    output.write_bytes(b'{"index": 1}\n{"index":')

    assert repair_trailing_record(output) == "truncated malformed tail"
    assert output.read_bytes() == b'{"index": 1}\n'
