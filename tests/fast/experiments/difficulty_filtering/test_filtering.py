import asyncio
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from experiments.src.difficulty_filtering.core.apply_filter import apply_filter
from experiments.src.difficulty_filtering.core.artifacts import (
    build_manifest,
    checkpoint_hashes,
    ensure_manifest,
    file_lock,
    read_records,
    repair_trailing_record,
    sha256_file,
    write_json,
)
from experiments.src.difficulty_filtering.core.config import FilterConfig
from experiments.src.difficulty_filtering.core.measure import measure_one, parse_outputs
from experiments.src.difficulty_filtering.core.records import build_record, keep_record, read_prompts, shard_indices, summarize
from experiments.src.difficulty_filtering.utils.preflight import validate_source
from experiments.src.difficulty_filtering.utils.progress import inspect_measurement


class FilteringTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = [
            {"prompt": [{"role": "user", "content": f"Question {i}"}], "label": "7", "metadata": {"original_index": i}}
            for i in range(5)
        ]
        self.source = self.root / "source.jsonl"
        self.source.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.config = FilterConfig(
            model_path=str(self.root / "model"),
            prompt_data=str(self.source),
            measurement_dir=str(self.root / "measurement"),
            filtered_output=str(self.root / "output.jsonl"),
            policy="test",
            num_shards=2,
        )
        self.records = [self.record(index, correct) for index, correct in enumerate([0, 1, 8, 15, 16])]

    def record(self, index, correct):
        return build_record(
            index, self.rows[index], [1.0] * correct + [0.0] * (16 - correct), [10] * 16, [False] * 16, self.config
        )

    def make_measurements(self):
        measurement = {
            "config": self.config.measurement_settings(),
            "input_sha256": sha256_file(self.source),
            "checkpoint_sha256": {"model.safetensors": "fixed"},
            "source_sha256": {"measure.py": "fixed"},
        }
        for index in range(self.config.num_shards):
            directory = self.root / "measurement" / f"part-{index:03d}"
            directory.mkdir(parents=True)
            assigned = shard_indices(len(self.rows), self.config.num_shards, index)
            write_json(
                directory / "manifest.json",
                {"measurement": measurement, "shard_index": index, "start": assigned.start, "stop": assigned.stop},
            )
            (directory / "pass_rates.jsonl").write_text("".join(json.dumps(self.records[i]) + "\n" for i in assigned))

    def test_boundaries_and_source_preservation(self):
        self.assertEqual([keep_record(record) for record in self.records], [False, True, True, True, False])
        self.make_measurements()
        summary = apply_filter(self.config)
        self.assertEqual(summary["kept_indices"], [1, 2, 3])
        selected = Path(self.config.filtered_output).read_text().splitlines()
        self.assertEqual(selected, self.source.read_text().splitlines()[1:4])
        self.assertEqual(apply_filter(self.config), summary)

    def test_manifest_covers_core_and_utils_sources(self):
        model = Path(self.config.model_path)
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"test weights")
        with patch("experiments.src.difficulty_filtering.core.artifacts.importlib.metadata.version", return_value="test"):
            manifest = build_manifest(self.config)
        root = Path(__file__).resolve().parents[4]
        package = root / "experiments/src/difficulty_filtering"
        expected = {str(path.relative_to(root)) for path in package.rglob("*.py")}
        sources = manifest["source_sha256"]
        actual = {name for name in sources if name.startswith("experiments/src/difficulty_filtering/")}
        self.assertEqual(actual, expected)
        self.assertIn("experiments/src/difficulty_filtering/core/measure.py", actual)
        self.assertIn("experiments/src/difficulty_filtering/utils/progress.py", actual)
        self.assertIn("release-lock.json", sources)
        self.assertIn("miles/rollout/rm_hub/deepscaler.py", sources)
        for relative, digest in sources.items():
            self.assertEqual(digest, sha256_file(root / relative))

    def test_unicode_line_separators_are_string_contents_through_merge(self):
        for row in self.rows:
            row["prompt"][0]["content"] += " code\u0085next\u2028paragraph\u2029end"
        lines = [(json.dumps(row, ensure_ascii=False) + "\r\n").encode() for row in self.rows]
        self.source.write_bytes(b"".join(lines))
        self.assertEqual(read_prompts(self.source), self.rows)
        self.records = [self.record(index, correct) for index, correct in enumerate([0, 1, 8, 15, 16])]
        self.make_measurements()
        summary = apply_filter(self.config)
        self.assertEqual(summary["kept_indices"], [1, 2, 3])
        self.assertEqual(Path(self.config.filtered_output).read_bytes(), b"".join(lines[1:4]))

    def test_preflight_rejects_bad_later_row_before_job_submission(self):
        self.source.write_text(self.source.read_text() + '{"prompt":"broken\n')
        with self.assertRaisesRegex(ValueError, r"source.jsonl:6"):
            validate_source(self.source, "deepscaler")

    def test_preflight_validates_native_metadata_for_every_row(self):
        for row in self.rows:
            row["metadata"] = {"verifier": "python_code", "unit_tests": {"inputs": ["2"], "outputs": ["4"]}}
        self.rows[-1]["metadata"]["unit_tests"]["outputs"] = []
        self.source.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        with self.assertRaisesRegex(ValueError, r"source.jsonl:5"):
            validate_source(self.source, "code_gen")

    def test_cached_validation_requires_identical_input_bytes(self):
        receipt = validate_source(self.source, "deepscaler")
        with patch("experiments.src.difficulty_filtering.utils.preflight.iter_prompts", side_effect=AssertionError):
            self.assertEqual(validate_source(self.source, "deepscaler", [receipt]), receipt)
        self.source.write_text(self.source.read_text().replace("Question 0", "Question 9"))
        with self.assertRaisesRegex(ValueError, "differs from the full validation"):
            validate_source(self.source, "deepscaler", [receipt])

    def test_partial_shard_never_publishes_dataset(self):
        self.make_measurements()
        path = self.root / "measurement/part-001/pass_rates.jsonl"
        path.write_text(json.dumps(self.records[3]) + "\n")
        with self.assertRaisesRegex(ValueError, "Incomplete shard"):
            apply_filter(self.config)
        self.assertFalse(Path(self.config.filtered_output).exists())

    def test_mixed_checkpoint_shards_are_rejected(self):
        self.make_measurements()
        path = self.root / "measurement/part-001/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["measurement"]["checkpoint_sha256"]["model.safetensors"] = "different"
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            apply_filter(self.config)

    def test_changed_source_or_settings_are_rejected(self):
        self.make_measurements()
        with self.assertRaisesRegex(ValueError, "Recipe differs"):
            apply_filter(replace(self.config, temperature=0.8))
        self.source.write_text(self.source.read_text().replace("Question 0", "Different question"))
        with self.assertRaisesRegex(ValueError, "source or sample count"):
            apply_filter(self.config)

    def test_existing_output_is_not_overwritten(self):
        self.make_measurements()
        output = Path(self.config.filtered_output)
        output.write_text("unrelated dataset\n")
        with self.assertRaises(FileExistsError):
            apply_filter(self.config)
        self.assertEqual(output.read_text(), "unrelated dataset\n")

    def test_duplicate_and_incomplete_groups_are_rejected(self):
        path = self.root / "records.jsonl"
        line = json.dumps(self.records[0]) + "\n"
        path.write_text(line + line)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_records(path, self.rows, self.config, range(5))
        record = {**self.records[0], "rewards": [0.0] * 15}
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "Incomplete response group"):
            read_records(path, self.rows, self.config, range(5))

    def test_only_incomplete_tail_is_repaired(self):
        path = self.root / "records.jsonl"
        first = json.dumps(self.records[0]) + "\n"
        path.write_text(first + '{"index":')
        repair_trailing_record(path)
        self.assertEqual(path.read_text(), first)
        path.write_text(first.rstrip())
        repair_trailing_record(path)
        self.assertEqual(path.read_text(), first)
        path.write_text("bad json\n" + first)
        repair_trailing_record(path)
        with self.assertRaises(json.JSONDecodeError):
            read_records(path, self.rows, self.config, range(5))

    def test_concurrent_writer_is_rejected(self):
        with file_lock(self.root / ".lock"):
            with self.assertRaisesRegex(RuntimeError, "Another job"):
                with file_lock(self.root / ".lock"):
                    pass

    def test_resume_manifest_and_model_content_identity(self):
        model = Path(self.config.model_path)
        model.mkdir()
        (model / "config.json").write_text("{}")
        weights = model / "model.safetensors"
        weights.write_bytes(b"original")
        first = checkpoint_hashes(model)
        weights.write_bytes(b"modified")
        self.assertNotEqual(first, checkpoint_hashes(model))
        path = self.root / "manifest.json"
        ensure_manifest(path, first)
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            ensure_manifest(path, checkpoint_hashes(model))

    def test_partition_coverage(self):
        for count, sizes in ((4, [4350, 4350, 4349, 4349]), (8, [2175] * 6 + [2174] * 2)):
            with self.subTest(shards=count):
                parts = [list(shard_indices(17398, count, index)) for index in range(count)]
                self.assertEqual([len(part) for part in parts], sizes)
                self.assertEqual([index for part in parts for index in part], list(range(17398)))

    def test_fractional_if_selection_and_metrics(self):
        config = replace(
            self.config,
            reward_backend="instruction_following",
            selection_rule="reward_variance",
            correct_threshold=1.0,
        )
        records = [
            build_record(i, self.rows[i], rewards, [10] * 16, [False] * 16, config)
            for i, rewards in enumerate([[0.0] * 16, [1.0] * 16, [0.5] * 16, [0.0] * 8 + [0.5] * 8])
        ]
        self.assertEqual([keep_record(record) for record in records], [False, False, False, True])
        result = summarize(records, 16)
        self.assertEqual(
            (result["all_zero"], result["all_one"], result["constant_partial"], result["kept"]), (1, 1, 1, 1)
        )
        self.assertEqual(result["mean_pass_rate"], 0.25)
        self.assertEqual(result["mean_reward"], 0.4375)

    def test_unscored_is_excluded_and_never_counted_as_wrong(self):
        config = replace(self.config, reward_backend="code_gen")
        diagnostics = [{"failure_kind": "execution_time_limit"}] + [{}] * 15
        record = build_record(0, self.rows[0], [None] + [0.0] * 15, [10] * 16, [False] * 16, config, diagnostics)
        self.assertFalse(keep_record(record))
        result = summarize([record], 16)
        self.assertIsNone(record["n_correct"])
        self.assertIsNone(result["mean_pass_rate"])
        self.assertEqual((result["unscored_groups"], result["unscored_samples"], result["all_zero"]), (1, 1, 0))
        self.assertEqual(result["failure_kinds"], {"execution_time_limit": 1})
        with self.assertRaisesRegex(ValueError, "Binary verifier"):
            build_record(0, self.rows[0], [0.5] * 16, [10] * 16, [False] * 16, config)

    def test_live_progress_ignores_only_partial_tail_without_mutating_journal(self):
        self.make_measurements()
        path = self.root / "measurement/part-001/pass_rates.jsonl"
        path.write_text(path.read_text() + '{"index":')
        before = path.read_bytes()
        result = inspect_measurement(self.root / "measurement")
        self.assertEqual((result["total"], result["all_zero"], result["all_one"]), (5, 1, 1))
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_config(self):
        for setting in (
            {"n_samples": 1},
            {"max_new_tokens": 40000},
            {"temperature": float("nan")},
            {"num_engines": 0},
            {"num_shards": 0},
        ):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                FilterConfig(**{**asdict(self.config), **setting})

    def test_http_groups_must_be_complete_and_successful(self):
        output = {"text": "answer", "meta_info": {"finish_reason": {"type": "length"}, "completion_tokens": 5}}
        self.assertEqual(parse_outputs([output], 1), (["answer"], [5], [True]))
        with self.assertRaisesRegex(ValueError, "Incomplete SGLang group"):
            parse_outputs([output], 2)
        output["meta_info"]["finish_reason"] = {"type": "abort"}
        with self.assertRaisesRegex(ValueError, "Unsuccessful"):
            parse_outputs([output], 1)

    def test_single_response_requests_and_context_cap(self):
        tokenizer = unittest.mock.Mock()
        tokenizer.apply_chat_template.return_value = "rendered"
        tokenizer.encode.return_value = [1] * (32768 - 123)
        output = {
            "text": "<think>work</think>\\boxed{7}",
            "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 10},
        }
        generate = AsyncMock(return_value=output)
        score = AsyncMock(return_value=[1.0] + [0.0] * 15)
        with patch("experiments.src.difficulty_filtering.core.measure.generate_request", generate), patch(
            "experiments.src.difficulty_filtering.core.measure.score_responses", score
        ):
            record, _ = asyncio.run(measure_one(None, ["http://test"], self.config, tokenizer, 0, self.rows[0]))
        self.assertEqual(generate.await_count, 16)
        self.assertEqual([call.args[3]["max_new_tokens"] for call in generate.await_args_list], [123] * 16)
        self.assertEqual(record["n_correct"], 1)

    def test_no_context_budget_is_all_truncated(self):
        tokenizer = unittest.mock.Mock()
        tokenizer.apply_chat_template.return_value = "rendered"
        tokenizer.encode.return_value = [1] * 32768
        generate = AsyncMock()
        score = AsyncMock(return_value=[0.0] * 16)
        with patch("experiments.src.difficulty_filtering.core.measure.generate_request", generate), patch(
            "experiments.src.difficulty_filtering.core.measure.score_responses", score
        ):
            record, _ = asyncio.run(measure_one(None, ["http://test"], self.config, tokenizer, 0, self.rows[0]))
        generate.assert_not_awaited()
        self.assertEqual(record["truncated_frac"], 1.0)
        self.assertFalse(keep_record(record))


if __name__ == "__main__":
    unittest.main()
