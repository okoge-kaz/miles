import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.src.difficulty_filtering.core.artifacts import sha256_file, write_json
from experiments.src.difficulty_filtering.core.config import FilterConfig
from experiments.src.difficulty_filtering.core.records import build_record, summarize
from experiments.src.difficulty_filtering.utils.smoke_gate import verify_output, verify_smokes


class SmokeGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.part = self.root / "measurement/part-000"
        self.part.mkdir(parents=True)
        self.config = FilterConfig(
            model_path="/model",
            prompt_data="/data/source.jsonl",
            measurement_dir="/data/measurement",
            filtered_output="/data/output.jsonl",
            policy="test",
            audit_prompts=8,
        )
        self.rows = [{"prompt": f"question-{index}", "label": "1"} for index in range(8)]
        (self.root / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.code = self.run / "source/measure.py"
        self.code.parent.mkdir()
        self.code.write_text("original\n")
        write_json(self.run / "config.json", asdict(self.config))
        write_json(self.run / "launch.json", {"smoke_prompts": 8})
        write_json(
            self.part / "manifest.json",
            {
                "measurement": {
                    "config": self.config.measurement_settings(),
                    "source_sha256": {"measure.py": sha256_file(self.code)},
                }
            },
        )
        records = [
            build_record(index, row, [0.0] * 8 + [1.0] * 8, [2] * 16, [False] * 16, self.config)
            for index, row in enumerate(self.rows)
        ]
        (self.part / "pass_rates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        audits = [{"index": index, "responses": [f"answer-{i}" for i in range(16)]} for index in range(8)]
        (self.part / "audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in audits))
        self.summary = summarize(records, 16)
        write_json(self.part / "summary.json", self.summary)
        write_json(self.part / "progress.json", self.summary)
        self.prepared = {"directory": self.run}

    def test_completed_smoke_requires_valid_records_source_and_audit(self):
        receipt = verify_output(self.prepared, self.root)
        self.assertEqual(receipt["summary"]["total"], 8)
        self.assertEqual(list(receipt["unique_responses"].values()), [16] * 8)
        self.code.write_text("modified\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            verify_output(self.prepared, self.root)

    def test_infrastructure_failure_is_not_a_passed_smoke(self):
        self.summary["failure_kinds"] = {"sandbox_start_failed": 1}
        write_json(self.part / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "infrastructure failure"):
            verify_output(self.prepared, self.root)

    def test_missing_audit_cannot_open_production_gate(self):
        (self.part / "audit.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "audit coverage"):
            verify_output(self.prepared, self.root)

    def test_running_or_failed_pbs_job_cannot_open_production_gate(self):
        records = [{"model": "4b", "dataset": "code", "action": "smoke", "job_id": "1.pbs"}]
        for status in [{"job_state": "R"}, {"job_state": "F", "Exit_status": 1}]:
            with patch(
                "experiments.src.difficulty_filtering.utils.smoke_gate.subprocess.run",
                return_value=SimpleNamespace(stdout=json.dumps({"Jobs": {"1.pbs": status}})),
            ):
                with self.assertRaisesRegex(ValueError, "not completed successfully"):
                    verify_smokes({("4b", "code"): self.prepared}, records, self.root)
