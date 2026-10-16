import contextlib
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from experiments.src.difficulty_filtering.utils.submit_suite import DATASETS, MODELS, Slots, schedule


class SuiteTest(unittest.TestCase):
    def test_all_models_share_sixteen_chains_including_cpu_merge(self):
        prepared = {}
        for dataset in DATASETS:
            for model in MODELS:
                if dataset == "dapo-math" and model != "qwen3-4b":
                    continue
                prepared[model, dataset] = {"config": {"num_shards": 64 if dataset == "nemotron-mcqa" else 16}}
        anchors = [{"job_id": f"old-{i}", "estimate": i / 10} for i in range(16)]
        # Old math merges own slots 0 and 8. No new job can occupy those slots first.
        anchors[0]["job_id"] = "merge-8b"
        anchors[8]["job_id"] = "merge-30b"
        state = {"anchors": anchors, "math_gates": {"qwen3-8b": "merge-8b", "qwen3-30b-a3b": "merge-30b"}, "jobs": []}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            schedule(root, prepared, state, root / "plan.json", submit=False)
            last = [anchor["job_id"] for anchor in anchors]
            history = [{job} for job in last]
            ids = set(last)
            lanes_4b = set()
            for record in state["jobs"]:
                lane = record["lane"]
                self.assertIn(last[lane], record["dependencies"])
                self.assertTrue(history[lane] <= set(record["dependencies"]))
                if last[lane] not in record["success_dependencies"]:
                    self.assertIn(last[lane], record["completion_dependencies"])
                self.assertEqual(record["success_dependencies"], [])
                self.assertTrue(set(record["dependencies"]) <= ids)
                last[lane] = record["job_id"]
                history[lane].add(record["job_id"])
                ids.add(record["job_id"])
                if record["model"] == "qwen3-4b":
                    lanes_4b.add(lane)
            self.assertGreater(len(lanes_4b), 8)
            self.assertEqual(sum(record["action"] == "merge" for record in state["jobs"]), 10)
            self.assertEqual(set(state["final_merges"]), set(MODELS))
            previous = deepcopy(state)
            schedule(root, prepared, state, root / "plan.json", submit=False)
            self.assertEqual(state, previous)

    def test_cuda_failure_and_incomplete_merge_do_not_cancel_successors(self):
        prepared = {
            (model, dataset): {"config": {"num_shards": 16}}
            for model in MODELS
            for dataset in ("nemotron-code", "instruction-following", "nemotron-mcqa")
        }
        state = {"anchors": [{"job_id": None, "estimate": 0}] * 16, "math_gates": {}, "jobs": []}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            schedule(Path(temporary), prepared, state, Path(temporary) / "plan.json", submit=False)
        finished, failed, cancelled = set(), set(), set()
        shards = {}
        for row in state["jobs"]:
            key = row["model"], row["dataset"]
            # PBS also propagates a predecessor's deletion before execution
            # through afterany. A nonzero exit after execution does not do so.
            if set(row["success_dependencies"]) & (failed | cancelled) or set(row["dependencies"]) & cancelled:
                cancelled.add(row["job_id"])
            self.assertTrue(set(row["dependencies"]) <= finished)
            if row["action"] == "filter":
                shards.setdefault(key, set()).add(row["job_id"])
                if key == ("qwen3-8b", "nemotron-code") and row["shard"] == 4:
                    failed.add(row["job_id"])
            if row["action"] == "merge":
                self.assertTrue(shards[key] <= set(row["completion_dependencies"]))
                if shards[key] & failed:
                    failed.add(row["job_id"])
            finished.add(row["job_id"])
        self.assertEqual(len(failed), 2)
        self.assertEqual(cancelled, set())
        self.assertEqual(len(finished), len(state["jobs"]))

    def test_cancelled_held_job_cannot_bypass_running_lane_predecessor(self):
        slots = Slots([{"job_id": None, "estimate": 0}] * 16)
        slots.assign("running", [], 10, lane=0)
        slots.assign("cancelled-held", ["failed-smoke"], 1, lane=0)
        _, success, completion = slots.assign("next", [], 1, lane=0)
        terminal = {"failed-smoke", "cancelled-held"}
        self.assertFalse(set(success + completion) <= terminal)
        terminal.add("running")
        self.assertTrue(set(success + completion) <= terminal)

    def test_smoke_stage_queues_no_production_or_merge_jobs(self):
        prepared = {(model, dataset): {"config": {"num_shards": 64}} for model in MODELS for dataset in DATASETS}
        state = {"anchors": [{"job_id": None, "estimate": 0}] * 16, "math_gates": {}, "jobs": []}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            path = Path(temporary) / "plan.json"
            schedule(Path(temporary), prepared, state, path, submit=False, stage="smoke")
            self.assertEqual(len(state["jobs"]), 12)
            self.assertEqual({row["action"] for row in state["jobs"]}, {"smoke"})
            with self.assertRaisesRegex(ValueError, "every smoke"):
                schedule(Path(temporary), prepared, state, path, submit=False, stage="full")
            state["verified_smokes"] = [row["job_id"] for row in state["jobs"]]
            schedule(Path(temporary), prepared, state, path, submit=False, stage="full")
            for row in state["jobs"]:
                self.assertFalse(set(row["dependencies"]) & set(state["verified_smokes"]))
            previous = deepcopy(state)
            schedule(Path(temporary), prepared, state, path, submit=False, stage="full")
            self.assertEqual(state, previous)
