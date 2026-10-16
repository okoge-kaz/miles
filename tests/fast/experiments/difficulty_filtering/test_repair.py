import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.src.difficulty_filtering.utils.repair_suite import (
    build_repair,
    logical_key,
    retire_waiting,
    submit_repair,
)


def record(job_id, action, shard, lane, dataset="dapo-math"):
    return dict(
        job_id=job_id,
        model="qwen3-4b",
        dataset=dataset,
        action=action,
        shard=shard,
        lane=lane,
        dependencies=[],
        success_dependencies=[],
        completion_dependencies=[],
    )


def job(state, exit_status=None, hold=""):
    return dict(job_state=state, Exit_status=exit_status, Hold_Types=hold, Resource_List={"nodect": 1})


class RepairTest(unittest.TestCase):
    def setUp(self):
        self.original = {
            "datasets": ["dapo-math", "nemotron-code"],
            "verified_smokes": ["smoke"],
            "jobs": [
                record("smoke", "smoke", 0, 0),
                record("failed", "filter", 0, 0),
                record("running", "filter", 1, 0),
                record("complete", "filter", 2, 1),
                record("cancelled-merge", "merge", 0, 0),
                record("cancelled-code", "filter", 0, 0, "nemotron-code"),
                record("held-code", "filter", 1, 1, "nemotron-code"),
                record("held-merge", "merge", 0, 1, "nemotron-code"),
            ],
        }
        self.jobs = {
            "smoke": job("F", 0),
            "failed": job("F", 1),
            "running": job("R"),
            "complete": job("F", 0),
            "cancelled-merge": job("F"),
            "cancelled-code": job("F"),
            "held-code": job("H", hold="us"),
            "held-merge": job("H", hold="us"),
        }

    def test_retry_waits_for_current_lane_and_merge_waits_for_retry(self):
        state = build_repair(self.original, self.jobs, "original-run")
        self.assertEqual({r["job_id"] for r in state["jobs"]}, {"smoke", "running", "complete"})
        self.assertEqual(set(state["retire"]), {"held-code", "held-merge"})
        retry, merge, code, _, code_merge = state["pending"]
        self.assertEqual(retry["dependencies"], ["running"])
        self.assertEqual(set(merge["dependencies"]), {"running", retry["alias"]})
        self.assertIn(merge["alias"], code["dependencies"])
        self.assertIn(code["alias"], code_merge["dependencies"])
        history = [{anchor["job_id"]} if anchor["job_id"] else set() for anchor in state["anchors"]]
        for row in state["pending"]:
            self.assertEqual(row["success_dependencies"], [])
            self.assertTrue(history[row["lane"]] <= set(row["dependencies"]))
            self.assertFalse({"failed", "cancelled-merge", "cancelled-code"} & set(row["dependencies"]))
            history[row["lane"]].add(row["alias"])
        self.assertEqual(
            {logical_key(row) for row in self.original["jobs"]},
            {logical_key(row) for row in state["jobs"] + state["pending"]},
        )

    def test_multiple_active_jobs_in_one_lane_are_rejected(self):
        self.jobs["failed"] = job("R")
        with self.assertRaisesRegex(ValueError, "at most one"):
            build_repair(self.original, self.jobs, "original-run")

    def test_unverified_smoke_prevents_repair(self):
        self.original["verified_smokes"] = []
        with self.assertRaisesRegex(ValueError, "validated smoke"):
            build_repair(self.original, self.jobs, "original-run")

    def test_retirement_never_deletes_running_or_unheld_jobs(self):
        for current in [job("R"), job("Q"), job("H", hold="s")]:
            with self.subTest(current=current), tempfile.TemporaryDirectory() as tmp:
                state = {"retire": ["candidate"]}
                with patch(
                    "experiments.src.difficulty_filtering.utils.repair_suite.read_jobs",
                    return_value={"candidate": current},
                ):
                    with patch("experiments.src.difficulty_filtering.utils.repair_suite.subprocess.run") as run:
                        with self.assertRaisesRegex(ValueError, "never delete"):
                            retire_waiting(state, Path(tmp) / "plan.json")
                        run.assert_not_called()

    def test_submission_reuses_snapshot_and_resume_does_not_duplicate_jobs(self):
        state = build_repair(self.original, self.jobs, "original-run")
        state["phase"] = "submitting"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "experiments.src.difficulty_filtering.utils.repair_suite.prepare_recipe",
                return_value={"directory": root},
            ) as prepare:
                with patch(
                    "experiments.src.difficulty_filtering.utils.repair_suite.enqueue",
                    side_effect=[f"new-{i}" for i in range(5)],
                ) as enqueue:
                    submit_repair(root, state, root / "plan.json")
                    submit_repair(root, state, root / "plan.json")
            self.assertEqual(enqueue.call_count, 5)
            self.assertEqual(prepare.call_count, 2)
            for call in prepare.call_args_list:
                self.assertTrue(call.kwargs["resume"])
            for row in state["jobs"]:
                self.assertFalse(any(dep.startswith("repair-") for dep in row["dependencies"]))
            self.assertEqual(state["phase"], "submitted")
            self.assertEqual(len(state["jobs"]), len(self.original["jobs"]))
