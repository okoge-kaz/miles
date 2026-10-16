import json
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "experiments/scripts/difficulty_filtering/qwen3-8b-sft/dapo-math.sh"


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_name = "test-filter-" + uuid.uuid4().hex
        self.run_dir = ROOT / "experiments/outputs/difficulty_filtering" / self.run_name
        self.addCleanup(shutil.rmtree, self.run_dir, ignore_errors=True)
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "FILTER_RUN_NAME": self.run_name,
            "PBS_PROJECT": "test-project",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def run_recipe(self, *args):
        return subprocess.run(
            ["bash", str(RECIPE), *args], cwd=ROOT, env=self.env, text=True, capture_output=True, check=True
        )

    def test_preview_checkpoint_resources_and_allowlist(self):
        self.env["WANDB_API_KEY"] = "must-not-appear"
        result = self.run_recipe("filter")
        self.assertIn("MINLR1.5e-6", result.stdout)
        self.assertIn("TP2-PP1-CP2-EP1", result.stdout)
        self.assertEqual(result.stdout.count("ngpus=8"), 8)
        self.assertIn("afterany:999900:999901:999902:999903:999904:999905:999906:999907", result.stdout)
        self.assertIn("run.sh", result.stdout)
        self.assertNotIn(" -V", result.stdout)
        self.assertNotIn("WANDB_API_KEY", result.stdout)
        self.assertNotIn("must-not-appear", result.stdout)
        self.assertFalse((self.run_dir / "config.json").exists())

    def test_smoke_is_one_job(self):
        result = self.run_recipe("smoke")
        self.assertEqual(result.stdout.count("ngpus=8"), 1)
        self.assertIn("smoke 0 32", result.stdout)
        self.assertNotIn("afterok:", result.stdout)

    def test_one_independent_shard_and_later_merge(self):
        result = self.run_recipe("filter", "--shard-index", "2")
        self.assertEqual(result.stdout.count("ngpus=8"), 1)
        self.assertIn("filter 2 32", result.stdout)
        self.assertNotIn("afterok:", result.stdout)
        result = self.run_recipe("merge")
        self.assertNotIn("ngpus=8", result.stdout)
        self.assertIn("select=1:ncpus=32:mpiprocs=1", result.stdout)
        self.assertIn("place=scatter ", result.stdout)
        self.assertNotIn("place=scatter:excl", result.stdout)
        self.assertIn("merge.sh", result.stdout)

    def test_all_shards_without_automatic_merge(self):
        result = self.run_recipe("filter", "--no-merge")
        self.assertEqual(result.stdout.count("ngpus=8"), 8)
        self.assertNotIn("afterok:", result.stdout)
        self.assertNotIn("select=1:ncpus=32:mpiprocs=1", result.stdout)

    def test_submission_snapshots_config_and_chains_merge(self):
        checkpoint = self.root / "hf/test-model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        dataset = self.root / "data/dapo-math-17k"
        dataset.mkdir(parents=True)
        (dataset / "dapo-math-17k.jsonl").write_text('{"prompt":"q","label":"7"}\n')
        image = self.root / "image.sif"
        image.touch()
        calls = self.root / "calls.jsonl"
        qsub = self.root / "qsub"
        qsub.write_text(
            "#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n"
            f"path = Path({str(calls)!r})\n"
            "lines = path.read_text().splitlines() if path.exists() else []\n"
            "with path.open('a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "print(str(1001 + len(lines)) + '.pbs')\n"
        )
        qsub.chmod(0o755)
        self.env.update(
            PBS_QSUB_BIN=str(qsub),
            HF_CKPT_DIR=str(self.root / "hf"),
            FILTER_HF_MODEL="test-model",
            RL_DATASET_DIR=str(self.root / "data"),
            CONTAINER_IMAGE=str(image),
        )
        # Preview may create the log directory; it must not consume the run name.
        self.run_recipe("filter")
        self.run_recipe("filter", "--submit")
        jobs = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(len(jobs), 9)
        self.assertIn(
            "depend=afterany:1001.pbs:1002.pbs:1003.pbs:1004.pbs:1005.pbs:1006.pbs:1007.pbs:1008.pbs", jobs[-1]
        )
        self.assertIn("select=1:ncpus=32:mpiprocs=1", jobs[-1])
        for job in jobs[:-1]:
            self.assertIn(str(self.run_dir / "run.sh"), job)
        self.assertIn(str(self.run_dir / "merge.sh"), jobs[-1])
        self.assertEqual(
            (self.run_dir / "run.sh").read_text(),
            (ROOT / "experiments/scripts/difficulty_filtering/run.sh").read_text(),
        )
        config = json.loads((self.run_dir / "config.json").read_text())
        self.assertEqual(config["model_path"], "/ckpt/hf/test-model")
        self.assertEqual(config["n_samples"], 16)
        self.assertEqual(config["num_shards"], 8)
        self.assertTrue(config["measurement_dir"].endswith("-8parts"))
        self.assertEqual(len((self.run_dir / "jobs.tsv").read_text().splitlines()), 9)
        self.assertTrue((self.run_dir / "source/experiments/src/nemo_gym/runtime/sources.lock.json").is_file())
        self.assertEqual(
            (self.run_dir / "source/experiments/src/difficulty_filtering/core/measure.py").read_bytes(),
            (ROOT / "experiments/src/difficulty_filtering/core/measure.py").read_bytes(),
        )

    def test_native_mcqa_64_shards_are_limited_to_eight_lanes(self):
        recipe = RECIPE.with_name("nemotron-mcqa.sh")
        result = subprocess.run(
            ["bash", str(recipe), "filter", "--dependency", "afterok:123.pbs"],
            cwd=ROOT,
            env=self.env,
            text=True,
            capture_output=True,
            check=True,
        )
        config, _ = json.JSONDecoder().raw_decode(result.stdout)
        self.assertEqual((config["reward_backend"], config["num_shards"]), ("mcqa", 64))
        jobs = [line for line in result.stdout.splitlines() if "ngpus=8" in line]
        self.assertEqual(len(jobs), 64)
        for i, line in enumerate(jobs):
            expected = "afterok:123.pbs" if i < 8 else f"afterany:{999900 + i - 8} "
            self.assertIn(expected, line)
            self.assertIn("afterok:123.pbs", line)

    def test_if_uses_native_fraction_with_training_source(self):
        result = subprocess.run(
            ["bash", str(RECIPE.with_name("instruction-following.sh")), "smoke"],
            cwd=ROOT,
            env=self.env,
            text=True,
            capture_output=True,
            check=True,
        )
        config, _ = json.JSONDecoder().raw_decode(result.stdout)
        self.assertEqual(config["selection_rule"], "reward_variance")
        self.assertEqual(config["correct_threshold"], 1.0)
        self.assertEqual(config["prompt_data"], "/data/multi_env/if-training.jsonl")

    def test_30b_checkpoint_and_parallelism(self):
        recipe = ROOT / "experiments/scripts/difficulty_filtering/qwen3-30b-a3b-sft/dapo-math.sh"
        result = subprocess.run(
            ["bash", str(recipe), "smoke"], cwd=ROOT, env=self.env, text=True, capture_output=True, check=True
        )
        config, _ = json.JSONDecoder().raw_decode(result.stdout)
        self.assertIn("TP4-PP2-CP2-EP4", config["model_path"])
        self.assertIn("MINLR2.0e-6", config["model_path"])
        self.assertEqual((config["tensor_parallel_size"], config["num_engines"]), (2, 4))
        self.assertEqual(config["concurrency"], 80)
        self.assertEqual(config["n_samples"], 16)
        self.assertEqual(config["num_shards"], 8)
        self.assertTrue(config["measurement_dir"].endswith("-8parts"))
        self.assertIn("qwen3-30b-a3b-sft-iter4000", config["measurement_dir"])


if __name__ == "__main__":
    unittest.main()
