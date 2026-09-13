import json
import math
import os
import subprocess
import sys

import pytest

from experiments.staleness import lightning_node_ratio as sweep
from experiments.tools.summarize_lightning_node_ratio import read_steps, summarize_arm


def test_all_requested_points_have_a_common_valid_batch(monkeypatch):
    monkeypatch.setenv("GLOBAL_BATCH_SIZE", "256")
    monkeypatch.setenv("MAX_WEIGHT_STALENESS", "99")
    plan = sweep.make_plan("test", list(sweep.POINTS), {})
    dps = []
    for arm in plan["arms"]:
        stale, train, rollout = arm["point"]
        env = arm["env"]
        dp = train * 4 // 2
        dps.append(dp)
        assert train + rollout == 32
        assert int(env["MAX_WEIGHT_STALENESS"]) == stale
        assert int(env["ROLLOUT_NUM_GPUS"]) == rollout * 4
        assert int(env["GLOBAL_BATCH_SIZE"]) == 3360
        assert 3360 % dp == 0
        assert int(env["ROLLOUT_BATCH_SIZE"]) * int(env["N_SAMPLES_PER_PROMPT"]) == 3360
        assert env["NUM_STEPS_PER_ROLLOUT"] == "1"
        assert env["NUM_ROLLOUT"] == "300"
        assert env["DEBUG_EXIT_AFTER_ROLLOUT"] == ""
        assert env["TRAINING_BUFFER_QUEUE_SIZE"] == "6000"
        assert env["ASYNC_MAX_CONCURRENT_SAMPLES"] == str(112 * arm["serving_profile"][0])
        assert env["MODEL_NAME"] == "Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000"
        assert env["SAVE_INTERVAL"] == env["HF_SAVE_INTERVAL"] == "10"
        assert env["SAVE_RETAIN_INTERVAL"] == "50"
        assert env["LR"] == "4e-6"
        assert env["ROLLOUT_TOP_P"] == "1.0"
        assert env["FUSE_ONE_STEP_ACTOR_LOGPROBS"] == "1"
        assert env["SGLANG_MEM_FRACTION"] == "0.70"
        assert (env["TIS_CLIP_LOW"], env["TIS_CLIP"]) == ("0.2", "5.0")
        assert env["USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER"] == "0"
        assert env["TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER_THRESHOLD"] == "2.0"
        assert (env["ADVANTAGE_CLIP_LOW"], env["ADVANTAGE_CLIP_HIGH"]) == ("-20", "20")
    assert math.lcm(*dps) == 3360
    assert len({arm["checkpoint"] for arm in plan["arms"]}) == 16
    assert tuple(arm["point"] for arm in plan["arms"])[-1] == [1, 16, 16]
    assert {tuple(arm["serving_profile"]) for arm in plan["arms"]} == {(32, 32), (64, 64)}
    for profile in sweep.SERVING_PROFILES:
        selected = [arm for arm in plan["arms"] if tuple(arm["serving_profile"]) == profile]
        assert [tuple(arm["point"]) for arm in selected] == list(sweep.POINTS)
        for arm in selected:
            assert arm["env"]["SGLANG_MAX_RUNNING_REQUESTS"] == str(profile[0])
            assert arm["env"]["SGLANG_CUDA_GRAPH_MAX_BS"] == str(profile[1])
            assert arm["env"]["RUN_NAME"].endswith(f"-req{profile[0]}-cg{profile[1]}")


@pytest.mark.parametrize(
    "settings",
    [
        {"GLOBAL_BATCH_SIZE": "256"},
        {"TRAINING_BUFFER_QUEUE_SIZE": "1000"},
        {"SAVE_HF": "0"},
        {"NUM_STEPS_PER_ROLLOUT": "2"},
        {"USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER": "2"},
        {"ASYNC_MAX_CONCURRENT_SAMPLES": "3585"},
        {"TYPO_BATCH_SIZE": "3360"},
    ],
)
def test_invalid_shared_settings_fail_before_submission(settings):
    with pytest.raises(ValueError):
        sweep.make_plan("invalid", list(sweep.POINTS), settings)


def test_preview_does_not_invoke_slurm(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("sbatch", "srun", "scontrol"):
        path = binary / name
        path.write_text("#!/bin/sh\nexit 91\n")
        path.chmod(0o755)
    plan_path = tmp_path / "plan.json"
    result = subprocess.run(
        [sys.executable, "-m", "experiments.staleness.lightning_node_ratio", "--plan-file", str(plan_path)],
        cwd=sweep.REPO,
        env=dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}"),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Preview only" in result.stdout
    assert len(json.loads(plan_path.read_text())["arms"]) == 16


@pytest.mark.parametrize("sequence_filter", ["0", "1"])
@pytest.mark.parametrize("serving_limit", [32, 64])
def test_train_shell_passes_async_filter_batch_and_save_arguments(tmp_path, serving_limit, sequence_filter):
    env = sweep.resolve_config(
        {
            "DEBUG_EXIT_AFTER_ROLLOUT": "20",
            "SGLANG_MAX_RUNNING_REQUESTS": str(serving_limit),
            "SGLANG_CUDA_GRAPH_MAX_BS": str(serving_limit),
            "USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER": sequence_filter,
        }
    )
    checkout = tmp_path / "checkout"
    (checkout / "experiments/common").mkdir(parents=True)
    (checkout / "experiments/container").mkdir(parents=True)
    (checkout / "experiments/common/ray_cluster.sh").write_text("")
    (checkout / "experiments/container/apply_oci_sglang_provenance.sh").write_text("")
    # Replace only the container mount prefix; execute the real argument builder.
    script = tmp_path / "train.sh"
    script.write_text((sweep.RECIPE / "train.sh").read_text().replace("/root/miles", str(checkout)))
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python").write_text('#!/bin/sh\nif [ "$1" = -m ]; then echo --mock-model; fi\n')
    (binary / "python").chmod(0o755)
    capture = tmp_path / "ray.json"
    (binary / "ray").write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['RATIO_ARGV_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    (binary / "ray").chmod(0o755)
    subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, **env, PATH=f"{binary}:{os.environ['PATH']}", RATIO_ARGV_CAPTURE=str(capture)),
    )
    argv = json.loads(capture.read_text())
    assert "train_async.py" in argv
    assert "--fully-async" in argv and "--colocate" not in argv and "--eval-interval" not in argv
    for flag, value in {
        "--global-batch-size": "3360",
        "--num-rollout": "300",
        "--debug-exit-after-rollout": "20",
        "--rollout-batch-size": "420",
        "--n-samples-per-prompt": "8",
        "--num-steps-per-rollout": "1",
        "--rollout-num-gpus": "112",
        "--actor-num-nodes": "4",
        "--max-weight-staleness": "8",
        "--staleness-reference": "prefill",
        "--async-max-concurrent-samples": str(112 * serving_limit),
        "--training-buffer-queue-size": "6000",
        "--dynamic-sampling-filter-path": env["DYNAMIC_SAMPLING_FILTER_PATH"],
        "--save-hf": env["CKPT_PATH"] + "/hf/{rollout_id}",
        "--hf-save-interval": "10",
        "--tis-clip": "5.0",
        "--tis-clip-low": "0.2",
        "--lr": "4e-6",
        "--lr-warmup-iters": "0",
        "--rollout-top-p": "1.0",
        "--advantage-clip-low": "-20",
        "--advantage-clip-high": "20",
        "--sglang-mem-fraction-static": "0.70",
        "--sglang-max-running-requests": str(serving_limit),
        "--sglang-cuda-graph-max-bs": str(serving_limit),
    }.items():
        assert argv[argv.index(flag) + 1] == value
    assert "--sglang-enable-response-weight-version-segments" in argv
    assert "--use-rollout-routing-replay" in argv
    assert "--use-wandb" in argv
    assert "--fuse-one-step-actor-logprobs" in argv
    assert ("--use-train-rollout-logprob-sequence-filter" in argv) == (sequence_filter == "1")
    threshold_flag = "--train-rollout-logprob-sequence-filter-threshold"
    if sequence_filter == "1":
        assert argv[argv.index(threshold_flag) + 1] == "2.0"
    else:
        assert threshold_flag not in argv
    assert "--zero-reward-on-truncated" not in argv
    assert "--zero-loss-on-truncated" not in argv
    assert "--skip-actor-forward-only" not in argv


def test_submission_records_exactly_one_four_hour_job_per_arm(monkeypatch, tmp_path):
    plan = sweep.make_plan("submit-test", list(sweep.POINTS), {"WS": str(tmp_path)})
    (tmp_path / "checkpoints/training").mkdir(parents=True)
    for relative in (
        f"checkpoints/hf/{plan['arms'][0]['env']['HF_MODEL_NAME']}/.download_complete",
        f"checkpoints/megatron/{plan['arms'][0]['env']['MODEL_NAME']}_torch_dist/latest_checkpointed_iteration.txt",
        "datasets/dapo-math-17k/dapo-math-17k.jsonl",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    image = tmp_path / "image.sqsh"
    image.touch()
    for arm in plan["arms"]:
        arm["env"]["SQSH_IMAGE"] = str(image)
    monkeypatch.setattr(sweep, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(sweep, "SLURM_LOG_ROOT", tmp_path / "slurm-logs")
    original_run = subprocess.run
    calls = []

    def run(argv, **kwargs):
        if argv[0] != "sbatch":
            return original_run(argv, **kwargs)
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=f"{9000 + len(calls)}\n", stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    sweep.submit_plan(plan, max_jobs=16, plan_file=tmp_path / "plan.json")
    assert len(calls) == 16
    assert all("--nodes=32" in cmd and "--time=04:00:00" in cmd for cmd in calls)
    assert all(not any(arg.startswith("--dependency") for arg in cmd) for cmd in calls)
    records = [json.loads(line) for line in (tmp_path / "logs/submit-test-submitted.jsonl").read_text().splitlines()]
    assert [record["job_id"] for record in records] == [str(i) for i in range(9001, 9017)]
    assert {tuple(record["serving_profile"]) for record in records} == {(32, 32), (64, 64)}
    with pytest.raises(FileExistsError):
        sweep.submit_plan(plan, max_jobs=16, plan_file=tmp_path / "plan.json")


def test_serving_profile_selection_and_submission_validation(tmp_path):
    plan = sweep.make_plan("selected", [(1, 16, 16)], {}, serving_profiles=[(64, 64)])
    assert len(plan["arms"]) == 1
    assert plan["arms"][0]["serving_profile"] == [64, 64]
    for profiles in ([], [(32, 64)], [(64, 64), (64, 64)]):
        with pytest.raises(ValueError, match="serving profiles"):
            sweep.make_plan("invalid", [(1, 16, 16)], {}, serving_profiles=profiles)
    with pytest.raises(ValueError, match="sweep-owned"):
        sweep.make_plan("invalid", [(1, 16, 16)], {"SGLANG_MAX_RUNNING_REQUESTS": "64"})
    plan["arms"] *= 2
    with pytest.raises(ValueError, match="duplicate arm"):
        sweep.submit_plan(plan, max_jobs=16, plan_file=tmp_path / "plan.json")
    plan["arms"] = plan["arms"][:1]
    plan["arms"][0]["env"]["SGLANG_MAX_RUNNING_REQUESTS"] = "32"
    with pytest.raises(ValueError, match="Serving profile and resolved limits disagree"):
        sweep.submit_plan(plan, max_jobs=16, plan_file=tmp_path / "plan.json")


def test_fixed_admission_override_applies_to_both_profiles():
    plan = sweep.make_plan("fixed", [(8, 4, 28)], {"ASYNC_MAX_CONCURRENT_SAMPLES": "3584"})
    assert len(plan["arms"]) == 2
    assert {arm["env"]["ASYNC_MAX_CONCURRENT_SAMPLES"] for arm in plan["arms"]} == {"3584"}


def test_summary_joins_writers_excludes_warmup_and_weights_rates(tmp_path):
    metric_dir = tmp_path / "metrics"
    metric_dir.mkdir()
    records = []
    for step, seconds, tokens in ((0, 1000, 10000), (1, 1, 100), (2, 9, 90)):
        records.extend(
            [
                dict(
                    ts=step * 10,
                    step_key="rollout/step",
                    step=step,
                    metrics={"perf/step_time": seconds, "perf/actor_train_time": seconds / 2},
                ),
                dict(
                    ts=step * 10 + 1,
                    step_key="rollout/step",
                    step=step,
                    metrics={
                        "perf/rollout_time": seconds * 2,
                        "throughput/window_seconds": seconds,
                        "throughput/generated_tokens": tokens,
                    },
                ),
            ]
        )
    # A completed rollout that was never trained must not enter the measurement.
    records.append(dict(ts=40, step_key="rollout/step", step=3, metrics={"perf/rollout_time": 999}))
    path = metric_dir / "20260912_00.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    arm = dict(
        env={
            "CONFIG_TAG": "test",
            "SGLANG_MAX_RUNNING_REQUESTS": "64",
            "SGLANG_CUDA_GRAPH_MAX_BS": "64",
            "ASYNC_MAX_CONCURRENT_SAMPLES": "3584",
        },
        point=[8, 4, 28],
        dashboard=str(tmp_path),
    )
    summary, rows = summarize_arm(arm, warmup_steps=1)
    assert summary["completed_steps"] == 3 and summary["measured_steps"] == 2
    assert summary["actor_train_mean_s"] == 2.5
    assert summary["rollout_drain_mean_s"] == 10
    assert summary["generated_tokens_per_s"] == 19  # (100 + 90) / (1 + 9), not (100 + 10) / 2
    assert summary["save_model_mean_s"] is None
    assert [row["rollout_step"] for row in rows] == [1, 2]
    assert summary["max_running_requests"] == summary["cuda_graph_max_bs"] == "64"
    assert all(row["max_running_requests"] == "64" for row in rows)
    with path.open("a") as output:
        output.write('{"unfinished":')
    with pytest.warns(UserWarning, match="unfinished"):
        assert len(read_steps(tmp_path)) == 4


def test_missing_metrics_are_not_reported_as_zero(tmp_path):
    summary, rows = summarize_arm(
        dict(env={"CONFIG_TAG": "empty"}, point=[1, 16, 16], dashboard=str(tmp_path)), warmup_steps=2
    )
    assert summary["status"] == "insufficient_data"
    assert summary["train_mean_s"] is None and summary["generated_tokens_per_s"] is None
    assert rows == []
