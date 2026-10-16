import base64
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from experiments.common.dataset_args import check_prompt_paths
from experiments.common.training import build_command, runtime_environment
from miles.utils.dataset_mixture_config import read_prompt_config

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("encoding", ["inline", "json", "yaml"])
def test_prompt_config_paths_resolve_mounts_slices_and_missing_sources(tmp_path, encoding):
    data_root = tmp_path / "data"
    data_root.mkdir()
    source = data_root / "math problems.jsonl"
    source.write_text('{"prompt":"x"}\n')
    value = {"datasets": [{"name": "math", "path": "/data/math problems.jsonl@[0:1]"}]}
    if encoding == "inline":
        config = "base64:" + base64.b64encode(json.dumps(value).encode()).decode()
    else:
        value["datasets"].append({"name": "relative", "path": "math problems.jsonl@[0:1]"})
        # JSON is also a valid YAML document; the extension selects the YAML loader.
        (data_root / f"mix.{encoding}").write_text(json.dumps(value))
        config = f"/data/mix.{encoding}"
    environment = {"PROMPT_DATA_CONFIG": config}
    check_prompt_paths(environment, data_root=data_root, repo_root=REPO_ROOT)
    source.unlink()
    with pytest.raises(ValueError, match="Missing prompt source"):
        check_prompt_paths(environment, data_root=data_root, repo_root=REPO_ROOT)


def _environment(tmp_path, **overrides):
    environment = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "WANDB_MODE": "disabled"}
    environment.update({"MILES_WORKSPACE_ROOT": str(tmp_path / "workspace"), **overrides})
    return environment


def _recipe_config(tmp_path, experiment="zero_loss_truncation", *, project="staleness_math", **overrides):
    command = """
set -euo pipefail
export MILES_JOB_NUM_NODES=8
source experiments/env.sh
source "experiments/scripts/$2${1:+/$1}/train.sh"
source experiments/common/run_identity.sh
python3 -c 'import json, os; print(json.dumps(dict(os.environ)))'
"""
    result = subprocess.run(
        ["bash", "-c", command, "_", experiment, project],
        cwd=REPO_ROOT,
        env=_environment(tmp_path, **overrides),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "project,experiment",
    [
        ("staleness_math", "zero_loss_truncation"),
        ("staleness_math", "zero_reward_truncation"),
        ("staleness_multi_env", "math_code_mcqa_if"),
    ],
)
@pytest.mark.parametrize("enabled", [None, "0"])
def test_staleness_policy_lag_logging_is_default_and_can_be_disabled(tmp_path, project, experiment, enabled):
    overrides = {} if enabled is None else {"LOG_POLICY_LAG_METRICS": enabled}
    command = build_command(_recipe_config(tmp_path, experiment, project=project, **overrides))
    assert ("--log-policy-lag-metrics" in command) == (enabled != "0")


def test_colocated_recipe_does_not_request_dedicated_rollout_gpus(tmp_path):
    command = build_command(_recipe_config(tmp_path, PLACEMENT="colocated"))
    assert "--colocate" in command
    assert "--fully-async" not in command and "--rollout-num-gpus" not in command


def test_separate_treatments_cannot_share_default_checkpoint_path(tmp_path):
    reward = _recipe_config(tmp_path, "zero_reward_truncation", RUN_NAME="same-study")
    loss = _recipe_config(tmp_path, "zero_loss_truncation", RUN_NAME="same-study")
    assert reward["CKPT_PATH"] != loss["CKPT_PATH"]


@pytest.mark.parametrize("truncation", ["none", "zero_reward"])
def test_math_builtin_reward_and_full_checkpoint_retention(tmp_path, truncation):
    config = _recipe_config(
        tmp_path, "zero_reward_truncation", TRUNCATION_MODE=truncation, SAVE_RETAIN_INTERVAL="none"
    )
    command = build_command(config)
    assert command[command.index("--rm-type") + 1] == "deepscaler"
    assert "--save-retain-interval" not in command
    assert "--save" in command
    if truncation == "none":
        assert "--custom-rm-path" not in command
    else:
        assert command[command.index("--custom-rm-path") + 1] == "experiments.src.training.truncation.zero_reward"


@pytest.mark.parametrize("retain_interval", ["0", "1", "15", "-10"])
def test_checkpoint_retention_rejects_incompatible_interval(tmp_path, retain_interval):
    config = _recipe_config(tmp_path, SAVE_INTERVAL="10", SAVE_RETAIN_INTERVAL=retain_interval)
    with pytest.raises(ValueError, match="positive multiple of SAVE_INTERVAL"):
        build_command(config)


@pytest.mark.parametrize(
    "project,experiment",
    [
        ("staleness_math", "zero_loss_truncation"),
        ("staleness_math", "zero_reward_truncation"),
        ("multi_env", ""),
        ("staleness_multi_env", "math_code_mcqa_if"),
    ],
)
def test_all_recipes_can_disable_checkpoint_deletion(tmp_path, project, experiment):
    config = _recipe_config(tmp_path, experiment, project=project, SAVE_RETAIN_INTERVAL="none")
    command = build_command(config)
    assert "--save-retain-interval" not in command
    assert "--save" in command and "--save-interval" in command


@pytest.mark.parametrize(
    "project,experiment,categories",
    [
        ("multi_env", "", {"math", "code", "mcqa", "ifevalg", "tool"}),
        ("staleness_multi_env", "math_code_mcqa_if", {"ifevalg", "math", "code", "mcqa"}),
    ],
)
def test_job_prompt_sources_preserve_reward_and_agent_dispatch(tmp_path, project, experiment, categories):
    command = build_command(_recipe_config(tmp_path, experiment, project=project))
    if "tool" in categories:
        assert command[command.index("--use-session-server") + 1] == "v2"
        assert command[command.index("--tito-model") + 1] == "qwen3"
    config, _ = read_prompt_config(command[command.index("--prompt-data-config") + 1])
    assert {source["category"] for source in config["datasets"]} == categories
    assert sum(source["ratio"] for source in config["datasets"]) == pytest.approx(1.0)
    for source in config["datasets"]:
        assert isinstance(source["ratio"], float) and 0 < source["ratio"] < 1
        if source["category"] == "math":
            assert source["rm_type"] == "deepscaler"
        else:
            assert source["custom_rm_path"] == "experiments.src.nemo_gym.bridge.reward.reward"
        if source["category"] == "tool":
            assert source["generate_function_path"] == "experiments.src.nemo_gym.bridge.generate.generate"
        if source["category"] == "ifevalg":
            assert source["path"] == "/data/multi_env/if-training.jsonl"


def test_job_explicit_prompt_files_override_the_default_mixture(tmp_path):
    command = build_command(
        _recipe_config(
            tmp_path,
            "math_code_mcqa_if",
            project="staleness_multi_env",
            PROMPT_DATA_FILES='["/data/a.jsonl", "/data/b.jsonl"]',
            PROMPT_DATA_RATIOS="[0.4, 0.6]",
        )
    )
    assert "--prompt-data-config" not in command
    index = command.index("--prompt-data-files")
    assert command[index + 1 : index + 3] == ["/data/a.jsonl", "/data/b.jsonl"]
    index = command.index("--prompt-data-ratios")
    assert command[index + 1 : index + 3] == ["0.4", "0.6"]


@pytest.mark.parametrize("experiment", ["zero_reward_truncation", "zero_loss_truncation"])
def test_checkpoint_scheduler_option_is_forwarded_and_validated(tmp_path, experiment):
    config = _recipe_config(tmp_path, experiment, USE_CHECKPOINT_OPT_PARAM_SCHEDULER="1")
    assert "--use-checkpoint-opt-param-scheduler" in build_command(config)
    config["USE_CHECKPOINT_OPT_PARAM_SCHEDULER"] = "0"
    assert "--use-checkpoint-opt-param-scheduler" not in build_command(config)
    config["USE_CHECKPOINT_OPT_PARAM_SCHEDULER"] = "invalid"
    with pytest.raises(ValueError, match="USE_CHECKPOINT_OPT_PARAM_SCHEDULER"):
        build_command(config)


def test_skip_actor_rejects_multi_step_shape(tmp_path):
    config = _recipe_config(
        tmp_path,
        ROLLOUT_BATCH_SIZE="2",
        N_SAMPLES_PER_PROMPT="4",
        NUM_STEPS_PER_ROLLOUT="2",
        GLOBAL_BATCH_SIZE="4",
        SKIP_ACTOR_FORWARD_ONLY="1",
    )
    with pytest.raises(ValueError, match="one optimizer step"):
        build_command(config)
    config["SKIP_ACTOR_FORWARD_ONLY"] = "0"
    assert "--skip-actor-forward-only" not in build_command(config)


@pytest.mark.parametrize(
    "recipe",
    [
        "staleness_math/zero_reward_truncation",
        "staleness_math/zero_loss_truncation",
        "multi_env",
        "staleness_multi_env/math_code_mcqa_if",
    ],
)
def test_sweep_previews_requested_grid_without_calling_qsub(tmp_path, recipe):
    environment = _environment(
        tmp_path,
        RUN_NAMESPACE="test-grid",
        CHAIN_JOBS="1",
        PBS_QSUB_BIN="must-not-run",
        STALENESS_LEVELS="0 2",
        RATIOS="1:1",
        TOTAL_NODES="2",
    )
    result = subprocess.run(
        ["bash", f"experiments/scripts/{recipe}/sweep.sh"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.count("must-not-run -q") == 2
    assert "-V" not in shlex.split(result.stdout)


@pytest.mark.parametrize("experiment", ["zero_loss_truncation", "zero_reward_truncation"])
def test_scheduler_exports_configuration_without_credentials(tmp_path, experiment):
    qsub = tmp_path / "qsub"
    capture = tmp_path / "qsub.args"
    qsub.write_text(f"#!/bin/bash\nprintf '%s\\0' \"$@\" > {shlex.quote(str(capture))}\nprintf '123.pbs1\\n'\n")
    qsub.chmod(0o755)
    result = subprocess.run(
        ["bash", "experiments/submit_training.sh", f"staleness_math/{experiment}", "test-export", "--nodes=2"],
        cwd=REPO_ROOT,
        env=_environment(
            tmp_path,
            PBS_QSUB_BIN=str(qsub),
            WANDB_API_KEY="fake-secret",
            LR="2e-6",
            RM_NUM_WORKERS="6",
            MODEL_PRESET="qwen3-30b-a3b-sft",
            EXPERT_PARALLEL_SIZE="4",
            ROLLOUT_EXPERT_PARALLEL_SIZE="2",
            USE_ROLLOUT_ROUTING_REPLAY="0",
            USE_REPLAY_DATA_BUFFER="1",
            REPLAY_DATA_BUFFER_ASYNC_RESTORE="1",
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    arguments = capture.read_bytes().decode().split("\0")
    assert result.stdout.strip() == "123.pbs1"
    assert "-V" not in arguments
    assert "LR" in arguments[arguments.index("-v") + 1]
    exported = arguments[arguments.index("-v") + 1].split(",")
    assert {"MODEL_PRESET", "EXPERT_PARALLEL_SIZE", "USE_ROLLOUT_ROUTING_REPLAY"} <= set(exported)
    assert "RM_NUM_WORKERS" in exported
    assert "ROLLOUT_EXPERT_PARALLEL_SIZE" in exported
    assert "USE_REPLAY_DATA_BUFFER" in exported
    assert "REPLAY_DATA_BUFFER_ASYNC_RESTORE" in exported
    assert "WANDB_API_KEY" not in " ".join(arguments) and "fake-secret" not in " ".join(arguments)


def test_ray_metadata_does_not_serialize_wandb_credentials(tmp_path):
    config = _recipe_config(tmp_path)
    config.update(WANDB_API_KEY="fake-secret", WANDB_MODE="online", MASTER_ADDR="10.0.0.1")
    assert "fake-secret" not in json.dumps(runtime_environment(config))
    assert "fake-secret" not in shlex.join(build_command(config))


def test_mixed_prompt_paths_ratios_and_optional_features_are_explicit_in_recipe(tmp_path):
    paths = ["/data/math easy.jsonl", "/data/code.jsonl"]
    config = _recipe_config(
        tmp_path,
        DATASET_PRESET="multi-env",
        PROMPT_DATA_FILES=json.dumps(paths),
        PROMPT_DATA_RATIOS="[0.75, 0.25]",
        PROMPT_DATA_EPOCH_SIZE="100",
        USE_REPLAY_DATA_BUFFER="1",
        REPLAY_DATA_BUFFER_ASYNC_RESTORE="1",
        LOG_CATEGORY_STALENESS="1",
        LOG_POLICY_LAG_METRICS="1",
        ASYNC_DATA_BUFFER_ORDER="oldest-policy-first",
        ASYNC_STALENESS_METRIC="first-prefill",
        CUSTOM_STALENESS_FUNCTION_PATH="example.clock",
        SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS="1",
    )
    command = build_command(config)
    start = command.index("--prompt-data-files")
    assert command[start + 1 : start + 3] == paths
    assert "--prompt-data" not in command
    assert command[command.index("--prompt-data-ratios") + 1 : command.index("--prompt-data-ratios") + 3] == [
        "0.75",
        "0.25",
    ]
    assert command[command.index("--custom-staleness-function-path") + 1] == "example.clock"
    for flag in (
        "--use-replay-data-buffer",
        "--replay-data-buffer-async-restore",
        "--log-category-staleness",
        "--log-policy-lag-metrics",
    ):
        assert flag in command


def test_replay_configuration_fails_before_ray_submission(tmp_path):
    config = _recipe_config(tmp_path, USE_REPLAY_DATA_BUFFER="0", REPLAY_DATA_BUFFER_ASYNC_RESTORE="1")
    with pytest.raises(ValueError, match="requires USE_REPLAY"):
        build_command(config)
    config.update(USE_REPLAY_DATA_BUFFER="1", ASYNC_DATA_BUFFER_CAPACITY_FACTOR="0.5")
    with pytest.raises(ValueError, match="at least one batch"):
        build_command(config)
    config.update(ASYNC_DATA_BUFFER_CAPACITY_FACTOR="1", PLACEMENT="colocated")
    with pytest.raises(ValueError, match="disaggregate placement"):
        build_command(config)


@pytest.mark.parametrize("experiment", ["zero_reward_truncation", "zero_loss_truncation"])
@pytest.mark.parametrize("legacy", [False, True])
def test_replay_data_buffer_environment_names_generate_canonical_cli(tmp_path, experiment, legacy):
    prefix = "REPLAY_BUFFER" if legacy else "REPLAY_DATA_BUFFER"
    config = _recipe_config(tmp_path, experiment, **{f"USE_{prefix}": "1", f"{prefix}_ASYNC_RESTORE": "1"})
    command = build_command(config)
    assert "--use-replay-data-buffer" in command
    assert "--replay-data-buffer-async-restore" in command
    assert "--use-replay-buffer" not in command
    assert "--replay-buffer-async-restore" not in command
    # Direct Python launch inputs also accept the old environment names.
    if legacy:
        config.pop("USE_REPLAY_DATA_BUFFER")
        config.pop("REPLAY_DATA_BUFFER_ASYNC_RESTORE")
        assert build_command(config) == command


def test_explicit_new_replay_data_buffer_environment_overrides_legacy(tmp_path):
    config = _recipe_config(
        tmp_path,
        USE_REPLAY_DATA_BUFFER="0",
        REPLAY_DATA_BUFFER_ASYNC_RESTORE="0",
        USE_REPLAY_BUFFER="1",
        REPLAY_BUFFER_ASYNC_RESTORE="1",
    )
    command = build_command(config)
    assert not any("replay-data-buffer" in token or "replay-buffer" in token for token in command)


def test_invalid_pipeline_allocation_fails_before_ray(tmp_path):
    result = subprocess.run(
        ["bash", "experiments/scripts/staleness_math/zero_loss_truncation/run.sh"],
        cwd=REPO_ROOT,
        env=_environment(
            tmp_path,
            MILES_LAUNCH_DRY_RUN="1",
            GPUS_PER_NODE="8",
            ACTOR_NUM_NODES="1",
            TENSOR_PARALLEL_SIZE="2",
            CONTEXT_PARALLEL_SIZE="1",
            PIPELINE_PARALLEL_SIZE="3",
        ),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2 and "TP * CP * PP" in result.stderr


def test_explicit_paths_and_names_preserve_spaces(tmp_path):
    config = _recipe_config(
        tmp_path,
        PROMPT_DATA="/data/custom cohort/train.jsonl",
        CKPT_PATH="/ckpt/training/custom run",
        WANDB_MODE="offline",
        WANDB_PROJECT="project with spaces",
    )
    command = build_command(config)
    for flag, value in {
        "--prompt-data": "/data/custom cohort/train.jsonl",
        "--load": "/ckpt/training/custom run",
        "--wandb-project": "project with spaces",
    }.items():
        assert command[command.index(flag) + 1] == value


@pytest.mark.parametrize("experiment", ["zero_loss_truncation", "zero_reward_truncation"])
def test_checkpoint_and_logging_controls_reach_training_command(tmp_path, experiment):
    config = _recipe_config(
        tmp_path,
        experiment,
        USE_REPLAY_DATA_BUFFER="1",
        REPLAY_DATA_BUFFER_MODE="inflight",
        HF_SAVE_INTERVAL="2",
        MAX_WEIGHT_STALENESS="28",
        SAVE_HF="1",
        LOG_ROLLOUT_REPETITION="0",
    )
    command = build_command(config)
    assert command[command.index("--max-weight-staleness") + 1] == "28"
    assert command[command.index("--hf-save-interval") + 1] == "2"
    assert command[command.index("--replay-data-buffer-mode") + 1] == "inflight"
    assert "--no-log-rollout-repetition" in command
    config["LOG_ROLLOUT_REPETITION"] = "1"
    assert "--log-rollout-repetition" in build_command(config)
    config["SAVE_HF"] = "0"
    with pytest.raises(ValueError, match="HF_SAVE_INTERVAL"):
        build_command(config)
    config["SAVE_HF"] = "1"
    config["USE_REPLAY_DATA_BUFFER"] = "0"
    with pytest.raises(ValueError, match="inflight"):
        build_command(config)
