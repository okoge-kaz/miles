import argparse
import math
import os
import re
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

from miles.utils.external_utils.model_args_utils import load_model_args
from experiments.common.dataset_args import prompt_arguments

REPO_ROOT = Path(__file__).resolve().parents[2]


def _validate_reward_runtime(environment: Mapping[str, str]) -> None:
    for name in ("CODE_EXEC_CONCURRENCY", "CODE_EXEC_MAX_TESTS", "CODE_EXEC_MAX_OUTPUT_BYTES"):
        if environment.get(name) and int(environment[name]) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("CODE_EXEC_TIMEOUT", "CODE_EXEC_MEMORY_GB"):
        if environment.get(name):
            value = float(environment[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def validate_config(environment: Mapping[str, str]) -> None:
    prompt_arguments(environment)
    _validate_reward_runtime(environment)
    if environment["PLACEMENT"] not in ("disaggregate", "colocated"):
        raise ValueError("PLACEMENT must be disaggregate or colocated")
    if environment["TRUNCATION_MODE"] not in ("none", "zero_reward", "zero_loss"):
        raise ValueError("TRUNCATION_MODE must be none, zero_reward, or zero_loss")
    for name in (
        "GPUS_PER_NODE",
        "ACTOR_NUM_NODES",
        "ACTOR_GPUS_PER_NODE",
        "ROLLOUT_NUM_GPUS_PER_ENGINE",
        "TENSOR_PARALLEL_SIZE",
        "CONTEXT_PARALLEL_SIZE",
        "PIPELINE_PARALLEL_SIZE",
        "EXPERT_PARALLEL_SIZE",
        "EXPERT_TENSOR_PARALLEL_SIZE",
        "ROLLOUT_BATCH_SIZE",
        "N_SAMPLES_PER_PROMPT",
        "GLOBAL_BATCH_SIZE",
        "NUM_STEPS_PER_ROLLOUT",
        "NUM_ROLLOUT",
        "MAX_RESPONSE_LEN",
        "MAX_CONTEXT_LEN",
        "MAX_TOKENS_PER_GPU",
        "SAVE_INTERVAL",
        "SAVE_RETAIN_INTERVAL",
    ):
        if int(environment[name]) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("SKIP_ACTOR_FORWARD_ONLY", "SAVE_HF", "DUMP_ROLLOUT_DATA", "USE_CHECKPOINT_OPT_PARAM_SCHEDULER"):
        if environment[name] not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
    for name in (
        "USE_REPLAY_DATA_BUFFER",
        "REPLAY_DATA_BUFFER_ASYNC_RESTORE",
        "LOG_ROLLOUT_REPETITION",
        "LOG_CATEGORY_STALENESS",
        "LOG_POLICY_LAG_METRICS",
        "SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS",
    ):
        if environment.get(name, "0") not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
    if (
        environment.get("REPLAY_DATA_BUFFER_ASYNC_RESTORE", "0") == "1"
        and environment.get("USE_REPLAY_DATA_BUFFER", "0") != "1"
    ):
        raise ValueError("REPLAY_DATA_BUFFER_ASYNC_RESTORE requires USE_REPLAY_DATA_BUFFER=1")
    if environment.get("USE_REPLAY_DATA_BUFFER", "0") == "1" and environment["PLACEMENT"] != "disaggregate":
        raise ValueError("ReplayDataBuffer requires disaggregate placement")
    hf_interval = environment.get("HF_SAVE_INTERVAL", "")
    if hf_interval and (int(hf_interval) <= 0 or environment["SAVE_HF"] != "1"):
        raise ValueError("HF_SAVE_INTERVAL must be positive and requires SAVE_HF=1")
    replay_mode = environment.get("REPLAY_DATA_BUFFER_MODE", "rollout")
    if replay_mode not in ("rollout", "inflight"):
        raise ValueError("REPLAY_DATA_BUFFER_MODE must be rollout or inflight")
    if replay_mode == "inflight" and environment.get("USE_REPLAY_DATA_BUFFER", "0") != "1":
        raise ValueError("REPLAY_DATA_BUFFER_MODE=inflight requires USE_REPLAY_DATA_BUFFER=1")
    order = environment.get("ASYNC_DATA_BUFFER_ORDER", "fifo")
    metric = environment.get("ASYNC_STALENESS_METRIC", "oldest")
    if order not in ("fifo", "oldest-policy-first"):
        raise ValueError("ASYNC_DATA_BUFFER_ORDER must be fifo or oldest-policy-first")
    if metric not in ("oldest", "newest", "token-mean", "submission", "first-prefill", "ready"):
        raise ValueError("Unknown ASYNC_STALENESS_METRIC")
    if metric == "first-prefill" and environment.get("SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS", "0") != "1":
        raise ValueError("first-prefill requires SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1")
    if environment["PLACEMENT"] != "disaggregate" and (
        order != "fifo" or metric != "oldest" or environment.get("CUSTOM_STALENESS_FUNCTION_PATH")
    ):
        raise ValueError("Async buffer ordering/staleness options require disaggregate placement")
    if (
        environment.get("REPLAY_DATA_BUFFER_ASYNC_RESTORE", "0") == "1"
        and float(environment["ASYNC_DATA_BUFFER_CAPACITY_FACTOR"]) < 1
    ):
        raise ValueError("Async ReplayDataBuffer restore requires capacity for at least one batch")
    samples = int(environment["ROLLOUT_BATCH_SIZE"]) * int(environment["N_SAMPLES_PER_PROMPT"])
    steps = int(environment["NUM_STEPS_PER_ROLLOUT"])
    if samples != int(environment["GLOBAL_BATCH_SIZE"]) * steps:
        raise ValueError(
            "ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT must equal GLOBAL_BATCH_SIZE * NUM_STEPS_PER_ROLLOUT"
        )
    if environment["SKIP_ACTOR_FORWARD_ONLY"] == "1" and steps != 1:
        raise ValueError("SKIP_ACTOR_FORWARD_ONLY=1 requires exactly one optimizer step per rollout")
    if int(environment["MAX_RESPONSE_LEN"]) > int(environment["MAX_CONTEXT_LEN"]):
        raise ValueError("MAX_RESPONSE_LEN must not exceed MAX_CONTEXT_LEN")
    if environment["IS_CORRECTION"] not in ("tis", "none"):
        raise ValueError("IS_CORRECTION must be tis or none")
    if environment["MEGATRON_TO_HF_MODE"] not in ("raw", "bridge"):
        raise ValueError("MEGATRON_TO_HF_MODE must be raw or bridge")
    mode = environment["WANDB_MODE"]
    if mode not in ("online", "offline", "disabled"):
        raise ValueError("WANDB_MODE must be online, offline, or disabled")
    if mode == "online" and not environment.get("WANDB_API_KEY"):
        raise ValueError("WANDB_MODE=online requires WANDB_API_KEY or the compute host's ~/.netrc")
    fraction = float(environment["SGLANG_MEM_FRACTION"])
    if not 0 < fraction < 1:
        raise ValueError("SGLANG_MEM_FRACTION must be between 0 and 1")
    if environment["PLACEMENT"] == "disaggregate":
        if float(environment["ASYNC_DATA_BUFFER_CAPACITY_FACTOR"]) <= 0:
            raise ValueError("ASYNC_DATA_BUFFER_CAPACITY_FACTOR must be positive")
        if environment["ASYNC_UNUSED_SAMPLES_HANDLER"] not in ("retry", "drop"):
            raise ValueError("ASYNC_UNUSED_SAMPLES_HANDLER must be retry or drop")
        if environment["MAX_WEIGHT_STALENESS"] != "none" and int(environment["MAX_WEIGHT_STALENESS"]) < 0:
            raise ValueError("MAX_WEIGHT_STALENESS must be nonnegative or none")
        if environment.get("ASYNC_MAX_CONCURRENT_SAMPLES") and int(environment["ASYNC_MAX_CONCURRENT_SAMPLES"]) <= 0:
            raise ValueError("ASYNC_MAX_CONCURRENT_SAMPLES must be positive")
        if environment["PAUSE_GENERATION_MODE"] not in ("in_place", "retract"):
            raise ValueError("PAUSE_GENERATION_MODE must be in_place or retract")
        if environment["UPDATE_WEIGHT_TRANSFER_MODE"] not in ("broadcast", "p2p"):
            raise ValueError("UPDATE_WEIGHT_TRANSFER_MODE must be broadcast or p2p")
        if environment["PAUSE_GENERATION_MODE"] == "in_place" and environment["UPDATE_WEIGHT_TRANSFER_MODE"] == "p2p":
            raise ValueError("in_place generation requires broadcast weight transfer")


def _recipe_args(environment: Mapping[str, str]) -> list[str]:
    project = environment["PROJECT_NAME"]
    experiment = environment["EXPERIMENT_NAME"]
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in (project, experiment)):
        raise ValueError("PROJECT_NAME and EXPERIMENT_NAME must be path-safe recipe names")
    recipe = REPO_ROOT / "experiments/scripts" / project / experiment / "train.sh"
    if not recipe.is_file():
        raise ValueError(f"unknown training recipe: {recipe}")
    result = subprocess.run(
        ["bash", str(recipe), "--print-args"],
        env={**environment, "MILES_REPO": str(REPO_ROOT)},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot build training arguments from {recipe}: {result.stderr.strip()}")
    if not result.stdout.endswith("\0"):
        raise ValueError(f"recipe must emit NUL-separated training arguments: {recipe}")
    return result.stdout[:-1].split("\0")


def build_command(environment: Mapping[str, str]) -> list[str]:
    environment = dict(environment)
    for name, legacy in (
        ("USE_REPLAY_DATA_BUFFER", "USE_REPLAY_BUFFER"),
        ("REPLAY_DATA_BUFFER_ASYNC_RESTORE", "REPLAY_BUFFER_ASYNC_RESTORE"),
    ):
        environment.setdefault(name, environment.get(legacy, "0"))
    validate_config(environment)
    train_script = "train_async.py" if environment["PLACEMENT"] == "disaggregate" else "train.py"
    return [
        "python3",
        str(REPO_ROOT / train_script),
        *shlex.split(load_model_args(environment["MODEL_PROFILE"])),
        *_recipe_args(environment),
    ]


def runtime_environment(environment: Mapping[str, str]) -> dict[str, dict[str, str]]:
    names = (
        "MASTER_ADDR",
        "no_proxy",
        "NCCL_IB_DISABLE",
        "NCCL_NET",
        "NCCL_NVLS_ENABLE",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "NCCL_DEBUG",
        "WANDB_MODE",
        "LLM_JUDGE_MODE",
        "LLM_JUDGE_ENDPOINT_FILE",
        "LLM_JUDGE_CATEGORIES",
        "LLM_JUDGE_TRIGGER",
        "LLM_JUDGE_ALLOW_HARD_FAILURE_OVERRIDE",
        "LLM_JUDGE_CONCURRENCY",
        "LLM_JUDGE_QUEUE_CAPACITY",
        "LLM_JUDGE_QUEUE_TIMEOUT",
        "LLM_JUDGE_REQUEST_TIMEOUT",
        "LLM_JUDGE_MAX_RETRIES",
        "LLM_JUDGE_MAX_TOKENS",
        "LLM_JUDGE_TEMPERATURE",
        "LLM_JUDGE_MAX_REQUEST_BYTES",
        "LLM_JUDGE_ON_ERROR",
        "LLM_JUDGE_CACHE_DIR",
        "LLM_JUDGE_CACHE_SIZE",
        "LLM_JUDGE_BASE_RM_PATH",
        "LLM_JUDGE_READY_TIMEOUT",
        "CODE_EXEC_SANDBOX",
        "CODE_EXEC_CONCURRENCY",
        "CODE_EXEC_TIMEOUT",
        "CODE_EXEC_MEMORY_GB",
        "CODE_EXEC_MAX_TESTS",
        "CODE_EXEC_MAX_OUTPUT_BYTES",
        "OPEN_INSTRUCT_PATH",
        "OPEN_INSTRUCT_DEPS_PATH",
    )
    return {
        "env_vars": {
            "PYTHONPATH": f"{REPO_ROOT}:/root/Megatron-LM",
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            **{name: environment[name] for name in names if environment.get(name)},
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and submit the selected ABCI training recipe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = build_command(os.environ)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    # The preview must also work on login hosts without Ray or GPU packages.
    from miles.utils.external_utils.ray_job import run_ray_job

    run_ray_job(entrypoint=shlex.join(command), runtime_env=runtime_environment(os.environ), job_lifetime="launcher")


if __name__ == "__main__":
    main()
