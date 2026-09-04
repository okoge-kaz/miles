from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]

RECIPES = (
    "experiments/scripts/code/async/nemotron3-nano-competitive-code/qwen3-4b",
    "experiments/scripts/instruction_following/async/nemotron3-nano-ifevalg/qwen3-4b",
    "experiments/scripts/multi_env/async/math-code-stem/qwen3-4b",
    "experiments/scripts/stem/async/nemotron3-nano-knowledge-mcqa-reasoning-gym/qwen3-4b",
    "experiments/scripts/tool_call_pivot/async/nemotron-agentic-conv-tooluse-pivot/qwen3-4b",
)
RECIPE_IDS = ("code", "instruction-following", "math-code-stem", "stem", "tool_call_pivot")
CUSTOM_RM_PATHS = {
    RECIPES[0]: "experiments.src.reward_sets.code.reward",
    RECIPES[1]: "experiments.src.reward_sets.instruction_following.reward",
    RECIPES[2]: "experiments.src.reward_sets.math_code_stem.reward",
    RECIPES[3]: "experiments.src.reward_sets.stem.reward",
    RECIPES[4]: "experiments.src.reward_sets.tool_call_pivot.reward",
}


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _resolve_segment_identity(
    run_script: str,
    slurm_job_id: str,
    num_rollout: str = "300",
) -> tuple[str, str, str]:
    task_family = next(
        line.removeprefix("TASK_FAMILY=") for line in run_script.splitlines() if line.startswith("TASK_FAMILY=")
    )
    dataset_tag = next(
        line.removeprefix("DATASET_TAG=") for line in run_script.splitlines() if line.startswith("DATASET_TAG=")
    )
    environment = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": slurm_job_id,
        "MODEL_NAME": "Qwen3-4B-Base-LR2e-5-Step4000",
        "DATASET_TAG": dataset_tag,
        "TASK_FAMILY": task_family,
        "PLACEMENT": "async",
        "ADVANTAGE_ESTIMATOR": "grpo",
        "EPS_CLIP": "0.2",
        "EPS_CLIP_HIGH": "0.28",
        "EPS_CLIP_C": "",
        "RATIO_DENOMINATOR": "actor",
        "IS_CORRECTION": "tis",
        "TIS_CLIP": "2.0",
        "TIS_CLIP_LOW": "0",
        "MIS_PROFILE": "",
        "USE_OPSM": "0",
        "M2PO_BUDGET": "0.04",
        "OPSM_DELTA": "1e-4",
        "KL_LOSS_COEF": "0.00",
        "LR": "1e-6",
        "MAX_RESPONSE_LEN": "16384",
        "NUM_ROLLOUT": num_rollout,
        "NUM_STEPS_PER_ROLLOUT": "1",
        "ROLLOUT_BATCH_SIZE": "192",
        "GLOBAL_BATCH_SIZE": "3072",
        "N_SAMPLES_PER_PROMPT": "16",
        "TRAIN_SEED": "1234",
        "ROLLOUT_SEED": "42",
        "QUEUE_TYPE": "queue-recycle",
        "QUEUE_FACTOR": "1",
        "MAX_WEIGHT_STALENESS": "4",
        "STALENESS_REFERENCE": "prefill",
        "ZERO_REWARD_ON_TRUNCATED": "1",
        "USE_REPLAY_BUFFER": "1",
        "REPLAY_BUFFER_TYPE": "inflight",
        "REPLAY_BUFFER_IDENTITY_TAG": "1",
        "CONFIG_TAG": f"4node-rollout-length-16k-lr1e-6-rbs192-gbs3072-n16-tseed1234-rseed42-nr{num_rollout}",
    }
    identity_path = REPO_ROOT / "experiments/common/run_identity.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n%s\\n%s\\n" "$RUN_NAME" "$CKPT_PATH" "$CONFIG_TAG"',
            "bash",
            str(identity_path),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    run_name, checkpoint_path, config_tag = result.stdout.splitlines()
    return run_name, checkpoint_path, config_tag


@pytest.mark.parametrize("recipe_dir", RECIPES, ids=RECIPE_IDS)
def test_domain_recipe_keeps_four_hour_segment_scheduler_and_production_shape(recipe_dir: str):
    run_script = _read(f"{recipe_dir}/run.sbatch")
    train_script = _read(f"{recipe_dir}/train.sh")

    for directive in (
        "#SBATCH --partition=batch",
        "#SBATCH --qos=interactive",
        "#SBATCH --nodes=4",
        "#SBATCH --time=04:00:00",
    ):
        assert directive in run_script

    for default in (
        ': "${MAX_RESPONSE_LEN:=16384}"',
        ': "${NUM_ROLLOUT:=300}"',
        ': "${ROLLOUT_BATCH_SIZE:=192}"',
        ': "${N_SAMPLES_PER_PROMPT:=16}"',
        ': "${GLOBAL_BATCH_SIZE:=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"',
        ': "${NUM_STEPS_PER_ROLLOUT:=1}"',
        ': "${ACTOR_NUM_NODES:=1}"',
        ': "${ACTOR_GPUS_PER_NODE:=8}"',
        ': "${ROLLOUT_NUM_GPUS:=24}"',
        ': "${TENSOR_PARALLEL_SIZE:=2}"',
        ': "${MAX_TOKENS_PER_GPU:=32768}"',
        ': "${EVAL_INTERVAL:=0}"',
    ):
        assert default in run_script

    assert "#SBATCH --partition=batch_long" not in run_script
    assert "#SBATCH --qos=normal" not in run_script
    for argument in (
        '--num-rollout "${NUM_ROLLOUT}"',
        '--rollout-batch-size "${ROLLOUT_BATCH_SIZE}"',
        '--n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"',
        '--rollout-max-response-len "${MAX_RESPONSE_LEN}"',
        '--global-batch-size "${GLOBAL_BATCH_SIZE}"',
        '--num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"',
    ):
        assert argument in train_script
    assert '[[ "${EVAL_INTERVAL}" == 0 ]] || {' in train_script


@pytest.mark.parametrize("recipe_dir", RECIPES, ids=RECIPE_IDS)
def test_domain_recipe_uses_inflight_replay_for_every_segment(recipe_dir: str):
    run_script = _read(f"{recipe_dir}/run.sbatch")
    train_script = _read(f"{recipe_dir}/train.sh")

    assert ': "${USE_REPLAY_BUFFER:=1}"' in run_script
    assert ': "${REPLAY_BUFFER_TYPE:=inflight}"' in run_script
    assert ': "${REPLAY_BUFFER_IDENTITY_TAG:=1}"' in run_script
    assert 'ROLLOUT_ARGS+=(--use-replay-buffer --replay-buffer-type "${REPLAY_BUFFER_TYPE}")' in train_script


@pytest.mark.parametrize("recipe_dir", RECIPES, ids=RECIPE_IDS)
def test_domain_recipe_segments_resume_the_same_checkpoint_identity(recipe_dir: str):
    run_script = _read(f"{recipe_dir}/run.sbatch")
    train_script = _read(f"{recipe_dir}/train.sh")
    identity_script = _read("experiments/common/run_identity.sh")
    clean_script = _read("experiments/common/clean_checkpoint.sh")

    identity_source = 'source "${REPO_ROOT}/experiments/common/run_identity.sh"'
    assert identity_source in run_script
    assert "export RUN_NAME CONFIG_TAG CKPT_PATH" in run_script
    assert 'source "${REPO_ROOT}/experiments/common/clean_checkpoint.sh"' in run_script
    assert "SLURM_JOB_ID" not in run_script[: run_script.index(identity_source)]

    config_tag = next(line for line in run_script.splitlines() if line.startswith(': "${CONFIG_TAG:='))
    for stable_axis in (
        "MAX_RESPONSE_LEN",
        "NUM_ROLLOUT",
        "LR",
        "ROLLOUT_BATCH_SIZE",
        "GLOBAL_BATCH_SIZE",
        "N_SAMPLES_PER_PROMPT",
        "TRAIN_SEED",
        "ROLLOUT_SEED",
    ):
        assert stable_axis in config_tag
    assert "SLURM_JOB_ID" not in config_tag

    assert 'RUN_NAME="${RUN_NAME:-' in identity_script
    assert 'CKPT_PATH="/ckpt/training/${TASK_FAMILY}/${DATASET_TAG}/' in identity_script
    assert "SLURM_JOB_ID" not in identity_script
    assert ': "${CLEAN_CHECKPOINT:=0}"' in clean_script
    assert "CLEAN_CHECKPOINT=1" not in run_script

    assert '--load "${CKPT_PATH}"' in train_script
    assert '--save "${CKPT_PATH}"' in train_script
    for default in (
        ': "${SAVE_INTERVAL:=10}"',
        ': "${SAVE_RETAIN_INTERVAL:=100}"',
        ': "${SAVE_HF:=1}"',
        ': "${HF_SAVE_INTERVAL:=10}"',
    ):
        assert default in run_script
    assert '--save-interval "${SAVE_INTERVAL}"' in train_script
    assert '--save-retain-interval "${SAVE_RETAIN_INTERVAL}"' in train_script
    assert '--hf-save-interval "${HF_SAVE_INTERVAL}"' in train_script

    first_identity = _resolve_segment_identity(run_script, slurm_job_id="100001")
    resumed_identity = _resolve_segment_identity(run_script, slurm_job_id="100002")
    assert first_identity == resumed_identity
    assert _resolve_segment_identity(run_script, slurm_job_id="100003", num_rollout="301") != first_identity
    run_name, checkpoint_path, config_tag = first_identity
    assert run_name
    assert len(run_name) + 9 <= 128
    assert config_tag.endswith("-nr300-zero-trunc-rb-inflight")
    assert "/max-weight-staleness-4-from-prefill/" in checkpoint_path
    assert checkpoint_path.endswith(f"/{config_tag}")


@pytest.mark.parametrize("recipe_dir", RECIPES, ids=RECIPE_IDS)
def test_domain_recipe_keeps_runtime_safety_contracts(recipe_dir: str):
    run_script = _read(f"{recipe_dir}/run.sbatch")
    train_script = _read(f"{recipe_dir}/train.sh")
    custom_rm_path = CUSTOM_RM_PATHS[recipe_dir]

    restriction = f"readonly VERIFIED_CUSTOM_RM_PATH={custom_rm_path}"
    assert restriction in run_script
    assert restriction in train_script
    assert '[[ -z "${RM_TYPE:-}"' in train_script
    assert "--custom-rm-path" in train_script

    assert ': "${ZERO_REWARD_ON_TRUNCATED:=1}"' in run_script
    assert "--zero-reward-on-truncated" in train_script
    assert ': "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS:=1}"' in run_script
    assert "--sglang-enable-response-weight-version-segments" in train_script

    for efa_contract in (
        ': "${MILES_NCCL_TRANSPORT:=efa}"',
        '[[ "${NCCL_IB_DISABLE:-0}" != 1 ]]',
        '[[ "${NCCL_NET_PLUGIN:-ofi}" == ofi ]]',
        'NCCL_NET="AWS Libfabric"',
        "FI_PROVIDER=efa",
        'echo "running EFA fail-closed preflight on every node"',
        '--nodes="${SLURM_JOB_NUM_NODES}"',
        '--ntasks="${SLURM_JOB_NUM_NODES}"',
        "bash /root/miles/experiments/common/check_efa.sh",
        "bash /root/miles/experiments/common/run_with_efa_env.sh",
    ):
        assert efa_contract in run_script


def test_tool_call_pivot_recipe_uses_current_sft_and_pinned_dataset() -> None:
    recipe_dir = RECIPES[4]
    run_script = _read(f"{recipe_dir}/run.sbatch")
    train_script = _read(f"{recipe_dir}/train.sh")

    assert "TASK_FAMILY=tool_call_pivot" in run_script
    assert "readonly ROLLOUT_SEMANTICS=static_single_turn_pivot" in run_script
    assert "tool_call_pivot rejects custom multi-turn generation" in train_script
    assert "--custom-generate-function-path" not in train_script
    assert "Qwen3-4B-Instruct-2507" not in run_script
    assert 'MODEL_NAME:=Qwen3-4B-Base-LR2e-5-Step4000' in run_script
    assert 'HF_MODEL_NAME:=iter_0004000' in run_script
    assert (
        "PROMPT_DATA=/data/nemotron-agentic-conv-tooluse-pivot/"
        "nemotron-agentic-conv-tooluse-pivot-train.jsonl"
    ) in run_script


@pytest.mark.parametrize(
    "relative_path",
    tuple(f"{recipe_dir}/{script_name}" for recipe_dir in RECIPES for script_name in ("run.sbatch", "train.sh")),
)
def test_domain_recipe_shell_syntax(relative_path: str):
    subprocess.run(["bash", "-n", str(REPO_ROOT / relative_path)], check=True)
