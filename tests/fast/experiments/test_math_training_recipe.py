from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = Path("experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b")


def _read(name: str) -> str:
    return (REPO_ROOT / RECIPE_DIR / name).read_text(encoding="utf-8")


def _resolve_segment_identity(slurm_job_id: str) -> tuple[str, str, str]:
    environment = {
        "PATH": os.environ["PATH"],
        "SLURM_JOB_ID": slurm_job_id,
        "MODEL_NAME": "Qwen3-4B-Base-LR2e-5-Step4000",
        "DATASET_TAG": "dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000",
        "TASK_FAMILY": "math",
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
        "NUM_ROLLOUT": "300",
        "NUM_STEPS_PER_ROLLOUT": "1",
        "ROLLOUT_BATCH_SIZE": "192",
        "GLOBAL_BATCH_SIZE": "3072",
        "N_SAMPLES_PER_PROMPT": "16",
        "TRAIN_SEED": "1234",
        "ROLLOUT_SEED": "42",
        "QUEUE_TYPE": "queue-recycle",
        "QUEUE_FACTOR": "1",
        "MAX_WEIGHT_STALENESS": "8",
        "STALENESS_REFERENCE": "prefill",
        "PAUSE_GENERATION_MODE": "in_place",
        "ZERO_REWARD_ON_TRUNCATED": "1",
        "USE_REPLAY_BUFFER": "1",
        "REPLAY_BUFFER_TYPE": "inflight",
        "REPLAY_BUFFER_IDENTITY_TAG": "1",
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


def test_math_recipe_uses_the_sft_checkpoint_and_production_shape():
    run_script = _read("run.sbatch")
    train_script = _read("train.sh")

    for directive in (
        "#SBATCH --partition=batch",
        "#SBATCH --qos=interactive",
        "#SBATCH --time=04:00:00",
    ):
        assert directive in run_script
    for default in (
        ': "${MODEL_NAME:=Qwen3-4B-Base-LR2e-5-Step4000}"',
        ': "${HF_MODEL_NAME:=iter_0004000}"',
        ': "${MAX_RESPONSE_LEN:=16384}"',
        ': "${NUM_ROLLOUT:=300}"',
        ': "${ROLLOUT_BATCH_SIZE:=192}"',
        ': "${N_SAMPLES_PER_PROMPT:=16}"',
        ': "${GLOBAL_BATCH_SIZE:=3072}"',
        ': "${EVAL_INTERVAL:=0}"',
    ):
        assert default in run_script
    assert "Qwen3-4B-Instruct-2507" not in run_script
    assert '--rollout-max-response-len "${MAX_RESPONSE_LEN}"' in train_script
    assert '--n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"' in train_script


def test_math_recipe_persists_and_restores_inflight_replay():
    run_script = _read("run.sbatch")
    train_script = _read("train.sh")

    assert ': "${USE_REPLAY_BUFFER:=1}"' in run_script
    assert ': "${REPLAY_BUFFER_TYPE:=inflight}"' in run_script
    assert ': "${REPLAY_BUFFER_IDENTITY_TAG:=1}"' in run_script
    assert 'ROLLOUT_ARGS+=(--use-replay-buffer --replay-buffer-type "${REPLAY_BUFFER_TYPE:-rollout}")' in train_script
    assert '--load          "${CKPT_PATH}"' in train_script
    assert '--save          "${CKPT_PATH}"' in train_script
    assert '--debug-exit-after-rollout "${DEBUG_EXIT_AFTER_ROLLOUT}"' in train_script


def test_math_recipe_is_efa_fail_closed():
    run_script = _read("run.sbatch")

    for contract in (
        ': "${MILES_NCCL_TRANSPORT:=efa}"',
        '[[ "${NCCL_IB_DISABLE:-0}" != 1 ]]',
        '[[ "${NCCL_NET_PLUGIN:-ofi}" == ofi ]]',
        'NCCL_NET="AWS Libfabric"',
        "FI_PROVIDER=efa",
        "bash /root/miles/experiments/common/check_efa.sh",
        "bash /root/miles/experiments/common/run_with_efa_env.sh",
    ):
        assert contract in run_script


def test_math_segments_keep_checkpoint_identity_across_job_ids():
    first = _resolve_segment_identity("307062")
    resumed = _resolve_segment_identity("307063")

    assert first == resumed
    run_name, checkpoint_path, config_tag = first
    assert run_name
    assert config_tag.endswith("-zero-trunc-rb-inflight")
    assert "/max-weight-staleness-8-from-prefill/" in checkpoint_path
    assert checkpoint_path.endswith(config_tag)


def test_math_recipe_shell_syntax():
    for name in ("run.sbatch", "train.sh"):
        subprocess.run(["bash", "-n", str(REPO_ROOT / RECIPE_DIR / name)], check=True)
