"""OCI launcher contracts; no scheduler submission or model download."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = sorted(
    path
    for path in (REPO / "experiments").rglob("*")
    if path.suffix in {".sh", ".sbatch"} and "outputs" not in path.parts
)
SCRIPTS.append(REPO / "tests/manual/run_oci_validation.sbatch")
SCRIPTS.append(REPO / "tests/manual/run_oci_step4000_smoke.sbatch")
SCRIPTS.append(REPO / "tests/manual/run_oci_async_smoke.sbatch")
SCRIPTS.append(REPO / "tests/manual/run_oci_prefill_smoke.sh")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: str(path.relative_to(REPO)))
def test_shell_syntax(path):
    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)


def test_default_account_architecture_and_gpu_shape(tmp_path):
    env = {
        "PATH": os.environ["PATH"],
        "USER": "validation",
        "MILES_REPO": str(tmp_path),
        "WS": str(tmp_path / "workspace"),
        "WANDB_API_KEY": "disabled",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$SLURM_ACCOUNT_NAME" "$GPU_PARTITION" "$INTERACTIVE_QOS" "$GPUS_PER_NODE" "$ACTOR_GPUS_PER_NODE" "$CONTAINER_ARCH" "$MILES_WORKER_PORT_START" "$MILES_WORKER_PORT_END"',
            "bash",
            str(REPO / "experiments/env.sh"),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "nemotron_sw_post",
        "batch",
        "interactive",
        "4",
        "4",
        "aarch64",
        "2000",
        "8999",
    ]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("custom_eval", [False, True])
@pytest.mark.parametrize("saved_scheduler", [False, True])
def test_eval_dataset_override_reaches_the_container(tmp_path, mode, custom_eval, saved_scheduler):
    env = {
        "PATH": os.environ["PATH"],
        "USER": "validation",
        "MILES_REPO": str(REPO),
        "WS": str(tmp_path),
        "SHARED_WS": str(tmp_path),
        "DATASET_DIR": str(tmp_path / "datasets"),
        "HF_CKPT_DIR": str(tmp_path / "hf"),
        "MEGATRON_CKPT_DIR": str(tmp_path / "megatron"),
        "CONTAINER_DIR": str(tmp_path / "container"),
        "OUTPUT_DIR": str(tmp_path / "outputs"),
        "SLURM_JOB_ID": "validation",
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_CPUS_PER_NODE": "128(x2)",
        "WANDB_API_KEY": "disabled",
    }
    expected = ["aime25", "/data/aime-2025/aime-2025.jsonl"]
    if custom_eval:
        expected = ["dapo-smoke", "/data/custom.jsonl@[0:2]"]
        env.update(EVAL_DATASET_NAME=expected[0], EVAL_PROMPT_DATA=expected[1])
    if saved_scheduler:
        env["USE_CHECKPOINT_OPT_PARAM_SCHEDULER"] = "1"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'srun() { printf "eval-contract:%s|%s\\n" "$EVAL_DATASET_NAME" "$EVAL_PROMPT_DATA"; '
            'printf "scheduler-contract:%s\\n" "$USE_CHECKPOINT_OPT_PARAM_SCHEDULER"; }; '
            'export -f srun; exec bash "$1"',
            "bash",
            str(REPO / f"experiments/scripts/math/{mode}/dapo-math-p10-90/qwen3-4b/run.sbatch"),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"eval-contract:{expected[0]}|{expected[1]}" in result.stdout.splitlines()
    assert f"scheduler-contract:{int(saved_scheduler)}" in result.stdout.splitlines()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("saved_scheduler", [False, True])
def test_checkpoint_scheduler_option_reaches_train_argv(mode, saved_scheduler):
    script = (REPO / f"experiments/scripts/math/{mode}/dapo-math-p10-90/qwen3-4b/train.sh").read_text()
    block = "CKPT_ARGS=(" + script.split("CKPT_ARGS=(", 1)[1].split("\nROLLOUT_ARGS=(", 1)[0]
    env = {
        "PATH": os.environ["PATH"],
        "HF_MODEL_NAME": "test-hf",
        "MODEL_NAME": "test-mcore",
        "CKPT_PATH": "/test/checkpoint",
        "SAVE_INTERVAL": "1",
        "SAVE_RETAIN_INTERVAL": "1",
        "SAVE_HF": "0",
        "USE_CHECKPOINT_OPT_PARAM_SCHEDULER": str(int(saved_scheduler)),
    }
    result = subprocess.run(
        ["bash", "-euc", block + '\nprintf "%s\\n" "${CKPT_ARGS[@]}"'],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert ("--use-checkpoint-opt-param-scheduler" in result.stdout.splitlines()) == saved_scheduler


@pytest.mark.parametrize(
    "path", [p for p in SCRIPTS if p.suffix == ".sbatch"], ids=lambda path: str(path.relative_to(REPO))
)
def test_sbatch_uses_current_partition_and_explicit_account(path):
    directives = dict(
        line.removeprefix("#SBATCH --").split("=", 1)
        for line in path.read_text().splitlines()
        if line.startswith("#SBATCH --") and "=" in line
    )
    assert directives["account"] == "nemotron_sw_post"
    assert directives["partition"] in {"batch", "batch_long", "cpu", "cpu_datamover"}
    if "gres" in directives:
        assert directives["gres"] == "gpu:4"
    if "gpus-per-node" in directives:
        assert directives["gpus-per-node"] == "4"
