from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _copied_env_script(tmp_path: Path) -> tuple[Path, Path]:
    experiments = Path(__file__).resolve().parents[3] / "experiments"
    source = experiments / "env.sh"
    checkout = tmp_path / "checkout"
    script = checkout / "experiments" / "env.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(source, script)
    common = script.parent / "common"
    common.mkdir()
    for helper_name in ("pbs.sh", "singularity.sh"):
        shutil.copy2(experiments / "common" / helper_name, common / helper_name)
    return checkout, script


def _source_environment(script: Path, environment: dict[str, str]) -> dict[str, str]:
    result = subprocess.run(
        ["bash", "-c", 'set -e; source "$1"; env -0', "bash", str(script)],
        check=False,
        env=environment,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
    return {
        key.decode(): value.decode()
        for item in result.stdout.split(b"\0")
        if item
        for key, value in (item.split(b"=", maxsplit=1),)
    }


def _base_environment(checkout: Path, workspace: Path, home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "MILES_REPO": str(checkout),
        "MILES_WORKSPACE_ROOT": str(workspace),
        "PATH": os.environ["PATH"],
        "USER": "test-user",
        "WANDB_MODE": "disabled",
    }


def test_experiments_env_does_not_read_repository_dotenv(tmp_path: Path) -> None:
    checkout, script = _copied_env_script(tmp_path)
    (checkout / ".env").write_text("MILES_DOTENV_SENTINEL=must-not-load\n", encoding="utf-8")

    environment = _base_environment(checkout, tmp_path / "workspace", tmp_path / "home")

    result = subprocess.run(
        ["bash", "-c", 'set -e; source "$1"; [[ -z "${MILES_DOTENV_SENTINEL:-}" ]]', "bash", str(script)],
        check=False,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_experiments_env_derives_layout_caches_and_mounts(tmp_path: Path) -> None:
    checkout, script = _copied_env_script(tmp_path)
    workspace = tmp_path / "workspace"
    resolved = _source_environment(
        script,
        _base_environment(checkout, workspace, tmp_path / "home"),
    )

    checkpoints = workspace / "checkpoints"
    datasets = workspace / "datasets"
    cache = workspace / "cache"
    expected = {
        "CHECKPOINT_ROOT": checkpoints,
        "CKPT_ROOT": checkpoints,
        "HF_CKPT_DIR": checkpoints / "hf",
        "MEGATRON_CKPT_DIR": checkpoints / "megatron",
        "TRAIN_CKPT_DIR": checkpoints / "training",
        "DATASET_ROOT": datasets,
        "PRETRAIN_DATASET_DIR": datasets / "pre-train",
        "RL_DATASET_DIR": datasets / "rl",
        "DATASET_DIR": datasets / "rl",
        "SFT_DATASET_DIR": datasets / "sft",
        "CONTAINER_DIR": workspace / "containers",
        "CACHE_DIR": cache,
        "SINGULARITY_CACHEDIR": cache / "singularity",
    }
    for name, path in expected.items():
        assert resolved[name] == str(path)
        assert path.is_dir()

    image = workspace / "containers" / "miles.sif"
    assert resolved["CONTAINER_IMAGE"] == str(image)
    assert resolved["ASYNC_CONTAINER_IMAGE_OVERRIDE"] == str(image)
    assert resolved["MILES_NCCL_TRANSPORT"] == "system"
    assert resolved["SGLANG_REPO"] == "okoge-kaz/sglang"
    assert resolved["SGLANG_BRANCH"] == "miles-staleness-weight-boundaries"
    assert resolved["SGLANG_COMMIT"] == "f994b9aedfd0b1465dbb8f4e2a02eb789fc76dce"
    assert resolved["PBS_CONTAINER_WALLTIME"] == "00:30:00"
    assert resolved["PBS_GPU_RESOURCE_TYPE"] == "rt_HF"
    assert resolved["PBS_CPU_RESOURCE_TYPE"] == "rt_HC"
    assert resolved["ABCI_HPCX_MODULE"] == "hpcx/2.20"
    assert resolved["SINGULARITY_TMPDIR"] == "/tmp"
    assert resolved["APPTAINER_TMPDIR"] == "/tmp"

    qwen_parent = checkpoints / "hf"
    assert resolved["QWEN3_4B_BASE_HF_ROOT"].startswith(str(qwen_parent))
    assert resolved["QWEN3_8B_BASE_HF_ROOT"].startswith(str(qwen_parent))
    assert resolved["QWEN3_30B_A3B_BASE_HF_ROOT"].startswith(str(qwen_parent))

    container_cache = "/cache"
    cache_variables = {
        "XDG_CACHE_HOME": f"{container_cache}/xdg",
        "HF_HOME": f"{container_cache}/huggingface",
        "HF_HUB_CACHE": f"{container_cache}/huggingface/hub",
        "HF_DATASETS_CACHE": f"{container_cache}/huggingface/datasets",
        "TRITON_CACHE_DIR": f"{container_cache}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{container_cache}/torchinductor",
        "TORCH_HOME": f"{container_cache}/torch",
        "CUDA_CACHE_PATH": f"{container_cache}/nv_compute",
        "VLLM_CACHE_ROOT": f"{container_cache}/vllm",
        "SGLANG_DG_CACHE_DIR": f"{container_cache}/sglang/deep_gemm",
    }
    for name, value in cache_variables.items():
        assert resolved[name] == value
        assert resolved[f"SINGULARITYENV_{name}"] == value

    assert resolved["CONTAINER_HOME"] == "/cache/home"
    assert resolved["SINGULARITY_HOME"] == f"{cache / 'home'}:/cache/home"
    assert resolved["APPTAINER_HOME"] == f"{cache / 'home'}:/cache/home"
    assert "SINGULARITYENV_HOME" not in resolved
    assert "APPTAINERENV_HOME" not in resolved
    assert (cache / "home").is_dir()
    assert (datasets / "rl/pre-train").is_dir()
    assert (datasets / "rl/sft").is_dir()

    mounts = set(resolved["CONTAINER_MOUNTS"].split(","))
    assert f"{checkout}:/root/miles" in mounts
    assert f"{datasets / 'rl'}:/data" in mounts
    assert f"{datasets / 'pre-train'}:/data/pre-train" in mounts
    assert f"{datasets / 'sft'}:/data/sft" in mounts
    assert f"{cache}:/cache" in mounts
    assert f"{cache}:/root/.cache" in mounts


def test_container_image_override_is_used_for_async_jobs(tmp_path: Path) -> None:
    checkout, script = _copied_env_script(tmp_path)
    image = tmp_path / "qualified.sif"
    environment = _base_environment(checkout, tmp_path / "workspace", tmp_path / "home")
    environment["CONTAINER_IMAGE"] = str(image)

    resolved = _source_environment(script, environment)

    assert resolved["CONTAINER_IMAGE"] == str(image)
    assert resolved["ASYNC_CONTAINER_IMAGE_OVERRIDE"] == str(image)
