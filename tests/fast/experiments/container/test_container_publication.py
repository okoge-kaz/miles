import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "code_sandbox,nemo_gym,failure_stage",
    [
        (False, False, None),
        (False, False, "base"),
        (True, False, None),
        (True, False, "sandbox"),
        (True, True, None),
        (True, True, "sandbox"),
        (True, True, "nemo_gym"),
    ],
)
def test_image_publication_checks_release_pins_and_preserves_link_on_smoke_failure(
    tmp_path, code_sandbox, nemo_gym, failure_stage
):
    singularity = tmp_path / "singularity"
    capture = tmp_path / "singularity.args"
    smoke_stages = tmp_path / "smoke-stages"
    singularity.write_text(
        """#!/bin/bash
printf '%s\\0' "$@" >> "${FAKE_CAPTURE}"
case "$1" in
    build)
        args=("$@")
        printf 'fake-sif\\n' > "${args[${#args[@]}-2]}"
        ;;
    exec)
        stage=base
        for argument in "$@"; do
            case "${argument}" in
                /opt/miles/verify_code_sandbox.py) stage=sandbox ;;
                experiments.src.nemo_gym.runtime.verify) stage=nemo_gym ;;
            esac
        done
        printf '%s\\n' "${stage}" >> "${FAKE_SMOKE_STAGES}"
        [[ "${FAKE_SMOKE_FAIL}" != "${stage}" ]] || exit 12
        ;;
    --version) printf 'singularity-ce version 4.4.1\\n' ;;
esac
"""
    )
    singularity.chmod(0o755)
    containers = tmp_path / "workspace/containers"
    containers.mkdir(parents=True)
    old_image = containers / "old.sif"
    old_image.write_text("old-image")
    stable = containers / "miles-v0.1.2.sif"
    stable.symlink_to(old_image)
    dated = containers / "candidate.sif"
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "WANDB_MODE": "disabled",
        "MILES_WORKSPACE_ROOT": str(tmp_path / "workspace"),
        "TMPDIR": str(tmp_path),
        "IMPORT_OUTPUT_IMAGE": str(dated),
        "DOCKER_IMAGE": "example/miles:test",
        "SINGULARITY_BIN": str(singularity),
        "INSTALL_CODE_SANDBOX": str(int(code_sandbox)),
        "INSTALL_NEMO_GYM": str(int(nemo_gym)),
        "FAKE_CAPTURE": str(capture),
        "FAKE_SMOKE_STAGES": str(smoke_stages),
        "FAKE_SMOKE_FAIL": failure_stage or "none",
    }
    result = subprocess.run(
        ["bash", "experiments/container/import_image.sh"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    arguments = capture.read_bytes().decode().split("\0")
    lock = json.loads((REPO_ROOT / "release-lock.json").read_text())
    assert f"SGLANG_COMMIT={lock['sglang_commit']}" in arguments
    assert f"MEGATRON_COMMIT={lock['megatron_commit']}" in arguments
    assert "DOCKER_IMAGE=example/miles:test" in arguments
    assert f"INSTALL_CODE_SANDBOX={int(code_sandbox)}" in arguments
    assert f"INSTALL_NEMO_GYM={int(nemo_gym)}" in arguments
    expected_stages = ["base", *(["sandbox"] if code_sandbox else []), *(["nemo_gym"] if nemo_gym else [])]
    if failure_stage:
        expected_stages = expected_stages[: expected_stages.index(failure_stage) + 1]
        assert result.returncode == 12
        assert stable.resolve() == old_image
        assert not dated.exists() and not Path(f"{dated}.sha256").exists()
        assert not Path(f"{dated}.provenance.env").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert stable.resolve() == dated
        assert dated.read_text() == "fake-sif\n"
        checksum = subprocess.run(
            ["sha256sum", "--check", f"{dated.name}.sha256"], cwd=containers, capture_output=True
        )
        assert checksum.returncode == 0
        provenance = Path(f"{dated}.provenance.env").read_text()
        assert f"sglang_commit={lock['sglang_commit']}" in provenance
        assert f"code_sandbox_enabled={int(code_sandbox)}" in provenance
        assert f"nemo_gym_enabled={int(nemo_gym)}" in provenance
    assert smoke_stages.read_text().splitlines() == expected_stages
    assert old_image.read_text() == "old-image"
    assert not list(containers.glob("*.partial-*")) and not list(containers.glob("*.publishing-*"))


def test_container_build_features_reach_the_pbs_job(tmp_path):
    result = subprocess.run(
        ["bash", "experiments/setup.sh", "container"],
        cwd=REPO_ROOT,
        env={
            "PATH": os.environ["PATH"],
            "WANDB_MODE": "disabled",
            "MILES_WORKSPACE_ROOT": str(tmp_path / "workspace"),
            "INSTALL_CODE_SANDBOX": "1",
            "INSTALL_NEMO_GYM": "1",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(result.stdout)
    exported = command[command.index("-v") + 1].split(",")
    assert {"INSTALL_CODE_SANDBOX", "INSTALL_NEMO_GYM"} <= set(exported)
    assert "-V" not in command


def test_checkpoint_cleanup_rejects_symlink_escape(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "run"
    protected.mkdir()
    (protected / "weights").write_text("keep")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "_", str(REPO_ROOT / "experiments/common/clean_checkpoint.sh")],
        env={
            "PATH": os.environ["PATH"],
            "TRAIN_CKPT_DIR": str(root),
            "CKPT_PATH": "/ckpt/training/linked/run",
            "CLEAN_CHECKPOINT": "1",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0 and "resolves outside" in result.stderr
    assert (protected / "weights").read_text() == "keep"
