import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("smoke_fails", [False, True])
def test_image_publication_checks_release_pins_and_preserves_link_on_smoke_failure(tmp_path, smoke_fails):
    singularity = tmp_path / "singularity"
    capture = tmp_path / "singularity.args"
    singularity.write_text("""#!/bin/bash
printf '%s\\0' "$@" >> "${FAKE_CAPTURE}"
case "$1" in
    build)
        args=("$@")
        printf 'fake-sif\\n' > "${args[${#args[@]}-2]}"
        ;;
    exec) [[ "${FAKE_SMOKE_FAIL}" == 0 ]] || exit 12 ;;
    --version) printf 'singularity-ce version 4.4.1\\n' ;;
esac
""")
    singularity.chmod(0o755)
    containers = tmp_path / "workspace/containers"
    containers.mkdir(parents=True)
    old_image = containers / "old.sif"
    old_image.write_text("old-image")
    stable = containers / "miles-v0.1.1.sif"
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
        "FAKE_CAPTURE": str(capture),
        "FAKE_SMOKE_FAIL": str(int(smoke_fails)),
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
    if smoke_fails:
        assert result.returncode == 12
        assert stable.resolve() == old_image
        assert not dated.exists() and not Path(f"{dated}.sha256").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert stable.resolve() == dated
        assert dated.read_text() == "fake-sif\n"
        checksum = subprocess.run(
            ["sha256sum", "--check", f"{dated.name}.sha256"], cwd=containers, capture_output=True
        )
        assert checksum.returncode == 0
        assert f"sglang_commit={lock['sglang_commit']}" in Path(f"{dated}.provenance.env").read_text()
    assert old_image.read_text() == "old-image"
    assert not list(containers.glob("*.partial-*")) and not list(containers.glob("*.publishing-*"))


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
