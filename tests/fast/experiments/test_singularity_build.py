from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
IMPORT_SCRIPT = REPO_ROOT / "experiments/container/import_image.sbatch"
DEFINITION_FILE = REPO_ROOT / "experiments/container/miles.def"


def _captured_arguments(path: Path) -> list[str]:
    return [item.decode() for item in path.read_bytes().split(b"\0") if item]


def test_definition_prepares_non_root_code_and_bind_paths() -> None:
    source = DEFINITION_FILE.read_text(encoding="utf-8")

    assert "Bootstrap: docker" in source
    assert "From: {{ DOCKER_IMAGE }}" in source
    assert "{{ SGLANG_REPO }}" in source
    assert "{{ SGLANG_BRANCH }}" in source
    assert "{{ SGLANG_COMMIT }}" in source
    assert 'python3 -m pip install -e "python[all]" --no-deps' in source
    assert "test_policy_weight_version.py" in source
    assert "apt-get purge -y ${efa_packages}" in source
    assert "rm -rf /opt/amazon" in source
    assert "libfabric1-aws libnccl-ofi-ngc-v3" in source
    assert "libnccl-net-ofi.so*" in source
    assert "libnccl-tuner-ofi.so*" in source
    assert "/etc/libibverbs.d/efa.driver" in source
    assert "/usr/include/infiniband/efadv.h" in source
    assert "/usr/include/rdma/efa-abi.h" in source
    assert "libefa-rdmav*.so" in source
    assert "libefa.so*" in source
    assert "test -r /etc/libibverbs.d/mlx5.driver" in source
    assert "test -r /usr/lib/x86_64-linux-gnu/libibverbs.so.1" in source
    assert "libmlx5-rdmav*.so" in source
    assert source.count("\\( -type f -o -type l \\)") >= 4
    assert "chmod 0755 /root /root/.cache" in source
    assert "chmod -R a+rX" in source
    assert 'mode="$(stat -c %a "${path}")"' in source
    assert '[ "${mode}" = 1777 ]' in source
    assert "fc0055dc4e0a316c3f83133267fbd6faaa770992" in source
    assert "export TAU2_DATA_DIR=/opt/tau3/data" in source
    assert "/opt/tau3/data/tau2/domains/retail/tasks.json" in source
    assert 'importlib.metadata.version("tau2") == "1.0.1"' in source
    for path in (
        "/root/miles",
        "/root/Megatron-LM",
        "/root/.cache",
        "/root/miles-baseline",
        "/data/pre-train",
        "/data/sft",
        "/ckpt/hf",
        "/ckpt/megatron",
        "/ckpt/training",
        "/cache",
        "/cache/home",
        "/checkpoint",
        "/results",
        "/evaluation-data",
        "/evaluation-cache",
        "/workspace/reasoning_eval",
        "/workspace/miles",
    ):
        assert path in source


def test_import_builds_definition_and_requests_unprivileged_smoke_test(tmp_path: Path) -> None:
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_singularity = fake_bin / "singularity"
    fake_singularity.write_text(
        """#!/bin/bash
set -euo pipefail
counter_file="${CAPTURE_DIR}/counter"
counter="$(cat "${counter_file}" 2>/dev/null || printf 0)"
printf '%s\\0' "$@" > "${CAPTURE_DIR}/${counter}.args"
printf '%s' "$((counter + 1))" > "${counter_file}"
if [[ "$1" == build ]]; then
    output="${@: -2:1}"
    printf '%s\n' "$PWD" > "${CAPTURE_DIR}/build.cwd"
    printf '%s\n' "${SINGULARITY_TMPDIR}" > "${CAPTURE_DIR}/build.tmpdir"
    printf '%s\n' "${SINGULARITY_CACHEDIR}" > "${CAPTURE_DIR}/build.cachedir"
    printf 'fake sif\n' > "${output}"
fi
""",
        encoding="utf-8",
    )
    fake_singularity.chmod(0o755)

    workspace = tmp_path / "workspace"
    local_root = tmp_path / "local"
    local_root.mkdir()
    image = workspace / "containers/miles-test.sif"
    stable_image = workspace / "containers/miles.sif"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CAPTURE_DIR": str(capture_dir),
            "MILES_WORKSPACE_ROOT": str(workspace),
            "MILES_REPO": str(REPO_ROOT),
            "DOCKER_IMAGE": "registry.example/miles:test",
            "IMPORT_OUTPUT_IMAGE": str(image),
            "SINGULARITY_LINK": str(stable_image),
            "PBS_O_WORKDIR": str(REPO_ROOT),
            "PBS_LOCALDIR": str(tmp_path / "missing-pbs-localdir"),
            "WANDB_MODE": "disabled",
            "TMPDIR": str(local_root),
        }
    )

    first_result = subprocess.run(
        ["bash", str(IMPORT_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first_result.returncode == 0, first_result.stdout + first_result.stderr

    invocations = [
        _captured_arguments(path)
        for path in sorted(capture_dir.glob("*.args"), key=lambda item: int(item.stem))
    ]
    build = next(arguments for arguments in invocations if arguments[0] == "build")
    smoke_invocations = [arguments for arguments in invocations if arguments[0] == "exec"]
    baked_smoke = next(arguments for arguments in smoke_invocations if "--bind" not in arguments)
    mounted_smoke = next(arguments for arguments in smoke_invocations if "--bind" in arguments)
    build_cwd = Path((capture_dir / "build.cwd").read_text(encoding="utf-8").strip())

    assert "--fakeroot" in build
    assert "--fix-perms" in build
    assert "DOCKER_IMAGE=registry.example/miles:test" in build
    assert "SGLANG_REPO=okoge-kaz/sglang" in build
    assert "SGLANG_BRANCH=miles-staleness-weight-boundaries" in build
    assert "SGLANG_COMMIT=f994b9aedfd0b1465dbb8f4e2a02eb789fc76dce" in build
    assert build[-1] == str(build_cwd / "miles.def")
    assert (build_cwd / "miles.def").parent == build_cwd
    assert "--no-eval" in baked_smoke
    assert "--no-home" in baked_smoke
    assert "--writable-tmpfs" not in baked_smoke
    assert "/root/miles" in baked_smoke[-1]
    assert "test ! -e /opt/amazon" in baked_smoke[-1]
    assert "libfabric1-aws libnccl-ofi-ngc-v3" in baked_smoke[-1]
    assert "libnccl-net-ofi.so*" in baked_smoke[-1]
    assert "test ! -e /etc/libibverbs.d/efa.driver" in baked_smoke[-1]
    assert "libefa-rdmav*.so" in baked_smoke[-1]
    assert "test -r /etc/libibverbs.d/mlx5.driver" in baked_smoke[-1]
    assert "libmlx5-rdmav*.so" in baked_smoke[-1]
    assert baked_smoke[-1].count("\\( -type f -o -type l \\)") >= 3
    assert "! -executable" in baked_smoke[-1]
    assert "! -readable" in baked_smoke[-1]
    assert "--writable-tmpfs" in mounted_smoke
    assert "--no-home" in mounted_smoke
    assert "--no-eval" in mounted_smoke
    assert "/root/miles/.env:ro" in mounted_smoke[mounted_smoke.index("--bind") + 1]
    assert "/usr/bin/bwrap:/usr/local/bin/bwrap:ro" in mounted_smoke[mounted_smoke.index("--bind") + 1]
    assert "/root/Megatron-LM" in mounted_smoke[-1]
    assert "test ! -e /opt/amazon" in mounted_smoke[-1]
    assert "libfabric1-aws libnccl-ofi-ngc-v3" in mounted_smoke[-1]
    assert "libnccl-net-ofi.so*" in mounted_smoke[-1]
    assert "test ! -e /etc/libibverbs.d/efa.driver" in mounted_smoke[-1]
    assert "libefa-rdmav*.so" in mounted_smoke[-1]
    assert "test -r /etc/libibverbs.d/mlx5.driver" in mounted_smoke[-1]
    assert "libmlx5-rdmav*.so" in mounted_smoke[-1]
    assert mounted_smoke[-1].count("\\( -type f -o -type l \\)") >= 3
    assert "test ! -w /root/miles/.env" in mounted_smoke[-1]
    assert 'test "${HOME}" = /cache/home' in mounted_smoke[-1]
    assert 'test "${TAU2_DATA_DIR}" = /opt/tau3/data' in mounted_smoke[-1]
    assert 'load_tasks("retail", "train")' in mounted_smoke[-1]
    assert "test_policy_weight_version.py" in mounted_smoke[-1]
    assert "enable_response_weight_version_segments" in mounted_smoke[-1]
    assert "/sgl-workspace/sglang/python" in mounted_smoke[-1]
    for path in (
        "/root/miles",
        "/root/.cache",
        "/data/pre-train",
        "/data/sft",
        "/ckpt/hf",
        "/ckpt/megatron",
        "/ckpt/training",
        "/cache",
    ):
        assert path in mounted_smoke[-1]
    assert build_cwd.parent == local_root
    assert (capture_dir / "build.tmpdir").read_text(encoding="utf-8").strip() == str(build_cwd)
    assert (capture_dir / "build.cachedir").read_text(encoding="utf-8").strip() == str(build_cwd)
    import_source = IMPORT_SCRIPT.read_text(encoding="utf-8")
    assert 'pbs_local_job_dir="/local/${pbs_job_id}"' in import_source
    assert 'pbs_local_short_job_dir="/local/${job_id}"' in import_source
    assert '"${PBS_LOCALDIR}" == /local/*' in import_source
    assert "PBS jobs require writable node-local storage" in import_source
    assert image.is_file()
    assert image.with_suffix(".sif.sha256").is_file()
    assert image.with_suffix(".sif.provenance.env").is_file()
    assert stable_image.is_symlink()
    assert stable_image.resolve() == image
    assert list(local_root.iterdir()) == []

    subprocess.run(
        ["bash", str(IMPORT_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    all_invocations = [
        _captured_arguments(path)
        for path in sorted(capture_dir.glob("*.args"), key=lambda item: int(item.stem))
    ]
    assert sum(arguments[0] == "build" for arguments in all_invocations) == 1
    assert list(local_root.iterdir()) == []

    provenance = image.with_suffix(".sif.provenance.env")
    provenance_source = provenance.read_text(encoding="utf-8")
    assert "sglang_repo=okoge-kaz/sglang" in provenance_source
    assert "sglang_branch=miles-staleness-weight-boundaries" in provenance_source
    assert "sglang_commit=f994b9aedfd0b1465dbb8f4e2a02eb789fc76dce" in provenance_source
    provenance.write_text(
        provenance_source.replace("definition_sha256=", "definition_sha256=stale-"),
        encoding="utf-8",
    )
    stale_result = subprocess.run(
        ["bash", str(IMPORT_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert stale_result.returncode != 0
    assert "different Singularity definition" in stale_result.stderr
    assert list(local_root.iterdir()) == []


def test_import_uses_native_pbs_without_project_option() -> None:
    source = IMPORT_SCRIPT.read_text(encoding="utf-8")

    assert "#PBS -q R9920261300" in source
    assert "#PBS -l walltime=00:30:00" in source
    assert "#SBATCH" not in source
    assert "#PBS -P" not in source
