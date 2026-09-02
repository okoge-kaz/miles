from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PBS_HELPER = REPO_ROOT / "experiments/common/pbs.sh"
SINGULARITY_HELPER = REPO_ROOT / "experiments/common/singularity.sh"


def _scheduler_env() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("MILES_JOB_") or name in {
            "MILES_NODE_RANK",
            "MILES_SUBMIT_DIR",
            "PBS_JOBID",
            "PBS_LOCALDIR",
            "PBS_NODEFILE",
            "PBS_O_WORKDIR",
        }:
            environment.pop(name)
    return environment


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _captured_arguments(path: Path) -> list[str]:
    return [item.decode() for item in path.read_bytes().split(b"\0") if item]


def test_native_job_headers_match_queue_chunk_contracts() -> None:
    job_files = tuple(
        path
        for root_name in ("experiments", "tests", "examples")
        for path in (REPO_ROOT / root_name).rglob("*.sbatch")
    )
    assert job_files

    for job_file in job_files:
        source = job_file.read_text(encoding="utf-8")
        assert "#PBS -q R9920261300\n" in source, job_file
        if ":ngpus=8:" in source:
            assert ":ncpus=192:ngpus=8:mpiprocs=1\n" in source, job_file
            assert "#PBS -l place=scatter:excl\n" in source, job_file
        else:
            assert "#PBS -l select=1:ncpus=32:mpiprocs=1\n" in source, job_file
            assert "#PBS -l place=scatter\n" in source, job_file
        assert "#PBS -P" not in source, job_file


def test_pbs_submit_translates_sbatch_headers_and_preserves_full_job_id(tmp_path: Path) -> None:
    capture = tmp_path / "qsub.args"
    cwd_capture = tmp_path / "qsub.cwd"
    fake_qsub = tmp_path / "qsub"
    _write_executable(
        fake_qsub,
        """#!/bin/bash
printf '%s\\0' "$@" > "${QSUB_CAPTURE}"
printf '%s\\n' "$PWD" > "${QSUB_CWD_CAPTURE}"
printf '31415.pbs1\\n'
""",
    )
    script = tmp_path / "training.sbatch"
    script.write_text(
        """#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --time=1-02:03:04
#SBATCH --job-name=header-name
#SBATCH --output=logs/%x-%j.log
#SBATCH --export=ALL,FOO=bar
true
""",
        encoding="utf-8",
    )
    submit_directory = tmp_path / "submit"
    submit_directory.mkdir()
    environment = _scheduler_env()
    environment.update(
        {
            "PBS_QSUB_BIN": str(fake_qsub),
            "QSUB_CAPTURE": str(capture),
            "QSUB_CWD_CAPTURE": str(cwd_capture),
            "MILES_SUBMIT_DIR": str(submit_directory),
        }
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; pbs_submit --parsable -A legacy --qos=normal '
            '--dependency=afterok:2718.pbs1 "$2"',
            "bash",
            str(PBS_HELPER),
            str(script),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = _captured_arguments(capture)
    resources = [arguments[index + 1] for index, value in enumerate(arguments) if value == "-l"]
    assert result.stdout.strip() == "31415.pbs1"
    assert cwd_capture.read_text(encoding="utf-8").strip() == str(submit_directory)
    assert "-P" not in arguments
    assert "-d" not in arguments
    assert "legacy" not in arguments
    assert arguments[arguments.index("-q") + 1] == "R9920261300"
    assert "select=2:ncpus=192:ngpus=8:mpiprocs=1" in resources
    assert "place=scatter:excl" in resources
    assert "walltime=26:03:04" in resources
    assert arguments[arguments.index("-N") + 1] == "header-name"
    assert arguments[arguments.index("-W") + 1] == "depend=afterok:2718.pbs1"
    assert arguments[arguments.index("-o") + 1] == f"{submit_directory}/logs/"
    assert "-V" in arguments
    assert arguments[arguments.index("-v") + 1] == "RTYPE=rt_HF,FOO=bar"
    command_index = arguments.index("--")
    assert arguments[command_index + 1 :] == [
        "/bin/bash",
        "-c",
        'cd -- "$1" && shift && exec /bin/bash "$@"',
        "miles-pbs-launch",
        str(submit_directory),
        str(script),
    ]


def test_pbs_submit_reads_native_pbs_cpu_profile(tmp_path: Path) -> None:
    capture = tmp_path / "qsub.args"
    fake_qsub = tmp_path / "qsub"
    _write_executable(
        fake_qsub,
        """#!/bin/bash
printf '%s\\0' "$@" > "${QSUB_CAPTURE}"
printf '42.pbs1\\n'
""",
    )
    container_directory = tmp_path / "container"
    container_directory.mkdir()
    script = container_directory / "import.sbatch"
    script.write_text(
        """#!/bin/bash
#PBS -N import-image
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l walltime=02:00:00
true
""",
        encoding="utf-8",
    )
    environment = _scheduler_env()
    environment.update(
        {
            "PBS_QSUB_BIN": str(fake_qsub),
            "QSUB_CAPTURE": str(capture),
            "MILES_SUBMIT_DIR": str(tmp_path),
        }
    )

    subprocess.run(
        [str(PBS_HELPER), "--parsable", "-N", "override-name", str(script)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = _captured_arguments(capture)
    resources = [arguments[index + 1] for index, value in enumerate(arguments) if value == "-l"]
    assert arguments[arguments.index("-q") + 1] == "R9920261300"
    assert "select=1:ncpus=32:mpiprocs=1" in resources
    assert "place=scatter" in resources
    assert "walltime=02:00:00" in resources
    assert arguments[arguments.index("-N") + 1] == "override-name"
    assert arguments[arguments.index("-v") + 1] == "RTYPE=rt_HC"


def test_pbs_context_uses_unique_nodefile_hosts_and_full_job_id(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node-a.domain\nnode-a.domain\nnode-b.domain\n", encoding="utf-8")
    submit_directory = tmp_path / "submit"
    job_tmp = tmp_path / "job-tmp"
    pbs_local = tmp_path / "pbs-local"
    environment = _scheduler_env()
    environment.update(
        {
            "PBS_JOBID": "1234.pbs1",
            "PBS_NODEFILE": str(nodefile),
            "PBS_O_WORKDIR": str(submit_directory),
            "PBS_LOCALDIR": str(pbs_local),
            "TMPDIR": str(job_tmp),
            "MILES_PBS_HOSTNAME": "node-b",
        }
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s|%s|%s|%s|%s\\n" "$MILES_JOB_ID" '
            '"$MILES_JOB_NUM_NODES" "$MILES_NODE_RANK" "$MILES_SUBMIT_DIR" '
            '"$MILES_JOB_TMPDIR"',
            "bash",
            str(PBS_HELPER),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == (
        f"1234.pbs1|2|1|{submit_directory}|{pbs_local}"
    )


def test_miles_srun_builds_singularity_command(tmp_path: Path) -> None:
    capture = tmp_path / "singularity.args"
    fake_singularity = tmp_path / "singularity"
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    _write_executable(
        fake_singularity,
        """#!/bin/bash
printf '%s\\0' "$@" > "${SINGULARITY_CAPTURE}"
""",
    )
    _write_executable(fake_nvidia_smi, "#!/bin/bash\nexit 0\n")
    environment = _scheduler_env()
    environment.update(
        {
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "SINGULARITY_BIN": str(fake_singularity),
            "SINGULARITY_CAPTURE": str(capture),
        }
    )

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; miles_srun --nodes=1 --ntasks=1 '
            '--image=image.sif --bind=/source:/target:ro '
            '--cwd=/work --writable-tmpfs --no-home --fakeroot '
            '--env=ALL,FOO=bar bash -lc "echo ok"',
            "bash",
            str(SINGULARITY_HELPER),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = _captured_arguments(capture)
    assert arguments[:6] == [
        "exec",
        "--no-eval",
        "--fakeroot",
        "--nv",
        "--writable-tmpfs",
        "--no-home",
    ]
    assert arguments[arguments.index("--pwd") + 1] == "/work"
    assert arguments[arguments.index("--bind") + 1] == "/source:/target:ro"
    assert arguments[arguments.index("--env") + 1] == "FOO=bar"
    assert arguments[-4:] == ["image.sif", "bash", "-lc", "echo ok"]


def test_container_exec_all_uses_unbound_mpi_process_per_unique_node(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node-a\nnode-a\nnode-b\n", encoding="utf-8")
    fake_mpirun = tmp_path / "mpirun"
    fake_singularity = tmp_path / "singularity"
    _write_executable(
        fake_mpirun,
        """#!/bin/bash
printf '%s\\0' "$@" > "${MPIRUN_CAPTURE}"
printf '%s\\n' "${FI_PROVIDER-unset}" > "${MPIRUN_FI_CAPTURE}"
rank_count=""
task_file=""
while (( $# > 0 )); do
    if [[ "$1" == -hostfile ]]; then
        cp -- "$2" "${MPIRUN_HOSTS_CAPTURE}"
        shift 2
    elif [[ "$1" == -np ]]; then
        rank_count="$2"
        shift 2
    elif [[ "$1" == /bin/bash ]]; then
        task_file="$2"
        break
    else
        shift
    fi
done
for (( rank = 0; rank < rank_count; rank++ )); do
    OMPI_COMM_WORLD_RANK="${rank}" /bin/bash "${task_file}"
done
""",
    )
    _write_executable(
        fake_singularity,
        """#!/bin/bash
printf '%s|%s|%s\\n' "$MILES_NODE_RANK" "$RUNTIME_ONLY" "${FI_PROVIDER-unset}" > "${SINGULARITY_CAPTURE}.${MILES_NODE_RANK}"
""",
    )
    environment = _scheduler_env()
    environment.update(
        {
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "PBS_JOBID": "987.pbs1",
            "PBS_NODEFILE": str(nodefile),
            "SINGULARITY_BIN": str(fake_singularity),
            "MPIRUN_CAPTURE": str(tmp_path / "mpirun.args"),
            "MPIRUN_HOSTS_CAPTURE": str(tmp_path / "mpirun.hosts"),
            "MPIRUN_FI_CAPTURE": str(tmp_path / "mpirun.fi-provider"),
            "SINGULARITY_CAPTURE": str(tmp_path / "rank"),
            "MILES_NODE_STATUS_ROOT": str(tmp_path / "task-status"),
            "MILES_CONTAINER_NV": "0",
            "ABCI_HPCX_MODULE": "hpcx/test",
        }
    )
    environment.pop("FI_PROVIDER", None)
    environment.pop("MPIRUN_BIN", None)

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; module() { export FI_PROVIDER=mlx; }; '
            'export RUNTIME_ONLY=from-head; '
            'miles_container_exec_all --image=image.sif -- true',
            "bash",
            str(SINGULARITY_HELPER),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = _captured_arguments(tmp_path / "mpirun.args")
    assert arguments[arguments.index("-np") + 1] == "2"
    assert arguments[arguments.index("-map-by") + 1] == "ppr:1:node"
    assert arguments[arguments.index("-bind-to") + 1] == "none"
    assert (tmp_path / "mpirun.hosts").read_text(encoding="utf-8") == (
        "node-a\nnode-b\n"
    )
    assert (tmp_path / "mpirun.fi-provider").read_text(encoding="utf-8").strip() == "mlx"
    assert (tmp_path / "rank.0").read_text(encoding="utf-8").strip() == (
        "0|from-head|unset"
    )
    assert (tmp_path / "rank.1").read_text(encoding="utf-8").strip() == (
        "1|from-head|unset"
    )


def test_container_exec_all_propagates_masked_remote_failure(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node-a\nnode-b\n", encoding="utf-8")
    fake_mpirun = tmp_path / "mpirun"
    fake_singularity = tmp_path / "singularity"
    _write_executable(
        fake_mpirun,
        """#!/bin/bash
rank_count=""
task_file=""
while (( $# > 0 )); do
    if [[ "$1" == -np ]]; then
        rank_count="$2"
        shift 2
    elif [[ "$1" == /bin/bash ]]; then
        task_file="$2"
        break
    else
        shift
    fi
done
for (( rank = 0; rank < rank_count; rank++ )); do
    OMPI_COMM_WORLD_RANK="${rank}" /bin/bash "${task_file}" || true
done
exit 0
""",
    )
    _write_executable(
        fake_singularity,
        """#!/bin/bash
[[ "$MILES_NODE_RANK" != 1 ]] || exit 19
""",
    )
    environment = _scheduler_env()
    environment.update(
        {
            "PBS_JOBID": "654.pbs1",
            "PBS_NODEFILE": str(nodefile),
            "MPIRUN_BIN": str(fake_mpirun),
            "SINGULARITY_BIN": str(fake_singularity),
            "MILES_NODE_STATUS_ROOT": str(tmp_path / "task-status"),
            "MILES_CONTAINER_NV": "0",
        }
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; miles_container_exec_all --image=image.sif -- true',
            "bash",
            str(SINGULARITY_HELPER),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 19
    assert "node rank 1 exited 19" in result.stderr
