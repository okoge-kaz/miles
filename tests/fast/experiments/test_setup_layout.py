from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_ROOT = REPO_ROOT / "experiments/setup"
SETUP_ENTRYPOINT = REPO_ROOT / "experiments/setup.sh"

CANONICAL_SETUP_DIRS = frozenset(
    {
        "datasets",
        "download",
        "environments",
        "manifests",
        "models",
    }
)
LEGACY_ROOT_SCRIPT_NAMES = (
    "build_math_jsonl.py",
    "prepare_gpqa.sbatch",
    "prepare_nemotron_rl_math.sbatch",
    "prepare_search_r1.sbatch",
    "prepare_tau_bench.sbatch",
)
SHELL_FILES = tuple(
    sorted(
        (*SETUP_ROOT.rglob("*.sh"), *SETUP_ROOT.rglob("*.sbatch"), SETUP_ENTRYPOINT),
        key=lambda path: path.as_posix(),
    )
)
PBS_JOB_FILES = tuple(
    path
    for path in SHELL_FILES
    if path.suffix == ".sbatch"
)
PBS_SINGULARITY_DOWNLOAD_WORKERS = (
    SETUP_ROOT / "download/download_model.sbatch",
    SETUP_ROOT / "download/download_dataset.sbatch",
    SETUP_ROOT / "download/download_swe_datasets.sbatch",
    SETUP_ROOT / "download/download_areal_tau2.sbatch",
)
PBS_SINGULARITY_CONVERT_WORKER = SETUP_ROOT / "models/convert_checkpoint.sbatch"
CPU_ASSET_WORKERS = (
    REPO_ROOT / "experiments/container/import_image.sbatch",
    *sorted((SETUP_ROOT / "download").glob("*.sbatch")),
)

ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")
SUBMISSION_COMMAND = re.compile(r"(?<![.\w-])(?:sbatch|pbs_submit)\s")
EXPORT_OPTION = re.compile(r"--export=(?:\"([^\"]*)\"|'([^']*)'|([^\s)]+))")
SETUP_PATH_REFERENCE = re.compile(
    r"experiments/setup/[A-Za-z0-9_./-]+\.(?:json|py|sbatch|sh|tsv|txt)\b"
)
SOURCE_SUFFIXES = frozenset(
    {".json", ".md", ".py", ".sbatch", ".sh", ".toml", ".tsv", ".txt", ".yaml", ".yml"}
)
SKIPPED_SOURCE_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules", "outputs"})


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _logical_shell_lines(source: str) -> tuple[str, ...]:
    without_continuations = re.sub(r"\\\r?\n[ \t]*", " ", source)
    return tuple(line.strip() for line in without_continuations.splitlines())


def _repository_source_files() -> tuple[Path, ...]:
    source_files: list[Path] = []
    for source_root in (REPO_ROOT / "experiments", REPO_ROOT / "tests"):
        for directory, directory_names, file_names in os.walk(source_root):
            directory_names[:] = sorted(name for name in directory_names if name not in SKIPPED_SOURCE_DIRS)
            for file_name in sorted(file_names):
                path = Path(directory) / file_name
                if path.suffix in SOURCE_SUFFIXES and path.resolve() != Path(__file__).resolve():
                    source_files.append(path)
    root_readme = REPO_ROOT / "README.md"
    if root_readme.is_file():
        source_files.append(root_readme)
    return tuple(source_files)


def _export_spec(command: str, source: str) -> str:
    match = EXPORT_OPTION.search(command)
    assert match is not None, f"scheduler submission lacks --export: {command}"
    spec = next(group for group in match.groups() if group is not None)

    if spec == "${EXPORT_NAMES}":
        declaration = re.search(
            r'^readonly EXPORT_NAMES="([A-Z0-9_,]+)"[ \t]*$',
            source,
            re.MULTILINE,
        )
        assert declaration is not None, "EXPORT_NAMES must be a literal, fixed-name allowlist"
        spec = declaration.group(1)

    if "${SETUP_PATH_EXPORTS}" in spec:
        declaration = re.search(
            r'^SETUP_PATH_EXPORTS="([A-Z0-9_,]+)"[ \t]*$',
            source,
            re.MULTILINE,
        )
        assert declaration is not None, "SETUP_PATH_EXPORTS must be a fixed-name allowlist"
        spec = spec.replace("${SETUP_PATH_EXPORTS}", declaration.group(1))

    if spec == "${exports}":
        required_names = (
            "MILES_WORKSPACE_ROOT",
            "MILES_REPO",
            "CONTAINER_DIR",
            "CACHE_DIR",
            "CONTAINER_IMAGE",
            "DOCKER_IMAGE",
            "SINGULARITY_LINK",
            "WANDB_MODE",
        )
        assignment = re.search(r'^    exports="([^"]+)"$', source, re.MULTILINE)
        assert assignment is not None, "exports must be constructed from a literal fixed-name list"
        for name in required_names:
            assert f"{name}=" in assignment.group(1)
        spec = ",".join((*required_names, "IMPORT_OUTPUT_IMAGE"))

    if "${HF_TOKEN_EXPORT}" in spec:
        assert 'HF_TOKEN_EXPORT=""' in source
        assert 'HF_TOKEN_EXPORT=",HF_TOKEN"' in source
        spec = spec.replace("${HF_TOKEN_EXPORT}", ",HF_TOKEN")

    return spec


def _active_manifest_rows(path: Path) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.partition("#")[0].strip()
        if line:
            rows.append(tuple(field.strip() for field in line.split("|")))
    return tuple(rows)


def _setup_test_environment(workspace: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MILES_WORKSPACE_ROOT": str(workspace),
            "MILES_REPO": str(REPO_ROOT),
            "CHECKPOINT_ROOT": str(workspace / "checkpoints"),
            "HF_CKPT_DIR": str(workspace / "checkpoints/hf"),
            "MEGATRON_CKPT_DIR": str(workspace / "checkpoints/megatron"),
            "TRAIN_CKPT_DIR": str(workspace / "checkpoints/training"),
            "DATASET_ROOT": str(workspace / "datasets"),
            "PRETRAIN_DATASET_DIR": str(workspace / "datasets/pre-train"),
            "RL_DATASET_DIR": str(workspace / "datasets/rl"),
            "SFT_DATASET_DIR": str(workspace / "datasets/sft"),
            "CONTAINER_DIR": str(workspace / "containers"),
            "CONTAINER_IMAGE": str(workspace / "containers/miles.sif"),
            "CACHE_DIR": str(workspace / "cache"),
            "SETUP_SUBMIT_DELAY_SECONDS": "0",
            "WANDB_MODE": "disabled",
        }
    )
    return environment


@pytest.mark.parametrize("shell_file", SHELL_FILES, ids=_relative)
def test_setup_shell_files_pass_bash_syntax(shell_file: Path):
    subprocess.run(
        ["bash", "-n", str(shell_file)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_setup_uses_only_canonical_top_level_layout():
    entries = {path.name: path for path in SETUP_ROOT.iterdir()}
    assert set(entries) == CANONICAL_SETUP_DIRS | {"README.md"}
    assert entries["README.md"].is_file()
    for directory_name in CANONICAL_SETUP_DIRS:
        assert entries[directory_name].is_dir()

    root_level_scripts = [
        _relative(path)
        for path in SETUP_ROOT.iterdir()
        if path.is_file() and path.suffix in {".py", ".sbatch", ".sh"}
    ]
    assert root_level_scripts == []


def test_unified_setup_entrypoint_lists_asset_groups_and_dry_runs(tmp_path: Path):
    workspace = tmp_path / "workspace"
    environment = _setup_test_environment(workspace)

    assert SETUP_ENTRYPOINT.is_file()
    assert os.access(SETUP_ENTRYPOINT, os.X_OK)
    listed = subprocess.run(
        [str(SETUP_ENTRYPOINT), "list"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for group in ("container", "models", "datasets", "sft", "all"):
        assert re.search(rf"^{group}\s", listed, re.MULTILINE)

    preview = subprocess.run(
        [str(SETUP_ENTRYPOINT), "container"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "dry-run: pbs_submit" in preview
    assert "--profile=cpu" in preview
    assert "--time=00:30:00" in preview
    assert "experiments/container/import_image.sbatch" in preview

    all_preview = subprocess.run(
        [str(SETUP_ENTRYPOINT), "all"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "would convert" in all_preview.stdout
    assert "preview continues with 3 missing external SFT source file(s)" in all_preview.stderr

    subprocess.run(
        [str(SETUP_ENTRYPOINT), "init"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    expected_directories = (
        "cache",
        "checkpoints/hf",
        "checkpoints/megatron",
        "checkpoints/training",
        "containers",
        "datasets/pre-train",
        "datasets/rl",
        "datasets/sft",
        "src",
    )
    for relative_directory in expected_directories:
        assert (workspace / relative_directory).is_dir()


def test_sif_and_download_workers_use_cpu_resources():
    for worker in CPU_ASSET_WORKERS:
        source = worker.read_text(encoding="utf-8")
        assert re.search(r"^#PBS -q R9920261300$", source, re.MULTILINE), _relative(worker)
        assert re.search(
            r"^#PBS -l select=1:ncpus=32:mpiprocs=1$",
            source,
            re.MULTILINE,
        ), _relative(worker)
        assert "ngpus=" not in source
        assert "RTYPE=" not in source


def test_dataset_setup_submits_only_cpu_jobs(tmp_path: Path):
    workspace = tmp_path / "workspace"
    environment = _setup_test_environment(workspace)
    environment.pop("HF_TOKEN", None)
    qsub_log = tmp_path / "qsub.log"
    fake_qsub = tmp_path / "qsub"
    fake_qsub.write_text(
        "#!/bin/bash\n"
        "printf '%q ' \"$@\" >>\"${QSUB_LOG}\"\n"
        "printf '\\n' >>\"${QSUB_LOG}\"\n"
        "printf '12345.pbs1\\n'\n",
        encoding="utf-8",
    )
    fake_qsub.chmod(0o700)
    environment["PBS_QSUB_BIN"] = str(fake_qsub)
    environment["QSUB_LOG"] = str(qsub_log)

    result = subprocess.run(
        [str(SETUP_ENTRYPOINT), "datasets", "--submit"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    submissions = qsub_log.read_text(encoding="utf-8").splitlines()
    manifest = SETUP_ROOT / "manifests/datasets.txt"
    public_dataset_count = sum(
        row[1] != "Idavidrein/gpqa" for row in _active_manifest_rows(manifest)
    )
    assert len(submissions) == 1 + public_dataset_count
    assert "Idavidrein/gpqa" not in qsub_log.read_text(encoding="utf-8")
    assert "HF_TOKEN is unset" in result.stderr
    for submission in submissions:
        assert "-q R9920261300" in submission
        assert "ngpus=" not in submission
        assert "RTYPE=rt_HC" in submission
    for submission in submissions[1:]:
        assert "depend=afterok:12345.pbs1" in submission


def test_dataset_setup_submits_gpqa_when_hf_token_is_available(tmp_path: Path):
    workspace = tmp_path / "workspace"
    environment = _setup_test_environment(workspace)
    qsub_log = tmp_path / "qsub.log"
    fake_qsub = tmp_path / "qsub"
    fake_qsub.write_text(
        "#!/bin/bash\n"
        "printf '%q ' \"$@\" >>\"${QSUB_LOG}\"\n"
        "printf '\\n' >>\"${QSUB_LOG}\"\n"
        "printf '12345.pbs1\\n'\n",
        encoding="utf-8",
    )
    fake_qsub.chmod(0o700)
    environment.update(
        {
            "HF_TOKEN": "test-token-must-not-appear",
            "PBS_QSUB_BIN": str(fake_qsub),
            "QSUB_LOG": str(qsub_log),
        }
    )

    subprocess.run(
        [str(SETUP_ENTRYPOINT), "datasets", "--submit"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    submissions = qsub_log.read_text(encoding="utf-8").splitlines()
    gpqa_submission = next(line for line in submissions if "Idavidrein/gpqa" in line)
    assert "HF_TOKEN" in gpqa_submission
    assert "test-token-must-not-appear" not in gpqa_submission


def test_dataset_status_requires_recursive_payload_and_provenance(tmp_path: Path):
    workspace = tmp_path / "workspace"
    environment = _setup_test_environment(workspace)
    target = workspace / "datasets/rl/dapo-math-17k"
    nested_payload = target / "data/snapshots/main/train.jsonl"
    nested_payload.parent.mkdir(parents=True)
    nested_payload.write_text('{"problem": "1 + 1"}\n', encoding="utf-8")

    incomplete = subprocess.run(
        [str(SETUP_ENTRYPOINT), "status"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dataset_count = len(_active_manifest_rows(SETUP_ROOT / "manifests/datasets.txt"))
    assert f"complete=0/{dataset_count}" in incomplete

    (target / "MILES_SOURCE_PROVENANCE").write_text(
        "repo=zhuzilin/dapo-math-17k\nrevision=main\n",
        encoding="utf-8",
    )
    complete = subprocess.run(
        [str(SETUP_ENTRYPOINT), "status"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert f"complete=1/{dataset_count}" in complete


def test_dataset_downloader_limits_retries_and_publishes_marker_atomically():
    source = (SETUP_ROOT / "download/download_dataset.sbatch").read_text(encoding="utf-8")

    assert 'HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-2}"' in source
    assert 'HF_DOWNLOAD_ATTEMPTS="${HF_DOWNLOAD_ATTEMPTS:-5}"' in source
    assert 'HF_DOWNLOAD_RETRY_DELAY_SECONDS="${HF_DOWNLOAD_RETRY_DELAY_SECONDS:-60}"' in source
    assert '--max-workers "${HF_DOWNLOAD_MAX_WORKERS}"' in source
    assert "-maxdepth" not in source
    assert 'MILES_SOURCE_PROVENANCE*' in source
    assert '_provenance_partial="${_provenance}.partial-' in source
    assert 'mv -f -- "${_provenance_partial}" "${_provenance}"' in source


def test_all_submit_preflights_sft_before_queueing(tmp_path: Path):
    workspace = tmp_path / "workspace"
    environment = _setup_test_environment(workspace)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub_marker = tmp_path / "qsub-called"
    fake_qsub = fake_bin / "qsub"
    fake_qsub.write_text(
        "#!/bin/bash\n"
        "touch \"${QSUB_MARKER}\"\n"
        "printf '12345.pbs1\\n'\n",
        encoding="utf-8",
    )
    fake_qsub.chmod(0o700)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["QSUB_MARKER"] = str(qsub_marker)

    result = subprocess.run(
        [str(SETUP_ENTRYPOINT), "all", "--submit"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing" in result.stderr
    assert not qsub_marker.exists()


def test_repository_does_not_reference_legacy_root_level_setup_paths():
    legacy_paths = tuple(f"experiments/setup/{name}" for name in LEGACY_ROOT_SCRIPT_NAMES)
    references: list[str] = []
    for source_file in _repository_source_files():
        source = source_file.read_text(encoding="utf-8", errors="ignore")
        for legacy_path in legacy_paths:
            if legacy_path in source:
                references.append(f"{_relative(source_file)} -> {legacy_path}")
    assert references == []


@pytest.mark.parametrize("shell_file", SHELL_FILES, ids=_relative)
def test_setup_shell_files_never_source_repository_dotenv(shell_file: Path):
    source = shell_file.read_text(encoding="utf-8")
    violations: list[str] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        source_command = re.match(r"^\s*(?:source|\.)\s+(.+)$", line)
        if source_command and re.search(r"(?:^|/)\.env(?:$|[\s\"'])", source_command.group(1)):
            violations.append(f"{_relative(shell_file)}:{line_number}: {line.strip()}")
    assert violations == []


def test_setup_submission_boundaries_use_fixed_name_exports():
    submissions: list[tuple[Path, str]] = []
    violations: list[str] = []
    for shell_file in SHELL_FILES:
        source = shell_file.read_text(encoding="utf-8")
        for command in _logical_shell_lines(source):
            if command.startswith("#") or not SUBMISSION_COMMAND.search(command):
                continue
            submissions.append((shell_file, command))
            try:
                spec = _export_spec(command, source)
            except AssertionError as error:
                violations.append(f"{_relative(shell_file)}: {error}")
                continue

            for field in spec.split(","):
                name = field.partition("=")[0]
                if name == "ALL":
                    violations.append(f"{_relative(shell_file)}: --export includes ALL")
                elif not ENVIRONMENT_NAME.fullmatch(name):
                    violations.append(
                        f"{_relative(shell_file)}: non-literal export name {name!r} in {command}"
                    )

    assert submissions, "expected setup submission wrappers to invoke pbs_submit"
    assert violations == []


@pytest.mark.parametrize("job_file", PBS_JOB_FILES, ids=_relative)
def test_direct_setup_jobs_use_native_pbs_without_project(job_file: Path):
    source = job_file.read_text(encoding="utf-8")
    assert re.search(r"^#PBS -q \S+$", source, re.MULTILINE)
    assert re.search(r"^#PBS -l select=", source, re.MULTILINE)
    assert re.search(r"^#PBS -l walltime=", source, re.MULTILINE)
    assert "#PBS -P" not in source
    assert "#SBATCH" not in source


@pytest.mark.parametrize("job_file", PBS_SINGULARITY_DOWNLOAD_WORKERS, ids=_relative)
def test_core_download_workers_use_pbs_and_singularity(job_file: Path):
    source = job_file.read_text(encoding="utf-8")
    assert re.search(r"^#PBS -q R9920261300$", source, re.MULTILINE)
    assert re.search(r"^#PBS -l select=1:ncpus=32:mpiprocs=1$", source, re.MULTILINE)
    assert "#PBS -P" not in source
    assert "experiments/common/singularity.sh" in source
    assert re.search(r"^miles_srun \\$", source, re.MULTILINE)
    assert "--image=\"${CONTAINER_IMAGE}\"" in source
    assert "SLURM_" not in source
    assert "--container-image" not in source


def test_checkpoint_conversion_worker_uses_pbs_gpu_and_singularity():
    source = PBS_SINGULARITY_CONVERT_WORKER.read_text(encoding="utf-8")
    assert re.search(r"^#PBS -q R9920261300$", source, re.MULTILINE)
    assert re.search(
        r"^#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1$",
        source,
        re.MULTILINE,
    )
    assert "#PBS -P" not in source
    assert "miles_srun" in source
    assert "--image=\"${CONTAINER_IMAGE}\"" in source
    assert "PBS_NODEFILE" in source
    assert "MILES_NODE_RANK" in source
    assert "SLURM_" not in source


def test_documented_and_literal_setup_paths_exist():
    readme = (SETUP_ROOT / "README.md").read_text(encoding="utf-8")
    documented_paths = re.findall(
        r"`((?:datasets|download|environments|manifests|models)/[^`\s]+)`",
        readme,
    )
    assert documented_paths
    for relative_path in documented_paths:
        assert (SETUP_ROOT / relative_path).exists(), f"missing documented setup path: {relative_path}"

    for source_file in SETUP_ROOT.rglob("*"):
        if not source_file.is_file() or source_file.suffix not in SOURCE_SUFFIXES:
            continue
        source = source_file.read_text(encoding="utf-8", errors="ignore")
        for referenced_path in SETUP_PATH_REFERENCE.findall(source):
            assert (REPO_ROOT / referenced_path).exists(), (
                f"{_relative(source_file)} references missing path {referenced_path}"
            )


def test_manifest_orchestrators_reference_existing_manifests():
    expected_references = {
        "download/stage_all.sh": ("manifests/datasets.txt", "manifests/models.txt"),
        "download/stage_model.sh": ("manifests/models.txt",),
        "download/stage_nemotron_rl_datasets.sh": ("manifests/nemotron_rl_datasets.tsv",),
        "models/stage_sft_checkpoints.sh": ("manifests/sft_checkpoints.txt",),
    }
    for consumer_path, manifest_paths in expected_references.items():
        consumer = (SETUP_ROOT / consumer_path).read_text(encoding="utf-8")
        for manifest_path in manifest_paths:
            assert (SETUP_ROOT / manifest_path).is_file()
            assert f"experiments/setup/{manifest_path}" in consumer


@pytest.mark.parametrize(
    ("manifest_path", "model_type_index"),
    (
        ("manifests/models.txt", 2),
        ("manifests/sft_checkpoints.txt", 3),
    ),
)
def test_model_manifest_entries_reference_existing_model_args(
    manifest_path: str,
    model_type_index: int,
):
    rows = _active_manifest_rows(SETUP_ROOT / manifest_path)
    assert rows
    for row in rows:
        assert len(row) > model_type_index
        model_type = row[model_type_index]
        model_args = REPO_ROOT / "scripts/models" / f"{model_type}.sh"
        assert model_args.is_file(), f"missing model args for {model_type}: {_relative(model_args)}"


def test_sft_checkpoint_sources_are_relative_to_hf_checkpoint_directory():
    rows = _active_manifest_rows(SETUP_ROOT / "manifests/sft_checkpoints.txt")
    assert rows
    for row in rows:
        assert len(row) == 4
        for field in row[1:3]:
            relative_path = Path(field)
            assert not relative_path.is_absolute()
            assert ".." not in relative_path.parts
