from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_ROOT = REPO_ROOT / "experiments/setup"

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
        (*SETUP_ROOT.rglob("*.sh"), *SETUP_ROOT.rglob("*.sbatch")),
        key=lambda path: path.as_posix(),
    )
)
SBATCH_FILES = tuple(path for path in SHELL_FILES if path.suffix == ".sbatch")

EXPECTED_ACCOUNT = "coreai_horizon_dilations"
ALLOWED_PARTITION_QOS_PAIRS = frozenset(
    {
        ("batch", "interactive"),
        ("batch", "normal"),
        ("batch_long", "normal"),
        ("cpu", "cpu-interactive"),
        ("cpu", "cpu-normal"),
        ("cpu_datamover", "cpu-datamover"),
    }
)
ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")
SBATCH_COMMAND = re.compile(r"(?<![.\w-])sbatch\s")
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


def _directive(source: str, option: str) -> str | None:
    match = re.search(rf"^#SBATCH[ \t]+--{re.escape(option)}=(\S+)[ \t]*$", source, re.MULTILINE)
    return match.group(1) if match else None


def _export_spec(command: str, source: str) -> str:
    match = EXPORT_OPTION.search(command)
    assert match is not None, f"sbatch submission lacks --export: {command}"
    spec = next(group for group in match.groups() if group is not None)

    if spec == "${EXPORT_NAMES}":
        declaration = re.search(
            r'^readonly EXPORT_NAMES="([A-Z0-9_,]+)"[ \t]*$',
            source,
            re.MULTILINE,
        )
        assert declaration is not None, "EXPORT_NAMES must be a literal, fixed-name allowlist"
        spec = declaration.group(1)

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


def test_sbatch_submission_boundaries_use_fixed_name_exports():
    submissions: list[tuple[Path, str]] = []
    violations: list[str] = []
    for shell_file in SHELL_FILES:
        source = shell_file.read_text(encoding="utf-8")
        for command in _logical_shell_lines(source):
            if command.startswith("#") or not SBATCH_COMMAND.search(command):
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

    assert submissions, "expected setup submission wrappers to invoke sbatch"
    assert violations == []


@pytest.mark.parametrize("job_file", SBATCH_FILES, ids=_relative)
def test_direct_setup_jobs_pin_account_partition_and_qos(job_file: Path):
    source = job_file.read_text(encoding="utf-8")
    account = _directive(source, "account")
    partition = _directive(source, "partition")
    qos = _directive(source, "qos")

    assert account == EXPECTED_ACCOUNT
    assert partition is not None
    assert qos is not None
    assert (partition, qos) in ALLOWED_PARTITION_QOS_PAIRS


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
        model_args = REPO_ROOT / "scripts/models" / f"{model_type}.py"
        assert model_args.is_file(), f"missing model args for {model_type}: {_relative(model_args)}"
