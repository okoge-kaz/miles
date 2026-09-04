from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

pytest.importorskip("e2b")
pytest.importorskip("harbor")

from harbor.environments.e2b import E2BEnvironment


def _semantic_record(
    *,
    task_id: str = "task-a",
    task_digest: str = "a" * 64,
    source_image: str = "registry.example/repo@sha256:" + "b" * 64,
    task_tree_sha256: str = "c" * 64,
    template_identity: str = "d" * 64,
) -> dict[str, object]:
    def evidence(suffix: str) -> dict[str, str]:
        return {
            "template_id": f"template-{suffix}",
            "build_id": f"build-{suffix}",
            "alias_sha256": "e" * 64,
            "template_identity_sha256": template_identity,
            "sandbox_id": f"sandbox-{suffix}",
        }

    agent = evidence("agent")
    verifier = evidence("verifier")
    source = {**verifier, "sandbox_id": "sandbox-source"}
    oracle = {**verifier, "sandbox_id": "sandbox-oracle"}
    return {
        "schema_version": "miles-swe-gym-admission-v1",
        "instance_id": task_id,
        "task_digest": task_digest,
        "source_image": source_image,
        "admitted_task_tree_sha256": task_tree_sha256,
        "template_evidence": {
            "source": source,
            "agent": agent,
            "empty_verifier": verifier,
            "oracle_verifier": oracle,
        },
        "checks": {
            "empty_reward": 0,
            "oracle_reward": 1,
            "no_new_privileges": True,
            "effective_capabilities_zero": True,
            "fresh_separate_verifier": True,
            "late_private_verifier_upload": True,
        },
    }


def _verified_semantic_record(
    *,
    template_identity: str = "d" * 64,
) -> dict[str, object]:
    from experiments.src.environments.swe import materialize
    from experiments.src.environments.swe import oci_image_lock

    record = _semantic_record(template_identity=template_identity)
    source_image = record["source_image"]
    record.update(
        {
            "schema_version": (
                "miles-swebench-verified-hardened-local-admission-v1"
            ),
            "source_schema": "swebench",
            "input_content_digest": "f" * 64,
            "locked_content_digest": "1" * 64,
            "content_digest": "1" * 64,
            "source_image_requested": source_image,
            "source_image_resolved": source_image,
            "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
            "base_commit": "2" * 40,
            "base_tree": "3" * 40,
            "oracle_patch_sha256": "4" * 64,
            "test_patch_sha256": "5" * 64,
            "model_path_policy_sha256": "6" * 64,
            "dataset_revision": materialize._SWEBENCH_VERIFIED_DATASET_REVISION,
            "harness_repository": materialize._SWEBENCH_HARNESS_REPOSITORY,
            "harness_commit": materialize._SWEBENCH_HARNESS_COMMIT,
            "harness_version": materialize._SWEBENCH_HARNESS_VERSION,
            "constants_sha256": materialize._SWEBENCH_CONSTANTS_SHA256,
            "log_parsers_sha256": materialize._SWEBENCH_LOG_PARSERS_SHA256,
            "grading_sha256": materialize._SWEBENCH_GRADING_SHA256,
            "score_semantics": materialize._SWEBENCH_HARDENED_SCORE_SEMANTICS,
            "harbor_adapter_commit": materialize._SWE_GYM_HARBOR_COMMIT,
            "harbor_adapter_sha256": materialize._SWE_GYM_HARBOR_ADAPTER_SHA256,
            "checks": dict(materialize._REPOSITORY_ADMISSION_CHECKS),
        }
    )
    return record


def _load_prebuild_module() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "examples" / "experimental" / "swe-agent-harbor-e2b" / "prebuild_templates.py"
    spec = importlib.util.spec_from_file_location("miles_e2b_prebuild", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task_selection_is_direct_and_fail_closed(tmp_path: Path) -> None:
    module = _load_prebuild_module()
    tasks_dir = tmp_path / "tasks"
    first = tasks_dir / "first"
    second = tasks_dir / "second"
    first.mkdir(parents=True)
    second.mkdir()
    tasks_dir.chmod(0o700)
    first.chmod(0o700)
    second.chmod(0o700)
    (first / "task.toml").write_text("", encoding="utf-8")
    (second / "task.toml").write_text("", encoding="utf-8")
    (first / "task.toml").chmod(0o400)
    (second / "task.toml").chmod(0o400)
    first.chmod(0o500)
    second.chmod(0o500)

    assert module._selected_task_dirs(tasks_dir, None) == [first, second]

    task_ids = tmp_path / "ids.txt"
    task_ids.write_text("second\n", encoding="utf-8")
    task_ids.chmod(0o600)
    assert module._selected_task_dirs(tasks_dir, task_ids) == [second]

    first.chmod(0o700)
    with pytest.raises(PermissionError, match="sealed read-only"):
        module._selected_task_dirs(tasks_dir, None)
    first.chmod(0o500)

    task_ids.write_text("../outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid task id"):
        module._selected_task_dirs(tasks_dir, task_ids)

    task_ids.write_text("first\n", encoding="utf-8")
    os.link(task_ids, tmp_path / "ids-copy.txt")
    with pytest.raises(PermissionError, match="task-id file"):
        module._selected_task_dirs(tasks_dir, task_ids)


def test_task_selection_rejects_hardlinked_tree_file(tmp_path: Path) -> None:
    module = _load_prebuild_module()
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task"
    task_dir.mkdir(parents=True, mode=0o700)
    tasks_dir.chmod(0o700)
    task_toml = task_dir / "task.toml"
    task_toml.write_text("", encoding="utf-8")
    os.link(task_toml, task_dir / "task-copy.toml")
    task_toml.chmod(0o400)
    (task_dir / "task-copy.toml").chmod(0o400)
    task_dir.chmod(0o500)

    with pytest.raises(PermissionError, match="unsafe entry"):
        module._selected_task_dirs(tasks_dir, None)


def test_semantic_manifest_is_owner_only_and_single_link(tmp_path: Path) -> None:
    module = _load_prebuild_module()
    manifest = tmp_path / "admissions.jsonl"
    manifest.write_text(
        json.dumps(_semantic_record()) + "\n",
        encoding="utf-8",
    )
    manifest.chmod(0o600)

    records = module._load_semantic_admissions(
        [manifest],
        selected_task_ids={"task-a"},
    )
    assert set(records) == {"task-a"}

    os.link(manifest, tmp_path / "admissions-copy.jsonl")
    with pytest.raises(PermissionError, match="single-link"):
        module._load_semantic_admissions(
            [manifest],
            selected_task_ids={"task-a"},
        )


def test_verified_semantic_schema_is_exact_and_pinned() -> None:
    module = _load_prebuild_module()
    record = _verified_semantic_record()

    assert module._validate_semantic_admission(record) == record

    unexpected = json.loads(json.dumps(record))
    unexpected["untrusted_extra"] = True
    with pytest.raises(ValueError, match="field set"):
        module._validate_semantic_admission(unexpected)

    wrong_pin = json.loads(json.dumps(record))
    wrong_pin["grading_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned dependency"):
        module._validate_semantic_admission(wrong_pin)

    unsupported = json.loads(json.dumps(record))
    unsupported["schema_version"] = "miles-swebench-verified-admission-v1"
    with pytest.raises(ValueError, match="unsupported semantic admission schema"):
        module._validate_semantic_admission(unsupported)


def test_private_task_id_file_rejects_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_prebuild_module()
    task_ids = tmp_path / "ids.txt"
    task_ids.write_text("task-a\n", encoding="utf-8")
    task_ids.chmod(0o600)
    real_open = os.open
    replaced = False

    def replace_after_open(path, flags, *args, **kwargs):
        nonlocal replaced
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == task_ids and not replaced:
            replaced = True
            task_ids.rename(tmp_path / "ids.original")
            task_ids.write_text("task-b\n", encoding="utf-8")
            task_ids.chmod(0o600)
        return descriptor

    monkeypatch.setattr(module.os, "open", replace_after_open)
    with pytest.raises(RuntimeError, match="changed while reading"):
        module._read_private_text_file(
            task_ids,
            max_bytes=module._MAX_TASK_IDS_BYTES,
            name="E2B task-id file",
        )


def test_semantic_manifest_rejects_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_prebuild_module()
    manifest = tmp_path / "admissions.jsonl"
    rendered = json.dumps(_semantic_record()) + "\n"
    manifest.write_text(rendered, encoding="utf-8")
    manifest.chmod(0o600)
    real_open = os.open
    replaced = False

    def replace_after_open(path, flags, *args, **kwargs):
        nonlocal replaced
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == manifest and not replaced:
            replaced = True
            manifest.rename(tmp_path / "admissions.original")
            manifest.write_text(rendered, encoding="utf-8")
            manifest.chmod(0o600)
        return descriptor

    monkeypatch.setattr(module.os, "open", replace_after_open)
    with pytest.raises(RuntimeError, match="path changed"):
        module._load_semantic_admissions(
            [manifest],
            selected_task_ids={"task-a"},
        )


def test_launcher_rejects_concurrency_mismatch_before_provider_access(
    tmp_path: Path,
) -> None:
    launcher = Path(__file__).parents[4] / "examples" / "experimental" / "swe-agent-harbor-e2b" / "launch_agent_server.sh"
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    environment = {
        **os.environ,
        "HARBOR_ROOT": str(tmp_path),
        "HARBOR_TASKS_DIR": str(tasks),
        "HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS": str(tmp_path / "admissions.jsonl"),
        "E2B_API_KEY": "provider-secret-must-not-appear",
        "HARBOR_RUN_SECRET": "r" * 32,
        "HARBOR_ADMIN_SECRET": "a" * 32,
        "MAX_CONCURRENT": "32",
        "ASYNC_MAX_CONCURRENT_SAMPLES": "64",
        "HARBOR_TIMEOUT_MULTIPLIER": "1",
        "HARBOR_TRIAL_WALL_TIMEOUT_SEC": "12600",
        "AGENT_TIMEOUT": "3600",
        "AGENT_SETUP_TIMEOUT": "1800",
        "HARBOR_VERIFIER_TIMEOUT_SEC": "2100",
        "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER": "1",
        "TRIALS_DIR": str(tmp_path / "trials"),
    }
    environment.pop("HARBOR_PYTHON", None)

    mismatch = subprocess.run(
        ["/bin/bash", str(launcher)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert mismatch.returncode == 2
    assert "MAX_CONCURRENT must equal ASYNC_MAX_CONCURRENT_SAMPLES" in mismatch.stderr
    assert "provider-secret-must-not-appear" not in mismatch.stdout + mismatch.stderr

    environment["MAX_CONCURRENT"] = "64"
    matched = subprocess.run(
        ["/bin/bash", str(launcher)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert matched.returncode == 2
    assert "Harbor Python is missing" in matched.stderr
    assert "MAX_CONCURRENT must equal" not in matched.stderr


@pytest.mark.parametrize(
    ("variable", "invalid", "expected_error"),
    [
        (
            "HARBOR_TIMEOUT_MULTIPLIER",
            "2",
            "HARBOR_TIMEOUT_MULTIPLIER is fixed to 1",
        ),
        (
            "HARBOR_TRIAL_WALL_TIMEOUT_SEC",
            "12599",
            "HARBOR_TRIAL_WALL_TIMEOUT_SEC is fixed to 12600 seconds",
        ),
        ("AGENT_TIMEOUT", "3601", "AGENT_TIMEOUT is fixed to 3600 seconds"),
        (
            "AGENT_SETUP_TIMEOUT",
            "1801",
            "AGENT_SETUP_TIMEOUT is fixed to 1800 seconds",
        ),
        (
            "HARBOR_VERIFIER_TIMEOUT_SEC",
            "2101",
            "HARBOR_VERIFIER_TIMEOUT_SEC is fixed to 2100 seconds",
        ),
        (
            "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER",
            "2",
            "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER is fixed to 1",
        ),
    ],
)
def test_launcher_rejects_timeout_contract_overrides_before_provider_access(
    tmp_path: Path,
    variable: str,
    invalid: str,
    expected_error: str,
) -> None:
    launcher = Path(__file__).parents[4] / "examples" / "experimental" / "swe-agent-harbor-e2b" / "launch_agent_server.sh"
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    secret = "provider-secret-must-not-appear"
    environment = {
        **os.environ,
        "HARBOR_ROOT": str(tmp_path),
        "HARBOR_TASKS_DIR": str(tasks),
        "HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS": str(
            tmp_path / "admissions.jsonl"
        ),
        "E2B_API_KEY": secret,
        "HARBOR_RUN_SECRET": "r" * 32,
        "HARBOR_ADMIN_SECRET": "a" * 32,
        "MAX_CONCURRENT": "64",
        "ASYNC_MAX_CONCURRENT_SAMPLES": "64",
        "HARBOR_TIMEOUT_MULTIPLIER": "1",
        "HARBOR_TRIAL_WALL_TIMEOUT_SEC": "12600",
        "AGENT_TIMEOUT": "3600",
        "AGENT_SETUP_TIMEOUT": "1800",
        "HARBOR_VERIFIER_TIMEOUT_SEC": "2100",
        "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER": "1",
        "TRIALS_DIR": str(tmp_path / "trials"),
        variable: invalid,
    }

    rejected = subprocess.run(
        ["/bin/bash", str(launcher)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert rejected.returncode == 2
    assert expected_error in rejected.stderr
    assert secret not in rejected.stdout + rejected.stderr


def test_timeout_budget_is_consistent_across_launcher_examples_and_preflight() -> None:
    root = Path(__file__).resolve().parents[4]
    e2b_root = root / "examples" / "experimental" / "swe-agent-harbor-e2b"
    launcher = (e2b_root / "launch_agent_server.sh").read_text(encoding="utf-8")
    example = (e2b_root / "config.example.sh").read_text(encoding="utf-8")
    preflight = (root / "tests" / "slurm" / "test_harbor_e2b_preflight.sbatch").read_text(
        encoding="utf-8"
    )

    for source in (launcher, example, preflight):
        assert "AGENT_TIMEOUT=3600" in source
        assert "AGENT_SETUP_TIMEOUT=1800" in source
        assert "HARBOR_TIMEOUT_MULTIPLIER=1" in source
        assert "HARBOR_TRIAL_WALL_TIMEOUT_SEC=12600" in source
        assert "HARBOR_VERIFIER_TIMEOUT_SEC=2100" in source
        assert "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER=1" in source


def test_server_lifetime_and_training_readiness_are_production_shaped() -> None:
    root = Path(__file__).resolve().parents[4]
    server_job = (
        root
        / "examples"
        / "experimental"
        / "swe-agent-harbor-e2b"
        / "run_agent_server.sbatch"
    ).read_text(encoding="utf-8")
    gate_job = (
        root
        / "examples"
        / "experimental"
        / "swe-agent-harbor-e2b"
        / "wait_agent_server.sbatch"
    ).read_text(encoding="utf-8")
    training_job = (
        root
        / "experiments"
        / "scripts"
        / "swe"
        / "async"
        / "swe-rebench-v2-swe-gym"
        / "qwen3-4b"
        / "run.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=cpu\n" in server_job
    assert "#SBATCH --qos=cpu-normal\n" in server_job
    assert "#SBATCH --cpus-per-task=32\n" in server_job
    assert "#SBATCH --time=1-00:00:00\n" in server_job
    assert (
        'prune_agent_server_environment.sh" \\\n'
        "    /bin/bash \\\n"
        '    "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/'
        'launch_agent_server.sh"'
        in server_job
    )
    assert "#SBATCH --partition=cpu\n" in gate_job
    assert "#SBATCH --qos=cpu-normal\n" in gate_job
    readiness = "wait_for_agent_server.py"
    assert readiness in training_job
    assert training_job.index(readiness) < training_job.index(
        'echo "running EFA fail-closed preflight on every node"'
    )
    readiness_start = training_job.rfind(
        "PYTHONPATH=",
        0,
        training_job.index(readiness),
    )
    readiness_call = training_job[
        readiness_start : training_job.index(readiness) + len(readiness)
    ]
    assert "HARBOR_RUN_SECRET" not in readiness_call
    assert 'SWE_AGENT_SERVER_READY_TIMEOUT_SEC:-60' in training_job
    assert "SWE_AGENT_SERVER_READY_TIMEOUT_SEC <= 60" in training_job
    assert (
        'ADMISSION_SUMMARY="/data/miles-swe/admitted/${DATASET_TAG}-train.summary.json"'
        in training_job
    )
    assert (
        'HOST_ADMISSION_SUMMARY="${DATASET_DIR}/miles-swe/admitted/'
        '${DATASET_TAG}-train.summary.json"'
        in training_job
    )
    assert '--admission-summary "${HOST_ADMISSION_SUMMARY}"' in training_job
    assert '--admission-summary "${ADMISSION_SUMMARY}"' not in training_job
    assert "SWE_HOST_ADMISSION_SUMMARY" in gate_job
    assert "SWE_ADMISSION_SUMMARY" not in gate_job


def test_submit_helper_uses_cpu_gate_and_fixed_exports(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    helper = (
        root
        / "experiments"
        / "scripts"
        / "swe"
        / "async"
        / "r2e-gym-swe-rebench-v2"
        / "qwen3-4b-instruct-2507"
        / "submit_when_ready.sh"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$*\" >> \"$SBATCH_LOG\"\n"
        "printf '%s\\n' \"${SWE_HOST_ADMISSION_SUMMARY:-}\" > \"$SUMMARY_LOG\"\n"
        "count=\"$(wc -l < \"$SBATCH_LOG\")\"\n"
        "if [[ \"$count\" == 1 ]]; then printf '12345\\n'; "
        "else printf '12346\\n'; fi\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o700)
    secret = "private-run-secret-" + "x" * 32
    environment = {
        **os.environ,
        "AGENT_SERVER_URL": "http://server.internal:11000",
        "HARBOR_RUN_SECRET": secret,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "SBATCH_LOG": str(sbatch_log),
        "SUMMARY_LOG": str(tmp_path / "summary.log"),
        "SHARED_WS": str(tmp_path / "shared"),
        "WS": str(tmp_path / "workspace"),
        "SLURM_SUBMIT_DIR": str(root),
        "MILES_SWE_UNRELATED_ENV_SENTINEL": "must-not-cross-sbatch",
    }

    result = subprocess.run(
        ["/bin/bash", str(helper)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "readiness_job_id=12345\ntraining_job_id=12346\n"
    )
    rendered = sbatch_log.read_text(encoding="utf-8")
    assert "--dependency=afterok:12345" in rendered
    assert "--export=ALL" not in rendered
    assert "--export=NONE" not in rendered
    assert rendered.count("--export=") == 2
    assert "MILES_SWE_FIXED_EXPORTS" in rendered
    gate_submission, training_submission = rendered.splitlines()
    assert "SWE_HOST_ADMISSION_SUMMARY" in gate_submission
    assert "SWE_HOST_ADMISSION_SUMMARY" not in training_submission
    assert (tmp_path / "summary.log").read_text(encoding="utf-8").strip() == str(
        tmp_path
        / "shared"
        / "datasets"
        / "miles-swe"
        / "admitted"
        / "swe-rebench-v2-filtered-verified-train.summary.json"
    )
    assert "MILES_SWE_UNRELATED_ENV_SENTINEL" not in rendered
    assert secret not in rendered + result.stdout + result.stderr


def _fake_sbatch_environment(
    tmp_path: Path,
    *,
    environment: dict[str, str],
) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$*\" >> \"$SBATCH_LOG\"\n"
        "printf '24680\\n'\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o700)
    return (
        {
            **os.environ,
            **environment,
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "SBATCH_LOG": str(sbatch_log),
            "MILES_SWE_UNRELATED_ENV_SENTINEL": "must-not-cross-sbatch",
        },
        sbatch_log,
    )


def test_agent_server_submitter_exports_only_fixed_names(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    submitter = (
        root
        / "examples"
        / "experimental"
        / "swe-agent-harbor-e2b"
        / "submit_agent_server.sh"
    )
    provider_secret = "provider-secret-" + "e" * 32
    run_secret = "run-secret-" + "r" * 32
    admin_secret = "admin-secret-" + "a" * 32
    environment, sbatch_log = _fake_sbatch_environment(
        tmp_path,
        environment={
            "SLURM_SUBMIT_DIR": str(root),
            "E2B_API_KEY": provider_secret,
            "HARBOR_ROOT": "/approved/harbor",
            "HARBOR_TASKS_DIR": "/approved/tasks",
            "HARBOR_E2B_PREBUILD_TASK_IDS_FILE": "/approved/task-ids",
            "HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS": "/approved/admission",
            "HARBOR_RUN_SECRET": run_secret,
            "HARBOR_ADMIN_SECRET": admin_secret,
            "MAX_CONCURRENT": "64",
        },
    )

    result = subprocess.run(
        [str(submitter)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "server_job_id=24680\n"
    rendered = sbatch_log.read_text(encoding="utf-8")
    assert "--export=ALL" not in rendered
    assert "--export=NONE" not in rendered
    for required_name in (
        "E2B_API_KEY",
        "HARBOR_RUN_SECRET",
        "HARBOR_ADMIN_SECRET",
        "MILES_SWE_FIXED_EXPORTS",
    ):
        assert required_name in rendered
    assert "MILES_SWE_UNRELATED_ENV_SENTINEL" not in rendered
    for secret in (provider_secret, run_secret, admin_secret):
        assert secret not in rendered + result.stdout + result.stderr


def test_live_e2b_submitter_exports_only_fixed_names(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    submitter = root / "tests" / "slurm" / "submit_harbor_e2b_live.sh"
    provider_secret = "provider-secret-" + "e" * 32
    unrelated_secret = "must-not-cross-live-probe"
    environment, sbatch_log = _fake_sbatch_environment(
        tmp_path,
        environment={
            "SLURM_SUBMIT_DIR": str(root),
            "E2B_API_KEY": provider_secret,
            "E2B_LIVE_SOURCE_IMAGE": "registry.example/repo@sha256:" + "a" * 64,
            "HARBOR_ROOT": str(tmp_path / "harbor"),
            "UNRELATED_SECRET": unrelated_secret,
        },
    )

    result = subprocess.run(
        [str(submitter)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "harbor_e2b_live_job_id=24680\n"
    rendered = sbatch_log.read_text(encoding="utf-8")
    assert "--export=ALL" not in rendered
    assert "--export=NONE" not in rendered
    assert "MILES_SWE_FIXED_EXPORTS" in rendered
    assert provider_secret not in rendered + result.stdout + result.stderr
    assert unrelated_secret not in rendered + result.stdout + result.stderr


@pytest.mark.parametrize(
    ("submitter_name", "expected_output"),
    [
        (
            "submit_admit_repository_swe_e2b.sh",
            "repository_admission_job_id=24680\n",
        ),
        ("submit_admit_r2e_e2b.sh", "r2e_admission_job_id=24680\n"),
    ],
)
def test_admission_submitters_export_only_fixed_names(
    tmp_path: Path,
    submitter_name: str,
    expected_output: str,
) -> None:
    root = Path(__file__).resolve().parents[4]
    submitter = root / "experiments" / "setup" / "environments" / submitter_name
    provider_secret = "provider-secret-" + "e" * 32
    environment, sbatch_log = _fake_sbatch_environment(
        tmp_path,
        environment={
            "SLURM_SUBMIT_DIR": str(root),
            "E2B_API_KEY": provider_secret,
            "HARBOR_ROOT": "/approved/harbor",
            "SWE_TASKSET": "swe-gym",
        },
    )

    result = subprocess.run(
        [str(submitter)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == expected_output
    rendered = sbatch_log.read_text(encoding="utf-8")
    assert "--export=ALL" not in rendered
    assert "--export=NONE" not in rendered
    assert "E2B_API_KEY" in rendered
    assert "DOCKERHUB_TOKEN" in rendered
    assert "DOCKERHUB_USERNAME" in rendered
    assert "MILES_SWE_FIXED_EXPORTS" in rendered
    assert "HARBOR_RUN_SECRET" not in rendered
    assert "MILES_SWE_UNRELATED_ENV_SENTINEL" not in rendered
    assert provider_secret not in rendered + result.stdout + result.stderr


def test_swe_prepare_submitter_exports_only_selector(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    submitter = (
        root
        / "experiments"
        / "setup"
        / "datasets"
        / "submit_prepare_swe_rl.sh"
    )
    environment, sbatch_log = _fake_sbatch_environment(
        tmp_path,
        environment={
            "SLURM_SUBMIT_DIR": str(root),
            "SWE_SOURCE": "swe-gym",
            "E2B_API_KEY": "must-not-cross-allocation",
        },
    )

    result = subprocess.run(
        [str(submitter)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "swe_prepare_job_id=24680\n"
    rendered = sbatch_log.read_text(encoding="utf-8")
    assert "--export=MILES_SWE_FIXED_EXPORTS,SWE_SOURCE" in rendered
    assert "--export=ALL" not in rendered
    assert "--export=NONE" not in rendered
    assert "E2B_API_KEY" not in rendered
    assert "MILES_SWE_UNRELATED_ENV_SENTINEL" not in rendered


@pytest.mark.parametrize(
    ("submitter_name", "required_environment", "expected_output"),
    [
        (
            "submit_lock_swe_oci_images.sh",
            {
                "SWE_PRIVATE_MANIFEST": "/private/input.jsonl",
                "SWE_LOCKED_MANIFEST": "/private/locked.jsonl",
                "SWE_IMAGE_LOCK_MANIFEST": "/private/locks.jsonl",
            },
            "swe_oci_lock_job_id=24680\n",
        ),
        (
            "submit_materialize_harbor_swe_tasks.sh",
            {"SWE_TASKSET": "swe-gym"},
            "swe_materialization_job_id=24680\n",
        ),
        (
            "submit_materialize_swebench_verified_eval.sh",
            {},
            "swebench_verified_materialization_job_id=24680\n",
        ),
    ],
)
def test_swe_provenance_submitters_use_fixed_names(
    tmp_path: Path,
    submitter_name: str,
    required_environment: dict[str, str],
    expected_output: str,
) -> None:
    root = Path(__file__).resolve().parents[4]
    submitter = (
        root / "experiments" / "setup" / "environments" / submitter_name
    )
    unrelated_secret = "must-not-cross-allocation"
    environment, sbatch_log = _fake_sbatch_environment(
        tmp_path,
        environment={
            "SLURM_SUBMIT_DIR": str(root),
            "E2B_API_KEY": unrelated_secret,
            **required_environment,
        },
    )

    result = subprocess.run(
        [str(submitter)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == expected_output
    rendered = sbatch_log.read_text(encoding="utf-8")
    assert "--export=ALL" not in rendered
    assert "--export=NONE" not in rendered
    assert "MILES_SWE_FIXED_EXPORTS" in rendered
    assert "MILES_SWE_UNRELATED_ENV_SENTINEL" not in rendered
    assert unrelated_secret not in rendered + result.stdout + result.stderr


def test_swe_slurm_jobs_require_fixed_submission_environment() -> None:
    root = Path(__file__).resolve().parents[4]
    jobs = [
        root
        / "examples"
        / "experimental"
        / "swe-agent-harbor-e2b"
        / "run_agent_server.sbatch",
        root
        / "examples"
        / "experimental"
        / "swe-agent-harbor-e2b"
        / "wait_agent_server.sbatch",
        root / "tests" / "slurm" / "test_harbor_e2b_live.sbatch",
        root
        / "experiments"
        / "scripts"
        / "swe"
        / "async"
        / "r2e-gym-swe-rebench-v2"
        / "qwen3-4b-instruct-2507"
        / "run.sbatch",
        root
        / "experiments"
        / "setup"
        / "environments"
        / "admit_repository_swe_e2b.sbatch",
        root
        / "experiments"
        / "setup"
        / "environments"
        / "admit_r2e_e2b.sbatch",
        root
        / "experiments"
        / "setup"
        / "datasets"
        / "prepare_swe_rl.sbatch",
        root
        / "experiments"
        / "setup"
        / "environments"
        / "lock_swe_oci_images.sbatch",
        root
        / "experiments"
        / "setup"
        / "environments"
        / "materialize_harbor_swe_tasks.sbatch",
        root
        / "experiments"
        / "setup"
        / "environments"
        / "materialize_swebench_verified_eval.sbatch",
        root
        / "experiments"
        / "scripts"
        / "swe"
        / "eval"
        / "swebench-verified"
        / "run.sbatch",
    ]

    for job in jobs:
        source = job.read_text(encoding="utf-8")
        assert "MILES_SWE_FIXED_EXPORTS" in source
        assert "MILES_SWE_UNRELATED_ENV_SENTINEL" in source
        assert "#SBATCH --export=NIL" in source
    training = jobs[3].read_text(encoding="utf-8")
    assert "E2B provider credentials are forbidden" in training
    preparation = jobs[6].read_text(encoding="utf-8")
    assert "--export=ALL" not in preparation
    assert (
        "--export=PATH,PYTHON_DOTENV_DISABLED,PYTHONDONTWRITEBYTECODE,"
        "SWE_SOURCE,WANDB_DISABLED,WANDB_MODE"
    ) in preparation


def test_credential_capable_swe_containers_mask_repository_dotenv() -> None:
    root = Path(__file__).resolve().parents[4]
    mask = root / "experiments" / "common" / "dotenv.disabled"
    assert mask.read_text(encoding="utf-8").splitlines() == [
        "# Deliberately empty: mount this over /root/miles/.env in credential-capable jobs."
    ]
    for path in (
        root
        / "experiments/scripts/swe/async/swe-rebench-v2-swe-gym/qwen3-4b/run.sbatch",
        root / "experiments/scripts/swe/eval/swebench-verified/run.sbatch",
        root / "experiments/setup/datasets/prepare_swe_rl.sbatch",
        root / "experiments/setup/environments/materialize_harbor_swe_tasks.sbatch",
        root
        / "experiments/setup/environments/materialize_swebench_verified_eval.sbatch",
    ):
        source = path.read_text(encoding="utf-8")
        assert "dotenv.disabled" in source
        assert ":/root/miles/.env:ro" in source


def test_agent_server_environment_pruner_drops_unrelated_secrets_and_hooks() -> None:
    helper = Path(__file__).parents[4] / "examples" / "experimental" / "swe-agent-harbor-e2b" / "prune_agent_server_environment.sh"
    run_secret = "r" * 32
    admin_secret = "a" * 32
    unrelated_secret = "must-not-reach-server-child"
    with __import__("tempfile").TemporaryDirectory() as temporary_root:
        trials_dir = Path(temporary_root) / "trials"
        trials_dir.mkdir(mode=0o700)
        fake_home = Path(temporary_root) / "submission-home"
        fake_home.mkdir(mode=0o700)
        (fake_home / ".netrc").write_text("fake-home-secret\n", encoding="utf-8")
        _assert_pruned_environment(
            helper,
            trials_dir=trials_dir,
            fake_home=fake_home,
            run_secret=run_secret,
            admin_secret=admin_secret,
            unrelated_secret=unrelated_secret,
        )


def test_agent_server_pruner_runs_nonexecutable_shell_launcher(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    helper = (
        root
        / "examples"
        / "experimental"
        / "swe-agent-harbor-e2b"
        / "prune_agent_server_environment.sh"
    )
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir(mode=0o700)
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/bash\nprintf 'launcher-ran\\n'\n", encoding="utf-8")
    launcher.chmod(0o600)

    result = subprocess.run(
        [str(helper), "/bin/bash", str(launcher)],
        env={
            **os.environ,
            "TRIALS_DIR": str(trials_dir),
            "WANDB_MODE": "offline",
            "PYTHON_DOTENV_DISABLED": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "launcher-ran\n"


def _assert_pruned_environment(
    helper: Path,
    *,
    trials_dir: Path,
    fake_home: Path,
    run_secret: str,
    admin_secret: str,
    unrelated_secret: str,
) -> None:
    code = (
        "import os; "
        "assert os.environ['HARBOR_RUN_SECRET']; "
        "assert os.environ['HARBOR_ADMIN_SECRET']; "
        "assert 'UNRELATED_SECRET' not in os.environ; "
        "assert 'BASH_ENV' not in os.environ; "
        "assert 'PYTHONPATH' not in os.environ; "
        "assert not any(name.startswith('BASH_FUNC_') for name in os.environ); "
        "assert 'HOSTILE_FUNCTION_RAN' not in os.environ; "
        "assert os.environ['WANDB_MODE'] == 'offline'; "
        "assert os.environ['PYTHON_DOTENV_DISABLED'] == '1'; "
        "assert os.environ['HARBOR_TIMEOUT_MULTIPLIER'] == '1'; "
        "assert os.environ['HARBOR_TRIAL_WALL_TIMEOUT_SEC'] == '12600'; "
        "assert os.environ['AGENT_TIMEOUT'] == '3600'; "
        "assert os.environ['AGENT_SETUP_TIMEOUT'] == '1800'; "
        "assert os.environ['HARBOR_VERIFIER_TIMEOUT_SEC'] == '2100'; "
        "assert os.environ['HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER'] == '1'; "
        "assert not os.path.exists(os.path.join(os.environ['HOME'], '.netrc')); "
        "assert os.environ['HOME'].startswith(os.environ['TRIALS_DIR'] + '/'); "
        "print('pruned')"
    )
    environment = {
        **os.environ,
        "HARBOR_RUN_SECRET": run_secret,
        "HARBOR_ADMIN_SECRET": admin_secret,
        "TRIALS_DIR": str(trials_dir),
        "HOME": str(fake_home),
        "UNRELATED_SECRET": unrelated_secret,
        "BASH_ENV": "/must/not/run",
        "PYTHONPATH": "/must/not/import",
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "offline",
        "WANDB_DISABLED": "true",
        "HARBOR_TIMEOUT_MULTIPLIER": "1",
        "HARBOR_TRIAL_WALL_TIMEOUT_SEC": "12600",
        "AGENT_TIMEOUT": "3600",
        "AGENT_SETUP_TIMEOUT": "1800",
        "HARBOR_VERIFIER_TIMEOUT_SEC": "2100",
        "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER": "1",
        "BASH_FUNC_hostile%%": "() { export HOSTILE_FUNCTION_RAN=1; }",
    }

    result = subprocess.run(
        [str(helper), sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pruned"
    assert run_secret not in result.stdout + result.stderr
    assert admin_secret not in result.stdout + result.stderr
    assert unrelated_secret not in result.stdout + result.stderr


def _write_production_task(task_dir: Path, *, agent_user: int = 1000) -> None:
    digest_image = "registry.example/repo@sha256:" + "a" * 64
    task_dir.mkdir(mode=0o700)
    (task_dir / "environment").mkdir(mode=0o700)
    (task_dir / "tests").mkdir(mode=0o700)
    (task_dir / "solution").mkdir(mode=0o700)
    (task_dir / "instruction.md").write_text("Fix it.\n", encoding="utf-8")
    (task_dir / "instruction.md").chmod(0o600)
    (task_dir / "environment" / "Dockerfile").write_text(
        f"FROM {digest_image}\n",
        encoding="utf-8",
    )
    (task_dir / "environment" / "Dockerfile").chmod(0o600)
    (task_dir / "tests" / ".harbor-e2b-late-tests").write_text(
        "late\n",
        encoding="utf-8",
    )
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/bash\nexit 0\n",
        encoding="utf-8",
    )
    for path in task_dir.joinpath("tests").iterdir():
        path.chmod(0o600)
    (task_dir / "task.toml").write_text(
        f'''schema_version = "1.3"
[metadata]
task_digest = "{"b" * 64}"
[agent]
user = {agent_user}
[environment]
network_mode = "no-network"
[verifier]
environment_mode = "separate"
user = 0
[[verifier.collect]]
command = "true"
required = true
[verifier.environment]
network_mode = "no-network"
docker_image = "{digest_image}"
''',
        encoding="utf-8",
    )
    (task_dir / "task.toml").chmod(0o600)
    for path in task_dir.rglob("*"):
        if path.is_dir():
            path.chmod(0o500)
        elif path.name == "test.sh":
            path.chmod(0o500)
        else:
            path.chmod(0o400)
    task_dir.chmod(0o500)


def test_template_specs_enforce_production_security_contract(tmp_path: Path) -> None:
    module = _load_prebuild_module()
    task_dir = tmp_path / "task"
    _write_production_task(task_dir)

    specs = module._template_specs([task_dir])

    assert [spec.role for spec in specs] == ["agent", "verifier"]
    (task_dir / "task.toml").chmod(0o600)
    (task_dir / "task.toml").write_text(
        (task_dir / "task.toml").read_text().replace("user = 1000", "user = 0", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="agent must run as UID 1000"):
        module._template_specs([task_dir])


def test_private_report_is_atomic_owner_only_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    module = _load_prebuild_module()
    report_dir = tmp_path / "reports"
    report = report_dir / "admission.json"

    module._write_private_report(report, {"schema_version": 2})

    assert report.stat().st_mode & 0o077 == 0
    report.unlink()
    target = tmp_path / "outside.json"
    target.write_text("outside\n", encoding="utf-8")
    report.symlink_to(target)
    with pytest.raises(PermissionError, match="target is unsafe"):
        module._write_private_report(report, {"schema_version": 2})

    report.unlink()
    report.write_text("old\n", encoding="utf-8")
    report.chmod(0o600)
    os.link(report, tmp_path / "report-copy.json")
    with pytest.raises(PermissionError, match="target is unsafe"):
        module._write_private_report(report, {"schema_version": 2})


def test_template_pin_writer_uses_same_explicit_cap_as_provider(
    tmp_path: Path,
) -> None:
    from harbor.environments import e2b

    module = _load_prebuild_module()
    assert module._MAX_TEMPLATE_PINS_BYTES == 64 * 1024 * 1024
    assert module._MAX_TEMPLATE_PINS_BYTES == e2b._MAX_TEMPLATE_PINS_BYTES
    output = tmp_path / "pins.json"

    with pytest.raises(ValueError, match="exceeds the explicit size cap"):
        module._write_private_report(
            output,
            {"schema_version": 2, "padding": "x" * 128},
            max_bytes=64,
        )

    assert not output.exists()


@pytest.mark.asyncio
async def test_prebuild_deduplicates_build_identity_and_pins_each_alias(
    tmp_path: Path,
) -> None:
    module = _load_prebuild_module()
    task_environment = MagicMock()
    task_environment.resolve_baseline.return_value = MagicMock()
    specs = [
        module.TemplateSpec(
            task_id="task-a",
            task_digest="a" * 64,
            source_image="registry.example/repo@sha256:" + "b" * 64,
            task_tree_sha256="c" * 64,
            role="agent",
            environment_dir=tmp_path,
            environment_name="task-a",
            task_environment=task_environment,
        ),
        module.TemplateSpec(
            task_id="task-a",
            task_digest="a" * 64,
            source_image="registry.example/repo@sha256:" + "b" * 64,
            task_tree_sha256="c" * 64,
            role="verifier",
            environment_dir=tmp_path,
            environment_name="task-a",
            task_environment=task_environment,
        ),
    ]

    first = object.__new__(E2BEnvironment)
    first._template_name = "agent-task-a"
    first._template_build_identity = "shared-build-identity"
    first._create_template = AsyncMock(
        return_value=MagicMock(
            template_id="template-id-123",
            build_id="build-id-456",
        )
    )
    second = object.__new__(E2BEnvironment)
    second._template_name = "agent-task-b"
    second._template_build_identity = "shared-build-identity"
    second._create_template = AsyncMock()

    with patch(
        "miles.rollout.harbor.environment_config.build_harbor_environment_config",
        return_value=MagicMock(),
    ):
        with patch(
            "harbor.environments.factory.EnvironmentFactory.create_environment_from_config",
            side_effect=[first, second],
        ):
            report, pins = await module._prebuild(
                specs,
                concurrency=16,
                allow_fresh_build=True,
            )

    assert report["task_count"] == 1
    assert report["task_set_sha256"] == module._task_set_sha256({"task-a"})
    expected_task_binding = module._task_binding_sha256(
        {
            "task-a": {
                "task_digest": "a" * 64,
                "task_tree_sha256": "c" * 64,
            }
        }
    )
    assert report["task_binding_sha256"] == expected_task_binding
    expected_task_runtime = module._task_runtime_sha256(
        {
            "task-a": {
                "task_digest": "a" * 64,
                "task_tree_sha256": "c" * 64,
            }
        }
    )
    assert report["task_runtime_sha256"] == expected_task_runtime
    assert report["template_count"] == 1
    assert report["runtime_alias_count"] == 2
    assert report["built_count"] == 1
    assert report["reused_count"] == 0
    assert report["semantic_admission_reuse"] is False
    assert report["template_id_access_checked"] is False
    assert report["template_ids_pinned"] is True
    assert report["skip_cache"] is True
    assert report["late_tests_post_start"] is True
    assert report["templates"] == [
        {
            "consumer_count": 2,
            "runtime_alias_count": 2,
            "roles": ["agent", "verifier"],
        }
    ]
    assert "task-a" not in str(report)
    assert "agent-task-a" not in str(report)
    assert pins == {
        "schema_version": 2,
        "pins": {
            "agent-task-a": {
                "template_id": "template-id-123",
                "build_id": "build-id-456",
            },
            "agent-task-b": {
                "template_id": "template-id-123",
                "build_id": "build-id-456",
            },
        },
        "tasks": {
            "task-a": {
                "task_digest": "a" * 64,
                "task_tree_sha256": "c" * 64,
            }
        },
        "task_count": 1,
        "task_set_sha256": module._task_set_sha256({"task-a"}),
        "task_binding_sha256": expected_task_binding,
        "task_runtime_sha256": expected_task_runtime,
    }
    first._create_template.assert_awaited_once()
    create_kwargs = first._create_template.await_args.kwargs
    assert create_kwargs["skip_cache"] is True
    assert create_kwargs["alias"].startswith("miles-swe-admit-")
    assert "agent-task-a" not in create_kwargs["alias"]
    second._create_template.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verified",
    [False, True],
    ids=["swe-gym", "swebench-verified"],
)
async def test_prebuild_reuses_semantic_template_ids_without_building(
    tmp_path: Path,
    *,
    verified: bool,
) -> None:
    module = _load_prebuild_module()
    task_environment = MagicMock()
    task_environment.resolve_baseline.return_value = MagicMock()
    identity = "d" * 64
    specs = [
        module.TemplateSpec(
            task_id="task-a",
            task_digest="a" * 64,
            source_image="registry.example/repo@sha256:" + "b" * 64,
            task_tree_sha256="c" * 64,
            role=role,
            environment_dir=tmp_path,
            environment_name=f"task-a-{role}",
            task_environment=task_environment,
        )
        for role in ("agent", "verifier")
    ]
    first = object.__new__(E2BEnvironment)
    first._template_name = "agent-alias"
    first._template_build_identity = identity
    first._create_template = AsyncMock()
    second = object.__new__(E2BEnvironment)
    second._template_name = "verifier-alias"
    second._template_build_identity = identity
    second._create_template = AsyncMock()
    record = (
        _verified_semantic_record(template_identity=identity)
        if verified
        else _semantic_record(template_identity=identity)
    )
    agent_evidence = record["template_evidence"]["agent"]
    record["template_evidence"]["source"] = {
        **agent_evidence,
        "sandbox_id": "sandbox-source",
    }
    record["template_evidence"]["empty_verifier"] = {
        **agent_evidence,
        "sandbox_id": "sandbox-verifier",
    }
    record["template_evidence"]["oracle_verifier"] = {
        **agent_evidence,
        "sandbox_id": "sandbox-oracle",
    }

    with (
        patch(
            "miles.rollout.harbor.environment_config.build_harbor_environment_config",
            return_value=MagicMock(),
        ),
        patch(
            "harbor.environments.factory.EnvironmentFactory.create_environment_from_config",
            side_effect=[first, second],
        ),
        patch("e2b.AsyncTemplate.get_tags", new=AsyncMock(return_value=[])) as get_tags,
    ):
        report, pins = await module._prebuild(
            specs,
            concurrency=4,
            semantic_admissions={"task-a": record},
        )

    assert report["built_count"] == 0
    assert report["reused_count"] == 1
    assert report["semantic_admission_reuse"] is True
    assert report["template_id_access_checked"] is True
    assert pins["pins"] == {
        "agent-alias": {
            "template_id": "template-agent",
            "build_id": "build-agent",
        },
        "verifier-alias": {
            "template_id": "template-agent",
            "build_id": "build-agent",
        },
    }
    assert pins["schema_version"] == 2
    assert pins["tasks"] == {
        "task-a": {
            "task_digest": "a" * 64,
            "task_tree_sha256": "c" * 64,
        }
    }
    assert pins["task_count"] == 1
    assert pins["task_set_sha256"] == module._task_set_sha256({"task-a"})
    assert pins["task_binding_sha256"] == module._task_binding_sha256(
        pins["tasks"]
    )
    assert pins["task_runtime_sha256"] == module._task_runtime_sha256(
        pins["tasks"]
    )
    get_tags.assert_has_awaits([call("template-agent")])
    first._create_template.assert_not_awaited()
    second._create_template.assert_not_awaited()
