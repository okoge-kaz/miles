"""Fail-closed tests for ReBench/SWE-Gym semantic E2B admission."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from experiments.src.environments.swe import admit_swe_gym
from experiments.src.environments.swe import admit_swe_rebench
from experiments.src.environments.swe import e2b_admission
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock
from experiments.src.environments.swe import semantic_admission

_BASE = "1" * 40
_TREE = "2" * 40
_IMAGE = "docker.io/swerebenchv2/test-task@sha256:" + "3" * 64
_GYM_IMAGE = (
    "docker.io/xingyaoww/sweb.eval.x86_64.test-task@sha256:" + "3" * 64
)
_ORACLE = (
    "diff --git a/pkg/app.py b/pkg/app.py\n"
    "--- a/pkg/app.py\n"
    "+++ b/pkg/app.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)
_TEST_PATCH = (
    "diff --git a/tests/test_app.py b/tests/test_app.py\n"
    "--- a/tests/test_app.py\n"
    "+++ b/tests/test_app.py\n"
    "@@ -1 +1 @@\n"
    "-assert False\n"
    "+assert True\n"
)


def test_source_inspection_discovers_workdir_repo_and_cleans_untracked() -> None:
    script = semantic_admission._SOURCE_INSPECTION_SCRIPT

    assert "repo=\"$(pwd -P)\"" in script
    assert 'source_gitdir="$(cd "${repo}/.git" && pwd -P)"' in script
    assert '-c safe.directory="${repo}"' in script
    assert '-C "${repo}" "$@"' in script
    assert "clean -ffdx" in script
    assert "ls-files --others -z" in script
    assert "exit 30" in script


def _candidate(schema: str, instance_id: str = "swe-task-1") -> dict:
    verifier: dict
    source_metadata: dict
    if schema == "swe-rebench-v2":
        verifier = {
            "kind": "swe-rebench-v2",
            "install_config": {
                "test_cmd": "pytest -q",
                "log_parser": "parse_log_pytest",
            },
            "fail_to_pass": ["tests/test_app.py::test_fix"],
            "pass_to_pass": ["tests/test_app.py::test_existing"],
            "test_patch": _TEST_PATCH,
        }
        source_metadata = {
            "split": "train",
            "language": "python",
            "interface": "cli",
        }
        source_dataset = "nebius/SWE-rebench-V2"
        requested_image = "docker.io/swerebenchv2/test-task:latest"
        resolved_image = _IMAGE
    else:
        verifier = {
            "kind": "swebench-harness-v1",
            "fail_to_pass": ["tests/test_app.py::test_fix"],
            "pass_to_pass": ["tests/test_app.py::test_existing"],
            "test_patch": _TEST_PATCH,
        }
        source_metadata = {"split": "train", "version": "1.0"}
        source_dataset = "SWE-Gym/SWE-Gym"
        requested_image = "docker.io/xingyaoww/sweb.eval.x86_64.test-task:latest"
        resolved_image = _GYM_IMAGE
    row = {
        "schema_version": "miles-swe-task-v1",
        "instance_id": instance_id,
        "source_dataset": source_dataset,
        "source_schema": schema,
        "repo": "django/django",
        "problem_statement": "Fix the repository behavior.",
        "base_commit": _BASE,
        "sandbox": {
            "source_image": requested_image,
            "backend_selector": "harbor",
        },
        "solution": {"oracle_patch": _ORACLE},
        "verifier": verifier,
        "source_metadata": source_metadata,
        "eval_only": False,
        "task_digest": hashlib.sha256(instance_id.encode()).hexdigest(),
    }
    row["content_digest"] = oci_image_lock._stable_digest_without_bindings(row)
    input_digest = row["content_digest"]
    row["sandbox"]["source_image"] = resolved_image
    row["sandbox"]["image_lock"] = {
        "schema_version": oci_image_lock.LOCK_SCHEMA,
        "source_image_requested": requested_image,
        "source_image_resolved": resolved_image,
        "input_content_digest": input_digest,
        "index_digest": "sha256:" + "4" * 64,
        "child_manifest_digest": "sha256:" + "3" * 64,
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    row["content_digest"] = oci_image_lock._stable_digest_without_bindings(row)
    return row


def _template_evidence(role: str, sandbox_index: int) -> dict[str, str]:
    shared = role in {"source", "verifier"}
    marker = "shared" if shared else "agentx"
    return {
        "template_id": f"template-{marker}",
        "build_id": f"build-{marker}",
        "alias_sha256": hashlib.sha256(marker.encode()).hexdigest(),
        "template_identity_sha256": hashlib.sha256(marker.encode()).hexdigest(),
        "sandbox_id": f"sandbox-{sandbox_index:06d}",
    }


@dataclass
class _FakeSandbox:
    spec: e2b_admission.SandboxSpec
    backend: _FakeBackend
    evidence: dict[str, str]
    patch: bytes = b""

    @property
    def template_evidence(self) -> dict[str, str]:
        return dict(self.evidence)

    async def exec(
        self,
        command: str,
        *,
        user: int,
        timeout_sec: float,
    ) -> e2b_admission.RemoteResult:
        self.backend.commands.append((self.spec.role, user, command, timeout_sec))
        if "1.1.1.1" in command:
            self.backend.events.append(f"{self.spec.role}:network")
        else:
            self.backend.events.append(f"{self.spec.role}:command")
        if "1.1.1.1" in command and self.backend.network_exposed:
            return e2b_admission.RemoteResult(71, "", "")
        return e2b_admission.RemoteResult(0, "fake-ok", "")

    async def upload_file(self, source: Path, destination: str) -> None:
        self.backend.uploads.append((self.spec.role, destination))
        if destination == "/tmp/miles-swe-model.patch":
            assert "verifier:private-package" in self.backend.events
            self.backend.events.append("verifier:patch")
            self.patch = source.read_bytes()
        else:
            self.backend.events.append(f"{self.spec.role}:upload")

    async def download_file(self, source: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.spec.role == "source":
            destination.write_text(f"{_BASE}\n{_TREE}\n", encoding="utf-8")
            destination.chmod(0o600)
            return
        reward = int(bool(self.patch))
        if self.backend.oracle_reward is not None and self.patch:
            reward = self.backend.oracle_reward
        if source.endswith("reward.txt"):
            destination.write_text(f"{reward}\n", encoding="utf-8")
            destination.chmod(0o600)
            return
        report = {"kind": self.backend.report_kind, "reward": reward}
        if self.backend.report_kind == "swe-rebench-v2":
            report["parser"] = "parse_log_pytest"
        destination.write_text(json.dumps(report), encoding="utf-8")
        destination.chmod(0o600)

    async def install_private_verifier(self, tests_dir: Path) -> None:
        assert self.spec.role == "verifier"
        assert tests_dir == self.spec.context_dir
        self.backend.events.append("verifier:private-package")

    async def close(self) -> None:
        self.backend.closed += 1


class _FakeBackend:
    def __init__(
        self,
        report_kind: str,
        *,
        oracle_reward: int | None = None,
        network_exposed: bool = False,
    ) -> None:
        self.report_kind = report_kind
        self.oracle_reward = oracle_reward
        self.network_exposed = network_exposed
        self.specs: list[e2b_admission.SandboxSpec] = []
        self.commands: list[tuple[str, int, str, float]] = []
        self.uploads: list[tuple[str, str]] = []
        self.events: list[str] = []
        self.closed = 0

    async def start(self, spec: e2b_admission.SandboxSpec) -> _FakeSandbox:
        self.specs.append(spec)
        self.events.append(f"{spec.role}:start")
        return _FakeSandbox(
            spec=spec,
            backend=self,
            evidence=_template_evidence(spec.role, len(self.specs)),
        )


def _config(tmp_path: Path, row: dict) -> semantic_admission.AdmissionConfig:
    private = tmp_path / "candidate.private.jsonl"
    private.write_text(json.dumps(row) + "\n", encoding="utf-8")
    private.chmod(0o600)
    return semantic_admission.AdmissionConfig(
        private_manifest=private,
        admitted_manifest=tmp_path / "state" / "admitted.tasks.private.jsonl",
        admission_manifest=tmp_path / "state" / "admission.private.jsonl",
        quarantine_manifest=tmp_path / "state" / "quarantine.private.jsonl",
        work_root=tmp_path / "work",
        concurrency=2,
    )


def _rebench_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> admit_swe_rebench.RebenchAdapter:
    parser = tmp_path / "log_parsers.py"
    constants = tmp_path / "swe_constants.py"
    official_eval = tmp_path / "eval.py"
    parser.write_text("NAME_TO_PARSER = {}\n", encoding="utf-8")
    constants.write_text("class TestStatus: pass\n", encoding="utf-8")
    official_eval.write_text("def evaluate_instance(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        materialize_module,
        "_REBENCH_LOG_PARSERS_SHA256",
        hashlib.sha256(parser.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        materialize_module,
        "_REBENCH_CONSTANTS_SHA256",
        hashlib.sha256(constants.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        materialize_module,
        "_REBENCH_EVAL_SHA256",
        hashlib.sha256(official_eval.read_bytes()).hexdigest(),
    )
    return admit_swe_rebench.RebenchAdapter(parser, constants, official_eval)


def _gym_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> admit_swe_gym.SweGymAdapter:
    harness = tmp_path / "harness" / "swegym" / "harness"
    harness.mkdir(parents=True)
    for name, constant in (
        ("constants", "_SWE_GYM_CONSTANTS_SHA256"),
        ("log_parsers", "_SWE_GYM_LOG_PARSERS_SHA256"),
        ("grading", "_SWE_GYM_GRADING_SHA256"),
    ):
        path = harness / f"{name}.py"
        path.write_text(f"# pinned fake {name}\n", encoding="utf-8")
        monkeypatch.setattr(
            materialize_module,
            constant,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return admit_swe_gym.SweGymAdapter(tmp_path / "harness")


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", ["swe-rebench-v2", "swe-gym"])
async def test_semantic_admission_binds_live_outcomes_and_fresh_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
) -> None:
    row = _candidate(schema)
    config = _config(tmp_path, row)
    if schema == "swe-rebench-v2":
        adapter = _rebench_adapter(tmp_path, monkeypatch)
        report_kind = "swe-rebench-v2"
    else:
        adapter = _gym_adapter(tmp_path, monkeypatch)
        report_kind = "swe-gym-pinned-v2.0.13"
    backend = _FakeBackend(report_kind)

    summary = await semantic_admission.admit_tasks(config, adapter, backend)

    assert summary == {
        "selected": 1,
        "admitted": 1,
        "quarantined": 0,
        "resumed": 0,
    }
    assert [spec.role for spec in backend.specs] == [
        "source",
        "agent",
        "verifier",
        "verifier",
    ]
    assert all(
        spec.expected_image == row["sandbox"]["source_image"]
        for spec in backend.specs
    )
    assert backend.closed == 4
    for index, event in enumerate(backend.events):
        if event.endswith(":start"):
            role = event.removesuffix(":start")
            assert backend.events[index + 1] == f"{role}:network"
    agent_commands = [
        command
        for role, _user, command, _timeout in backend.commands
        if role == "agent"
    ]
    if schema == "swe-rebench-v2":
        assert any("agent-public-baseline" in command for command in agent_commands)
    else:
        assert any("swe_gym_run.py --public-baseline" in command for command in agent_commands)
        assert ("agent", "/tmp/miles-swe-public/config.json") in backend.uploads
        assert all("test_patch.diff" not in destination for _, destination in backend.uploads)
    admission = json.loads(config.admission_manifest.read_text(encoding="utf-8"))
    admitted = json.loads(config.admitted_manifest.read_text(encoding="utf-8"))
    assert admitted == row
    assert admission["schema_version"] == adapter.admission_schema
    assert admission["checks"]["empty_reward"] == 0
    assert admission["checks"]["oracle_reward"] == 1
    evidence = admission["template_evidence"]
    assert evidence["empty_verifier"]["template_id"] == evidence["oracle_verifier"]["template_id"]
    assert evidence["empty_verifier"]["sandbox_id"] != evidence["oracle_verifier"]["sandbox_id"]
    assert config.quarantine_manifest.read_text(encoding="utf-8") == ""
    assert config.admission_manifest.stat().st_mode & 0o077 == 0
    assert config.quarantine_manifest.stat().st_mode & 0o077 == 0
    assert config.admitted_manifest.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_golden_mismatch_is_reasoned_quarantine_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _candidate("swe-rebench-v2"))
    adapter = _rebench_adapter(tmp_path, monkeypatch)
    first = _FakeBackend("swe-rebench-v2", oracle_reward=0)

    summary = await semantic_admission.admit_tasks(config, adapter, first)

    assert summary["quarantined"] == 1
    assert config.admission_manifest.read_text(encoding="utf-8") == ""
    assert config.admitted_manifest.read_text(encoding="utf-8") == ""
    quarantine = json.loads(config.quarantine_manifest.read_text(encoding="utf-8"))
    assert quarantine["reason_code"] == "golden_outcome_mismatch"
    resumed = _FakeBackend("swe-rebench-v2")
    summary = await semantic_admission.admit_tasks(config, adapter, resumed)
    assert summary == {
        "selected": 1,
        "admitted": 0,
        "quarantined": 0,
        "resumed": 1,
    }
    assert resumed.specs == []


@pytest.mark.asyncio
async def test_public_network_exposure_is_fatal_not_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _candidate("swe-rebench-v2"))
    adapter = _rebench_adapter(tmp_path, monkeypatch)

    with pytest.raises(
        semantic_admission.SystemicAdmissionError,
        match="public-network isolation",
    ):
        await semantic_admission.admit_tasks(
            config,
            adapter,
            _FakeBackend("swe-rebench-v2", network_exposed=True),
        )

    assert not config.admission_manifest.exists()
    assert not config.admitted_manifest.exists()
    assert not config.quarantine_manifest.exists()


@pytest.mark.asyncio
async def test_checkpoint_compaction_is_linear_and_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_candidate("swe-rebench-v2", f"swe-task-{index}") for index in range(5)]
    config = _config(tmp_path, rows[0])
    config.private_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    config.private_manifest.chmod(0o600)
    config = replace(config, concurrency=2)
    adapter = _rebench_adapter(tmp_path, monkeypatch)

    summary = await semantic_admission.admit_tasks(
        config,
        adapter,
        _FakeBackend("swe-rebench-v2"),
    )

    assert summary["selected"] == 5
    checkpoint_dir = config.admission_manifest.with_name(
        config.admission_manifest.name + ".d"
    )
    checkpoints = list(checkpoint_dir.glob("*.json"))
    assert len(checkpoints) == 5
    assert len(config.admission_manifest.read_text().splitlines()) == 5
    assert len(config.admitted_manifest.read_text().splitlines()) == 5
    value = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    value["record"]["checks"]["oracle_reward"] = 0
    checkpoints[0].write_text(json.dumps(value) + "\n", encoding="utf-8")
    checkpoints[0].chmod(0o600)
    with pytest.raises(ValueError, match="required live check"):
        await semantic_admission.admit_tasks(
            config,
            adapter,
            _FakeBackend("swe-rebench-v2"),
        )


@pytest.mark.asyncio
async def test_native_template_cache_builds_once_with_random_skip_cache_alias() -> None:
    class BuildInfo:
        template_id = "template-id-123"
        build_id = "build-id-123"

    class Environment:
        class Config:
            docker_image = _IMAGE

        task_env_config = Config()
        _effective_cpus = 4
        _effective_memory_mb = 8192
        environment_id = "dockerfile-content-id"

        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        async def _create_template(self, *, alias: str, skip_cache: bool):
            self.calls.append((alias, skip_cache))
            return BuildInfo()

    backend = object.__new__(e2b_admission.NativeHarborE2BBackend)
    backend._template_pins = {}
    backend._template_locks = {}
    backend._template_locks_guard = asyncio.Lock()
    first = Environment()
    second = Environment()

    pins = await asyncio.gather(
        backend._fresh_template_pin(first),
        backend._fresh_template_pin(second),
    )

    assert len(first.calls) + len(second.calls) == 1
    alias, skip_cache = (first.calls or second.calls)[0]
    assert alias.startswith("miles-swe-admit-")
    assert skip_cache is True
    assert pins[0] == pins[1]
    assert pins[0].alias_sha256 == hashlib.sha256(alias.encode()).hexdigest()
    expected_identity = e2b_admission._template_build_identity(first)
    assert pins[0].template_identity_sha256 == hashlib.sha256(
        expected_identity.encode()
    ).hexdigest()


def test_source_and_late_verifier_share_exact_image_resource_identity() -> None:
    class Config:
        docker_image = _IMAGE

    class Environment:
        task_env_config = Config()
        _effective_cpus = 4
        _effective_memory_mb = 8192

        def __init__(self, environment_id: str) -> None:
            self.environment_id = environment_id

    source = Environment("source-specific-name")
    verifier = Environment("verifier-specific-name")

    assert e2b_admission._template_build_identity(source) == (
        e2b_admission._template_build_identity(verifier)
    )


def test_native_backend_requires_process_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(ValueError, match="process environment"):
        e2b_admission.NativeHarborE2BBackend()


def test_swe_gym_dependency_validation_is_local_and_hash_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dependency validation attempted network access")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    adapter = _gym_adapter(tmp_path, monkeypatch)
    adapter.validate_dependencies()

    constants = tmp_path / "harness" / "swegym" / "harness" / "constants.py"
    constants.write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        adapter.validate_dependencies()


def test_private_evidence_readback_rejects_hardlinks(tmp_path: Path) -> None:
    evidence = tmp_path / "reward.txt"
    evidence.write_text("1\n", encoding="utf-8")
    evidence.chmod(0o600)
    linked = tmp_path / "reward-hardlink.txt"
    linked.hardlink_to(evidence)

    with pytest.raises(
        semantic_admission.SystemicAdmissionError,
        match="unsafe private E2B readback",
    ):
        semantic_admission._read_regular_bounded(evidence, maximum_bytes=8)


def test_private_evidence_readback_rejects_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "report.json"
    evidence.write_text('{"reward":1}\n', encoding="utf-8")
    evidence.chmod(0o600)
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"reward":0}\n', encoding="utf-8")
    replacement.chmod(0o600)
    original_read = semantic_admission.os.read
    replaced = False

    def replace_after_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        content = original_read(descriptor, size)
        if not replaced:
            replaced = True
            replacement.replace(evidence)
        return content

    monkeypatch.setattr(semantic_admission.os, "read", replace_after_read)
    with pytest.raises(
        semantic_admission.SystemicAdmissionError,
        match="changed",
    ):
        semantic_admission._read_regular_bounded(evidence, maximum_bytes=128)


def test_private_evidence_readback_rejects_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "report.json"
    evidence.write_text('{"reward":1}\n', encoding="utf-8")
    evidence.chmod(0o600)
    original_read = semantic_admission.os.read
    truncated = False

    def truncate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal truncated
        content = original_read(descriptor, size)
        if not truncated:
            truncated = True
            evidence.write_bytes(b"")
            evidence.chmod(0o600)
        return content

    monkeypatch.setattr(semantic_admission.os, "read", truncate_after_read)
    with pytest.raises(
        semantic_admission.SystemicAdmissionError,
        match="changed",
    ):
        semantic_admission._read_regular_bounded(evidence, maximum_bytes=128)


@pytest.mark.asyncio
async def test_sandbox_teardown_failure_preserves_cancellation() -> None:
    class FailingCloseSandbox:
        async def close(self) -> None:
            raise RuntimeError("teardown failed")

    class Backend:
        async def start(
            self,
            _spec: e2b_admission.SandboxSpec,
        ) -> FailingCloseSandbox:
            return FailingCloseSandbox()

    spec = e2b_admission.SandboxSpec(
        role="source",
        name="cancel-test",
        context_dir=Path("."),
        source_image=_IMAGE,
        expected_image=_IMAGE,
    )
    with pytest.raises(asyncio.CancelledError) as caught:
        async with e2b_admission.sandbox(Backend(), spec):
            raise asyncio.CancelledError

    assert isinstance(caught.value.__cause__, e2b_admission.LifecycleCleanupError)


def test_swe_setup_wires_admitted_names_and_candidate_locations() -> None:
    root = Path(__file__).parents[3]
    materialize_setup = (
        root / "experiments/setup/environments/materialize_harbor_swe_tasks.sbatch"
    ).read_text(encoding="utf-8")
    prepare_setup = (
        root / "experiments/setup/datasets/prepare_swe_rl.sbatch"
    ).read_text(encoding="utf-8")
    verifier_download = (
        root / "experiments/setup/download/download_swe_verifiers.sbatch"
    ).read_text(encoding="utf-8")
    training = (
        root
        / "experiments/scripts/swe/async/swe-rebench-v2-swe-gym/"
        "qwen3-4b/run.sbatch"
    ).read_text(encoding="utf-8")

    assert '${admitted_root}/${SWE_TASKSET}-train.jsonl' in materialize_setup
    assert '${admitted_root}/${SWE_TASKSET}-train.summary.json' in materialize_setup
    assert '${admitted_root}/${SWE_TASKSET}-train.task-ids.txt' in materialize_setup
    assert '--task-ids-output "${task_ids_output}"' in materialize_setup
    assert '${private_root}/${SWE_TASKSET}.admitted.tasks.private.jsonl' in (
        materialize_setup
    )
    assert "SWE_MATERIALIZE_LIMIT must be a positive integer" in materialize_setup
    assert 'dry_manifest_default="${production_manifest}"' in materialize_setup
    assert "R2E requires live-derived admitted rows" in materialize_setup
    assert (
        "production SWE_MATERIALIZE_LIMIT requires SWE_FINALIZE_ALLOW_SUBSET=1"
        in materialize_setup
    )
    assert "--export=ALL" not in materialize_setup
    assert "E2B_API_KEY" not in materialize_setup
    assert '${candidate_root}/${name}-${usage}.not-admitted.jsonl' in prepare_setup
    assert '${unadmitted_root}/${name}-${usage}.gold-free.jsonl' in prepare_setup
    assert "/data/miles-swe/admitted/${DATASET_TAG}-train.jsonl" in training
    assert "/data/miles-swe/admitted/${DATASET_TAG}-train.summary.json" in training
    assert "--export=ALL" not in verifier_download
    assert "--export=NONE" in verifier_download
    assert "env -i PATH=" in verifier_download
    assert "unset WANDB_API_KEY" in verifier_download
    assert verifier_download.index("export WANDB_MODE=offline") < (
        verifier_download.index('source "${REPO_ROOT}/experiments/env.sh"')
    )
    assert "trap cleanup_active_partial EXIT" in verifier_download
    assert "trap \"exit 143\" TERM" in verifier_download
    assert "--connect-timeout 15 --max-time 120 --retry 3" in verifier_download
