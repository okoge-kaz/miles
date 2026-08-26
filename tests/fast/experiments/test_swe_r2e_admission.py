"""Fail-closed tests for R2E-Gym native-E2B live admission."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import urllib.error
from dataclasses import dataclass, replace
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.src.environments.swe import admit_r2e
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock

_BASE = "1" * 40
_GOLD = "2" * 40
_IMAGE = "docker.io/namanjain12/numpy_final@sha256:" + "3" * 64
_PATCH = (
    "diff --git a/numpy.py b/numpy.py\n"
    "index 257cc56b9f16d38ece55bdbb3f55b08eb16476a2..5716ca5987cbf97d6bb54920bea6adde242d87e6 100644\n"
    "--- a/numpy.py\n"
    "+++ b/numpy.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)


def _candidate() -> dict:
    row = {
        "schema_version": "miles-swe-task-v1",
        "instance_id": "r2e-" + "4" * 32,
        "source_dataset": "R2E-Gym/R2E-Gym-V1",
        "source_schema": "r2e-gym-v1",
        "repo": "numpy",
        "problem_statement": "Fix array conversion.",
        "base_commit": None,
        "sandbox": {
            "source_image": _IMAGE,
            "backend_selector": "harbor",
        },
        "solution": {"oracle_patch": None},
        "verifier": {
            "kind": "r2e-expected-pytest-map-v1",
            "gold_commit": _GOLD,
            "expected_output": {"test_array": "PASSED"},
        },
        "source_metadata": {
            "split": "train",
            "repo_name": "numpy",
            "published_instance_id": None,
        },
        "eval_only": False,
        "task_digest": "5" * 64,
    }
    row["content_digest"] = admit_r2e._stable_digest_without_bindings(row)
    input_content_digest = row["content_digest"]
    row["sandbox"]["image_lock"] = {
        "schema_version": oci_image_lock.LOCK_SCHEMA,
        "source_image_requested": _IMAGE,
        "source_image_resolved": _IMAGE,
        "input_content_digest": input_content_digest,
        "index_digest": None,
        "child_manifest_digest": "sha256:" + "3" * 64,
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    row["content_digest"] = admit_r2e._stable_digest_without_bindings(row)
    return row


@dataclass
class _FakeSandbox:
    spec: admit_r2e.SandboxSpec
    backend: _FakeBackend
    sandbox_number: int
    patch: bytes | None = None
    closed: bool = False

    @property
    def template_evidence(self) -> dict[str, str]:
        template_role = (
            "image" if self.spec.role in {"source", "verifier"} else self.spec.role
        )
        return {
            "template_id": f"template-{template_role}-000001",
            "build_id": f"build-{template_role}-000001",
            "alias_sha256": hashlib.sha256(template_role.encode()).hexdigest(),
            "template_identity_sha256": hashlib.sha256(
                f"identity-{template_role}".encode()
            ).hexdigest(),
            "sandbox_id": f"sandbox-{self.sandbox_number:06d}",
        }

    async def exec(
        self,
        command: str,
        *,
        user: int,
        timeout_sec: float,
    ) -> admit_r2e.RemoteResult:
        self.backend.commands.append((self.spec.role, user, command, timeout_sec))
        if command == admit_r2e._NETWORK_DENIAL_SCRIPT:
            self.backend.events.append(f"{self.spec.role}:network-denied")
        if command == "test ! -e /tests/.harbor-e2b-late-tests":
            self.backend.events.append("verifier:late-tests-absent")
        if (
            self.spec.role == "verifier"
            and self.backend.fail_verifier
            and "/tests/test.sh" in command
        ):
            return admit_r2e.RemoteResult(2, "", "fake verifier failure")
        if (
            self.spec.role == "source"
            and self.backend.fail_source
            and "/tmp/miles-r2e-inspect.sh" in command
        ):
            return admit_r2e.RemoteResult(23, "", "fake source incompatibility")
        return admit_r2e.RemoteResult(0, f"{self.spec.role}-ok", "")

    async def upload_file(self, source: Path, destination: str) -> None:
        self.backend.uploads.append((self.spec.role, destination))
        if self.spec.role == "verifier" and destination.endswith("model.patch"):
            assert self.backend.events[-1] == "verifier:private-package"
            self.backend.events.append("verifier:patch")
            self.patch = source.read_bytes()

    async def download_file(self, source: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.spec.role == "source" and source.endswith("source-result"):
            if self.backend.invalid_source_evidence:
                destination.write_bytes(b"\xff")
                return
            patch_digest = hashlib.sha256(_PATCH.encode()).hexdigest()
            destination.write_text(f"{_BASE}\n{patch_digest}\n", encoding="utf-8")
            return
        if self.spec.role == "source" and source.endswith("oracle.patch"):
            destination.write_text(_PATCH, encoding="utf-8")
            return
        if self.spec.role == "verifier" and source.endswith("reward.txt"):
            reward = self._reward()
            destination.write_text(f"{reward}\n", encoding="utf-8")
            return
        if self.spec.role == "verifier" and source.endswith("report.json"):
            destination.write_text(
                json.dumps({"reward": self._reward(), "resolved": bool(self._reward())}),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"unexpected fake download: {self.spec.role} {source}")

    def _reward(self) -> int:
        if self.backend.oracle_reward is not None and self.patch == _PATCH.encode():
            return self.backend.oracle_reward
        return int(bool(self.patch))

    async def install_private_verifier(self, tests_dir: Path) -> None:
        assert self.spec.role == "verifier"
        assert tests_dir == self.spec.context_dir
        self.backend.events.append("verifier:private-package")

    async def close(self) -> None:
        self.closed = True
        self.backend.closed += 1


class _FakeBackend:
    def __init__(
        self,
        *,
        oracle_reward: int | None = None,
        fail_verifier: bool = False,
        fail_source: bool = False,
        invalid_source_evidence: bool = False,
        provider_failure: bool = False,
    ) -> None:
        self.oracle_reward = oracle_reward
        self.fail_verifier = fail_verifier
        self.fail_source = fail_source
        self.invalid_source_evidence = invalid_source_evidence
        self.provider_failure = provider_failure
        self.specs: list[admit_r2e.SandboxSpec] = []
        self.commands: list[tuple[str, int, str, float]] = []
        self.uploads: list[tuple[str, str]] = []
        self.events: list[str] = []
        self.closed = 0

    async def start(self, spec: admit_r2e.SandboxSpec) -> _FakeSandbox:
        if self.provider_failure:
            raise RuntimeError("E2B authentication failed")
        self.specs.append(spec)
        self.events.append(f"{spec.role}:start")
        return _FakeSandbox(
            spec=spec,
            backend=self,
            sandbox_number=len(self.specs),
        )


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[admit_r2e.AdmissionConfig, dict]:
    row = _candidate()
    private_manifest = tmp_path / "candidate.private.jsonl"
    private_manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    private_manifest.chmod(0o600)
    parser = tmp_path / "execution_log_parser.py"
    parser.write_text(
        "def parse_log_pytest(log): return {}\n"
        "def decolor_dict_keys(value): return value\n",
        encoding="utf-8",
    )
    parser_digest = hashlib.sha256(parser.read_bytes()).hexdigest()
    monkeypatch.setattr(materialize_module, "_R2E_PARSER_SHA256", parser_digest)
    config = admit_r2e.AdmissionConfig(
        private_manifest=private_manifest,
        admission_manifest=tmp_path / "state" / "r2e-admission.private.jsonl",
        admitted_manifest=tmp_path / "state" / "r2e-admitted.private.jsonl",
        quarantine_manifest=tmp_path / "state" / "r2e-quarantine.private.jsonl",
        work_root=tmp_path / "work",
        r2e_execution_log_parser=parser,
    )
    return config, row


def _raw_generic_task(instance_id: str, source_image: str, source_schema: str) -> dict:
    row = _candidate()
    row["instance_id"] = instance_id
    row["source_schema"] = source_schema
    if source_schema == "swe-rebench-v2":
        row["source_dataset"] = "nebius/SWE-rebench-V2"
        row["source_metadata"] = {
            "split": "train",
            "language": "python",
            "interface": "cli",
        }
    elif source_schema == "swe-gym":
        row["source_dataset"] = "SWE-Gym/SWE-Gym"
        row["source_metadata"] = {"split": "train", "version": "1.0"}
    row["sandbox"]["source_image"] = source_image
    row["sandbox"].pop("image_lock")
    row["task_digest"] = hashlib.sha256(instance_id.encode()).hexdigest()
    row.pop("content_digest")
    row["content_digest"] = admit_r2e._stable_digest_without_bindings(row)
    return row


def _available_lock(source_image: str) -> dict:
    registry, display_registry, repository, reference = oci_image_lock.parse_image_reference(
        source_image
    )
    digest = "sha256:" + hashlib.sha256(source_image.encode()).hexdigest()
    return {
        "schema_version": oci_image_lock.LOCK_SCHEMA,
        "status": "available",
        "source_image_requested": source_image,
        "source_image_resolved": f"{display_registry}/{repository}@{digest}",
        "registry": registry,
        "repository": repository,
        "reference": reference,
        "reference_kind": "digest" if reference.startswith("sha256:") else "tag",
        "index_digest": None,
        "child_manifest_digest": digest,
        "platform": {"os": "linux", "architecture": "amd64"},
        "resolved_at": "2026-08-26T00:00:00+00:00",
    }


@pytest.mark.parametrize(
    ("source_schema", "source_image"),
    [
        ("r2e-gym-v1", "docker.io/namanjain12/numpy_final:gold"),
        (
            "swe-gym",
            "docker.io/xingyaoww/sweb.eval.x86_64.test-task:latest",
        ),
        ("swe-rebench-v2", "docker.io/swerebenchv2/test-task:latest"),
    ],
)
def test_source_schema_image_publisher_policies_accept_only_expected_namespaces(
    source_schema: str,
    source_image: str,
) -> None:
    trusted = _raw_generic_task("trusted-task", source_image, source_schema)
    oci_image_lock._validate_task(trusted)

    attacker = json.loads(json.dumps(trusted))
    attacker["sandbox"]["source_image"] = "docker.io/attacker/task:latest"
    attacker["content_digest"] = admit_r2e._stable_digest_without_bindings(attacker)
    with pytest.raises(ValueError, match="publisher policy"):
        oci_image_lock._validate_task(attacker)

    wrong_dataset = json.loads(json.dumps(trusted))
    wrong_dataset["source_dataset"] = "attacker/private-copy"
    wrong_dataset["content_digest"] = admit_r2e._stable_digest_without_bindings(
        wrong_dataset
    )
    with pytest.raises(ValueError, match="publisher policy"):
        oci_image_lock._validate_task(wrong_dataset)


@pytest.mark.asyncio
async def test_admission_derives_oracle_runs_fresh_verifiers_and_writes_private_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, original = _inputs(tmp_path, monkeypatch)
    backend = _FakeBackend()

    summary = await admit_r2e.admit_r2e_tasks(config, backend)

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
    assert backend.specs[0].source_image == _IMAGE
    assert all(spec.expected_image == _IMAGE for spec in backend.specs)
    assert backend.specs[1].context_dir.name == "environment"
    assert all(spec.context_dir.name == "tests" for spec in backend.specs[2:])
    assert backend.closed == 4
    verifier_events = [event for event in backend.events if event.startswith("verifier:")]
    assert verifier_events == [
        "verifier:start",
        "verifier:network-denied",
        "verifier:late-tests-absent",
        "verifier:private-package",
        "verifier:patch",
        "verifier:start",
        "verifier:network-denied",
        "verifier:late-tests-absent",
        "verifier:private-package",
        "verifier:patch",
    ]
    assert [
        role
        for role, _, command, _ in backend.commands
        if command == admit_r2e._NETWORK_DENIAL_SCRIPT
    ] == ["source", "agent", "verifier", "verifier"]
    admission = json.loads(config.admission_manifest.read_text(encoding="utf-8"))
    admitted = json.loads(config.admitted_manifest.read_text(encoding="utf-8"))
    assert admission["schema_version"] == "miles-r2e-admission-v1"
    assert admission["input_content_digest"] != original["content_digest"]
    assert admission["locked_content_digest"] == original["content_digest"]
    assert admission["source_image"] == _IMAGE
    assert admission["base_commit"] == _BASE
    assert len(admission["admitted_task_tree_sha256"]) == 64
    assert admission["checks"]["empty_reward"] == 0
    assert admission["checks"]["oracle_reward"] == 1
    evidence = admission["e2b_sandbox_evidence"]
    assert set(evidence) == {
        "source",
        "agent",
        "empty_verifier",
        "oracle_verifier",
    }
    assert (
        evidence["empty_verifier"]["template_id"]
        == evidence["oracle_verifier"]["template_id"]
    )
    assert evidence["source"]["template_id"] == evidence["empty_verifier"]["template_id"]
    assert (
        evidence["empty_verifier"]["sandbox_id"]
        != evidence["oracle_verifier"]["sandbox_id"]
    )
    assert len({item["sandbox_id"] for item in evidence.values()}) == 4
    assert all(
        value is True
        for key, value in admission["checks"].items()
        if key not in {"empty_reward", "oracle_reward"}
    )
    assert admitted["base_commit"] == _BASE
    assert admitted["solution"]["oracle_patch"] == _PATCH
    assert admitted["task_digest"] == original["task_digest"]
    assert admitted["content_digest"] != original["content_digest"]
    assert config.admission_manifest.stat().st_mode & 0o077 == 0
    assert config.admitted_manifest.stat().st_mode & 0o077 == 0
    production = materialize_module.materialize(
        SimpleNamespace(
            manifest=config.admitted_manifest,
            output=tmp_path / "production-harbor-task",
            r2e_execution_log_parser=config.r2e_execution_log_parser,
            r2e_admission_manifest=config.admission_manifest,
            swe_rebench_log_parsers=None,
            swe_rebench_constants=None,
            swe_gym_harness_root=None,
            swe_gym_admission_manifest=None,
            allow_mutable_images=False,
            allow_unadmitted_r2e_dry_run=False,
            allow_unadmitted_swe_gym_dry_run=False,
            limit=None,
        )
    )
    assert production["tasks"] == 1
    assert admission["admitted_task_tree_sha256"] == materialize_module._task_tree_sha256(
        tmp_path / "production-harbor-task" / admitted["instance_id"]
    )


@pytest.mark.asyncio
async def test_resume_requires_no_new_sandbox_and_preserves_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _inputs(tmp_path, monkeypatch)
    first_backend = _FakeBackend()
    await admit_r2e.admit_r2e_tasks(config, first_backend)
    admission_before = config.admission_manifest.read_bytes()
    admitted_before = config.admitted_manifest.read_bytes()
    resumed_backend = _FakeBackend()

    summary = await admit_r2e.admit_r2e_tasks(config, resumed_backend)

    assert summary == {
        "selected": 1,
        "admitted": 0,
        "quarantined": 0,
        "resumed": 1,
    }
    assert resumed_backend.specs == []
    assert config.admission_manifest.read_bytes() == admission_before
    assert config.admitted_manifest.read_bytes() == admitted_before


@pytest.mark.asyncio
async def test_resume_rejects_tampered_checkpoint_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _inputs(tmp_path, monkeypatch)
    await admit_r2e.admit_r2e_tasks(config, _FakeBackend())
    checkpoint_dir = config.admission_manifest.with_name(config.admission_manifest.name + ".d")
    checkpoint = next(checkpoint_dir.glob("*.json"))
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    value["admission"]["checks"]["runtime_smoke"] = False
    checkpoint.write_text(json.dumps(value) + "\n", encoding="utf-8")
    checkpoint.chmod(0o600)

    with pytest.raises(ValueError, match="runtime_smoke"):
        await admit_r2e.admit_r2e_tasks(config, _FakeBackend())


@pytest.mark.asyncio
async def test_resume_rejects_reused_verifier_sandbox_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _inputs(tmp_path, monkeypatch)
    await admit_r2e.admit_r2e_tasks(config, _FakeBackend())
    checkpoint_dir = config.admission_manifest.with_name(config.admission_manifest.name + ".d")
    checkpoint = next(checkpoint_dir.glob("*.json"))
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    evidence = value["admission"]["e2b_sandbox_evidence"]
    evidence["oracle_verifier"]["sandbox_id"] = evidence["empty_verifier"]["sandbox_id"]
    checkpoint.write_text(json.dumps(value) + "\n", encoding="utf-8")
    checkpoint.chmod(0o600)

    with pytest.raises(ValueError, match="fresh sandboxes"):
        await admit_r2e.admit_r2e_tasks(config, _FakeBackend())


@pytest.mark.asyncio
async def test_resume_rejects_invalid_task_tree_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _inputs(tmp_path, monkeypatch)
    await admit_r2e.admit_r2e_tasks(config, _FakeBackend())
    checkpoint_dir = config.admission_manifest.with_name(config.admission_manifest.name + ".d")
    checkpoint = next(checkpoint_dir.glob("*.json"))
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    value["admission"]["admitted_task_tree_sha256"] = "invalid"
    checkpoint.write_text(json.dumps(value) + "\n", encoding="utf-8")
    checkpoint.chmod(0o600)

    with pytest.raises(ValueError, match="admitted_task_tree_sha256"):
        await admit_r2e.admit_r2e_tasks(config, _FakeBackend())


@pytest.mark.asyncio
async def test_admission_uses_per_record_checkpoints_and_single_stream_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, template = _inputs(tmp_path, monkeypatch)
    rows = []
    for index in range(5):
        row = json.loads(json.dumps(template))
        row["instance_id"] = f"r2e-scale-{index}"
        row["task_digest"] = hashlib.sha256(row["instance_id"].encode()).hexdigest()
        row.pop("content_digest")
        row["content_digest"] = admit_r2e._stable_digest_without_bindings(row)
        rows.append(row)
    config.private_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    config.private_manifest.chmod(0o600)
    config = replace(config, concurrency=2)
    writes = {config.admission_manifest: 0, config.admitted_manifest: 0}
    original_write = admit_r2e._atomic_write_jsonl

    def counted_write(path: Path, values, **kwargs) -> None:
        if path in writes:
            writes[path] += 1
        original_write(path, values, **kwargs)

    monkeypatch.setattr(admit_r2e, "_atomic_write_jsonl", counted_write)

    summary = await admit_r2e.admit_r2e_tasks(config, _FakeBackend())

    checkpoint_dir = config.admission_manifest.with_name(config.admission_manifest.name + ".d")
    assert summary == {
        "selected": 5,
        "admitted": 5,
        "quarantined": 0,
        "resumed": 0,
    }
    assert len(list(checkpoint_dir.glob("*.json"))) == 5
    assert writes == {config.admission_manifest: 1, config.admitted_manifest: 1}
    assert len(config.admission_manifest.read_text().splitlines()) == 5
    assert len(config.admitted_manifest.read_text().splitlines()) == 5


@pytest.mark.asyncio
async def test_failed_oracle_is_reasoned_quarantine_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _inputs(tmp_path, monkeypatch)
    backend = _FakeBackend(oracle_reward=0)

    summary = await admit_r2e.admit_r2e_tasks(config, backend)

    assert summary == {
        "selected": 1,
        "admitted": 0,
        "quarantined": 1,
        "resumed": 0,
    }
    assert config.admission_manifest.read_text(encoding="utf-8") == ""
    assert config.admitted_manifest.read_text(encoding="utf-8") == ""
    quarantine = json.loads(
        config.quarantine_manifest.read_text(encoding="utf-8")
    )
    assert quarantine["reason"] == "golden_outcome_mismatch"
    assert quarantine["reason_detail"] == (
        "empty/oracle verifier outcomes do not match 0/1"
    )
    assert quarantine["observed_empty_reward"] == 0
    assert quarantine["observed_oracle_reward"] == 0
    assert backend.closed == 4

    resumed_backend = _FakeBackend(oracle_reward=1)
    resumed = await admit_r2e.admit_r2e_tasks(config, resumed_backend)
    assert resumed == {
        "selected": 1,
        "admitted": 0,
        "quarantined": 0,
        "resumed": 1,
    }
    assert resumed_backend.specs == []


@pytest.mark.asyncio
async def test_one_golden_mismatch_does_not_abort_other_admissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, template = _inputs(tmp_path, monkeypatch)
    rows = []
    for instance_id in ("r2e-batch-good-a", "r2e-batch-bad", "r2e-batch-good-b"):
        row = json.loads(json.dumps(template))
        row["instance_id"] = instance_id
        row["task_digest"] = hashlib.sha256(instance_id.encode()).hexdigest()
        row.pop("content_digest")
        row["content_digest"] = admit_r2e._stable_digest_without_bindings(row)
        rows.append(row)
    config.private_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    config.private_manifest.chmod(0o600)
    original_reward = _FakeSandbox._reward

    def selective_reward(sandbox: _FakeSandbox) -> int:
        if (
            sandbox.patch == _PATCH.encode()
            and sandbox.spec.name.endswith("r2e-batch-bad")
        ):
            return 0
        return original_reward(sandbox)

    monkeypatch.setattr(_FakeSandbox, "_reward", selective_reward)

    summary = await admit_r2e.admit_r2e_tasks(
        replace(config, concurrency=2),
        _FakeBackend(),
    )

    assert summary == {
        "selected": 3,
        "admitted": 2,
        "quarantined": 1,
        "resumed": 0,
    }
    admitted = [
        json.loads(line)
        for line in config.admitted_manifest.read_text(encoding="utf-8").splitlines()
    ]
    quarantined = [
        json.loads(line)
        for line in config.quarantine_manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["instance_id"] for row in admitted] == [
        "r2e-batch-good-a",
        "r2e-batch-good-b",
    ]
    assert [row["instance_id"] for row in quarantined] == ["r2e-batch-bad"]


@pytest.mark.asyncio
async def test_infrastructure_failure_does_not_issue_partial_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _inputs(tmp_path, monkeypatch)
    backend = _FakeBackend(provider_failure=True)

    with pytest.raises(RuntimeError, match="E2B authentication failed"):
        await admit_r2e.admit_r2e_tasks(config, backend)

    assert not config.admission_manifest.exists()
    assert not config.admitted_manifest.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "reason"),
    [
        (_FakeBackend(fail_source=True), "source_image_unsupported"),
        (_FakeBackend(invalid_source_evidence=True), "source_evidence_invalid"),
        (_FakeBackend(fail_verifier=True), "verifier_incompatible"),
    ],
)
async def test_task_local_incompatibility_is_checkpointed_as_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: _FakeBackend,
    reason: str,
) -> None:
    config, _ = _inputs(tmp_path, monkeypatch)

    summary = await admit_r2e.admit_r2e_tasks(config, backend)

    assert summary == {
        "selected": 1,
        "admitted": 0,
        "quarantined": 1,
        "resumed": 0,
    }
    record = json.loads(config.quarantine_manifest.read_text(encoding="utf-8"))
    assert record["schema_version"] == "miles-r2e-admission-quarantine-v2"
    assert record["reason"] == reason
    assert record["reason_detail"]
    assert record["observed_empty_reward"] is None
    assert record["observed_oracle_reward"] is None

    resumed = await admit_r2e.admit_r2e_tasks(config, _FakeBackend())
    assert resumed == {
        "selected": 1,
        "admitted": 0,
        "quarantined": 0,
        "resumed": 1,
    }


@pytest.mark.asyncio
async def test_workspace_cleanup_failure_does_not_issue_admission_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _inputs(tmp_path, monkeypatch)

    def fail_cleanup(workspace: Path) -> None:
        raise RuntimeError(f"cleanup failed for {workspace.name}")

    monkeypatch.setattr(admit_r2e, "_remove_admission_workspace", fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await admit_r2e.admit_r2e_tasks(config, _FakeBackend())

    checkpoint_dir = config.admission_manifest.with_name(config.admission_manifest.name + ".d")
    assert list(checkpoint_dir.glob("*.json")) == []
    assert not config.admission_manifest.exists()
    assert not config.admitted_manifest.exists()


def test_mutable_source_image_is_rejected_before_live_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, row = _inputs(tmp_path, monkeypatch)
    row["sandbox"]["source_image"] = "docker.io/namanjain12/numpy_final:gold"
    row["sandbox"].pop("image_lock")
    row["content_digest"] = admit_r2e._stable_digest_without_bindings(row)
    config.private_manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config.private_manifest.chmod(0o600)

    with pytest.raises(ValueError, match="immutable"):
        admit_r2e._validate_candidate(row)


@pytest.mark.asyncio
async def test_mutable_image_is_bound_to_atomic_linux_amd64_digest_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, row = _inputs(tmp_path, monkeypatch)
    source_tag = "docker.io/namanjain12/numpy_final:gold"
    row["sandbox"]["source_image"] = source_tag
    row["sandbox"].pop("image_lock")
    row["content_digest"] = admit_r2e._stable_digest_without_bindings(row)
    config.private_manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config.private_manifest.chmod(0o600)
    lock_path = tmp_path / "state" / "images.private.jsonl"
    locked_manifest = tmp_path / "state" / "locked.private.jsonl"

    def resolve(image: str) -> dict:
        assert image == source_tag
        return {
            "schema_version": oci_image_lock.LOCK_SCHEMA,
            "status": "available",
            "source_image_requested": source_tag,
            "source_image_resolved": _IMAGE,
            "registry": "registry-1.docker.io",
            "repository": "namanjain12/numpy_final",
            "reference": "gold",
            "reference_kind": "tag",
            "index_digest": "sha256:" + "6" * 64,
            "child_manifest_digest": "sha256:" + "3" * 64,
            "platform": {"os": "linux", "architecture": "amd64"},
            "resolved_at": "2026-08-26T00:00:00+00:00",
        }

    monkeypatch.setattr(oci_image_lock, "resolve_image_reference", resolve)
    summary = oci_image_lock.lock_private_tasks(
        oci_image_lock.ImageLockConfig(
            private_manifest=config.private_manifest,
            locked_manifest=locked_manifest,
            image_lock_manifest=lock_path,
            resolve_missing=True,
        )
    )
    config = admit_r2e.AdmissionConfig(
        private_manifest=locked_manifest,
        admission_manifest=config.admission_manifest,
        admitted_manifest=config.admitted_manifest,
        quarantine_manifest=config.quarantine_manifest,
        work_root=config.work_root,
        r2e_execution_log_parser=config.r2e_execution_log_parser,
    )
    backend = _FakeBackend()

    await admit_r2e.admit_r2e_tasks(config, backend)

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    admission = json.loads(config.admission_manifest.read_text(encoding="utf-8"))
    assert summary == {
        "tasks": 1,
        "emitted": 1,
        "missing": 0,
        "resolved": 1,
        "reused": 0,
        "refreshed_missing": 0,
        "cached_missing": 0,
    }
    assert lock["source_image_resolved"] == _IMAGE
    assert lock["platform"] == {"os": "linux", "architecture": "amd64"}
    assert lock_path.stat().st_mode & 0o077 == 0
    assert backend.specs[0].source_image == _IMAGE
    assert admission["source_image_requested"] == source_tag
    assert admission["source_image_resolved"] == _IMAGE
    assert admission["input_content_digest"] == row["content_digest"]
    assert admission["locked_content_digest"] != row["content_digest"]


@pytest.mark.parametrize("status", [401, 429, 500])
def test_registry_non_404_failures_are_never_classified_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    headers = Message()
    if status == 401:
        headers["WWW-Authenticate"] = "Basic realm=unsupported"

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://registry-1.docker.io/v2/r2e/numpy/manifests/gold",
            status,
            "failure",
            headers,
            io.BytesIO(),
        )

    monkeypatch.setattr(oci_image_lock.urllib.request, "urlopen", fail)

    with pytest.raises(RuntimeError):
        oci_image_lock.get_manifest("registry-1.docker.io", "r2e/numpy", "gold")


def test_registry_404_is_the_only_missing_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://registry-1.docker.io/v2/r2e/numpy/manifests/gold",
            404,
            "not found",
            Message(),
            io.BytesIO(),
        )

    monkeypatch.setattr(oci_image_lock.urllib.request, "urlopen", missing)

    with pytest.raises(oci_image_lock.ImageNotFoundError):
        oci_image_lock.get_manifest("registry-1.docker.io", "r2e/numpy", "gold")


def test_dockerhub_credentials_are_process_env_only_and_basic_is_token_request_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    class Response:
        headers = Message()

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self, limit: int) -> bytes:
            return b'{"token":"registry-bearer"}'

    def open_token(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setenv("DOCKERHUB_USERNAME", "private-user")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "private-token")
    monkeypatch.setattr(oci_image_lock.urllib.request, "urlopen", open_token)

    token = oci_image_lock._request_bearer_token(
        'Bearer realm="https://auth.docker.io/token",'
        'service="registry.docker.io",scope="repository:r2e/numpy:pull"',
        registry="registry-1.docker.io",
    )

    expected = base64.b64encode(b"private-user:private-token").decode()
    assert token == "registry-bearer"
    assert captured["authorization"] == f"Basic {expected}"
    assert "private-token" not in str(captured["url"])


def test_non_docker_registry_cannot_forward_dockerhub_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    class Response:
        headers = Message()

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self, limit: int) -> bytes:
            return b'{"token":"anonymous-bearer"}'

    def open_token(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setenv("DOCKERHUB_USERNAME", "private-user")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "private-token")
    monkeypatch.setattr(oci_image_lock.urllib.request, "urlopen", open_token)

    token = oci_image_lock._request_bearer_token(
        'Bearer realm="https://auth.docker.io/token",'
        'service="attacker.invalid",scope="repository:attacker/image:pull"',
        registry="attacker.invalid",
    )

    assert token == "anonymous-bearer"
    assert captured["authorization"] is None


def test_generic_lock_quarantines_only_404_and_emits_other_envs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available_image = "docker.io/swerebenchv2/test-task:available"
    missing_image = "docker.io/xingyaoww/sweb.eval.x86_64.test-task:missing"
    rows = [
        _raw_generic_task("rebench-task", available_image, "swe-rebench-v2"),
        _raw_generic_task("swe-gym-task", missing_image, "swe-gym"),
    ]
    private = tmp_path / "all-envs.private.jsonl"
    private.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    private.chmod(0o600)

    def resolve(image: str) -> dict:
        if image == missing_image:
            return oci_image_lock._missing_image_lock(image)
        return _available_lock(image)

    monkeypatch.setattr(oci_image_lock, "resolve_image_reference", resolve)
    locked = tmp_path / "state" / "locked.private.jsonl"
    locks = tmp_path / "state" / "images.private.jsonl"

    summary = oci_image_lock.lock_private_tasks(
        oci_image_lock.ImageLockConfig(
            private_manifest=private,
            locked_manifest=locked,
            image_lock_manifest=locks,
            resolve_missing=True,
            concurrency=2,
            checkpoint_batch_size=2,
        )
    )

    emitted = [json.loads(line) for line in locked.read_text().splitlines()]
    evidence = [json.loads(line) for line in locks.read_text().splitlines()]
    assert summary == {
        "tasks": 2,
        "emitted": 1,
        "missing": 1,
        "resolved": 2,
        "reused": 0,
        "refreshed_missing": 0,
        "cached_missing": 0,
    }
    assert [row["instance_id"] for row in emitted] == ["rebench-task"]
    assert {row["status"] for row in evidence} == {"available", "missing"}


def test_generic_lock_checkpoints_in_batches_and_streams_large_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for index in range(33):
        row = _raw_generic_task(
            f"task-{index}",
            f"docker.io/swerebenchv2/test-task-{index}:v{index}",
            "swe-rebench-v2",
        )
        row["solution"]["oracle_patch"] = "x" * 100_000
        row["content_digest"] = admit_r2e._stable_digest_without_bindings(row)
        rows.append(row)
    private = tmp_path / "large.private.jsonl"
    private.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    private.chmod(0o600)
    monkeypatch.setattr(oci_image_lock, "resolve_image_reference", _available_lock)
    lock_manifest = tmp_path / "state" / "images.private.jsonl"
    compact_writes = 0
    original_write = oci_image_lock._atomic_write_jsonl

    def counted_write(path: Path, values, **kwargs) -> None:
        nonlocal compact_writes
        if path == lock_manifest:
            compact_writes += 1
        original_write(path, values, **kwargs)

    monkeypatch.setattr(oci_image_lock, "_atomic_write_jsonl", counted_write)

    summary = oci_image_lock.lock_private_tasks(
        oci_image_lock.ImageLockConfig(
            private_manifest=private,
            locked_manifest=tmp_path / "state" / "locked.private.jsonl",
            image_lock_manifest=lock_manifest,
            resolve_missing=True,
            concurrency=4,
            checkpoint_batch_size=8,
        )
    )

    assert summary["emitted"] == 33
    assert summary["resolved"] == 33
    assert compact_writes == 1
    checkpoints = list((tmp_path / "state" / "images.private.jsonl.d").glob("*.json"))
    assert len(checkpoints) == 33


def test_generic_lock_detects_input_mutation_before_issuing_compacted_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _raw_generic_task(
        "task",
        "docker.io/xingyaoww/sweb.eval.x86_64.test-task:tag",
        "swe-gym",
    )
    private = tmp_path / "tasks.private.jsonl"
    private.write_text(json.dumps(row) + "\n", encoding="utf-8")
    private.chmod(0o600)

    def mutate_then_resolve(source_image: str) -> dict:
        private.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
        private.chmod(0o600)
        return _available_lock(source_image)

    monkeypatch.setattr(oci_image_lock, "resolve_image_reference", mutate_then_resolve)
    locked = tmp_path / "state" / "locked.private.jsonl"
    locks = tmp_path / "state" / "images.private.jsonl"

    with pytest.raises(RuntimeError, match="changed during OCI"):
        oci_image_lock.lock_private_tasks(
            oci_image_lock.ImageLockConfig(
                private_manifest=private,
                locked_manifest=locked,
                image_lock_manifest=locks,
                resolve_missing=True,
            )
        )

    assert not locked.exists()
    assert not locks.exists()


def test_cached_404_requires_explicit_refresh_before_image_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "docker.io/xingyaoww/sweb.eval.x86_64.test-task:later"
    row = _raw_generic_task("task", image, "swe-gym")
    private = tmp_path / "tasks.private.jsonl"
    private.write_text(json.dumps(row) + "\n", encoding="utf-8")
    private.chmod(0o600)
    locked = tmp_path / "state" / "locked.private.jsonl"
    locks = tmp_path / "state" / "images.private.jsonl"
    monkeypatch.setattr(
        oci_image_lock,
        "resolve_image_reference",
        lambda source_image: oci_image_lock._missing_image_lock(source_image),
    )
    first = oci_image_lock.lock_private_tasks(
        oci_image_lock.ImageLockConfig(
            private_manifest=private,
            locked_manifest=locked,
            image_lock_manifest=locks,
            resolve_missing=True,
        )
    )
    assert first["missing"] == 1

    def must_not_retry(source_image: str) -> dict:
        raise AssertionError(f"unexpected implicit retry for {source_image}")

    monkeypatch.setattr(oci_image_lock, "resolve_image_reference", must_not_retry)
    cached = oci_image_lock.lock_private_tasks(
        oci_image_lock.ImageLockConfig(
            private_manifest=private,
            locked_manifest=locked,
            image_lock_manifest=locks,
        )
    )
    assert cached["cached_missing"] == 1

    def now_available(source_image: str) -> dict:
        value = _available_lock(source_image)
        value["resolved_at"] = "2999-01-01T00:00:00+00:00"
        return value

    monkeypatch.setattr(oci_image_lock, "resolve_image_reference", now_available)
    refreshed = oci_image_lock.lock_private_tasks(
        oci_image_lock.ImageLockConfig(
            private_manifest=private,
            locked_manifest=locked,
            image_lock_manifest=locks,
            resolve_missing=True,
            refresh_missing=True,
        )
    )

    assert refreshed["emitted"] == 1
    assert refreshed["refreshed_missing"] == 1
    assert json.loads(locks.read_text())["status"] == "available"


def test_generic_lock_rejects_symlink_input_and_output_parent(tmp_path: Path) -> None:
    row = _raw_generic_task(
        "task",
        "docker.io/xingyaoww/sweb.eval.x86_64.test-task:tag",
        "swe-gym",
    )
    real_input = tmp_path / "real.private.jsonl"
    real_input.write_text(json.dumps(row) + "\n", encoding="utf-8")
    real_input.chmod(0o600)
    input_link = tmp_path / "input-link.private.jsonl"
    input_link.symlink_to(real_input)
    with pytest.raises(PermissionError, match="non-symlink"):
        oci_image_lock._validate_private(input_link)
    hardlink_input = tmp_path / "hardlink.private.jsonl"
    hardlink_input.hardlink_to(real_input)
    with pytest.raises(PermissionError, match="non-symlink"):
        oci_image_lock._validate_private(hardlink_input)
    hardlink_input.unlink()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PermissionError, match="non-symlink"):
        oci_image_lock._atomic_write_jsonl(parent_link / "output.jsonl", [row])

    with pytest.raises(ValueError, match="alias"):
        oci_image_lock.lock_private_tasks(
            oci_image_lock.ImageLockConfig(
                private_manifest=real_input,
                locked_manifest=real_input,
                image_lock_manifest=tmp_path / "images.private.jsonl",
            )
        )


def test_single_manifest_platform_is_verified_from_content_digest_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.dumps({"os": "linux", "architecture": "arm64"}).encode()
    config_digest = "sha256:" + hashlib.sha256(config).hexdigest()
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": config_digest,
                "size": len(config),
            },
            "layers": [],
        }
    ).encode()
    manifest_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    monkeypatch.setattr(
        oci_image_lock,
        "get_manifest",
        lambda *args, **kwargs: (manifest, manifest_digest),
    )
    monkeypatch.setattr(oci_image_lock, "_get_blob", lambda *args, **kwargs: config)

    with pytest.raises(RuntimeError, match="not linux/amd64"):
        oci_image_lock.resolve_image_reference("docker.io/shared/task:arm")


def test_immutable_reference_must_match_registry_content_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": "sha256:" + "9" * 64,
                "size": 1,
            },
            "layers": [],
        }
    ).encode()
    returned = "sha256:" + hashlib.sha256(body).hexdigest()
    requested = "sha256:" + "8" * 64
    monkeypatch.setattr(oci_image_lock, "get_manifest", lambda *args, **kwargs: (body, returned))

    with pytest.raises(RuntimeError, match="requested immutable digest"):
        oci_image_lock.resolve_image_reference(f"docker.io/shared/task@{requested}")


def test_manifest_schema_and_media_type_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps({"schemaVersion": 1, "mediaType": "application/json"}).encode()
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(oci_image_lock, "get_manifest", lambda *args, **kwargs: (body, digest))

    with pytest.raises(RuntimeError, match="unsupported OCI manifest mediaType"):
        oci_image_lock.resolve_image_reference("docker.io/shared/task:malformed")


def test_private_readback_rejects_symlink_and_size_overflow(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"1234")
    target.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(target)
    hardlink = tmp_path / "hardlink"
    hardlink.hardlink_to(target)

    with pytest.raises(RuntimeError, match="non-symlink"):
        admit_r2e._read_regular_bounded(link, maximum_bytes=8)
    with pytest.raises(RuntimeError, match="non-symlink"):
        admit_r2e._read_regular_bounded(hardlink, maximum_bytes=8)
    hardlink.unlink()
    with pytest.raises(RuntimeError, match="size limit"):
        admit_r2e._read_regular_bounded(target, maximum_bytes=3)


def test_native_backend_fails_closed_without_process_e2b_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    monkeypatch.setenv("HARBOR_E2B_NO_NEW_PRIVS_USERS", "1000")

    with pytest.raises(ValueError, match="E2B_API_KEY"):
        admit_r2e.NativeHarborE2BBackend()


def test_source_git_commands_disable_local_executable_configuration() -> None:
    script = admit_r2e._SOURCE_INSPECTION_SCRIPT
    for required in (
        "GIT_CONFIG_NOSYSTEM=1",
        "GIT_CONFIG_GLOBAL=/dev/null",
        "GIT_NO_REPLACE_OBJECTS=1",
        "-c core.fsmonitor=false",
        "-c core.hooksPath=/dev/null",
        "-c diff.external=",
        "--no-ext-diff --no-textconv",
        "--no-color --no-renames --no-indent-heuristic --diff-algorithm=myers",
        "--src-prefix=a/ --dst-prefix=b/ --unified=3 --binary --full-index",
    ):
        assert required in script
    assert 'gold="$(cat /tmp/miles-r2e-expected-gold)"' in script
    assert '[[ "${head}" == "${base}" ]]' in script
    assert '[[ "${head}" == "${gold}" ]]' not in script


def test_agent_live_check_requires_visible_rootfs_attestation() -> None:
    script = admit_r2e._AGENT_ROOT_CHECK_SCRIPT
    assert "miles-r2e-visible-rootfs-attestation-v1" in script
    assert "r2e-runtime-inventory.sha256" in script
    assert "find / -xdev" in script
    assert "-path /tmp" not in script
    assert 'grep -a -F -q -- "${gold}"' in script

    user_script = admit_r2e._AGENT_USER_CHECK_SCRIPT
    assert "r2e-runtime-imports" in user_script
    assert "importlib.import_module" in user_script


def test_hardened_source_diff_does_not_execute_local_git_drivers(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Miles test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "miles@example.invalid"],
        check=True,
    )
    source = repo / "module.py"
    source.write_text("before = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "module.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("before = False\n", encoding="utf-8")
    (repo / ".gitattributes").write_text("*.py diff=evil\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "module.py", ".gitattributes"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "gold"], check=True)
    gold = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    canary = tmp_path / "git-driver-executed"
    executable = tmp_path / "evil-driver"
    executable.write_text(f"#!/bin/sh\ntouch '{canary}'\nexit 99\n", encoding="utf-8")
    executable.chmod(0o700)
    subprocess.run(
        ["git", "-C", str(repo), "config", "diff.evil.command", str(executable)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.fsmonitor", str(executable)],
        check=True,
    )
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "empty-home"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_COUNT": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_EXTERNAL_DIFF": "",
    }
    result = subprocess.run(
        [
            "git",
            "--no-pager",
            "-c",
            f"safe.directory={repo}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.pager=cat",
            "-c",
            "pager.diff=false",
            "-c",
            "diff.external=",
            "-C",
            str(repo),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            f"{base}..{gold}",
            "--",
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert "module.py" in result.stdout
    assert not canary.exists()
