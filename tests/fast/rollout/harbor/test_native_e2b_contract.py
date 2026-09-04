from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("e2b")
pytest.importorskip("harbor")
server_module = pytest.importorskip("miles_agent_server")

from agent_server.trial_runner import (
    _reset_runtime_task_attestations_for_tests,
    _runtime_task_attestation_error,
    _task_tree_attestation_matches,
)
from e2b import FileType

from harbor.environments.base import ExecResult
from harbor.environments.e2b import E2BEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig as TrialEnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import AgentInfo
from harbor.trial.trial import Trial


def test_e2b_task_rejects_requests_before_startup_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    _reset_runtime_task_attestations_for_tests()
    try:
        assert (
            _runtime_task_attestation_error("bound-task", "a" * 64)
            == "TaskAttestationUnavailable"
        )
        assert not _task_tree_attestation_matches(
            tmp_path / "bound-task",
            "bound-task",
            "a" * 64,
        )
    finally:
        _reset_runtime_task_attestations_for_tests()


def test_non_e2b_backend_preserves_native_task_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARBOR_ENV_TYPE", "docker")
    assert _runtime_task_attestation_error("legacy-task", None) is None
    assert _task_tree_attestation_matches(
        tmp_path / "legacy-task",
        "legacy-task",
        None,
    )


def test_e2b_server_startup_skips_local_docker_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    monkeypatch.setenv("E2B_API_KEY", "present")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "miles_agent_server.py",
            "--dashboard-port",
            "0",
            "--trials-dir",
            str(tmp_path),
            "--dashboard-log-path",
            str(tmp_path / "requests.jsonl"),
        ],
    )

    with patch.object(server_module, "_check_docker_login") as docker_login:
        with patch.object(server_module.uvicorn, "run") as uvicorn_run:
            server_module.main()

    docker_login.assert_not_called()
    uvicorn_run.assert_called_once()


def _make_environment(
    tmp_path: Path,
    *,
    task_environment: EnvironmentConfig | None = None,
) -> E2BEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()
    return E2BEnvironment(
        environment_dir=environment_dir,
        environment_name="swe-test",
        session_id="session",
        trial_paths=trial_paths,
        task_env_config=task_environment or EnvironmentConfig(env={"TASK_SETTING": "safe"}),
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
    )


@pytest.mark.asyncio
async def test_dockerfile_template_uses_task_directory_as_build_context(tmp_path: Path) -> None:
    environment = _make_environment(tmp_path)
    template = MagicMock()
    template_builder = MagicMock()
    template_builder.from_dockerfile.return_value = template

    with (
        patch("harbor.environments.e2b.Template", return_value=template_builder) as constructor,
        patch("harbor.environments.e2b.AsyncTemplate.build", new=AsyncMock()) as build,
    ):
        await environment._create_template()

    constructor.assert_called_once_with(file_context_path=str(environment.environment_dir))
    template_builder.from_dockerfile.assert_called_once_with(dockerfile_content_or_path=str(environment.environment_dir / "Dockerfile"))
    build.assert_awaited_once_with(
        template=template,
        alias=environment._template_name,
        skip_cache=False,
    )


@pytest.mark.asyncio
async def test_source_image_takes_precedence_over_task_dockerfile(tmp_path: Path) -> None:
    source_image = "registry.example.invalid/swe-base@sha256:" + "b" * 64
    environment = _make_environment(
        tmp_path,
        task_environment=EnvironmentConfig(docker_image=source_image),
    )
    template = MagicMock()
    template_builder = MagicMock()
    template_builder.from_image.return_value = template

    with (
        patch("harbor.environments.e2b.Template", return_value=template_builder) as constructor,
        patch("harbor.environments.e2b.AsyncTemplate.build", new=AsyncMock()) as build,
    ):
        await environment._create_template()

    constructor.assert_called_once_with()
    template_builder.from_image.assert_called_once_with(image=source_image)
    template_builder.from_dockerfile.assert_not_called()
    build.assert_awaited_once_with(
        template=template,
        alias=environment._template_name,
        skip_cache=False,
    )


@pytest.mark.asyncio
async def test_create_and_stop_are_ephemeral_and_do_not_forward_provider_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "must-not-enter-sandbox")
    environment = _make_environment(tmp_path)
    sandbox = MagicMock()
    sandbox.kill = AsyncMock()

    with patch(
        "harbor.environments.e2b.AsyncSandbox.create",
        new=AsyncMock(return_value=sandbox),
    ) as create:
        await environment._create_sandbox()

    assert environment._sandbox is sandbox
    assert create.await_args.kwargs["envs"] == {"TASK_SETTING": "safe"}
    assert create.await_args.kwargs["metadata"] == {
        "environment_name": "swe-test",
        "session_id": "session",
    }
    assert "E2B_API_KEY" not in create.await_args.kwargs["envs"]
    assert "must-not-enter-sandbox" not in repr(create.await_args.kwargs)

    await environment.stop(delete=True)

    sandbox.kill.assert_awaited_once_with()
    assert environment._sandbox is None


@pytest.mark.asyncio
async def test_file_upload_and_download_round_trip_through_native_sdk_contract(
    tmp_path: Path,
) -> None:
    environment = _make_environment(tmp_path)
    sandbox = MagicMock()
    sandbox.files.write = AsyncMock()
    sandbox.files.read = AsyncMock(return_value=b"diff --git a/a.py b/a.py\n")
    environment._sandbox = sandbox

    local_patch = tmp_path / "agent.patch"
    local_patch.write_bytes(b"diff --git a/a.py b/a.py\n")
    await environment.upload_file(local_patch, "/tmp/agent.patch")

    downloaded = tmp_path / "downloaded.patch"
    await environment.download_file("/tmp/agent.patch", downloaded)

    sandbox.files.write.assert_awaited_once_with("/tmp/agent.patch", b"diff --git a/a.py b/a.py\n")
    sandbox.files.read.assert_awaited_once_with("/tmp/agent.patch", format="bytes")
    assert downloaded.read_bytes() == local_patch.read_bytes()


@pytest.mark.asyncio
async def test_patch_command_uses_requested_workspace_user_and_timeout(tmp_path: Path) -> None:
    environment = _make_environment(tmp_path)
    handle = MagicMock()
    handle.wait = AsyncMock(return_value=MagicMock(stdout="applied", stderr="", exit_code=0))
    sandbox = MagicMock()
    sandbox.commands.run = AsyncMock(return_value=handle)
    environment._sandbox = sandbox

    result = await environment.exec(
        "git apply --whitespace=nowarn /tmp/agent.patch",
        cwd="/workspace/repo",
        timeout_sec=120,
        user="root",
    )

    sandbox.commands.run.assert_awaited_once_with(
        cmd="git apply --whitespace=nowarn /tmp/agent.patch",
        background=True,
        cwd="/workspace/repo",
        envs={"TASK_SETTING": "safe"},
        timeout=120,
        user="root",
    )
    assert (result.return_code, result.stdout, result.stderr) == (0, "applied", "")


@pytest.mark.asyncio
async def test_artifact_directory_is_downloaded_before_sandbox_teardown(
    tmp_path: Path,
) -> None:
    environment = _make_environment(tmp_path)
    artifact = MagicMock(path="/logs/artifacts/verifier.json", type=FileType.FILE)
    sandbox = MagicMock()
    sandbox.files.list = AsyncMock(return_value=[artifact])
    environment._sandbox = sandbox
    environment.download_file = AsyncMock()

    local_artifacts = tmp_path / "artifacts"
    await environment.download_dir("/logs/artifacts", local_artifacts)

    environment.download_file.assert_awaited_once_with(
        source_path="/logs/artifacts/verifier.json",
        target_path=str(local_artifacts / "verifier.json"),
    )


def _make_separate_verifier_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('artifacts = ["/workspace/agent.patch"]\n[agent]\ntimeout_sec = 10.0\n[verifier]\ntimeout_sec = 10.0\nenvironment_mode = "separate"\n[verifier.environment]\n[environment]\n')
    (task_dir / "instruction.md").write_text("Fix the repository and publish a patch.\n")
    source_image = "ubuntu:24.04@sha256:" + "a" * 64
    environment_dir = task_dir / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(f"FROM {source_image}\nRUN git -C /workspace/repo reset --hard 1111111111111111111111111111111111111111\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text(f"FROM {source_image}\nCOPY test.sh /tests/test.sh\n")
    (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task_dir


def _offline_e2b_instance(
    environment: E2BEnvironment,
    *,
    role: str,
    events: list[str],
) -> None:
    async def start(*, force_build: bool) -> None:
        assert force_build is False
        events.append(f"{role}:start")

    async def stop(*, delete: bool) -> None:
        assert delete is True
        events.append(f"{role}:stop")

    async def download_file(source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"captured from {source_path}\n")

    async def download_dir(source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        if role == "agent" and source_dir == "/logs/artifacts":
            (target / "agent-evidence.json").write_text('{"captured":true}\n')
        if role == "verifier" and source_dir == "/logs/verifier":
            (target / "reward.txt").write_text("1.0\n")

    environment.start = AsyncMock(side_effect=start)
    environment.stop = AsyncMock(side_effect=stop)
    environment.exec = AsyncMock(return_value=ExecResult(stdout="/", stderr="", return_code=0))
    environment.upload_file = AsyncMock()
    environment.upload_dir = AsyncMock()
    environment.download_file = AsyncMock(side_effect=download_file)
    environment.download_dir = AsyncMock(side_effect=download_dir)
    environment.is_dir = AsyncMock(side_effect=lambda path, user=None: path == "/logs/artifacts")
    environment.is_file = AsyncMock(side_effect=lambda path, user=None: path != "/logs/artifacts")


@pytest.mark.asyncio
async def test_separate_verifier_uses_fresh_e2b_and_transfers_agent_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "offline-placeholder")
    task_dir = _make_separate_verifier_task(tmp_path)
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    environments: list[E2BEnvironment] = []
    events: list[str] = []
    create_environment = EnvironmentFactory.create_environment_from_config

    def create_offline_environment(**kwargs) -> E2BEnvironment:
        environment = create_environment(**kwargs)
        assert isinstance(environment, E2BEnvironment)
        role = "agent" if not environments else "verifier"
        _offline_e2b_instance(environment, role=role, events=events)
        environments.append(environment)
        return environment

    agent = MagicMock(
        name=lambda: "oracle",
        version=lambda: "1.0",
        SUPPORTS_ATIF=False,
        SUPPORTS_WINDOWS=True,
        setup=AsyncMock(),
        run=AsyncMock(),
        to_agent_info=lambda: AgentInfo(name="oracle", version="1.0"),
    )
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trial_name="e2b-separate",
        trials_dir=trials_dir,
        agent=AgentConfig(name="oracle"),
        environment=TrialEnvironmentConfig(type="e2b", delete=True),
        verifier=VerifierConfig(),
    )

    with (
        patch(
            "harbor.trial.trial.EnvironmentFactory.create_environment_from_config",
            side_effect=create_offline_environment,
        ),
        patch(
            "harbor.trial.trial.AgentFactory.create_agent_from_config",
            return_value=agent,
        ),
        patch("harbor.trial.trial.AgentName") as agent_name,
    ):
        agent_name.ORACLE.value = "oracle"
        trial = await Trial.create(config)
        result = await trial.run()

    assert len(environments) == 2
    agent_environment, verifier_environment = environments
    assert agent_environment is not verifier_environment
    assert agent_environment.environment_dir == task_dir / "environment"
    assert verifier_environment.environment_dir == (task_dir / "tests").resolve()
    assert agent_environment.task_env_config.docker_image is None
    assert verifier_environment.task_env_config.docker_image is None
    assert verifier_environment.session_id.endswith("__verifier__trial")
    assert events.index("agent:stop") < events.index("verifier:start")
    assert events[-1] == "verifier:stop"
    verifier_environment.upload_dir.assert_awaited_once_with(
        source_dir=trial.paths.artifacts_dir / "logs" / "artifacts",
        target_dir="/logs/artifacts",
    )
    verifier_environment.upload_file.assert_awaited_once_with(
        source_path=trial.paths.artifacts_dir / "workspace" / "agent.patch",
        target_path="/workspace/agent.patch",
    )
    assert result.verifier_result.rewards["reward"] == 1.0
