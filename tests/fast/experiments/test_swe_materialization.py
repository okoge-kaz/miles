"""Security and configuration tests for Harbor SWE task materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.src.datasets.swe.schema import normalize_swe_row
from experiments.src.environments.swe import materialize as materialize_module
from experiments.src.environments.swe import oci_image_lock
from experiments.src.environments.swe import timeouts

_BASE_COMMIT = "1" * 40
_GOLD_COMMIT = "2" * 40


def _arguments(manifest: Path, output: Path, **overrides) -> argparse.Namespace:
    values = {
        "manifest": manifest,
        "output": output,
        "admission_evidence": None,
        "r2e_execution_log_parser": None,
        "r2e_admission_manifest": None,
        "swe_rebench_log_parsers": None,
        "swe_rebench_constants": None,
        "swe_rebench_eval": None,
        "swe_rebench_admission_manifest": None,
        "swe_gym_harness_root": None,
        "swe_gym_admission_manifest": None,
        "allow_mutable_images": True,
        "allow_unadmitted_r2e_dry_run": True,
        "allow_unadmitted_swe_rebench_dry_run": True,
        "allow_unadmitted_swe_gym_dry_run": False,
        "limit": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_manifest(path: Path, task) -> None:
    path.write_text(json.dumps(task.to_task_manifest()) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _fake_dependency(tmp_path: Path, name: str, content: str) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True)


def _assert_owner_only_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        mode = path.lstat().st_mode
        assert mode & 0o077 == 0, path
        assert stat.S_IMODE(mode) in (
            {0o500} if path.is_dir() else {0o400, 0o500}
        ), path


def _r2e_task():
    return normalize_swe_row(
        {
            "repo_name": "numpy",
            "docker_image": "docker.io/namanjain12/numpy_final:gold",
            "commit_hash": _GOLD_COMMIT,
            "base_commit": _BASE_COMMIT,
            "problem_statement": "Fix array conversion.",
            "patch": "diff --git a/numpy.py b/numpy.py\n",
            "expected_output_json": json.dumps({"test_array": "PASSED"}),
        },
        dataset_id="R2E-Gym/R2E-Gym-V1",
        usage="train",
    )


def _rebench_task():
    return normalize_swe_row(
        {
            "instance_id": "python-markdown__markdown-1529",
            "repo": "Python-Markdown/markdown",
            "base_commit": _BASE_COMMIT,
            "problem_statement": "Fix list parsing.",
            "patch": "diff --git a/markdown.py b/markdown.py\n",
            "test_patch": "diff --git a/tests/test_list.py b/tests/test_list.py\n",
            "FAIL_TO_PASS": ["tests/test_list.py::test_nested"],
            "PASS_TO_PASS": ["tests/test_core.py::test_basic"],
            "image_name": "docker.io/swerebenchv2/markdown:base",
            "install_config": {
                "test_cmd": ["pytest tests/unit -q", "pytest tests/integration -q"],
                "log_parser": "parse_log_pytest",
            },
        },
        dataset_id="nebius/SWE-rebench-V2",
        usage="train",
    )


def test_agent_runtime_policy_is_narrow_and_explicit() -> None:
    rebench = _rebench_task().to_task_manifest()
    rebench["source_image"] = rebench["sandbox"]["source_image"]
    rebench["source_metadata"]["language"] = "ts"
    rebench["verifier"]["install_config"]["install"] = ["npm ci --quiet"]
    assert materialize_module._agent_runtime_policy(rebench) == (
        "npm-node-modules-v2"
    )
    dockerfile = materialize_module._agent_dockerfile(rebench)
    assert "MILES_SWE_RUNTIME_POLICY=npm-node-modules-v2" in dockerfile

    unsupported = _rebench_task().to_task_manifest()
    assert materialize_module._agent_runtime_policy(unsupported) == "none"

    swe_gym = _swe_gym_task().to_task_manifest()
    swe_gym["source_image"] = swe_gym["sandbox"]["source_image"]
    assert materialize_module._agent_runtime_policy(swe_gym) == (
        "python-editable-metadata-v1"
    )
    dockerfile = materialize_module._agent_dockerfile(swe_gym)
    assert "MILES_SWE_RUNTIME_POLICY=python-editable-metadata-v1" in dockerfile

    preparation = (
        Path(__file__).parents[3]
        / "experiments/src/environments/swe/templates/prepare_agent.sh"
    ).read_text(encoding="utf-8")
    assert "published npm runtime tree exceeds" in preparation
    assert "published npm dist runtime exceeds" in preparation
    assert "npm-repo-runtime.inventory.sha256" in preparation
    assert "npm-repo-runtime-paths" in preparation
    assert "published Python editable metadata exceeds" in preparation
    assert "git -C \"${repo}\" clean -ffdx" in preparation

    playwright_sealer = (
        Path(__file__).parents[3]
        / "experiments/src/environments/swe/templates/"
        "seal_playwright_runtime.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "maximum_bytes=8589934592",
        "maximum_entries=200000",
        "source image contains ambiguous Playwright browser caches",
        "contains a symlink or special file",
        "hard link escapes its sealed tree",
        "playwright-runtime.inventory.sha256",
        'chmod -R a-w "${target}"',
    ):
        assert required in playwright_sealer

    admission = (
        Path(__file__).parents[3]
        / "experiments/src/environments/swe/semantic_admission.py"
    ).read_text(encoding="utf-8")
    assert "recomputed_npm_inventory" in admission
    assert "recomputed_playwright_inventory" in admission
    assert "npm ls --all --json" not in admission
    assert "JSON.parse(fs.readFileSync" in admission


def test_materialized_timeout_budget_fits_the_four_hour_pilot() -> None:
    manifest = _rebench_task().to_task_manifest()
    manifest["source_image"] = manifest["sandbox"]["source_image"]
    task_toml = tomllib.loads(
        materialize_module._task_toml(manifest)
    )

    assert task_toml["environment"]["build_timeout_sec"] == (
        timeouts.AGENT_ENVIRONMENT_START_TIMEOUT_SEC
    )
    assert task_toml["agent"]["timeout_sec"] == (
        timeouts.AGENT_EXECUTION_TIMEOUT_SEC
    )
    assert task_toml["verifier"]["collect"][0]["timeout_sec"] == (
        timeouts.COLLECT_TIMEOUT_SEC
    )
    assert task_toml["verifier"]["environment"]["build_timeout_sec"] == (
        timeouts.VERIFIER_ENVIRONMENT_START_TIMEOUT_SEC
    )
    assert task_toml["verifier"]["timeout_sec"] == (
        timeouts.VERIFIER_EXECUTION_TIMEOUT_SEC
    )
    assert timeouts.TRIAL_PHASE_BUDGET_SEC == 11_220
    assert (
        timeouts.TRIAL_PHASE_BUDGET_SEC
        < timeouts.TRIAL_WALL_TIMEOUT_SEC
        < timeouts.TRIAL_REQUEST_TIMEOUT_SEC
        < timeouts.FOUR_HOUR_JOB_TIMEOUT_SEC
    )

    repository = Path(__file__).parents[3]
    training = (
        repository
        / "experiments/scripts/swe/async/r2e-gym-swe-rebench-v2/"
        "qwen3-4b-instruct-2507/run.sbatch"
    ).read_text(encoding="utf-8")
    evaluation = (
        repository / "experiments/scripts/swe/eval/swebench-verified/run.sbatch"
    ).read_text(encoding="utf-8")
    assert (
        f'SWE_TRIAL_REQUEST_TIMEOUT_SEC="${{SWE_TRIAL_REQUEST_TIMEOUT_SEC:-'
        f'{timeouts.TRIAL_REQUEST_TIMEOUT_SEC}}}"'
    ) in training
    assert f"--request-timeout {timeouts.TRIAL_REQUEST_TIMEOUT_SEC}" in evaluation


def _swe_gym_task(*, image: str | None = None):
    row = {
        "instance_id": "pandas-dev__pandas-12345",
        "repo": "pandas-dev/pandas",
        "version": "2.0",
        "base_commit": _BASE_COMMIT,
        "problem_statement": "Fix dataframes.",
        "patch": ("diff --git a/pandas/core/frame.py b/pandas/core/frame.py\n--- a/pandas/core/frame.py\n+++ b/pandas/core/frame.py\n"),
        "test_patch": ("diff --git a/pandas/tests/test_frame.py b/pandas/tests/test_frame.py\n--- a/pandas/tests/test_frame.py\n+++ b/pandas/tests/test_frame.py\n"),
        "FAIL_TO_PASS": ["pandas/tests/test_frame.py::test_fix"],
        "PASS_TO_PASS": ["pandas/tests/test_frame.py::test_existing"],
    }
    if image is not None:
        row["docker_image"] = image
    return normalize_swe_row(
        row,
        dataset_id="SWE-Gym/SWE-Gym",
        usage="train",
    )


def _fake_swe_gym_harness(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "swe-gym-harness"
    harness = root / "swegym" / "harness"
    harness.mkdir(parents=True)
    files = {
        "constants.py": "class TestStatus: pass\n",
        "log_parsers.py": "MAP_REPO_TO_PARSER = {}\n",
        "grading.py": "def get_eval_tests_report(*args): return {}\n",
    }
    names = {
        "constants.py": "_SWE_GYM_CONSTANTS_SHA256",
        "log_parsers.py": "_SWE_GYM_LOG_PARSERS_SHA256",
        "grading.py": "_SWE_GYM_GRADING_SHA256",
    }
    for filename, content in files.items():
        path = harness / filename
        path.write_text(content, encoding="utf-8")
        monkeypatch.setattr(
            materialize_module,
            names[filename],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return root


def _locked_manifest(value: dict, *, requested: str, resolved: str) -> dict:
    locked = json.loads(json.dumps(value))
    sandbox = locked["sandbox"]
    sandbox["source_image"] = requested
    sandbox.pop("image_lock", None)
    input_payload = dict(locked)
    input_payload.pop("task_digest", None)
    input_payload.pop("content_digest", None)
    input_digest = materialize_module._stable_digest(input_payload)
    locked["content_digest"] = input_digest
    child_digest = resolved.rsplit("@", 1)[1]
    sandbox["source_image"] = resolved
    sandbox["image_lock"] = {
        "schema_version": "miles-oci-image-lock-v1",
        "source_image_requested": requested,
        "source_image_resolved": resolved,
        "input_content_digest": input_digest,
        "index_digest": None,
        "child_manifest_digest": child_digest,
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    locked_payload = dict(locked)
    locked_payload.pop("task_digest", None)
    locked_payload.pop("content_digest", None)
    locked["content_digest"] = materialize_module._stable_digest(locked_payload)
    return locked


def _template_roles() -> dict[str, dict[str, str]]:
    shared = {
        "template_id": "template-shared",
        "build_id": "build-shared",
        "alias_sha256": "4" * 64,
        "template_identity_sha256": "5" * 64,
    }
    agent = {
        "template_id": "template-agent",
        "build_id": "build-agent",
        "alias_sha256": "6" * 64,
        "template_identity_sha256": "7" * 64,
    }
    return {
        "source": {**shared, "sandbox_id": "sandbox-source"},
        "agent": {**agent, "sandbox_id": "sandbox-agent"},
        "empty_verifier": {**shared, "sandbox_id": "sandbox-empty"},
        "oracle_verifier": {**shared, "sandbox_id": "sandbox-oracle"},
    }


def _r2e_admission(manifest: dict, task_tree_sha256: str) -> dict:
    image_lock = manifest["sandbox"]["image_lock"]
    oracle_patch = manifest["solution"]["oracle_patch"]
    return {
        "schema_version": "miles-r2e-admission-v1",
        "instance_id": manifest["instance_id"],
        "task_digest": manifest["task_digest"],
        "input_content_digest": image_lock["input_content_digest"],
        "locked_content_digest": manifest["content_digest"],
        "content_digest": manifest["content_digest"],
        "source_image_requested": image_lock["source_image_requested"],
        "source_image_resolved": image_lock["source_image_resolved"],
        "source_image": image_lock["source_image_resolved"],
        "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
        "base_commit": manifest["base_commit"],
        "oracle_patch_sha256": hashlib.sha256(
            oracle_patch.encode("utf-8")
        ).hexdigest(),
        "admitted_task_tree_sha256": task_tree_sha256,
        "e2b_sandbox_evidence": _template_roles(),
        "checks": {
            "publisher_namespace_policy": True,
            "source_head_matches_base": True,
            "unique_parent_matches_base": True,
            "empty_reward": 0,
            "oracle_reward": 1,
            "runtime_smoke": True,
            "tool_smoke": True,
            "no_new_privileges": True,
            "effective_capabilities_zero": True,
            "suid_sgid_absent": True,
            "file_capabilities_absent": True,
            "fresh_separate_verifier": True,
            "source_network_denied": True,
            "agent_network_denied": True,
            "empty_verifier_network_denied": True,
            "oracle_verifier_network_denied": True,
            "late_verifier_tests_absent_before_upload": True,
            "gold_history_absent": True,
            "known_gold_artifacts_absent": True,
            "gold_blob_content_absent": True,
            "gold_commit_text_absent": True,
        },
    }


def _repository_admission(
    manifest: dict,
    task_dir: Path,
    *,
    swe_gym: bool,
) -> dict:
    image_lock = manifest["sandbox"]["image_lock"]
    tests_dir = task_dir / "tests"
    value = {
        "schema_version": (
            "miles-swe-gym-admission-v1"
            if swe_gym
            else "miles-swe-rebench-admission-v1"
        ),
        "instance_id": manifest["instance_id"],
        "source_schema": manifest["source_schema"],
        "task_digest": manifest["task_digest"],
        "input_content_digest": image_lock["input_content_digest"],
        "locked_content_digest": manifest["content_digest"],
        "content_digest": manifest["content_digest"],
        "source_image_requested": image_lock["source_image_requested"],
        "source_image_resolved": image_lock["source_image_resolved"],
        "source_image": image_lock["source_image_resolved"],
        "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
        "base_commit": manifest["base_commit"],
        "base_tree": "8" * 40,
        "oracle_patch_sha256": hashlib.sha256(
            manifest["solution"]["oracle_patch"].encode("utf-8")
        ).hexdigest(),
        "test_patch_sha256": hashlib.sha256(
            manifest["verifier"]["test_patch"].encode("utf-8")
        ).hexdigest(),
        "model_path_policy_sha256": hashlib.sha256(
            (tests_dir / "model_path_policy.json").read_bytes()
        ).hexdigest(),
        "admitted_task_tree_sha256": materialize_module._task_tree_sha256(
            task_dir
        ),
        "template_evidence": _template_roles(),
        "checks": dict(materialize_module._REPOSITORY_ADMISSION_CHECKS),
    }
    if swe_gym:
        value.update(
            {
                "dataset_revision": materialize_module._SWE_GYM_DATASET_REVISION,
                "harness_commit": materialize_module._SWE_GYM_HARNESS_COMMIT,
                "harness_version": materialize_module._SWE_GYM_HARNESS_VERSION,
                "constants_sha256": materialize_module._SWE_GYM_CONSTANTS_SHA256,
                "log_parsers_sha256": materialize_module._SWE_GYM_LOG_PARSERS_SHA256,
                "grading_sha256": materialize_module._SWE_GYM_GRADING_SHA256,
                "harbor_adapter_commit": materialize_module._SWE_GYM_HARBOR_COMMIT,
                "harbor_adapter_sha256": (
                    materialize_module._SWE_GYM_HARBOR_ADAPTER_SHA256
                ),
            }
        )
    else:
        value.update(
            {
                "rebench_commit": materialize_module._REBENCH_COMMIT,
                "log_parsers_sha256": materialize_module._REBENCH_LOG_PARSERS_SHA256,
                "constants_sha256": materialize_module._REBENCH_CONSTANTS_SHA256,
                "eval_sha256": materialize_module._REBENCH_EVAL_SHA256,
            }
        )
    return value


def test_r2e_materialization_uses_private_fresh_verifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _r2e_task()
    manifest = tmp_path / "tasks.private.jsonl"
    _write_manifest(manifest, task)
    parser, digest = _fake_dependency(
        tmp_path,
        "execution_log_parser.py",
        "def parse_log_pytest(log): return {}\ndef decolor_dict_keys(value): return value\n",
    )
    monkeypatch.setattr(materialize_module, "_R2E_PARSER_SHA256", digest)
    output = tmp_path / "tasks"

    summary = materialize_module.materialize(
        _arguments(
            manifest,
            output,
            r2e_execution_log_parser=parser,
        )
    )

    task_dir = output / task.instance_id
    config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    assert summary["separate_verifier"] is True
    assert config["metadata"]["task_digest"] == task.to_task_manifest()["task_digest"]
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["environment"]["network_mode"] == "no-network"
    assert config["environment"]["network_mode"] == "no-network"
    assert config["agent"]["user"] == 1000
    assert config["verifier"]["user"] == 0
    assert "docker_image" not in config["environment"]
    assert config["verifier"]["environment"]["docker_image"] == task.source_image
    assert config["artifacts"] == [
        {"source": "/logs/artifacts", "exclude": ["*"]},
        {
            "source": "/opt/miles-swe/collected/model.patch",
            "destination": "model.patch",
        },
    ]
    collect = config["verifier"]["collect"][0]
    assert collect["required"] is True
    assert collect["user"] == 0
    assert "timeout --signal=TERM --kill-after=5s 100s" in collect["command"]
    agent_dockerfile = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "install -d -o root -g root -m 0700 /opt/miles-swe/collected" in (agent_dockerfile)
    assert "install -d -o root -g root -m 0700 /opt/miles-swe/root-home" in (agent_dockerfile)
    assert "MILES_SWE_AGENT_STATE_INVALID" in agent_dockerfile
    assert f"MILES_SWE_GOLD_COMMIT={_GOLD_COMMIT}" in agent_dockerfile
    collector = (task_dir / "environment" / "collect_agent_patch.sh").read_text(encoding="utf-8")
    assert collector.index("kill -STOP") < collector.index("safe_git add -A")
    assert '--git-dir="${gitdir}" --work-tree="${repo}"' in collector
    assert "GIT_CONFIG_NOSYSTEM=1" in collector
    assert "GIT_CONFIG_GLOBAL=/dev/null" in collector
    assert "GIT_CONFIG_COUNT=0" in collector
    assert "core.fsmonitor=false" in collector
    assert "core.hooksPath=/dev/null" in collector
    assert "--no-ext-diff --no-textconv" in collector
    agent_prep = (task_dir / "environment" / "prepare_agent.sh").read_text(encoding="utf-8")
    assert 'rm -rf -- "${repo}/.git"' in agent_prep
    assert "--no-reflogs --unreachable --no-progress" in agent_prep
    assert "pack-objects --stdout --revs" in agent_prep
    assert 'mv -- "${fresh_git}" /opt/miles-swe/agent-git' in agent_prep
    assert "printf '%s\\n' /opt/miles-swe/agent-git >/opt/miles-swe/gitdir" in (agent_prep)
    tests_dir = task_dir / "tests"
    assert not (tests_dir / "Dockerfile").exists()
    assert (tests_dir / ".harbor-e2b-late-tests").is_file()
    assert (tests_dir / "r2e_execution_log_parser.py").is_file()
    assert (tests_dir / "gold_commit.txt").read_text(encoding="utf-8") == (
        _GOLD_COMMIT + "\n"
    )
    test_script = (tests_dir / "test.sh").read_text(encoding="utf-8")
    assert "prepare_r2e_verifier.sh" in test_script
    assert 'MILES_SWE_GOLD_COMMIT="${gold_commit}"' in test_script
    assert "strip_agent_privileges.py" in test_script
    assert "/usr/bin/setpriv" in test_script
    assert "--no-new-privs" in test_script
    assert "/opt/miles-swe/collected/model.patch" in test_script
    _assert_owner_only_tree(output / task.instance_id)
    assert "expected_output" not in agent_dockerfile
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    assert _GOLD_COMMIT not in instruction
    assert "test_array" not in instruction
    assert task_dir.stat().st_mode & 0o077 == 0


def test_r2e_agent_hardening_binds_parent_and_attests_visible_rootfs() -> None:
    script = (materialize_module._ASSET_DIR / "prepare_agent.sh").read_text(
        encoding="utf-8"
    )
    for required in (
        'gold_commit="${MILES_SWE_GOLD_COMMIT:-}"',
        'rev-parse "${gold_commit}:${relative}"',
        '[[ "$(git -C "${repo}" rev-parse HEAD)" == "${base_commit}" ]]',
        'rm -rf -- /r2e_tests "${repo}/r2e_tests"',
        'rm -rf -- "${repo}/.git"',
        "miles-r2e-visible-rootfs-attestation-v1",
        "r2e-runtime-inventory.sha256",
        "R2E editable runtime overlay exceeds",
        "R2E runtime hard link escapes its sealed /opt tree",
        "find /tmp -xdev -mindepth 1 -delete",
        "R2E agent rootfs retains exact gold-file content",
        "R2E agent rootfs retains the gold commit text",
    ):
        assert required in script


def test_collector_ignores_agent_git_config_and_external_attributes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.email", "test@miles.invalid", cwd=repo)
    _run("git", "config", "user.name", "Miles Test", cwd=repo)
    (repo / "sample.txt").write_text("base\n", encoding="utf-8")
    _run("git", "add", "sample.txt", cwd=repo)
    _run("git", "commit", "-qm", "base", cwd=repo)
    trusted_gitdir = tmp_path / "trusted-agent-git"
    (repo / ".git").rename(trusted_gitdir)
    trusted_gitdir.chmod(0o755)

    marker = tmp_path / "external-driver-ran"
    driver = tmp_path / "evil-driver.sh"
    driver.write_text(
        '#!/bin/sh\nprintf hit >"$MILES_TEST_MARKER"\ncat\n',
        encoding="utf-8",
    )
    driver.chmod(0o700)
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    (agent_home / ".gitconfig").write_text(
        f'[core]\n\tfsmonitor = {driver}\n[diff "agent"]\n\tcommand = {driver}\n[filter "agent"]\n\tclean = {driver}\n\trequired = true\n',
        encoding="utf-8",
    )
    # The worktree-local .git is fully agent-controlled. The collector must
    # ignore both this config and any hook it names.
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text(
        f'[core]\n\tfsmonitor = {driver}\n[diff "agent"]\n\tcommand = {driver}\n[filter "agent"]\n\tclean = {driver}\n\trequired = true\n',
        encoding="utf-8",
    )
    (repo / ".gitattributes").write_text(
        "*.txt diff=agent filter=agent\n",
        encoding="utf-8",
    )
    (repo / "sample.txt").write_text("changed\n", encoding="utf-8")

    collected = tmp_path / "collected"
    collected.mkdir()
    target = collected / "model.patch"
    target.write_text("MILES_SWE_AGENT_STATE_INVALID\n", encoding="utf-8")
    root_home = tmp_path / "root-home"
    (root_home / "xdg").mkdir(parents=True)
    root_home.chmod(0o700)
    workdir = tmp_path / "workdir"
    workdir.write_text(f"{repo}\n", encoding="utf-8")
    gitdir = tmp_path / "gitdir"
    gitdir.write_text(f"{trusted_gitdir}\n", encoding="utf-8")
    production = materialize_module._ASSET_DIR / "collect_agent_patch.sh"
    collector = tmp_path / "collect_agent_patch.sh"
    script = production.read_text(encoding="utf-8")
    script = script.replace("/opt/miles-swe/root-home", str(root_home))
    script = script.replace(
        "/opt/miles-swe/collected/.model.patch.XXXXXX",
        str(collected / ".model.patch.XXXXXX"),
    )
    script = script.replace(
        "/opt/miles-swe/collected/model.patch",
        str(target),
    )
    script = script.replace("/opt/miles-swe/workdir", str(workdir))
    script = script.replace("/opt/miles-swe/gitdir", str(gitdir))
    script = script.replace("== 0:700", f"== {os.getuid()}:700")
    script = script.replace("== 0 ]]", f"== {os.getuid()} ]]")
    collector.write_text(script, encoding="utf-8")
    collector.chmod(0o700)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(agent_home),
            "XDG_CONFIG_HOME": str(agent_home),
            "MILES_TEST_MARKER": str(marker),
        }
    )

    _run("bash", str(collector), cwd=repo, env=env)

    assert not marker.exists()
    patch = target.read_text(encoding="utf-8")
    assert "sample.txt" in patch
    assert "changed" in patch


def test_source_git_config_is_removed_before_root_checkout() -> None:
    for filename in (
        "prepare_agent.sh",
        "prepare_verifier_user.sh",
        "prepare_r2e_verifier.sh",
    ):
        script = (materialize_module._ASSET_DIR / filename).read_text(
            encoding="utf-8"
        )
        sanitize = script.index(
            'install -o root -g root -m 0600 /dev/null "${source_gitdir}/config"'
        )
        checkout = script.index('git -C "${repo}" checkout')
        assert sanitize < checkout
        assert 'rm -f -- "${source_gitdir}/config.worktree"' in script
        assert '"${source_gitdir}/info/attributes"' in script
        assert "GIT_CONFIG_NOSYSTEM=1" in script
        assert "GIT_CONFIG_GLOBAL=/dev/null" in script
        assert "core.fsmonitor" in script
        assert "core.hooksPath" in script


def test_fresh_git_repository_cannot_read_source_or_alternate_objects(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _run("git", "init", "-q", cwd=source)
    _run("git", "config", "user.email", "test@miles.invalid", cwd=source)
    _run("git", "config", "user.name", "Miles Test", cwd=source)
    (source / "implementation.py").write_text("base = True\n", encoding="utf-8")
    _run("git", "add", "implementation.py", cwd=source)
    _run("git", "commit", "-qm", "base", cwd=source)
    base_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (source / "implementation.py").write_text("gold = True\n", encoding="utf-8")
    _run("git", "commit", "-qam", "gold", cwd=source)
    gold_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _run("git", "checkout", "-q", base_commit, cwd=source)
    old_git = tmp_path / "source-git"
    (source / ".git").rename(old_git)

    _run("git", "init", "-q", "--template=", cwd=source)
    _run("git", "config", "user.email", "test@miles.invalid", cwd=source)
    _run("git", "config", "user.name", "Miles Test", cwd=source)
    _run("git", "add", "-A", cwd=source)
    _run("git", "commit", "-qm", "synthetic base", cwd=source)

    for old_commit in (base_commit, gold_commit):
        result = subprocess.run(
            ("git", "cat-file", "-e", f"{old_commit}^{{commit}}"),
            cwd=source,
            check=False,
            capture_output=True,
        )
        assert result.returncode != 0
    fsck = subprocess.run(
        ("git", "fsck", "--no-reflogs", "--unreachable", "--no-progress"),
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    assert fsck.stdout == ""
    assert fsck.stderr == ""


def test_collect_timeout_keeps_invalid_sentinel_authoritative(tmp_path: Path) -> None:
    task = _r2e_task()
    manifest = task.to_task_manifest()
    manifest["source_image"] = task.source_image
    command = materialize_module._task_toml(manifest)

    assert "124|137" in command
    assert "grep -qx 'MILES_SWE_AGENT_STATE_INVALID'" in command
    assert command.index("grep -qx 'MILES_SWE_AGENT_STATE_INVALID'") < command.index("exit 0\n        ;;")


def test_rebench_materialization_preserves_list_commands_and_parser_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _rebench_task()
    manifest = tmp_path / "tasks.private.jsonl"
    _write_manifest(manifest, task)
    parser, parser_digest = _fake_dependency(
        tmp_path,
        "log_parsers.py",
        "NAME_TO_PARSER = {}\ndef parse_log_pytest(log): return {}\n",
    )
    constants, constants_digest = _fake_dependency(
        tmp_path,
        "swe_constants.py",
        "class TestStatus: pass\n",
    )
    official_eval, eval_digest = _fake_dependency(
        tmp_path,
        "eval.py",
        "def evaluate_instance(): pass\n",
    )
    monkeypatch.setattr(
        materialize_module,
        "_REBENCH_LOG_PARSERS_SHA256",
        parser_digest,
    )
    monkeypatch.setattr(
        materialize_module,
        "_REBENCH_CONSTANTS_SHA256",
        constants_digest,
    )
    monkeypatch.setattr(materialize_module, "_REBENCH_EVAL_SHA256", eval_digest)
    output = tmp_path / "tasks"

    materialize_module.materialize(
        _arguments(
            manifest,
            output,
            swe_rebench_log_parsers=parser,
            swe_rebench_constants=constants,
            swe_rebench_eval=official_eval,
        )
    )

    tests_dir = output / task.instance_id / "tests"
    verifier = json.loads((tests_dir / "verifier_config.json").read_text(encoding="utf-8"))
    assert verifier["test_commands"] == [
        "pytest tests/unit -q",
        "pytest tests/integration -q",
    ]
    assert (tests_dir / "swe_rebench_run.py").is_file()
    assert (tests_dir / "seal_playwright_runtime.sh").is_file()
    assert (tests_dir / "lib" / "agent" / "log_parsers.py").is_file()
    assert (tests_dir / "lib" / "agent" / "swe_constants.py").is_file()
    assert (tests_dir / "official_eval.py").read_bytes() == official_eval.read_bytes()
    assert verifier["eval_sha256"] == eval_digest
    assert not (tests_dir / "Dockerfile").exists()
    assert (tests_dir / ".harbor-e2b-late-tests").is_file()
    test_script = (tests_dir / "test.sh").read_text(encoding="utf-8")
    assert "prepare_verifier_user.sh" in test_script
    assert "strip_agent_privileges.py" in test_script
    assert "/usr/bin/setpriv" in test_script
    assert "--no-new-privs" in test_script
    assert "/opt/miles-swe/collected/model.patch" in test_script
    agent_prep = (
        output / task.instance_id / "environment" / "prepare_agent.sh"
    ).read_text(encoding="utf-8")
    assert (
        output
        / task.instance_id
        / "environment"
        / "seal_playwright_runtime.sh"
    ).is_file()
    assert 'git -C "${repo}" clean -ffdx' in agent_prep
    assert "contains forbidden untracked content" in agent_prep
    assert "runtime_excludes" not in agent_prep


def test_materializer_rejects_mutable_source_image_by_default(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.private.jsonl"
    _write_manifest(manifest, _r2e_task())

    with pytest.raises(ValueError, match="mutable"):
        materialize_module.materialize(
            _arguments(
                manifest,
                tmp_path / "tasks",
                allow_mutable_images=False,
                allow_unadmitted_r2e_dry_run=True,
            )
        )


def test_materializer_rejects_unadmitted_r2e_in_production(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.private.jsonl"
    _write_manifest(manifest, _r2e_task())

    with pytest.raises(ValueError, match="not admitted for production"):
        materialize_module.materialize(
            _arguments(
                manifest,
                tmp_path / "tasks",
                allow_unadmitted_r2e_dry_run=False,
            )
        )


def test_materializer_accepts_only_exact_live_r2e_admission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = "docker.io/namanjain12/numpy_final@sha256:" + "a" * 64
    task = replace(_r2e_task(), source_image=image)
    private_manifest = task.to_task_manifest()
    private_manifest["solution"]["oracle_patch"] += "\n"
    requested_image = "docker.io/namanjain12/numpy_final:gold"
    private_manifest = _locked_manifest(
        private_manifest,
        requested=requested_image,
        resolved=image,
    )
    manifest = tmp_path / "tasks.private.jsonl"
    manifest.write_text(json.dumps(private_manifest) + "\n", encoding="utf-8")
    manifest.chmod(0o600)
    parser, digest = _fake_dependency(
        tmp_path,
        "execution_log_parser.py",
        "def parse_log_pytest(log): return {}\ndef decolor_dict_keys(value): return value\n",
    )
    monkeypatch.setattr(materialize_module, "_R2E_PARSER_SHA256", digest)
    dry_output = tmp_path / "dry-tasks"
    materialize_module.materialize(
        _arguments(
            manifest,
            dry_output,
            r2e_execution_log_parser=parser,
            allow_mutable_images=False,
            allow_unadmitted_r2e_dry_run=True,
        )
    )
    admitted_tree = materialize_module._task_tree_sha256(
        dry_output / private_manifest["instance_id"]
    )
    admission_record = _r2e_admission(private_manifest, admitted_tree)
    admission = tmp_path / "r2e-admissions.private.jsonl"
    admission.write_text(json.dumps(admission_record) + "\n", encoding="utf-8")
    admission.chmod(0o600)

    evidence = tmp_path / "evidence" / "materialization.private.jsonl"
    summary = materialize_module.materialize(
        _arguments(
            manifest,
            tmp_path / "tasks",
            admission_evidence=evidence,
            r2e_execution_log_parser=parser,
            r2e_admission_manifest=admission,
            allow_mutable_images=False,
            allow_unadmitted_r2e_dry_run=False,
            allow_unadmitted_swe_rebench_dry_run=False,
        )
    )

    assert summary["tasks"] == 1
    assert summary["admission_evidence_records"] == 1
    evidence_record = json.loads(evidence.read_text(encoding="utf-8"))
    assert evidence_record["schema_version"] == (
        "miles-swe-materialization-evidence-v1"
    )
    assert evidence_record["instance_id"] == private_manifest["instance_id"]
    assert evidence_record["task_digest"] == private_manifest["task_digest"]
    assert evidence_record["content_digest"] == private_manifest["content_digest"]
    assert evidence_record["source_image"] == image
    assert evidence_record["private_manifest_record_sha256"] == (
        materialize_module._stable_digest(private_manifest)
    )
    assert evidence_record["semantic_admission_record_sha256"] == (
        materialize_module._stable_digest(admission_record)
    )
    task_dir = tmp_path / "tasks" / private_manifest["instance_id"]
    assert evidence_record["task_tree_sha256"] == (
        materialize_module._task_tree_sha256(task_dir)
    )
    assert all(evidence_record["checks"].values())
    assert evidence.stat().st_mode & 0o077 == 0


def test_materialization_evidence_rejects_dry_run_flags(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.private.jsonl"
    _write_manifest(manifest, _r2e_task())

    with pytest.raises(ValueError, match="dry-run flags"):
        materialize_module.materialize(
            _arguments(
                manifest,
                tmp_path / "tasks",
                admission_evidence=tmp_path / "evidence.private.jsonl",
            )
        )


def test_materializer_rejects_symlink_output_root(tmp_path: Path) -> None:
    manifest = tmp_path / "tasks.private.jsonl"
    _write_manifest(manifest, _r2e_task())
    real_output = tmp_path / "real-tasks"
    real_output.mkdir()
    output = tmp_path / "tasks"
    output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(PermissionError, match="output directory is unsafe"):
        materialize_module.materialize(_arguments(manifest, output))


def test_task_tree_digest_rejects_symlinks_and_hardlinks(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir(mode=0o700)
    source = task_dir / "source"
    source.write_text("trusted\n", encoding="utf-8")
    source.chmod(0o600)
    link = task_dir / "link"
    link.symlink_to(source.name)
    task_dir.chmod(0o500)
    source.chmod(0o400)

    with pytest.raises(ValueError, match="symlink, special file, or hardlink"):
        materialize_module._task_tree_sha256(task_dir)

    task_dir.chmod(0o700)
    link.unlink()
    os.link(source, task_dir / "hardlink")
    task_dir.chmod(0o500)
    with pytest.raises(ValueError, match="symlink, special file, or hardlink"):
        materialize_module._task_tree_sha256(task_dir)


def test_materialization_evidence_requires_live_semantic_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "docker.io/swerebenchv2/markdown@sha256:" + "d" * 64
    task = replace(_rebench_task(), source_image=image)
    manifest = tmp_path / "tasks.private.jsonl"
    _write_manifest(manifest, task)
    parser, parser_digest = _fake_dependency(
        tmp_path,
        "log_parsers.py",
        "NAME_TO_PARSER = {}\ndef parse_log_pytest(log): return {}\n",
    )
    constants, constants_digest = _fake_dependency(
        tmp_path,
        "swe_constants.py",
        "class TestStatus: pass\n",
    )
    official_eval, eval_digest = _fake_dependency(
        tmp_path,
        "eval.py",
        "def evaluate_instance(): pass\n",
    )
    monkeypatch.setattr(
        materialize_module,
        "_REBENCH_LOG_PARSERS_SHA256",
        parser_digest,
    )
    monkeypatch.setattr(
        materialize_module,
        "_REBENCH_CONSTANTS_SHA256",
        constants_digest,
    )
    monkeypatch.setattr(materialize_module, "_REBENCH_EVAL_SHA256", eval_digest)

    with pytest.raises(ValueError, match="not admitted for production"):
        materialize_module.materialize(
            _arguments(
                manifest,
                tmp_path / "tasks",
                admission_evidence=tmp_path / "evidence.private.jsonl",
                swe_rebench_log_parsers=parser,
                swe_rebench_constants=constants,
                swe_rebench_eval=official_eval,
                allow_mutable_images=False,
                allow_unadmitted_r2e_dry_run=False,
                allow_unadmitted_swe_rebench_dry_run=False,
            )
        )
    assert not (tmp_path / "tasks" / task.instance_id).exists()


def test_materializer_rejects_unadmitted_swe_gym(
    tmp_path: Path,
) -> None:
    task = _swe_gym_task()
    manifest = tmp_path / "tasks.private.jsonl"
    _write_manifest(manifest, task)

    with pytest.raises(ValueError, match="not admitted for production"):
        materialize_module.materialize(_arguments(manifest, tmp_path / "tasks"))


def test_swe_gym_materialization_uses_pinned_separate_exact_verifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = "docker.io/xingyaoww/sweb.eval.x86_64.task@sha256:" + "a" * 64
    task = _swe_gym_task(image=image)
    private = _locked_manifest(
        task.to_task_manifest(),
        requested="docker.io/xingyaoww/sweb.eval.x86_64.task:latest",
        resolved=image,
    )
    manifest = tmp_path / "tasks.private.jsonl"
    manifest.write_text(json.dumps(private) + "\n", encoding="utf-8")
    manifest.chmod(0o600)
    harness = _fake_swe_gym_harness(tmp_path, monkeypatch)
    dry_output = tmp_path / "dry-tasks"
    materialize_module.materialize(
        _arguments(
            manifest,
            dry_output,
            swe_gym_harness_root=harness,
            allow_mutable_images=False,
            allow_unadmitted_swe_gym_dry_run=True,
        )
    )
    admission_record = _repository_admission(
        private,
        dry_output / task.instance_id,
        swe_gym=True,
    )
    admission = tmp_path / "swe-gym-admission.private.jsonl"
    admission.write_text(json.dumps(admission_record) + "\n", encoding="utf-8")
    admission.chmod(0o600)
    output = tmp_path / "tasks"

    summary = materialize_module.materialize(
        _arguments(
            manifest,
            output,
            swe_gym_harness_root=harness,
            swe_gym_admission_manifest=admission,
            allow_mutable_images=False,
            allow_unadmitted_swe_gym_dry_run=False,
        )
    )

    task_dir = output / task.instance_id
    config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    assert summary["schemas"] == {"swe-gym": 1}
    assert config["verifier"]["environment_mode"] == "separate"
    assert "docker_image" not in config["environment"]
    assert config["verifier"]["environment"]["docker_image"] == image
    agent_from = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8").splitlines()[0]
    assert agent_from == f"FROM {image}"
    tests_dir = task_dir / "tests"
    assert not (tests_dir / "Dockerfile").exists()
    assert (tests_dir / ".harbor-e2b-late-tests").is_file()
    verifier = json.loads((tests_dir / "verifier_config.json").read_text())
    assert verifier["harness_commit"] == materialize_module._SWE_GYM_HARNESS_COMMIT
    assert verifier["harbor_adapter_commit"] == materialize_module._SWE_GYM_HARBOR_COMMIT
    assert (tests_dir / "lib" / "swegym" / "harness" / "grading.py").is_file()
    assert (tests_dir / "lib" / "swegym" / "harness" / "log_parsers.py").is_file()
    assert "get_eval_tests_report" in (tests_dir / "swe_gym_grader.py").read_text()
    assert "MAP_REPO_VERSION_TO_SPECS" in (tests_dir / "swe_gym_run.py").read_text()
    policy = json.loads((tests_dir / "model_path_policy.json").read_text())
    assert policy["allowed_paths"] == ["pandas/core/frame.py"]
    assert "pandas/tests/test_frame.py" not in json.dumps(policy)
    assert "gold" not in (task_dir / "instruction.md").read_text()
    agent_prep = (task_dir / "environment" / "prepare_agent.sh").read_text(
        encoding="utf-8"
    )
    assert 'git -C "${repo}" clean -ffdx' in agent_prep
    assert "contains forbidden untracked content" in agent_prep
    assert "runtime_excludes" not in agent_prep
    _assert_owner_only_tree(task_dir)


def test_swe_gym_admission_requires_empty_zero_and_oracle_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad-admission.private.jsonl"
    checks = dict(materialize_module._REPOSITORY_ADMISSION_CHECKS)
    checks["empty_reward"] = 1
    value = {
        "schema_version": "miles-swe-gym-admission-v1",
        "source_schema": "swe-gym",
        "dataset_revision": materialize_module._SWE_GYM_DATASET_REVISION,
        "harness_commit": materialize_module._SWE_GYM_HARNESS_COMMIT,
        "harness_version": materialize_module._SWE_GYM_HARNESS_VERSION,
        "constants_sha256": materialize_module._SWE_GYM_CONSTANTS_SHA256,
        "log_parsers_sha256": materialize_module._SWE_GYM_LOG_PARSERS_SHA256,
        "grading_sha256": materialize_module._SWE_GYM_GRADING_SHA256,
        "harbor_adapter_commit": materialize_module._SWE_GYM_HARBOR_COMMIT,
        "harbor_adapter_sha256": materialize_module._SWE_GYM_HARBOR_ADAPTER_SHA256,
        "instance_id": "pandas-dev__pandas-12345",
        "task_digest": "a" * 64,
        "input_content_digest": "b" * 64,
        "locked_content_digest": "c" * 64,
        "content_digest": "c" * 64,
        "source_image_requested": "example.invalid/task:latest",
        "source_image_resolved": "example.invalid/task@sha256:" + "d" * 64,
        "source_image": "example.invalid/task@sha256:" + "d" * 64,
        "image_publisher_policy": oci_image_lock.IMAGE_PUBLISHER_POLICY,
        "base_commit": "e" * 40,
        "base_tree": "f" * 40,
        "oracle_patch_sha256": "1" * 64,
        "test_patch_sha256": "2" * 64,
        "model_path_policy_sha256": "3" * 64,
        "admitted_task_tree_sha256": "4" * 64,
        "template_evidence": _template_roles(),
        "checks": checks,
    }
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="live checks are invalid"):
        materialize_module._load_swe_gym_admissions(path)


@pytest.mark.parametrize(
    "oracle_patch",
    [
        "diff --git a/tests/test_agent.py b/tests/test_agent.py\n",
        "diff --git a/conftest.py b/conftest.py\n",
        "diff --git a/pyproject.toml b/pyproject.toml\n",
        "diff --git a/src/a.py b/src/b.py\nrename from src/a.py\nrename to src/b.py\n",
        "diff --git a/src/a.bin b/src/a.bin\nGIT binary patch\n",
        "diff --git a/src/link b/src/link\nnew file mode 120000\n",
    ],
)
def test_model_path_policy_rejects_test_config_rename_binary_and_symlink(
    oracle_patch: str,
) -> None:
    manifest = _r2e_task().to_task_manifest()
    manifest["oracle_patch"] = oracle_patch

    with pytest.raises(ValueError):
        materialize_module._model_path_policy(manifest)
