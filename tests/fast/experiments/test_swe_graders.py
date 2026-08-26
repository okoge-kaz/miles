"""Adversarial tests for private SWE grader result handling."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_ASSETS = Path(__file__).parents[3] / "experiments" / "src" / "environments" / "swe" / "templates"
_HIDDEN_TEST = "private/test_module.py::test_oracle_behavior"


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ASSETS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _redirect_paths(monkeypatch: pytest.MonkeyPatch, module: ModuleType, root: Path) -> None:
    real_path = Path
    mapping = {
        "/tests/verifier_config.json": root / "verifier_config.json",
        "/logs/verifier/test-output.log": root / "test-output.log",
        "/logs/verifier/report.json": root / "report.json",
        "/logs/verifier/reward.txt": root / "reward.txt",
    }
    monkeypatch.setattr(module, "Path", lambda raw: mapping.get(str(raw), real_path(raw)))


def _write_inputs(root: Path, config: dict[str, object]) -> None:
    (root / "verifier_config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "test-output.log").write_text("model-controlled output\n", encoding="utf-8")


def _assert_private_result_is_reward_zero(root: Path) -> None:
    assert (root / "reward.txt").read_text(encoding="utf-8") == "0\n"
    rendered = (root / "report.json").read_text(encoding="utf-8")
    assert _HIDDEN_TEST not in rendered
    report = json.loads(rendered)
    assert report["parser_error"] is True
    assert report["reward"] == 0
    assert report["resolved"] is False


def test_swe_gym_pristine_hidden_patch_failure_is_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load("test_swe_gym_prepare_pristine", "swe_gym_prepare.py")
    responses = iter((0, 1))
    monkeypatch.setattr(
        module,
        "_git",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=next(responses)),
    )
    model_patch = tmp_path / "model.patch"
    model_patch.write_bytes(b"")

    with pytest.raises(module.InfrastructureError, match="pristine base"):
        module.prepare(
            tmp_path,
            model_patch,
            tmp_path / "hidden.patch",
            "a" * 40,
        )


def test_swe_gym_post_model_hidden_conflict_is_reward_zero_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load("test_swe_gym_prepare_model_conflict", "swe_gym_prepare.py")
    responses = iter((0, 0, 0, 0, 1))
    monkeypatch.setattr(
        module,
        "_git",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=next(responses)),
    )
    model_patch = tmp_path / "model.patch"
    model_patch.write_text("model diff\n", encoding="utf-8")

    with pytest.raises(module.RejectedPatch, match="conflicts"):
        module.prepare(
            tmp_path,
            model_patch,
            tmp_path / "hidden.patch",
            "a" * 40,
        )


@pytest.mark.parametrize(
    "filename",
    ["r2e_test.sh", "swe_rebench_test.sh", "swe_gym_test.sh"],
)
def test_untrusted_verifier_runner_has_bounded_resources_and_output(
    filename: str,
) -> None:
    script = (_ASSETS / filename).read_text(encoding="utf-8")

    assert "ulimit -c 0" in script
    assert "ulimit -f 65536" in script
    assert "ulimit -n 4096" in script
    assert "ulimit -u 1024" in script
    assert "model_test_output_limit_exceeded" in script
    assert "test_output_size" in script
    assert "cat /logs/verifier/test-output.log" not in script


def test_file_size_limit_stops_adversarial_output_writer(tmp_path: Path) -> None:
    output = tmp_path / "untrusted-output.log"
    with output.open("wb") as stream:
        completed = subprocess.run(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                'ulimit -c 0; ulimit -f 64; exec python3 -c \'import os; data=b"x"*4096; exec("while True:\\n os.write(1, data)")\'',
            ],
            stdout=stream,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    assert completed.returncode != 0
    assert output.stat().st_size <= 65_536


def test_r2e_parser_exception_is_model_reward_zero_without_test_id_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_module = ModuleType("r2e_execution_log_parser")

    def raise_from_untrusted_log(_log: str):
        raise RuntimeError("model-triggered parser failure")

    parser_module.parse_log_pytest = raise_from_untrusted_log
    parser_module.decolor_dict_keys = lambda value: value
    monkeypatch.setitem(sys.modules, parser_module.__name__, parser_module)
    grader = _load("test_r2e_grader_adversarial", "r2e_grader.py")
    _redirect_paths(monkeypatch, grader, tmp_path)
    _write_inputs(tmp_path, {"expected_output": {_HIDDEN_TEST: "PASSED"}})

    grader.main()

    _assert_private_result_is_reward_zero(tmp_path)


def _install_rebench_modules(monkeypatch: pytest.MonkeyPatch, parser) -> None:
    package = ModuleType("lib")
    package.__path__ = []
    agent = ModuleType("lib.agent")
    agent.__path__ = []
    log_parsers = ModuleType("lib.agent.log_parsers")
    log_parsers.NAME_TO_PARSER = {"parse_log_pytest": parser}

    class TestStatus(str, Enum):
        PASSED = "PASSED"
        FAILED = "FAILED"
        SKIPPED = "SKIPPED"
        ERROR = "ERROR"

    constants = ModuleType("lib.agent.swe_constants")
    constants.TestStatus = TestStatus
    agent.log_parsers = log_parsers
    agent.swe_constants = constants
    for module in (package, agent, log_parsers, constants):
        monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize(
    "parser",
    [
        lambda _log: (_ for _ in ()).throw(RuntimeError("parser failed")),
        lambda _log: {_HIDDEN_TEST: "NOT_A_STATUS"},
        lambda _log: [_HIDDEN_TEST],
    ],
)
def test_rebench_invalid_parser_output_is_model_reward_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parser,
) -> None:
    _install_rebench_modules(monkeypatch, parser)
    grader = _load("test_rebench_grader_adversarial", "swe_rebench_grader.py")
    _redirect_paths(monkeypatch, grader, tmp_path)
    _write_inputs(
        tmp_path,
        {
            "log_parser": "parse_log_pytest",
            "fail_to_pass": [_HIDDEN_TEST],
            "pass_to_pass": [],
        },
    )

    grader.main()

    _assert_private_result_is_reward_zero(tmp_path)


@pytest.mark.parametrize(
    ("parsed", "fail_to_pass", "pass_to_pass", "expected_reward"),
    [
        (
            {"tests/test_fix.py::test_fix": "PASSED"},
            ["tests/test_fix.py::test_fix"],
            [],
            1,
        ),
        (
            {
                "tests/test_fix.py::test_fix": "PASSED",
                "tests/test_extra.py::test_extra": "PASSED",
            },
            ["tests/test_fix.py::test_fix"],
            [],
            0,
        ),
        (
            {"tests/test_fix.py::test_fix": "FAILED"},
            ["tests/test_fix.py::test_fix"],
            [],
            0,
        ),
        (
            {"tests/test_fix.py::test_fix [1.25 ms]": "PASSED"},
            ["tests/test_fix.py::test_fix [99 ms]"],
            [],
            1,
        ),
    ],
)
def test_rebench_grader_matches_pinned_eval_passed_match_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parsed: dict[str, str],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    expected_reward: int,
) -> None:
    _install_rebench_modules(monkeypatch, lambda _log: parsed)
    grader = _load("test_rebench_pinned_eval_parity", "swe_rebench_grader.py")
    _redirect_paths(monkeypatch, grader, tmp_path)
    _write_inputs(
        tmp_path,
        {
            "log_parser": "parse_log_pytest",
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
        },
    )

    grader.main()

    assert (tmp_path / "reward.txt").read_text(encoding="utf-8") == (
        f"{expected_reward}\n"
    )


def _install_swe_gym_modules(monkeypatch: pytest.MonkeyPatch, parser) -> None:
    package = ModuleType("swegym")
    package.__path__ = []
    harness = ModuleType("swegym.harness")
    harness.__path__ = []

    class ResolvedStatus(Enum):
        NO = "RESOLVED_NO"
        FULL = "RESOLVED_FULL"

    class TestStatus(Enum):
        PASSED = "PASSED"
        FAILED = "FAILED"
        SKIPPED = "SKIPPED"
        ERROR = "ERROR"
        XFAIL = "XFAIL"

    constants = ModuleType("swegym.harness.constants")
    constants.FAIL_TO_PASS = "FAIL_TO_PASS"
    constants.PASS_TO_PASS = "PASS_TO_PASS"
    constants.ResolvedStatus = ResolvedStatus
    constants.TestStatus = TestStatus
    grading = ModuleType("swegym.harness.grading")
    grading.get_eval_tests_report = lambda _actual, _gold: {}
    grading.get_resolution_status = lambda _report: ResolvedStatus.NO.value
    log_parsers = ModuleType("swegym.harness.log_parsers")
    log_parsers.MAP_REPO_TO_PARSER = {"owner/repo": parser}
    for module in (package, harness, constants, grading, log_parsers):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def test_swe_gym_public_baseline_uses_pinned_command_without_hidden_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = ModuleType("swegym")
    package.__path__ = []
    harness = ModuleType("swegym.harness")
    harness.__path__ = []
    constants = ModuleType("swegym.harness.constants")
    constants.MAP_REPO_VERSION_TO_SPECS = {
        "owner/repo": {
            "1.0": {
                "eval_commands": ["export BASELINE_SETUP=1"],
                "install": "python -m pip install -e .",
                "test_cmd": "pytest -n0 -rA",
            }
        }
    }
    constants.NON_TEST_EXTS = []
    for module in (package, harness, constants):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = _load("test_swe_gym_public_baseline", "swe_gym_run.py")

    script = runner._public_baseline_script(
        {
            "repo": "owner/repo",
            "version": "1.0",
            "pass_to_pass": ["tests/test_public.py::test_existing"],
        },
        tmp_path,
    )

    assert "export BASELINE_SETUP=1" in script
    assert "pytest -n0 -rA tests/test_public.py::test_existing" in script
    assert "python -m pip install -e ." not in script
    assert _HIDDEN_TEST not in script


@pytest.mark.parametrize(
    "parser",
    [
        lambda _log: (_ for _ in ()).throw(RuntimeError("parser failed")),
        lambda _log: {_HIDDEN_TEST: "NOT_A_STATUS"},
        lambda _log: [_HIDDEN_TEST],
    ],
)
def test_swe_gym_invalid_parser_output_is_model_reward_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parser,
) -> None:
    _install_swe_gym_modules(monkeypatch, parser)
    grader = _load("test_swe_gym_grader_adversarial", "swe_gym_grader.py")
    _redirect_paths(monkeypatch, grader, tmp_path)
    _write_inputs(
        tmp_path,
        {
            "repo": "owner/repo",
            "fail_to_pass": [_HIDDEN_TEST],
            "pass_to_pass": [],
        },
    )

    grader.main()

    _assert_private_result_is_reward_zero(tmp_path)
