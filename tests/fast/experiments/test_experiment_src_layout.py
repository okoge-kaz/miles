from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REMOVED_NAMESPACES = (
    "calendar_env",
    "nemo_blends",
    "tau_bench",
    "tool_call_env",
    "workplace_env",
)
CANONICAL_DATASET_CLIS = (
    "experiments.src.datasets.calendar.prepare",
    "experiments.src.datasets.common.audit",
    "experiments.src.datasets.common.merge",
    "experiments.src.datasets.nemotron.convert",
    "experiments.src.datasets.nemotron.restore",
    "experiments.src.datasets.areal_tau2.prepare",
    "experiments.src.datasets.tau_bench.prepare",
    "experiments.src.datasets.tool_call_pivot.prepare",
    "experiments.src.datasets.workplace.prepare",
)


def test_removed_namespace_directories_are_absent():
    source_root = Path(__file__).resolve().parents[3] / "experiments" / "src"
    present = [name for name in REMOVED_NAMESPACES if (source_root / name).exists()]
    assert not present, present


@pytest.mark.parametrize("module_name", CANONICAL_DATASET_CLIS)
def test_canonical_dataset_cli_keeps_help_contract(module_name):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_canonical_layout_does_not_import_removed_namespaces():
    program = """
import sys
import experiments.src.datasets.calendar.prepare
import experiments.src.datasets.areal_tau2.prepare
import experiments.src.datasets.common.merge
import experiments.src.datasets.nemotron.convert
import experiments.src.datasets.tau_bench.prepare
import experiments.src.datasets.tool_call_pivot.prepare
import experiments.src.datasets.workplace.prepare
import experiments.src.environments.tau_bench.generator
import experiments.src.environments.areal_tau2.generator
import experiments.src.environments.workplace.generator
import experiments.src.evaluators.livecodebench
import experiments.src.protocols.openai_responses
import experiments.src.reward_sets.all_domains
import experiments.tools.reasoning_eval.suite
removed_prefixes = (
    'experiments.src.calendar_env',
    'experiments.src.nemo_blends',
    'experiments.src.tau_bench',
    'experiments.src.tool_call_env',
    'experiments.src.workplace_env',
)
loaded = sorted(name for name in sys.modules if name.startswith(removed_prefixes))
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_environment_generators_do_not_import_each_other():
    source_root = Path(__file__).resolve().parents[3] / "experiments" / "src" / "environments"
    tau_source = (source_root / "tau_bench" / "generator.py").read_text()
    workplace_source = (source_root / "workplace" / "generator.py").read_text()

    assert "environments.workplace" not in tau_source
    assert "environments.tau_bench" not in workplace_source
    assert "environments.common.observations" in tau_source
    assert "environments.common.observations" in workplace_source
