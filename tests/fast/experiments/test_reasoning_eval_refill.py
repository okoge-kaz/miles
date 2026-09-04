from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
REFILL_SCRIPT = REPO_ROOT / "experiments" / "scripts" / "reasoning_eval" / "refill-snapshot.sbatch"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("queue_snapshot", "expected_lane"),
    [
        ("", "batch|interactive|04:00:00"),
        ("101|batch|interactive|RUNNING|q3e-existing\n", "batch|short|02:00:00"),
        (
            "101|batch|interactive|RUNNING|q3e-interactive\n"
            "102|batch|short|RUNNING|q3e-short\n",
            "batch|normal|02:30:00",
        ),
    ],
)
def test_refill_routes_submissions_by_qos(
    tmp_path: Path,
    queue_snapshot: str,
    expected_lane: str,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    lane_log = tmp_path / "lanes.log"
    launcher = mock_bin / "launcher"
    _write_executable(
        launcher,
        """#!/bin/bash
set -euo pipefail
printf '%s|%s|%s|%s\n' "$PARTITION" "$QOS" "$WALL" "$*" >> "$LANE_LOG"
if [[ " $* " != *" --submit "* ]]; then
    printf 'status: available=1 complete=0 pending=1\n'
fi
""",
    )
    _write_executable(
        mock_bin / "squeue",
        """#!/bin/bash
set -euo pipefail
printf '%s' "${MOCK_QUEUE_SNAPSHOT:-}"
""",
    )

    shared = tmp_path / "shared"
    for relative_path in (
        "datasets",
        "checkpoints/huggingface",
        "checkpoints/megatron",
        "containers",
    ):
        (shared / relative_path).mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{mock_bin}:{environment['PATH']}",
            "SLURM_SUBMIT_DIR": str(REPO_ROOT),
            "SLURM_JOB_USER": "test-user",
            "LAUNCHER": str(launcher),
            "LANE_LOG": str(lane_log),
            "MOCK_QUEUE_SNAPSHOT": queue_snapshot,
            "TRAINING_ROOT": str(tmp_path / "training"),
            "RUN_NAMESPACE": "test-refill",
            "SNAPSHOT_ARM_MAX_STEPS": "test-arm=1",
            "EXPECTED_CHECKPOINTS": "1",
            "BATCH_INFLIGHT_TARGET": "1",
            "REGULAR_INFLIGHT_TARGET": "1",
            "INTERACTIVE_INFLIGHT_TARGET": "1",
            "REFILL_ONCE": "1",
            "SHARED_WS": str(shared),
            "WS": str(tmp_path / "workspace"),
            "WANDB_MODE": "disabled",
        }
    )

    result = subprocess.run(
        ["bash", str(REFILL_SCRIPT)],
        check=False,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    lanes = lane_log.read_text(encoding="utf-8").splitlines()
    assert lanes[0].startswith("batch|short|02:00:00|")
    assert lanes[-1].startswith(f"{expected_lane}|--submit --max-submissions 1")


def test_refill_header_uses_aws_pdx_cpu_partition_and_qos() -> None:
    text = REFILL_SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --qos=cpu-long" in text
    assert "batch_short" not in text
    assert 'INTERACTIVE_PARTITION="${INTERACTIVE_PARTITION:-${GPU_PARTITION}}"' in text
