from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REFILL_SCRIPT = REPO_ROOT / "experiments" / "scripts" / "reasoning_eval" / "refill-snapshot.sbatch"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_refill_maintains_pbs_inflight_target(tmp_path: Path) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    lane_log = tmp_path / "lanes.log"
    launcher = mock_bin / "launcher"
    _write_executable(
        launcher,
        """#!/bin/bash
set -euo pipefail
printf '%s|%s\n' "$WALL" "$*" >> "$LANE_LOG"
if [[ " $* " != *" --submit "* ]]; then
    printf 'status: available=1 complete=0 pending=1\n'
fi
""",
    )
    _write_executable(
        mock_bin / "qselect",
        """#!/bin/bash
set -euo pipefail
exit 0
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
            "PBS_O_WORKDIR": str(REPO_ROOT),
            "USER": "test-user",
            "LAUNCHER": str(launcher),
            "LANE_LOG": str(lane_log),
            "TRAINING_ROOT": str(tmp_path / "training"),
            "RUN_NAMESPACE": "test-refill",
            "SNAPSHOT_ARM_MAX_STEPS": "test-arm=1",
            "EXPECTED_CHECKPOINTS": "1",
            "INFLIGHT_TARGET": "1",
            "REFILL_ONCE": "1",
            "MILES_WORKSPACE_ROOT": str(shared),
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
    assert lanes[0] == "04:00:00|"
    assert lanes[-1] == "04:00:00|--submit --max-submissions 1"


def test_refill_header_uses_pbs_cpu_queue_without_project() -> None:
    text = REFILL_SCRIPT.read_text(encoding="utf-8")

    assert "#PBS -q R9920261300" in text
    assert "#PBS -l walltime=24:00:00" in text
    assert "#PBS -P" not in text
    assert "qselect -u" in text
    assert "qstat -f -F json" in text
