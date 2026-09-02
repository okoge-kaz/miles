from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from experiments.tools.replay_buffer_validation.tau2 import analyze

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_load_segments_uses_seeded_manifest_keys(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[int, str, str, str, Path]] = []

    def fake_parse_segment(
        seed: int,
        mode: str,
        phase: str,
        job_id: str,
        log_dir: Path,
    ) -> tuple[int, str, str]:
        calls.append((seed, mode, phase, job_id, log_dir))
        return seed, mode, phase

    monkeypatch.setattr(analyze, "_parse_segment", fake_parse_segment)
    manifest = {"LOG_DIR": str(tmp_path), "SEEDS": "41 42"}
    for seed in (41, 42):
        for _, key in analyze.JOB_SPECS:
            for phase in ("FRESH", "RESUME"):
                manifest[f"SEED_{seed}_{key}_{phase}_JOB"] = f"{seed}-{key}-{phase}"

    segments = analyze._load_segments(manifest)

    assert len(segments) == 16
    assert calls[0] == (41, "no-replay", "fresh", "41-NO_REPLAY-FRESH", tmp_path)
    assert calls[-1] == (
        42,
        "inflight-overlap",
        "resume",
        "42-INFLIGHT_OVERLAP-RESUME",
        tmp_path,
    )


def test_fresh_failure_and_resume_exit_patterns_are_distinct() -> None:
    failure = "debug_failure_after_rollout=10 reached at rollout_id=9"
    clean_exit = "debug_exit_after_rollout=6 reached at rollout_id=15, exiting"

    assert analyze.DEBUG_FAILURE.search(failure)
    assert not analyze.DEBUG_EXIT.search(failure)
    assert analyze.DEBUG_EXIT.search(clean_exit)
    assert not analyze.DEBUG_FAILURE.search(clean_exit)


def test_recovery_submission_reuses_fresh_job_and_exports_valid_save_intervals(
    tmp_path: Path,
) -> None:
    namespace = "pytest-rbresume"
    reused_job = "812345.pbs1"
    fake_repo = tmp_path / "miles"
    output_root = fake_repo / "experiments/outputs"
    manifest_dir = output_root / "replay_buffer_validation/tau2"
    log_dir = (
        output_root
        / "training/tau_bench/areal-tau2/qwen3-4b-agentic-sft-953/"
        "replay-resume-ablation"
    )
    manifest_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    (manifest_dir / f"{namespace}.jobs").write_text(
        "\n".join(
            (
                f"VALIDATION_NAMESPACE={namespace}",
                "SEEDS=42",
                "FRESH_UPDATES=10",
                "RESUME_UPDATES=6",
                "ROLLOUT_BATCH_SIZE=8",
                "N_SAMPLES_PER_PROMPT=16",
                "GLOBAL_BATCH_SIZE=128",
                "ASYNC_MAX_CONCURRENT_SAMPLES=96",
                f"SEED_42_NO_REPLAY_FRESH_JOB={reused_job}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (log_dir / f"{reused_job}.OU").write_text(
        "debug_failure_after_rollout=10 reached at rollout_id=9\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    mock_qsub = fake_bin / "qsub"
    mock_qsub.write_text(
        """#!/bin/bash
set -euo pipefail
job_id=900000
if [[ -s "${MOCK_QSUB_STATE}" ]]; then
    job_id="$(< "${MOCK_QSUB_STATE}")"
fi
job_id=$((job_id + 1))
printf '%s\n' "${job_id}" >"${MOCK_QSUB_STATE}"
printf '%q ' "$@" >>"${MOCK_QSUB_LOG}"
printf '\n' >>"${MOCK_QSUB_LOG}"
printf '%s.pbs1\n' "${job_id}"
""",
        encoding="utf-8",
    )
    mock_qsub.chmod(0o755)
    qsub_log = tmp_path / "qsub.log"
    qsub_state = tmp_path / "qsub.state"
    launcher = (
        REPO_ROOT
        / "experiments/tools/replay_buffer_validation/tau2/"
        "submit_replay_resume_ablation.sh"
    )
    environment = {
        **os.environ,
        "MILES_REPO": str(fake_repo),
        "MILES_WORKSPACE_ROOT": str(tmp_path / "workspace"),
        "VALIDATION_NAMESPACE": namespace,
        "RECOVERY_TAG": "save-retain-fix",
        "WANDB_MODE": "online",
        "WANDB_API_KEY": "test-key",
        "PBS_QSUB_BIN": str(mock_qsub),
        "MOCK_QSUB_LOG": str(qsub_log),
        "MOCK_QSUB_STATE": str(qsub_state),
    }

    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--submit",
            "--reuse-first-fresh-job",
            reused_job,
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    calls = [shlex.split(line) for line in qsub_log.read_text().splitlines()]
    assert len(calls) == 8
    assert "depend=afterany:812345.pbs1" in calls[0]
    assert "depend=afterok:900001.pbs1" in calls[1]
    for index, call in enumerate(calls[:7]):
        exported = next(argument for argument in call if "SAVE_INTERVAL=" in argument)
        if index % 2 == 0:
            assert "SAVE_INTERVAL=1000,SAVE_RETAIN_INTERVAL=1000" in exported
            assert "DEBUG_EXIT_AFTER_ROLLOUT=6" in exported
        else:
            assert "SAVE_INTERVAL=10,SAVE_RETAIN_INTERVAL=100" in exported
            assert "DEBUG_FAIL_AFTER_ROLLOUT=10" in exported

    recovery_manifest = manifest_dir / f"{namespace}.save-retain-fix.jobs"
    manifest_text = recovery_manifest.read_text(encoding="utf-8")
    assert "SEED_42_NO_REPLAY_FRESH_JOB=812345.pbs1" in manifest_text
    assert "SEED_42_NO_REPLAY_RESUME_JOB=900001.pbs1" in manifest_text
    assert "SEED_42_INFLIGHT_OVERLAP_RESUME_JOB=900007.pbs1" in manifest_text
    assert "SUMMARY_JOB=900008.pbs1" in manifest_text
    assert "reused    seed=42" in result.stdout
    assert "summary=900008.pbs1" in result.stdout


def test_tau_recipe_rejects_incompatible_save_intervals_before_launch() -> None:
    recipe = (
        REPO_ROOT
        / "experiments/scripts/tau_bench/async/areal-tau2/"
        "qwen3-4b-agentic-sft-953/run.sbatch"
    ).read_text(encoding="utf-8")

    assert "SAVE_RETAIN_INTERVAL % SAVE_INTERVAL == 0" in recipe
    assert "must be a multiple of SAVE_INTERVAL" in recipe
