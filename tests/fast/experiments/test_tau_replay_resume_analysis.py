from pathlib import Path

from experiments.tools.replay_buffer_validation.tau2 import analyze


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
