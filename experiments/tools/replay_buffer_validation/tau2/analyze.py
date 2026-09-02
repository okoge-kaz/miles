"""Summarize the four-mode Tau2 replay/resume ablation."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.tools.replay_buffer_validation import log_parser

JOB_SPECS = (
    ("no-replay", "NO_REPLAY"),
    ("rollout", "ROLLOUT"),
    ("inflight", "INFLIGHT"),
    ("inflight-overlap", "INFLIGHT_OVERLAP"),
)
ENTRY_EPOCH = re.compile(r"MILES_RL_ENTRY_EPOCH=(?P<value>[0-9]+(?:\.[0-9]+)?)")
DEBUG_EXIT = re.compile(r"debug_exit_after_rollout=(?P<count>\d+) reached at rollout_id=(?P<id>\d+)")
DEBUG_FAILURE = re.compile(
    r"debug_failure_after_rollout=(?P<count>\d+) reached at rollout_id=(?P<id>\d+)"
)
CHECKPOINT = re.compile(r"replay resume checkpoint (?P<id>\d+): (?P<payload>\{.*\})")
TAU_OVERLAP = re.compile(
    r"Tau resume overlap sample=\S+ attempts=(?P<attempts>\d+) "
    r"policy_request=(?P<policy>[0-9.]+)s db_prefill_overlap=(?P<overlap>[0-9.]+)s "
    r"db_restore_unhidden=(?P<unhidden>[0-9.]+)s"
)


@dataclass(frozen=True)
class Segment:
    seed: int
    mode: str
    phase: str
    job_id: str
    path: Path | None
    rollout_steps: tuple[int, ...]
    train_steps: tuple[int, ...]
    first_rollout_seconds: float | None
    first_optimizer_seconds: float | None
    first_rollout_wait_seconds: float | None
    checkpoint_metrics: dict[str, float]
    resume_metrics: dict[str, float]
    overlap_attempts: int
    overlap_seconds: float
    restore_unhidden_seconds: float
    stopped_as_expected: bool


def _parse_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


def _find_log(log_dir: Path, job_id: str) -> Path | None:
    matches = sorted(
        [
            *log_dir.glob(f"{job_id}.OU"),
            *log_dir.glob(f"*-{job_id}.log"),
        ]
    )
    if len(matches) > 1:
        raise RuntimeError(f"multiple logs found for job {job_id}: {matches}")
    return matches[0] if matches else None


def _first_metric_seconds(records: list[dict], entry: float | None, step_key: str) -> float | None:
    if entry is None:
        return None
    candidates = [record for record in records if record["step_key"] == step_key]
    if step_key == "rollout/step":
        candidates = [record for record in candidates if "perf/rollout_time" in record["metrics"]]
    if not candidates:
        return None
    return min(float(record["ts"]) for record in candidates) - entry


def _parse_segment(seed: int, mode: str, phase: str, job_id: str, log_dir: Path) -> Segment:
    path = _find_log(log_dir, job_id)
    if path is None:
        return Segment(
            seed,
            mode,
            phase,
            job_id,
            None,
            (),
            (),
            None,
            None,
            None,
            {},
            {},
            0,
            0.0,
            0.0,
            False,
        )

    text = log_parser.ANSI.sub("", path.read_text(errors="replace"))
    records = log_parser.parse_log(path)
    rollout_records = [
        record
        for record in records
        if record["step_key"] == "rollout/step" and "perf/rollout_time" in record["metrics"]
    ]
    train_records = [record for record in records if record["step_key"] == "train/step"]
    entries = [float(match["value"]) for match in ENTRY_EPOCH.finditer(text)]
    entry = entries[0] if entries else None
    checkpoint_metrics: dict[str, float] = {}
    for match in CHECKPOINT.finditer(text):
        checkpoint_metrics = {
            key: float(value) for key, value in log_parser.parse_payload(match["payload"]).items()
        }
    resume_metrics: dict[str, float] = {}
    for record in rollout_records:
        for key, value in record["metrics"].items():
            if key.startswith("resume/benchmark/") or key.startswith("resume/replay_buffer/"):
                resume_metrics[key] = float(value)
        if resume_metrics:
            break
    overlaps = list(TAU_OVERLAP.finditer(text))
    first_wait = (
        float(rollout_records[0]["metrics"]["perf/rollout_time"])
        if rollout_records
        else None
    )
    return Segment(
        seed=seed,
        mode=mode,
        phase=phase,
        job_id=job_id,
        path=path,
        rollout_steps=tuple(sorted({int(record["step"]) for record in rollout_records})),
        train_steps=tuple(sorted({int(record["step"]) for record in train_records})),
        first_rollout_seconds=_first_metric_seconds(records, entry, "rollout/step"),
        first_optimizer_seconds=_first_metric_seconds(records, entry, "train/step"),
        first_rollout_wait_seconds=first_wait,
        checkpoint_metrics=checkpoint_metrics,
        resume_metrics=resume_metrics,
        overlap_attempts=sum(int(match["attempts"]) for match in overlaps),
        overlap_seconds=sum(float(match["overlap"]) for match in overlaps),
        restore_unhidden_seconds=sum(float(match["unhidden"]) for match in overlaps),
        stopped_as_expected=bool(
            DEBUG_FAILURE.search(text) if phase == "fresh" else DEBUG_EXIT.search(text)
        ),
    )


def _value(metrics: dict[str, float], suffix: str) -> float | None:
    return metrics.get(f"resume/benchmark/{suffix}")


def _format(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _render_checkpoint_table(segments: list[Segment]) -> list[str]:
    lines = [
        "| seed | mode | outstanding samples | carried | lost | completed groups reused | partial groups | partial tokens | regenerate groups | replay save (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for segment in segments:
        metrics = segment.checkpoint_metrics
        lines.append(
            f"| {segment.seed} | {segment.mode} "
            f"| {_format(_value(metrics, 'checkpoint/outstanding_samples'), 0)} "
            f"| {_format(_value(metrics, 'checkpoint/carried_samples'), 0)} "
            f"| {_format(_value(metrics, 'checkpoint/lost_samples'), 0)} "
            f"| {_format(_value(metrics, 'checkpoint/completed_groups_reused'), 0)} "
            f"| {_format(_value(metrics, 'checkpoint/partial_groups_continued'), 0)} "
            f"| {_format(_value(metrics, 'checkpoint/partial_response_tokens_continued'), 0)} "
            f"| {_format(_value(metrics, 'checkpoint/groups_to_regenerate'), 0)} "
            f"| {_format(_value(metrics, 'checkpoint/replay_total_seconds'))} |"
        )
    return lines


def _render_resume_table(segments: list[Segment]) -> list[str]:
    lines = [
        "| seed | mode | state load (s) | entry→first rollout (s) | entry→first optimizer (s) | first rollout wait (s) | regenerated active groups | inflight groups/tokens | DB overlap/unhidden (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for segment in segments:
        metrics = segment.resume_metrics
        inflight_groups = metrics.get("resume/replay_buffer/inflight_groups_restored")
        inflight_tokens = metrics.get("resume/replay_buffer/inflight_tokens_restored")
        inflight = (
            "-"
            if inflight_groups is None or inflight_tokens is None
            else f"{inflight_groups:.0f}/{inflight_tokens:.0f}"
        )
        lines.append(
            f"| {segment.seed} | {segment.mode} "
            f"| {_format(_value(metrics, 'load/total_seconds'))} "
            f"| {_format(segment.first_rollout_seconds)} "
            f"| {_format(segment.first_optimizer_seconds)} "
            f"| {_format(segment.first_rollout_wait_seconds)} "
            f"| {_format(metrics.get('resume/replay_buffer/regenerated_active_groups'), 0)} "
            f"| {inflight} "
            f"| {segment.overlap_seconds:.2f}/{segment.restore_unhidden_seconds:.2f} |"
        )
    return lines


def _render_speedup(segments: list[Segment]) -> list[str]:
    lines = [
        "| seed | mode | first-rollout speedup vs no replay | first-optimizer speedup vs no replay |",
        "|---:|---|---:|---:|",
    ]
    for segment in segments:
        baseline = next(
            candidate
            for candidate in segments
            if candidate.seed == segment.seed and candidate.mode == "no-replay"
        )
        rollout_speedup = _ratio(baseline.first_rollout_seconds, segment.first_rollout_seconds)
        optimizer_speedup = _ratio(baseline.first_optimizer_seconds, segment.first_optimizer_seconds)
        lines.append(
            f"| {segment.seed} | {segment.mode} "
            f"| {_format(rollout_speedup, 3)}x | {_format(optimizer_speedup, 3)}x |"
        )
    return lines


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _audit(segments: list[Segment], fresh_updates: int, resume_updates: int) -> list[str]:
    failures: list[str] = []
    for segment in segments:
        if segment.path is None:
            failures.append(f"seed={segment.seed} {segment.mode}/{segment.phase}: log missing")
            continue
        start = 0 if segment.phase == "fresh" else fresh_updates
        count = fresh_updates if segment.phase == "fresh" else resume_updates
        expected = tuple(range(start, start + count))
        if segment.rollout_steps != expected:
            failures.append(
                f"seed={segment.seed} {segment.mode}/{segment.phase}: "
                f"rollout steps={segment.rollout_steps}, expected={expected}"
            )
        if segment.train_steps != expected:
            failures.append(
                f"seed={segment.seed} {segment.mode}/{segment.phase}: "
                f"train steps={segment.train_steps}, expected={expected}"
            )
        if not segment.stopped_as_expected:
            expected_stop = "intentional debug failure" if segment.phase == "fresh" else "clean debug exit"
            failures.append(
                f"seed={segment.seed} {segment.mode}/{segment.phase}: {expected_stop} missing"
            )
        if segment.phase == "fresh" and not segment.checkpoint_metrics:
            failures.append(
                f"seed={segment.seed} {segment.mode}/fresh: checkpoint conservation metrics missing"
            )
        if segment.phase == "resume" and _value(segment.resume_metrics, "load/total_seconds") is None:
            failures.append(f"seed={segment.seed} {segment.mode}/resume: load timing metric missing")

    seeds = sorted({segment.seed for segment in segments})
    for seed in seeds:
        fresh = {
            segment.mode: segment
            for segment in segments
            if segment.seed == seed and segment.phase == "fresh"
        }
        no_replay = fresh.get("no-replay")
        if no_replay and _value(no_replay.checkpoint_metrics, "checkpoint/lost_samples") == 0:
            failures.append(f"seed={seed} no-replay/fresh: checkpoint reports no lost samples")
        for mode in ("rollout", "inflight", "inflight-overlap"):
            segment = fresh.get(mode)
            if segment and _value(segment.checkpoint_metrics, "checkpoint/lost_samples") != 0:
                failures.append(f"seed={seed} {mode}/fresh: replay checkpoint lost samples")
        overlap = next(
            (
                segment
                for segment in segments
                if segment.seed == seed
                and segment.mode == "inflight-overlap"
                and segment.phase == "resume"
            ),
            None,
        )
        if overlap and overlap.overlap_attempts == 0:
            failures.append(
                f"seed={seed} inflight-overlap/resume: no DB/prefill overlap attempt observed"
            )
    return failures


def _load_segments(manifest: dict[str, str]) -> list[Segment]:
    log_dir = Path(manifest["LOG_DIR"])
    seeds = [int(seed) for seed in manifest["SEEDS"].split()]
    return [
        _parse_segment(
            seed,
            mode,
            phase,
            manifest[f"SEED_{seed}_{key}_{phase.upper()}_JOB"],
            log_dir,
        )
        for seed in seeds
        for mode, key in JOB_SPECS
        for phase in ("fresh", "resume")
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = _parse_manifest(args.manifest)
    segments = _load_segments(manifest)
    fresh = [segment for segment in segments if segment.phase == "fresh"]
    resumes = [segment for segment in segments if segment.phase == "resume"]
    failures = _audit(
        segments,
        fresh_updates=int(manifest["FRESH_UPDATES"]),
        resume_updates=int(manifest["RESUME_UPDATES"]),
    )
    lines = [
        f"# Tau2 replay resume ablation: {manifest['VALIDATION_NAMESPACE']}",
        "",
        f"W&B project: `{manifest['WANDB_PROJECT']}`",
        "",
        f"Seeds: `{manifest['SEEDS']}`",
        "",
        "## Checkpoint sample conservation",
        "",
        *_render_checkpoint_table(fresh),
        "",
        "## Resume latency",
        "",
        *_render_resume_table(resumes),
        "",
        "## Acceleration",
        "",
        *_render_speedup(resumes),
        "",
        "## Audit",
        "",
    ]
    lines.extend(f"- FAIL: {failure}" for failure in failures)
    if not failures:
        lines.append(
            f"- PASS: all {len(segments) // 2} fresh/resume pairs completed with comparable telemetry."
        )
    rendered = "\n".join(lines) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
