"""Summarize the rollout-versus-inflight fresh/resume validation logs."""

from __future__ import annotations

import argparse
import math
import re
import shlex
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.src.offpolicy_acceleration import log_source


JOB_SPECS = (
    ("rollout_fresh", "ROLLOUT_FRESH_JOB", "rollout", "fresh"),
    ("rollout_resume", "ROLLOUT_RESUME_JOB", "rollout", "resume"),
    ("inflight_fresh", "INFLIGHT_FRESH_JOB", "inflight", "fresh"),
    ("inflight_resume", "INFLIGHT_RESUME_JOB", "inflight", "resume"),
)
PUBLISHED = re.compile(
    r"Published replay buffer (?P<path>\S+) \((?P<bytes>\d+) bytes, "
    r"capture (?P<capture>[0-9.]+) seconds, write (?P<write>[0-9.]+) seconds, "
    r"total (?P<total>[0-9.]+) seconds\)"
)
LOADED = re.compile(
    r"Loaded replay buffer from (?P<path>\S+) at rollout (?P<rollout>\d+) "
    r"\(read (?P<read>[0-9.]+) seconds, restore (?P<restore>[0-9.]+) seconds, "
    r"total (?P<total>[0-9.]+) seconds\)"
)
START_ROLLOUT = re.compile(r"start_rollout_id\s+\.*\s+(?P<value>\d+)")
SAVED_ITERATION = re.compile(r"successfully saved checkpoint from iteration\s+(?P<value>\d+)")
RL_ENTRY_EPOCH = re.compile(r"MILES_RL_ENTRY_EPOCH=(?P<value>[0-9]+(?:\.[0-9]+)?)")
CHECKPOINT_LOAD = re.compile(r"(?:^|\s)--load\s+(?P<path>\S+)")


@dataclass(frozen=True)
class SaveSample:
    size_bytes: int
    capture_seconds: float
    write_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class LoadSample:
    rollout_id: int
    read_seconds: float
    restore_seconds: float
    total_seconds: float


@dataclass
class Segment:
    name: str
    job_id: str
    buffer_type: str
    phase: str
    log_path: Path | None
    checkpoint_paths: list[str]
    start_rollout_id: int | None
    entry_epoch: float | None
    first_rollout_wall_seconds: float | None
    first_train_wall_seconds: float | None
    rollout_steps: list[int]
    train_steps: list[int]
    train_losses: list[float]
    grad_norms: list[float]
    learning_rates: list[float]
    raw_rewards: list[float]
    truncated_ratios: list[float]
    rollout_times: list[float]
    step_times: list[float]
    train_wait_times: list[float]
    trained_staleness_means: list[float]
    trained_staleness_maxes: list[float]
    offered_staleness_maxes: list[float]
    bound_exceeded_groups: list[float]
    saves: list[SaveSample]
    loads: list[LoadSample]
    saved_iterations: list[int]
    resume_metrics: dict[str, float]
    clean_debug_exit: bool
    missing_config_args: list[str]


def parse_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


def find_log(log_dir: Path, job_id: str) -> Path | None:
    matches = sorted(log_dir.glob(f"*-{job_id}.log"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"multiple logs found for job {job_id}: {matches}")
    return matches[0]


def metric_values(records: list[dict], key: str) -> list[float]:
    return [float(record["metrics"][key]) for record in records if key in record["metrics"]]


def checkpoint_load_paths(text: str) -> list[str]:
    """Return concrete checkpoint paths, ignoring unexpanded shell templates."""
    paths = set()
    for match in CHECKPOINT_LOAD.finditer(text):
        path = match["path"].strip("'\"")
        if "$" not in path:
            paths.add(path)
    return sorted(paths)


def parse_segment(name: str, job_id: str, buffer_type: str, phase: str, log_dir: Path) -> Segment:
    path = find_log(log_dir, job_id)
    if path is None:
        return Segment(
            name=name,
            job_id=job_id,
            buffer_type=buffer_type,
            phase=phase,
            log_path=None,
            checkpoint_paths=[],
            start_rollout_id=None,
            entry_epoch=None,
            first_rollout_wall_seconds=None,
            first_train_wall_seconds=None,
            rollout_steps=[],
            train_steps=[],
            train_losses=[],
            grad_norms=[],
            learning_rates=[],
            raw_rewards=[],
            truncated_ratios=[],
            rollout_times=[],
            step_times=[],
            train_wait_times=[],
            trained_staleness_means=[],
            trained_staleness_maxes=[],
            offered_staleness_maxes=[],
            bound_exceeded_groups=[],
            saves=[],
            loads=[],
            saved_iterations=[],
            resume_metrics={},
            clean_debug_exit=False,
            missing_config_args=[],
        )

    text = log_source.ANSI.sub("", path.read_text(errors="replace"))
    records = log_source.merge_step_records(log_source.parse_log(path))
    rollout_records = [record for record in records if record["step_key"] == "rollout/step"]
    train_records = [record for record in records if record["step_key"] == "train/step"]
    trained_records = [
        record
        for record in rollout_records
        if "perf/step_time" in record["metrics"] or "rollout/raw_reward" in record["metrics"]
    ]
    entry_epochs = [float(match["value"]) for match in RL_ENTRY_EPOCH.finditer(text)]
    entry_epoch = entry_epochs[0] if entry_epochs else None

    def first_wall_seconds(records_for_stream: list[dict]) -> float | None:
        if entry_epoch is None or not records_for_stream:
            return None
        return min(float(record["ts"]) for record in records_for_stream) - entry_epoch

    start_ids = [int(match["value"]) for match in START_ROLLOUT.finditer(text)]
    saves = [
        SaveSample(
            size_bytes=int(match["bytes"]),
            capture_seconds=float(match["capture"]),
            write_seconds=float(match["write"]),
            total_seconds=float(match["total"]),
        )
        for match in PUBLISHED.finditer(text)
    ]
    loads = [
        LoadSample(
            rollout_id=int(match["rollout"]),
            read_seconds=float(match["read"]),
            restore_seconds=float(match["restore"]),
            total_seconds=float(match["total"]),
        )
        for match in LOADED.finditer(text)
    ]
    resume_metrics: dict[str, float] = {}
    for record in rollout_records:
        for key, value in record["metrics"].items():
            if key.startswith("resume/replay_buffer/"):
                resume_metrics[key.removeprefix("resume/replay_buffer/")] = float(value)

    required_args = (
        "--hf-checkpoint /ckpt/hf/iter_0004000",
        "--ref-load /ckpt/megatron/Qwen3-4B-Base-LR2e-5-Step4000_torch_dist",
        "--rollout-max-response-len 16384",
        "--rollout-max-context-len 32768",
        "--num-rollout 8",
        "--rollout-batch-size 192",
        "--n-samples-per-prompt 16",
        "--global-batch-size 3072",
        "--num-steps-per-rollout 1",
        "--fully-async-queue-policy queue-recycle",
        "--rm-type deepscaler",
        "--zero-reward-on-truncated",
        "--max-weight-staleness 8",
        "--staleness-reference prefill",
        "--use-replay-buffer",
        f"--replay-buffer-type {buffer_type}",
        "--save-interval 1",
        "--wandb-project async-rl-miles-replay-buffer",
    )

    trained_steps = sorted({int(record["step"]) for record in trained_records})
    return Segment(
        name=name,
        job_id=job_id,
        buffer_type=buffer_type,
        phase=phase,
        log_path=path,
        checkpoint_paths=checkpoint_load_paths(text),
        start_rollout_id=start_ids[0] if start_ids else (trained_steps[0] if trained_steps else None),
        entry_epoch=entry_epoch,
        first_rollout_wall_seconds=first_wall_seconds(rollout_records),
        first_train_wall_seconds=first_wall_seconds(train_records),
        rollout_steps=trained_steps,
        train_steps=sorted({int(record["step"]) for record in train_records}),
        train_losses=metric_values(train_records, "train/loss"),
        grad_norms=metric_values(train_records, "train/grad_norm"),
        learning_rates=metric_values(train_records, "train/lr-pg_0"),
        raw_rewards=metric_values(trained_records, "rollout/raw_reward"),
        truncated_ratios=metric_values(trained_records, "rollout/truncated_ratio"),
        rollout_times=metric_values(trained_records, "perf/rollout_time"),
        step_times=metric_values(trained_records, "perf/step_time"),
        train_wait_times=metric_values(trained_records, "perf/train_wait_time"),
        trained_staleness_means=metric_values(trained_records, "staleness/bound/train/mean"),
        trained_staleness_maxes=metric_values(trained_records, "staleness/bound/train/max"),
        offered_staleness_maxes=metric_values(trained_records, "staleness/bound/rollout/max"),
        bound_exceeded_groups=metric_values(trained_records, "staleness/bound_exceeded_groups"),
        saves=saves,
        loads=loads,
        saved_iterations=[int(match["value"]) for match in SAVED_ITERATION.finditer(text)],
        resume_metrics=resume_metrics,
        clean_debug_exit="debug_exit_after_rollout=" in text and "reached at rollout_id=" in text,
        missing_config_args=[arg for arg in required_args if arg not in text],
    )


def mean(values: list[float]) -> str:
    return f"{statistics.fmean(values):.4f}" if values else "-"


def median(values: list[float]) -> str:
    return f"{statistics.median(values):.3f}" if values else "-"


def first(values: list[float]) -> str:
    return f"{values[0]:.3f}" if values else "-"


def optional_seconds(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def maximum(values: list[float]) -> str:
    return f"{max(values):.3f}" if values else "-"


def render_segment_table(segments: list[Segment]) -> list[str]:
    lines = [
        "| segment | job | start | rollout steps | train steps | reward mean | trunc. mean | "
        "first rollout s | step mean s | "
        "entry→rollout s | entry→train s | save count | capture med. s | write med. s | total med. s | "
        "size med. MiB | load total s | loss mean | grad norm mean | train stale mean | train stale max | "
        "offered stale max | "
        "recycled groups | clean exit |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for segment in segments:
        sizes = [save.size_bytes / 2**20 for save in segment.saves]
        start = segment.start_rollout_id if segment.start_rollout_id is not None else "-"
        lines.append(
            f"| {segment.name} | {segment.job_id} | {start} "
            f"| {','.join(map(str, segment.rollout_steps)) or '-'} "
            f"| {','.join(map(str, segment.train_steps)) or '-'} | {mean(segment.raw_rewards)} "
            f"| {mean(segment.truncated_ratios)} | {first(segment.rollout_times)} "
            f"| {mean(segment.step_times)} | {optional_seconds(segment.first_rollout_wall_seconds)} "
            f"| {optional_seconds(segment.first_train_wall_seconds)} | {len(segment.saves)} "
            f"| {median([save.capture_seconds for save in segment.saves])} "
            f"| {median([save.write_seconds for save in segment.saves])} "
            f"| {median([save.total_seconds for save in segment.saves])} "
            f"| {median(sizes)} | {first([load.total_seconds for load in segment.loads])} "
            f"| {mean(segment.train_losses)} | {mean(segment.grad_norms)} "
            f"| {mean(segment.trained_staleness_means)} | {maximum(segment.trained_staleness_maxes)} "
            f"| {maximum(segment.offered_staleness_maxes)} | {sum(segment.bound_exceeded_groups):.0f} "
            f"| {'yes' if segment.clean_debug_exit else 'no'} |"
        )
    return lines


def render_resume_table(segments: list[Segment]) -> list[str]:
    resumes = [segment for segment in segments if segment.phase == "resume"]
    keys = sorted({key for segment in resumes for key in segment.resume_metrics})
    lines = ["| metric | " + " | ".join(segment.name for segment in resumes) + " |"]
    lines.append("|---|" + "---:|" * len(resumes))
    for key in keys:
        values = [
            f"{segment.resume_metrics[key]:.3f}" if key in segment.resume_metrics else "-" for segment in resumes
        ]
        lines.append(f"| {key} | " + " | ".join(values) + " |")
    return lines


def render_comparison(segments: list[Segment]) -> list[str]:
    by_name = {segment.name: segment for segment in segments}
    saves = {
        buffer_type: [
            save
            for segment in segments
            if segment.buffer_type == buffer_type
            for save in segment.saves
        ]
        for buffer_type in ("rollout", "inflight")
    }

    def median_or_none(values: list[float]) -> float | None:
        return statistics.median(values) if values else None

    def mean_or_none(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    def max_or_none(values: list[float]) -> float | None:
        return max(values) if values else None

    def sum_or_none(values: list[float]) -> float | None:
        return sum(values) if values else None

    def metric_by_buffer(field: str, buffer_type: str) -> list[float]:
        return [
            value
            for segment in segments
            if segment.buffer_type == buffer_type
            for value in getattr(segment, field)
        ]

    def indexed_or_none(values: list[float], index: int) -> float | None:
        return values[index] if len(values) > index else None

    def load_value(segment_name: str, field: str) -> float | None:
        loads = by_name[segment_name].loads
        return float(getattr(loads[0], field)) if loads else None

    rollout_load_total = load_value("rollout_resume", "total_seconds")
    inflight_load_total = load_value("inflight_resume", "total_seconds")
    rollout_first_wait = indexed_or_none(by_name["rollout_resume"].rollout_times, 0)
    inflight_first_wait = indexed_or_none(by_name["inflight_resume"].rollout_times, 0)

    rows = (
        (
            "replay save capture median (s)",
            median_or_none([sample.capture_seconds for sample in saves["rollout"]]),
            median_or_none([sample.capture_seconds for sample in saves["inflight"]]),
        ),
        (
            "replay save write median (s)",
            median_or_none([sample.write_seconds for sample in saves["rollout"]]),
            median_or_none([sample.write_seconds for sample in saves["inflight"]]),
        ),
        (
            "replay save total median (s)",
            median_or_none([sample.total_seconds for sample in saves["rollout"]]),
            median_or_none([sample.total_seconds for sample in saves["inflight"]]),
        ),
        (
            "replay size median (MiB)",
            median_or_none([sample.size_bytes / 2**20 for sample in saves["rollout"]]),
            median_or_none([sample.size_bytes / 2**20 for sample in saves["inflight"]]),
        ),
        (
            "trained staleness mean (all steps)",
            mean_or_none(metric_by_buffer("trained_staleness_means", "rollout")),
            mean_or_none(metric_by_buffer("trained_staleness_means", "inflight")),
        ),
        (
            "trained staleness max (all steps)",
            max_or_none(metric_by_buffer("trained_staleness_maxes", "rollout")),
            max_or_none(metric_by_buffer("trained_staleness_maxes", "inflight")),
        ),
        (
            "offered staleness max (all steps)",
            max_or_none(metric_by_buffer("offered_staleness_maxes", "rollout")),
            max_or_none(metric_by_buffer("offered_staleness_maxes", "inflight")),
        ),
        (
            "bound-exceeded groups (all steps)",
            sum_or_none(metric_by_buffer("bound_exceeded_groups", "rollout")),
            sum_or_none(metric_by_buffer("bound_exceeded_groups", "inflight")),
        ),
        (
            "resume entry to first rollout metric (s)",
            by_name["rollout_resume"].first_rollout_wall_seconds,
            by_name["inflight_resume"].first_rollout_wall_seconds,
        ),
        (
            "resume entry to first optimizer metric (s)",
            by_name["rollout_resume"].first_train_wall_seconds,
            by_name["inflight_resume"].first_train_wall_seconds,
        ),
        (
            "resume replay read (s)",
            load_value("rollout_resume", "read_seconds"),
            load_value("inflight_resume", "read_seconds"),
        ),
        (
            "resume replay restore (s)",
            load_value("rollout_resume", "restore_seconds"),
            load_value("inflight_resume", "restore_seconds"),
        ),
        (
            "resume replay load total (s)",
            rollout_load_total,
            inflight_load_total,
        ),
        (
            "resume load + first rollout wait (s)",
            rollout_load_total + rollout_first_wait
            if rollout_load_total is not None and rollout_first_wait is not None
            else None,
            inflight_load_total + inflight_first_wait
            if inflight_load_total is not None and inflight_first_wait is not None
            else None,
        ),
        (
            "resume first rollout wait (s)",
            rollout_first_wait,
            inflight_first_wait,
        ),
        (
            "resume second rollout wait (s)",
            indexed_or_none(by_name["rollout_resume"].rollout_times, 1),
            indexed_or_none(by_name["inflight_resume"].rollout_times, 1),
        ),
        (
            "resume train wait mean (s)",
            statistics.fmean(by_name["rollout_resume"].train_wait_times)
            if by_name["rollout_resume"].train_wait_times
            else None,
            statistics.fmean(by_name["inflight_resume"].train_wait_times)
            if by_name["inflight_resume"].train_wait_times
            else None,
        ),
        (
            "fresh raw reward mean",
            statistics.fmean(by_name["rollout_fresh"].raw_rewards)
            if by_name["rollout_fresh"].raw_rewards
            else None,
            statistics.fmean(by_name["inflight_fresh"].raw_rewards)
            if by_name["inflight_fresh"].raw_rewards
            else None,
        ),
        (
            "resume raw reward mean",
            statistics.fmean(by_name["rollout_resume"].raw_rewards)
            if by_name["rollout_resume"].raw_rewards
            else None,
            statistics.fmean(by_name["inflight_resume"].raw_rewards)
            if by_name["inflight_resume"].raw_rewards
            else None,
        ),
    )
    lines = [
        "| measure | rollout | inflight | inflight - rollout | inflight / rollout |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, rollout_value, inflight_value in rows:
        if rollout_value is None or inflight_value is None:
            lines.append(f"| {label} | - | - | - | - |")
            continue
        ratio = inflight_value / rollout_value if rollout_value != 0 else None
        ratio_text = f"{ratio:.3f}x" if ratio is not None else "-"
        lines.append(
            f"| {label} | {rollout_value:.4f} | {inflight_value:.4f} "
            f"| {inflight_value - rollout_value:+.4f} | {ratio_text} |"
        )
    return lines


def audit(segments: list[Segment], manifest: dict[str, str]) -> list[str]:
    fresh_count = int(manifest["FRESH_ROLLOUTS"])
    resume_count = int(manifest["RESUME_ROLLOUTS"])
    failures: list[str] = []
    for segment in segments:
        if segment.log_path is None:
            failures.append(f"{segment.name}: log not present")
            continue
        if len(segment.checkpoint_paths) != 1:
            failures.append(f"{segment.name}: checkpoint paths={segment.checkpoint_paths}, expected one")
        else:
            checkpoint_path = segment.checkpoint_paths[0]
            if manifest["VALIDATION_NAMESPACE"] not in checkpoint_path:
                failures.append(
                    f"{segment.name}: checkpoint path does not contain validation namespace: {checkpoint_path}"
                )
            if f"-rb-{segment.buffer_type}" not in checkpoint_path:
                failures.append(
                    f"{segment.name}: checkpoint path does not identify {segment.buffer_type}: {checkpoint_path}"
                )
        expected_start = 0 if segment.phase == "fresh" else fresh_count
        expected_count = fresh_count if segment.phase == "fresh" else resume_count
        expected_steps = list(range(expected_start, expected_start + expected_count))
        effective_start = segment.start_rollout_id
        if effective_start is None and segment.rollout_steps:
            effective_start = segment.rollout_steps[0]
        if effective_start != expected_start:
            failures.append(
                f"{segment.name}: start_rollout_id={effective_start}, expected {expected_start}"
            )
        if segment.rollout_steps != expected_steps:
            failures.append(f"{segment.name}: rollout steps={segment.rollout_steps}, expected {expected_steps}")
        if segment.train_steps != expected_steps:
            failures.append(f"{segment.name}: train steps={segment.train_steps}, expected {expected_steps}")
        if not segment.clean_debug_exit:
            failures.append(f"{segment.name}: clean debug exit not found")
        if segment.entry_epoch is None:
            failures.append(f"{segment.name}: RL entry timestamp missing")
        if segment.first_rollout_wall_seconds is None or segment.first_rollout_wall_seconds < 0:
            failures.append(
                f"{segment.name}: invalid entry-to-rollout time {segment.first_rollout_wall_seconds}"
            )
        if segment.first_train_wall_seconds is None or segment.first_train_wall_seconds < 0:
            failures.append(f"{segment.name}: invalid entry-to-train time {segment.first_train_wall_seconds}")
        if len(segment.saves) != expected_count:
            failures.append(f"{segment.name}: replay-buffer saves={len(segment.saves)}, expected {expected_count}")
        if sorted(set(segment.saved_iterations)) != expected_steps:
            failures.append(
                f"{segment.name}: saved iterations={sorted(set(segment.saved_iterations))}, "
                f"expected {expected_steps}"
            )
        if segment.phase == "resume" and not segment.resume_metrics:
            failures.append(f"{segment.name}: resume metrics missing")
        if segment.phase == "resume" and segment.resume_metrics:
            required_resume_metrics = {
                "prepared_batches_restored",
                "warm_prepared_batch_hit",
                "applied_weight_version_restored",
                "current_applied_weight_version",
                "regenerated_active_groups",
                "inflight_groups_restored",
                "inflight_tokens_restored",
            }
            missing_resume_metrics = sorted(required_resume_metrics - segment.resume_metrics.keys())
            if missing_resume_metrics:
                failures.append(f"{segment.name}: missing resume metrics {missing_resume_metrics}")
            else:
                metrics = segment.resume_metrics
                if metrics["prepared_batches_restored"] < 1 or metrics["warm_prepared_batch_hit"] != 1:
                    failures.append(
                        f"{segment.name}: prepared replay batch was not restored and reused: {metrics}"
                    )
                expected_current_version = metrics["applied_weight_version_restored"] + 1
                if metrics["current_applied_weight_version"] != expected_current_version:
                    failures.append(
                        f"{segment.name}: restored weight version was not advanced by the initial push: {metrics}"
                    )
                if segment.buffer_type == "rollout":
                    if metrics["regenerated_active_groups"] < 1:
                        failures.append(f"{segment.name}: rollout replay regenerated no active groups")
                    if metrics["inflight_groups_restored"] != 0 or metrics["inflight_tokens_restored"] != 0:
                        failures.append(f"{segment.name}: rollout replay unexpectedly restored inflight state")
                elif metrics["inflight_groups_restored"] < 1 or metrics["inflight_tokens_restored"] < 1:
                    failures.append(f"{segment.name}: inflight replay restored no partial generation state")
        if segment.phase == "resume" and len(segment.loads) != 1:
            failures.append(f"{segment.name}: replay-buffer loads={len(segment.loads)}, expected 1")
        if (
            segment.phase == "resume"
            and len(segment.loads) == 1
            and segment.loads[0].rollout_id != fresh_count - 1
        ):
            failures.append(
                f"{segment.name}: loaded rollout={segment.loads[0].rollout_id}, expected {fresh_count - 1}"
            )
        if segment.phase == "fresh" and segment.loads:
            failures.append(f"{segment.name}: unexpected replay-buffer load in a fresh run")
        required_metric_series = {
            "raw rewards": segment.raw_rewards,
            "truncated ratios": segment.truncated_ratios,
            "rollout times": segment.rollout_times,
            "step times": segment.step_times,
            "train wait times": segment.train_wait_times,
            "trained staleness means": segment.trained_staleness_means,
            "trained staleness maxes": segment.trained_staleness_maxes,
            "offered staleness maxes": segment.offered_staleness_maxes,
            "bound-exceeded groups": segment.bound_exceeded_groups,
            "train losses": segment.train_losses,
            "gradient norms": segment.grad_norms,
            "learning rates": segment.learning_rates,
        }
        for metric_name, values in required_metric_series.items():
            if len(values) != expected_count:
                failures.append(f"{segment.name}: {metric_name}={len(values)}, expected {expected_count}")
            elif not all(math.isfinite(value) for value in values):
                failures.append(f"{segment.name}: {metric_name} contains a non-finite value")
        invalid_trained_staleness = [
            value for value in segment.trained_staleness_maxes if value < 0 or value > 8
        ]
        if invalid_trained_staleness:
            failures.append(
                f"{segment.name}: trained staleness exceeded [0, 8]: {invalid_trained_staleness}"
            )
        if segment.grad_norms and not any(value > 0 for value in segment.grad_norms):
            failures.append(f"{segment.name}: no positive gradient norm: {segment.grad_norms}")
        if segment.learning_rates and not all(value > 0 for value in segment.learning_rates):
            failures.append(f"{segment.name}: non-positive learning rate: {segment.learning_rates}")
        if segment.missing_config_args:
            failures.append(f"{segment.name}: missing config args {segment.missing_config_args}")

    by_name = {segment.name: segment for segment in segments}
    for buffer_type in ("rollout", "inflight"):
        fresh_paths = by_name[f"{buffer_type}_fresh"].checkpoint_paths
        resume_paths = by_name[f"{buffer_type}_resume"].checkpoint_paths
        if len(fresh_paths) == 1 and len(resume_paths) == 1 and fresh_paths[0] != resume_paths[0]:
            failures.append(
                f"{buffer_type}: fresh/resume checkpoint mismatch: {fresh_paths[0]} != {resume_paths[0]}"
            )
    rollout_paths = by_name["rollout_fresh"].checkpoint_paths
    inflight_paths = by_name["inflight_fresh"].checkpoint_paths
    if len(rollout_paths) == 1 and len(inflight_paths) == 1 and rollout_paths[0] == inflight_paths[0]:
        failures.append(f"rollout/inflight share checkpoint path: {rollout_paths[0]}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("experiments/outputs/training/math/dapo-math-p10-90/qwen3-4b"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = parse_manifest(args.manifest)
    segments = [
        parse_segment(name, manifest[key], buffer_type, phase, args.log_dir)
        for name, key, buffer_type, phase in JOB_SPECS
    ]
    failures = audit(segments, manifest)

    lines = [
        f"# Replay-buffer validation: {manifest['VALIDATION_NAMESPACE']}",
        "",
        f"W&B project: `{manifest['WANDB_PROJECT']}`",
        "",
        *render_segment_table(segments),
        "",
        "## Direct comparison",
        "",
        *render_comparison(segments),
        "",
        "## Resume restoration",
        "",
        *render_resume_table(segments),
        "",
        "## Audit",
        "",
    ]
    lines.extend(f"- FAIL: {failure}" for failure in failures)
    if not failures:
        lines.append(
            "- PASS: all four segments completed the expected fresh/resume steps and persisted replay buffers."
        )
    rendered = "\n".join(lines) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
