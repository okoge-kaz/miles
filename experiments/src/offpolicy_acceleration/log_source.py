"""Recover the metric stream from a plain Slurm job log.

miles logs every ``tracking.log`` payload to stdout before it ships it anywhere,
as one line per step per stream:

    [2026-08-04 21:58:08.454 rollout_manager]  metrics.py:79   - perf 0:   {...}
    [2026-08-04 21:54:45.347 rollout_manager]  metrics.py:53   - eval 0:   {...}
    [2026-08-04 22:01:46.242 actor_cell0_rank0] log_utils.py:460 - step 0:  {...}
    [2026-08-04 22:01:46.242 actor_cell0_rank0] log_utils.py:... - rollout 0: {...}

so ``experiments/outputs/training/.../<job>.log`` is a complete, timestamped copy
of everything wandb received -- and it exists for every run ever launched here,
including the ones started before ``--use-miles-dashboard``. That makes it the
most portable source for the study, and the fallback whenever a dump directory
has been cleaned up.

Two things it does **not** carry, and no amount of parsing will recover:

* **per-prompt eval rewards.** ``log_eval_rollout_data`` reduces each benchmark
  to a single mean before logging, so the paired-over-prompts bootstrap needs the
  ``--dump-details`` eval dumps. From the log alone the interval can only be taken
  across training seeds.
* **the realized-lag distribution.** ``rollout/weight_version/{mean,median,min,max}``
  are statistics of the *absolute* oldest weight version per rollout batch, not of
  the lag; the lag needs a reference version at drain time. See
  ``lag_from_weight_versions`` for what can and cannot be reconstructed.

Ray's log deduplication (``[repeated 126x across cluster]``) collapses identical
lines across workers. Metric lines carry a step id so they are never identical
and always survive; **sglang's own ``Decode batch, ...`` lines do not** -- they
are per-engine, frequently identical, and therefore lossy in the log. Engine-side
series have to come from the dashboard's scraper instead.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# "[2026-08-04 21:58:08.454 rollout_manager] file.py:79 - <kind> <id>: {<dict>}"
LINE = re.compile(
    r"\[(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) [^\]]*\]\s*"
    r"\S+:\d+\s*-\s*(?P<kind>eval|perf|rollout|step) (?P<id>\d+): (?P<payload>\{.*\})\s*$"
)
NUMBER = re.compile(r"'(?P<key>[^']+)':\s*(?P<value>-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf|-inf))")

# Which log line each metric stream comes from, and the step key it is logged
# against -- the same pairing tracking.log uses, so records parsed from a log and
# records read from dashboard/metrics.jsonl are interchangeable downstream.
KIND_STEP_KEY = {
    "eval": "eval/step",
    "perf": "rollout/step",
    "rollout": "rollout/step",
    "step": "train/step",
}


def parse_payload(payload: str) -> dict[str, Any]:
    """Parse the logged dict, tolerating ``nan``/``inf`` that ``literal_eval`` rejects."""
    try:
        parsed = ast.literal_eval(payload)
        if isinstance(parsed, dict):
            return {k: v for k, v in parsed.items() if isinstance(v, (int, float))}
    except (ValueError, SyntaxError):
        pass
    return {m["key"]: float(m["value"]) for m in NUMBER.finditer(payload)}


def parse_log(path: Path) -> list[dict[str, Any]]:
    """Metric records in the shape ``extract_run`` expects from ``metrics.jsonl``.

    Timestamps are naive local time as miles printed them; they are converted to
    epoch seconds with the reader's local timezone. Only *differences* are ever
    used downstream (wall-clock since run start), so a timezone that disagrees
    with the cluster's shifts every stamp equally and cancels.
    """
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line in _iter_lines(path):
        match = LINE.search(ANSI.sub("", line))
        if not match:
            continue
        key = (match["kind"], int(match["id"]))
        if key in seen:
            continue  # a second training cell logging the same step
        seen.add(key)
        metrics = parse_payload(match["payload"])
        if not metrics:
            continue
        step_key = KIND_STEP_KEY[match["kind"]]
        metrics.setdefault(step_key, int(match["id"]))
        records.append(
            {
                "ts": datetime.strptime(match["stamp"], "%Y-%m-%d %H:%M:%S.%f").timestamp(),
                "step_key": step_key,
                "step": int(match["id"]),
                "metrics": metrics,
            }
        )
    records.sort(key=lambda r: r["ts"])
    return records


def merge_step_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold the ``perf``/``rollout``/``step`` lines of one id into one record.

    They are three separate log lines for the same optimizer step, and every
    consumer downstream wants them as one row keyed by step.
    """
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record["step_key"], record["step"])
        if key not in merged:
            merged[key] = {**record, "metrics": dict(record["metrics"])}
        else:
            merged[key]["metrics"].update(record["metrics"])
            merged[key]["ts"] = max(merged[key]["ts"], record["ts"])
    return sorted(merged.values(), key=lambda r: r["ts"])


def run_start_ts(records: list[dict[str, Any]]) -> float:
    """Earliest timestamp in the log.

    This is the first *metric* line, not job submission: model load, checkpoint
    conversion and engine warmup all precede it. It therefore understates
    wall-clock time to a target by a constant startup cost, identical across arms
    with the same model and geometry -- which is why every arm in a comparison
    must share them, and why the absolute numbers are only comparable within a
    study, not against another paper's.
    """
    assert records, "no metric lines found in the log"
    return min(r["ts"] for r in records)


def lag_from_weight_versions(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Best-effort realized lag per rollout step, from the logged version stats.

    ``rollout/weight_version/{min,mean,max}`` are the *absolute* oldest weight
    version of each sample in the batch, aggregated. Lag needs a reference: the
    engine's current version when the batch was drained. The running maximum of
    ``weight_version/max`` over the run is the best available stand-in, so

        lag_min ~ running_max - weight_version/max
        lag_max ~ running_max - weight_version/min

    bracket the true per-batch lag. This is a bracket, not the distribution: it
    cannot produce percentiles, and it silently reads zero whenever the batch
    happens to contain no fresh sample. The exact per-batch mean is
    ``rollout/fully_async/avg_staleness``, which miles only computes when
    ``--max-weight-staleness`` is set (``fully_async_rollout.py:202``) -- run the
    unbounded arm with a bound so large it never binds rather than unset, or this
    is all there is.
    """
    rows = []
    running_max = None
    for record in records:
        metrics = record["metrics"]
        if "rollout/weight_version/max" not in metrics:
            continue
        newest = float(metrics["rollout/weight_version/max"])
        running_max = newest if running_max is None else max(running_max, newest)
        rows.append(
            {
                "step": float(record["step"]),
                "lag_min": running_max - newest,
                "lag_max": running_max - float(metrics.get("rollout/weight_version/min", newest)),
                "lag_mean_proxy": running_max - float(metrics.get("rollout/weight_version/mean", newest)),
                "mixed_version_ratio": float(metrics.get("rollout/weight_version/mixed_version_ratio", 0.0)),
                "avg_staleness": float(metrics.get("rollout/fully_async/avg_staleness", float("nan"))),
            }
        )
    return rows


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open(errors="replace") as handle:
        yield from handle
