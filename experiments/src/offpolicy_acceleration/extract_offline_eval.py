"""Turn offline checkpoint evaluations into an extract ``analyze.py`` can read.

``src/offline_eval/run_eval.sbatch`` evaluates one checkpoint on every AIME year
and writes ``<OUT_DIR>/<benchmark>.jsonl``, one ``PassRateRecord`` per prompt.
That is the path that produces reportable numbers: 4 years x n=16 = 120 problems,
against the in-training eval's single year at n=8. This module stacks those
per-checkpoint directories into the ``run.json`` + ``eval_<benchmark>.npz`` shape
that ``analyze.py`` consumes, so the equivalence protocol runs on the good
measurement rather than on the cheap one.

Three joins have to be right, and each is a way to get a wrong answer quietly:

**Prompts.** Records carry ``rewards``, the per-sample list, not just the
``pass_rate`` mean -- so the rollout level of the bootstrap survives and the
matrix is ``(prompts, samples)``. When a benchmark's records disagree on sample
count (a resumed measurement that was killed mid-prompt), the whole benchmark
falls back to a ``(prompts, 1)`` matrix of ``pass_rate`` and says so, rather than
padding to a rectangle that was never measured.

**Time.** The evaluation job's own clock is meaningless here -- it ran days after
training. Wall-clock comes from the *training* log, by mapping each checkpoint's
step to the elapsed training time at that step, with inter-allocation queue gaps
removed (``log_source.active_elapsed_hours``).

**Steps.** ``run_eval.sbatch`` names each output directory after the checkpoint
(``TAG``), which ends in the ``--save-hf`` rollout id. That trailing number is the
step. A directory whose step has no matching training record is skipped and
reported, never silently placed at time zero.

Usage:

    python -m experiments.src.offpolicy_acceleration.extract_offline_eval \\
      --eval-root   /data/offline_eval \\
      --match       Qwen3-4B-Instruct-2507_async-on-1step \\
      --slurm-log   experiments/outputs/training/.../<job>.log \\
      --out         $WS/offpolicy-study/extracts \\
      --arm on-policy --seed 0 --total-gpus 16 \\
      --base-eval   /data/offline_eval/base_Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from experiments.src.offpolicy_acceleration import log_source
from experiments.src.offpolicy_acceleration.extract_run import (
    RunExtract,
    guard_series_on_eval_grid,
    read_meta,
    read_metric_records,
    resolve_factors,
    total_gpus,
    write_extract,
)

# TAG is the checkpoint path with separators replaced, so it ends in the rollout
# id that --save-hf stamped on the directory (".../hf/120" -> "..._hf_120").
STEP_SUFFIX = re.compile(r"_(?P<step>\d+)$")


def parse_step(directory_name: str) -> int | None:
    match = STEP_SUFFIX.search(directory_name)
    return int(match["step"]) if match else None


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def matrix_from_records(records: list[dict[str, Any]]) -> tuple[np.ndarray, bool]:
    """(prompts, samples) rewards, and whether the per-sample axis is real.

    Sorted by prompt ``index`` so the row order is the benchmark file's order and
    therefore identical across checkpoints and across arms -- which is what makes
    the bootstrap's shared prompt draw a *paired* comparison rather than an
    accidental one.
    """
    ordered = sorted(records, key=lambda r: int(r["index"]))
    sample_counts = {len(r.get("rewards") or []) for r in ordered}

    if len(sample_counts) == 1 and sample_counts != {0}:
        return np.array([[float(x) for x in r["rewards"]] for r in ordered], dtype=float), True
    # Ragged or absent: fall back to the prompt-level mean. Padding to a rectangle
    # would invent samples, and dropping prompts would change the held-out set.
    return np.array([[float(r["pass_rate"])] for r in ordered], dtype=float), False


def collect_checkpoints(eval_root: Path, match: str | None) -> dict[int, Path]:
    """``{step: directory}`` for every evaluated checkpoint under ``eval_root``."""
    found: dict[int, Path] = {}
    for directory in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        if match and match not in directory.name:
            continue
        step = parse_step(directory.name)
        if step is None:
            continue
        found[step] = directory
    return found


def read_all(
    checkpoints: dict[int, Path], base_eval: Path | None
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, bool], list[str]]:
    """Per-benchmark ``{step: (P, N)}``, whether each stayed per-sample, and notes."""
    by_step = dict(checkpoints)
    if base_eval is not None:
        # The base model is step 0 by definition: it is the policy before any
        # optimizer step, and Q0 anchors the whole q_p target ladder.
        assert 0 not in by_step, "step 0 is already present; --base-eval would overwrite it"
        by_step[0] = base_eval

    matrices: dict[str, dict[int, np.ndarray]] = {}
    per_sample: dict[str, bool] = {}
    notes: list[str] = []
    for step, directory in sorted(by_step.items()):
        for path in sorted(directory.glob("*.jsonl")):
            if path.name.endswith(".meta.jsonl"):
                continue
            name = path.stem
            records = load_records(path)
            if not records:
                notes.append(f"step {step}: {name} is empty; skipped")
                continue
            matrix, sampled = matrix_from_records(records)
            matrices.setdefault(name, {})[step] = matrix
            per_sample[name] = per_sample.get(name, True) and sampled
    return matrices, per_sample, notes


def align_benchmarks(
    matrices: dict[str, dict[int, np.ndarray]], per_sample: dict[str, bool]
) -> tuple[dict[str, dict[int, np.ndarray]], list[str]]:
    """Drop the per-sample axis for any benchmark that lost it at some step.

    A benchmark that is ``(P, 16)`` at most steps and ``(P, 1)`` at one cannot be
    stacked, and silently keeping the 16 would make the interval at that step
    narrower than the data supports. Collapsing the whole benchmark to prompt
    means is the honest resolution, and it is recorded.
    """
    notes: list[str] = []
    aligned: dict[str, dict[int, np.ndarray]] = {}
    for name, by_step in matrices.items():
        widths = {matrix.shape[1] for matrix in by_step.values()}
        if len(widths) == 1 and per_sample.get(name, False):
            aligned[name] = by_step
            continue
        if len(widths) > 1:
            notes.append(f"{name}: sample counts differ across steps {sorted(widths)}; using prompt means")
        else:
            notes.append(f"{name}: no per-sample rewards; using prompt means (rollout bootstrap disabled)")
        aligned[name] = {step: matrix.mean(axis=1, keepdims=True) for step, matrix in by_step.items()}

    heights = {name: {matrix.shape[0] for matrix in by_step.values()} for name, by_step in aligned.items()}
    for name, counts in heights.items():
        assert len(counts) == 1, f"{name}: prompt count changes across checkpoints {sorted(counts)}"
    return aligned, notes


def elapsed_by_step(records: list[dict[str, Any]], gap_seconds: float) -> tuple[dict[int, float], dict[str, Any]]:
    """``{training step: elapsed hours}`` from the training log."""
    hours, report = log_source.active_elapsed_hours(records, gap_seconds)
    mapping: dict[int, float] = {}
    for record, elapsed in zip(records, hours, strict=True):
        step = record.get("step")
        if step is not None:
            # Later records for the same step win: a resumed allocation replayed it.
            mapping[int(step)] = elapsed
    return mapping, report


def place_on_time_axis(steps: list[int], step_hours: dict[int, float]) -> tuple[list[int], list[float], list[str]]:
    """Keep only checkpoints whose step exists in the training log."""
    kept_steps, kept_hours, notes = [], [], []
    for step in steps:
        if step in step_hours:
            kept_steps.append(step)
            kept_hours.append(step_hours[step])
        elif step == 0:
            kept_steps.append(step)
            kept_hours.append(0.0)  # the base model, before any training time
        else:
            notes.append(f"step {step}: no training record, so it has no wall-clock; dropped")
    return kept_steps, kept_hours, notes


def build(args: argparse.Namespace) -> tuple[RunExtract, dict[str, dict[int, np.ndarray]]]:
    checkpoints = collect_checkpoints(Path(args.eval_root), args.match)
    assert checkpoints or args.base_eval, (
        f"no checkpoint directories under {args.eval_root}"
        + (f" matching {args.match!r}" if args.match else "")
        + "; run_eval.sbatch names them after the checkpoint, ending in the rollout id"
    )

    matrices, per_sample, notes = read_all(checkpoints, Path(args.base_eval) if args.base_eval else None)
    matrices, align_notes = align_benchmarks(matrices, per_sample)
    notes += align_notes
    assert matrices, "no benchmark jsonl found under the checkpoint directories"

    dump_dir = Path(args.dump_details) if args.dump_details else None
    records = read_metric_records(dump_dir, args.slurm_log)
    step_hours, time_report = elapsed_by_step(records, args.max_step_gap_minutes * 60)

    all_steps = sorted({step for by_step in matrices.values() for step in by_step})
    steps, wall_h, place_notes = place_on_time_axis(all_steps, step_hours)
    notes += place_notes
    assert steps, "no evaluated checkpoint could be placed on the training time axis"

    # Every benchmark must cover every kept step, or the pooled set would change
    # composition from one step to the next.
    for name, by_step in list(matrices.items()):
        missing = [step for step in steps if step not in by_step]
        if missing:
            notes.append(f"{name}: missing at steps {missing}; benchmark dropped")
            del matrices[name]
    assert matrices, "no benchmark covers every step that has a wall-clock"

    meta = read_meta(dump_dir)
    gpus = total_gpus(meta, args.total_gpus)
    factors, sources = resolve_factors(args.factor, args.manifest, meta.get("run_name", ""), meta)
    notes.append(
        f"time base: {time_report['active_h']:.2f} active h over {time_report['n_allocations']} allocation(s); "
        f"{time_report['excluded_h']:.2f} h of inter-allocation gaps excluded"
    )
    notes.append("quality from offline checkpoint evaluation, not the in-training eval")

    extract = RunExtract(
        run_id=args.run_id or (args.match or Path(args.eval_root).name),
        arm=args.arm,
        seed=args.seed,
        run_name=meta.get("run_name", ""),
        start_ts=log_source.run_start_ts(records),
        total_gpus=gpus,
        benchmarks=sorted(matrices),
        eval_steps=steps,
        eval_wall_clock_h=wall_h,
        eval_gpu_hours=[hours * gpus for hours in wall_h],
        logged_means={
            name: [float(matrices[name][step].mean()) for step in steps] for name in sorted(matrices)
        },
        prompt_level={name: True for name in sorted(matrices)},
        guards=_guards_at_steps(records, step_hours, steps),
        lag_steps=[],
        time_report=time_report,
        factors=factors,
        factor_sources=sources,
        notes=notes,
    )
    trimmed = {name: {step: by_step[step] for step in steps} for name, by_step in matrices.items()}
    return extract, trimmed


def _guards_at_steps(
    records: list[dict[str, Any]], step_hours: dict[int, float], steps: list[int]
) -> dict[str, list[float | None]]:
    """Guard series sampled at the training timestamps of the evaluated steps.

    The convergence guards live on the training clock, so they are read at the
    moment each checkpoint was written rather than when it was later evaluated.
    """
    stamps: list[float] = []
    for step in steps:
        matching = [record["ts"] for record in records if record.get("step") == step]
        stamps.append(max(matching) if matching else min(record["ts"] for record in records))
    return guard_series_on_eval_grid(records, stamps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-root", required=True, help="directory holding one subdirectory per evaluated checkpoint")
    p.add_argument("--match", default=None, help="substring selecting this run's checkpoint directories")
    p.add_argument("--base-eval", default=None, help="offline eval of the base model; placed at step 0 for Q0")
    p.add_argument("--out", required=True, type=Path, help="extract root, read by analyze.py")
    p.add_argument("--arm", required=True, help="arm label shared by the seeds of one configuration")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--slurm-log", type=Path, action="append", default=[], help="repeat once per allocation")
    p.add_argument("--dump-details", default=None, help="the training run's dump dir, if it still exists")
    p.add_argument("--factor", action="append", default=[], help="K=V, repeatable")
    p.add_argument("--manifest", type=Path, default=None, help="experiments/sweep.py manifest to join on")
    p.add_argument("--total-gpus", type=int, default=None)
    p.add_argument("--max-step-gap-minutes", type=float, default=log_source.DEFAULT_GAP_SECONDS / 60)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    extract, matrices = build(args)
    run_dir = write_extract(Path(args.out), extract, matrices, {})

    widths = {name: next(iter(by_step.values())).shape for name, by_step in matrices.items()}
    print(f"run        {extract.run_id}  (arm={extract.arm}, seed={extract.seed})")
    print(f"checkpoints {len(extract.eval_steps)} steps: {extract.eval_steps}")
    print(f"benchmarks {', '.join(f'{name} {shape}' for name, shape in sorted(widths.items()))}")
    pooled = sum(shape[0] for shape in widths.values())
    print(f"pooled     {pooled} held-out prompts if analysed together")
    print(f"time       {extract.eval_wall_clock_h[0]:.2f}..{extract.eval_wall_clock_h[-1]:.2f} h")
    for note in extract.notes:
        print(f"  note: {note}")
    print(f"wrote      {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
