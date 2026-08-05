"""Turn one run's ``--dump-details`` directory into a compact, torch-free extract.

This is the expensive half, split from the analysis for the same reason
``difficulty_filter`` splits measurement from filtering: reading the eval dumps
needs ``torch`` and the training image, while every question asked of the numbers
afterwards needs only numpy. Run this once per run, inside the container; run
``analyze.py`` and ``figures.py`` as often as you like, anywhere.

What it recovers, and from where:

| quantity | source |
|---|---|
| per-prompt, per-rollout eval rewards | ``rollout_data/eval_<rid>.pt`` (needs ``--dump-details``) |
| evaluation wall-clock | per-record deltas with inter-allocation gaps removed (``log_source.active_elapsed_hours``) |
| collapse guards (entropy, KL, reward, truncation) | ``dashboard/metrics.jsonl`` |
| realized policy lag P(L) | per-sample ``weight_versions`` in ``rollout_data/<rid>.pt`` |
| GPU count for GPU-hours | ``meta.json`` args snapshot |

The eval dump concatenates every eval dataset into one file with per-dataset
sample indices restarting at zero and no dataset label on the samples, so the
blocks are split at each index reset and then **matched to benchmarks by
reproducing the logged ``eval/<name>`` mean**. Matching on the recorded number
rather than on declaration order is what makes it safe when two benchmarks have
the same prompt count -- as aime24 and aime25 do, at 30 each.

Factors (the axes of the robustness surface) are *not* in ``meta.json``: its args
snapshot carries 16 keys, none of them ``max_weight_staleness``,
``num_steps_per_rollout``, ``pause_generation_mode`` or the seed. They are taken
from ``--factor K=V`` or from a ``experiments/sweep.py`` manifest, and every
factor records where it came from, so an analysis can never quietly attribute a
run to the wrong cell of the design.

Usage (inside the training container):

    python -m experiments.src.offpolicy_acceleration.extract_run \\
      --dump-details /ckpt/training/math/dapo-math/Qwen3-4B/<config-tag>/dump \\
      --out          /lustre/.../offpolicy-study/extracts \\
      --arm          stale2-retract --seed 0 \\
      --factor max_weight_staleness=2 --factor pause_generation_mode=retract
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.src.offpolicy_acceleration import log_source

# Metrics pulled onto the evaluation grid for the convergence guards and for the
# realized-lag cross-check. Prefixes are matched exactly; anything missing is
# recorded as absent rather than defaulted, because "the guard could not be
# evaluated" and "the guard passed" are different findings.
GUARD_METRICS = (
    # collapse guards
    "train/entropy_loss",  # identically 0 unless --observe-training-entropy is passed
    "rollout/raw_reward",
    "rollout/truncated_ratio",
    "rollout/repetition_frac",
    "train/kl_loss",
    "train/grad_norm",
    # train/rollout mismatch and off-policy drift
    "train/train_rollout_logprob_abs_diff",
    "train/train_rollout_kl",
    "train/ppo_kl",
    "train/ess_ratio",
    "train/pg_clipfrac",
    "train/tis_abs",
    "train/tis_clipfrac",
    # realized staleness and wasted generation
    "rollout/weight_version/min",
    "rollout/weight_version/max",
    "rollout/weight_version/mean",
    "rollout/weight_version/mixed_version_ratio",
    "rollout/fully_async/avg_staleness",
    "rollout/fully_async/max_staleness",
    "rollout/fully_async/stale_groups_recycled",
    "rollout/fully_async/aborted_groups_recycled",
    # throughput, for the wall-clock story behind a speedup
    "rollout/response_len/mean",
    "perf/rollout_time",
    "perf/train_wait_time",
    "perf/wait_time_ratio",
    "perf/tokens_per_gpu_per_sec",
)

EVAL_STEP_KEY = "eval/step"
BENCHMARK_MEAN_RE = re.compile(r"^eval/(?P<name>[^/-]+)$")


@dataclass
class RunExtract:
    """Everything ``analyze.py`` needs about one training run."""

    run_id: str
    arm: str
    seed: int
    run_name: str
    start_ts: float
    total_gpus: int
    benchmarks: list[str]
    eval_steps: list[int]
    eval_wall_clock_h: list[float]
    eval_gpu_hours: list[float]
    logged_means: dict[str, list[float]]
    prompt_level: dict[str, bool]
    guards: dict[str, list[float | None]]
    lag_steps: list[int]
    time_report: dict[str, Any]
    factors: dict[str, str]
    factor_sources: dict[str, str]
    notes: list[str]


# --------------------------------------------------------------------------
# dashboard streams
# --------------------------------------------------------------------------


def read_meta(dump_dir: Path | None) -> dict[str, Any]:
    """The dashboard's run metadata, or an empty stand-in in log-only mode.

    ``meta.json`` is the only place a run records its own start time and GPU
    geometry. Without it (``--slurm-log`` alone) the wall-clock origin becomes the
    first metric line and ``--total-gpus`` has to be supplied by hand.
    """
    if dump_dir is None:
        return {}
    path = dump_dir / "dashboard" / "meta.json"
    assert path.is_file(), (
        f"{path} is missing: the run was not started with --use-miles-dashboard. Pass --slurm-log "
        "to read the metric stream from the job log instead, together with --total-gpus."
    )
    return json.loads(path.read_text())


def read_metric_records(dump_dir: Path | None, slurm_logs: list[Path]) -> list[dict[str, Any]]:
    """Every ``tracking.log`` payload, with its wall-clock ``ts``.

    Prefers ``dashboard/metrics.jsonl`` -- a flat, unpartitioned stream, so one
    pass is the whole file -- and falls back to parsing the job log, which carries
    the identical payloads with the identical timestamps. The two are
    interchangeable by construction: both are copies of what ``tracking.log`` was
    handed.
    """
    if dump_dir is not None:
        path = dump_dir / "dashboard" / "metrics.jsonl"
        if path.is_file():
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            return log_source.stitch([[json.loads(line) for line in lines]])
    assert slurm_logs, "no dashboard/metrics.jsonl and no --slurm-log: there is no metric stream to read"
    return log_source.stitch([log_source.merge_step_records(log_source.parse_log(p)) for p in slurm_logs])


def eval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records carrying an evaluation, in step order (ties broken by timestamp)."""
    hits = [r for r in records if r.get("step_key") == EVAL_STEP_KEY or EVAL_STEP_KEY in r.get("metrics", {})]
    return sorted(hits, key=lambda r: (r["metrics"].get(EVAL_STEP_KEY, r.get("step") or 0), r["ts"]))


def benchmark_names(eval_recs: list[dict[str, Any]]) -> list[str]:
    """Benchmark keys as logged, e.g. ``eval/aime24`` -> ``aime24``.

    Derived from the metrics themselves rather than from the recipe, so a run
    whose ``--eval-prompt-data`` was edited mid-study is described by what it
    actually evaluated.
    """
    names: list[str] = []
    for record in eval_recs:
        for key in record["metrics"]:
            match = BENCHMARK_MEAN_RE.match(key)
            if match and match["name"] not in names and match["name"] != "step":
                names.append(match["name"])
    return names


def guard_series_on_eval_grid(records: list[dict[str, Any]], eval_ts: list[float]) -> dict[str, list[float | None]]:
    """Last observed value of each guard metric at or before each evaluation.

    Guards are logged on the training cadence and evaluations on their own; a
    last-value-carried-forward join is the honest alignment, since the question
    a guard answers is "what state was the run in when this evaluation was
    taken".
    """
    series: dict[str, list[float | None]] = {}
    for metric in GUARD_METRICS:
        observations = [(r["ts"], r["metrics"][metric]) for r in records if metric in r.get("metrics", {})]
        observations.sort()
        if not observations:
            continue
        stamps = np.array([ts for ts, _ in observations])
        values = np.array([float(v) for _, v in observations])
        aligned: list[float | None] = []
        for ts in eval_ts:
            position = int(np.searchsorted(stamps, ts, side="right")) - 1
            aligned.append(float(values[position]) if position >= 0 else None)
        series[metric] = aligned
    return series


def total_gpus(meta: dict[str, Any], override: int | None) -> int:
    """GPUs held by the job, for the GPU-hours axis.

    Rollout and training GPUs are both charged: an asynchronous run holds its
    rollout GPUs for the whole job whether or not they are busy, and a
    GPU-efficiency claim that ignores them would credit asynchrony with hardware
    it is actually occupying.
    """
    if override is not None:
        return override
    args = meta.get("args", {})
    actor = int(args.get("actor_num_nodes", 0)) * int(args.get("actor_num_gpus_per_node", 0))
    rollout = 0 if args.get("colocate") else int(args.get("rollout_num_gpus", 0))
    assert actor + rollout > 0, (
        "cannot derive the GPU count from meta.json; pass --total-gpus explicitly "
        f"(args snapshot: {args})"
    )
    return actor + rollout


# --------------------------------------------------------------------------
# eval dumps -> per-prompt reward matrices
# --------------------------------------------------------------------------


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Sample dicts from one dump. ``Sample.to_dict`` is a plain ``__dict__``
    copy, so the payload is JSON-ish and needs no miles import to interpret."""
    import torch  # deferred: the analysis half must stay importable without torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload["samples"])


def split_eval_blocks(samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split the concatenated eval dump at each ``index`` reset.

    ``eval_rollout_single_dataset`` numbers its samples from zero per dataset and
    the driver concatenates the datasets, so an index that does not increase is
    exactly a dataset boundary.
    """
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous = -1
    for sample in samples:
        index = int(sample["index"])
        if index <= previous and current:
            blocks.append(current)
            current = []
        current.append(sample)
        previous = index
    if current:
        blocks.append(current)
    return blocks


def block_matrix(block: list[dict[str, Any]], n_samples_per_prompt: int) -> tuple[np.ndarray, int]:
    """(P, N) rewards for one benchmark, plus the count of ``None`` rewards.

    ``None`` becomes 0.0, matching ``log_eval_rollout_data``: an errored or
    aborted trial is a failed attempt, and dropping it instead would quietly
    grade the arm on the subset of prompts it managed to finish.
    """
    ordered = sorted(block, key=lambda s: int(s["index"]))
    assert len(ordered) % n_samples_per_prompt == 0, (
        f"eval block of {len(ordered)} samples is not divisible by n_samples_per_eval_prompt "
        f"{n_samples_per_prompt}; pass the value this run actually used"
    )
    n_prompts = len(ordered) // n_samples_per_prompt
    rewards = np.zeros((n_prompts, n_samples_per_prompt))
    n_none = 0
    for position, sample in enumerate(ordered):
        reward = sample.get("reward")
        if not isinstance(reward, (int, float)):
            n_none += 1
            reward = 0.0
        rewards[position // n_samples_per_prompt, position % n_samples_per_prompt] = float(reward)
    return rewards, n_none


def match_blocks_to_benchmarks(
    blocks: list[list[dict[str, Any]]],
    logged: dict[str, float],
    n_samples_per_prompt: int,
    tolerance: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Assign each block to the benchmark whose logged mean it reproduces.

    Declaration order is not trusted as the primary key: it is right in practice
    but unverifiable from the dump, whereas the recomputed mean is a checksum
    that also proves the block was read correctly. A block that matches nothing
    is an error, not a silently-dropped benchmark.
    """
    matrices: dict[str, np.ndarray] = {}
    remaining = dict(logged)
    for block in blocks:
        rewards, _ = block_matrix(block, n_samples_per_prompt)
        mean = float(rewards.mean())
        best = min(remaining, key=lambda name: abs(remaining[name] - mean), default=None)
        assert best is not None, f"eval dump has more blocks than logged benchmarks ({list(logged)})"
        gap = abs(remaining[best] - mean)
        assert gap <= max(tolerance, 1e-4), (
            f"eval block mean {mean:.6f} matches no logged benchmark (closest {best} at "
            f"{remaining[best]:.6f}, gap {gap:.2e}); the dump and the metrics stream disagree"
        )
        matrices[best] = rewards
        del remaining[best]
    return matrices


def extract_eval_matrices(
    dump_dir: Path, eval_recs: list[dict[str, Any]], n_samples_per_prompt: int
) -> tuple[dict[str, dict[int, np.ndarray]], list[str]]:
    """Per-benchmark ``{step: (P, N)}`` matrices, plus notes on what was missing."""
    matrices: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    notes: list[str] = []
    for record in eval_recs:
        step = int(record["metrics"].get(EVAL_STEP_KEY, record.get("step") or 0))
        path = dump_dir / "rollout_data" / f"eval_{step}.pt"
        if not path.is_file():
            notes.append(f"step {step}: no eval dump at {path.name}; falling back to the logged mean")
            continue
        logged = {
            match["name"]: float(value)
            for key, value in record["metrics"].items()
            if (match := BENCHMARK_MEAN_RE.match(key)) and match["name"] != "step"
        }
        for name, rewards in match_blocks_to_benchmarks(split_eval_blocks(load_samples(path)), logged, n_samples_per_prompt).items():
            matrices[name][step] = rewards
    return dict(matrices), notes


# --------------------------------------------------------------------------
# realized policy lag
# --------------------------------------------------------------------------


def _numeric_versions(sample: dict[str, Any]) -> list[int]:
    return [int(v) for v in (sample.get("weight_versions") or []) if str(v).isdigit()]


def realized_lag(dump_dir: Path, max_steps: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-sample realized lag from the training dumps: (lags, step_of_each_lag, notes).

    Lag is ``current_version - oldest_version_the_sample_was_generated_under``.
    The trainer's own ``current`` is the engine's version at drain time, which is
    not dumped; the closest recoverable proxy is the newest version seen anywhere
    in the same rollout batch. It agrees with the logged
    ``rollout/fully_async/avg_staleness`` whenever the batch contains at least
    one fresh sample, which is the normal case -- ``analyze.py`` reports both so
    the proxy is checkable rather than assumed.

    Two censorings apply and are stated in the notes, not corrected for: groups
    recycled for exceeding ``--max-weight-staleness`` never reach the dump, so
    the distribution is truncated at the bound; and a sample whose versions are
    all non-numeric contributes nothing.
    """
    rollout_dir = dump_dir / "rollout_data"
    paths = sorted(
        (p for p in rollout_dir.glob("*.pt") if not p.stem.startswith("eval_")),
        key=lambda p: int(p.stem),
    )
    if max_steps is not None:
        paths = paths[:max_steps]

    lags: list[int] = []
    steps: list[int] = []
    notes: list[str] = []
    for path in paths:
        samples = load_samples(path)
        versions = [_numeric_versions(s) for s in samples]
        newest = max((v[-1] for v in versions if v), default=None)
        if newest is None:
            notes.append(f"step {path.stem}: no numeric weight_versions; lag not recoverable")
            continue
        for version_list in versions:
            if version_list:
                lags.append(max(newest - min(version_list), 0))
                steps.append(int(path.stem))
    return np.array(lags, dtype=np.int32), np.array(steps, dtype=np.int32), notes


# --------------------------------------------------------------------------
# factors
# --------------------------------------------------------------------------


def resolve_factors(
    explicit: list[str], manifest: Path | None, run_name: str, meta: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge factor sources, recording provenance for each key.

    Priority: explicit ``--factor`` beats a sweep manifest, which beats the
    ``meta.json`` args snapshot. Nothing is inferred from the run name -- the
    config tag is built from a subset of the knobs and reading factors out of it
    would invent values the run never had.
    """
    factors: dict[str, str] = {}
    sources: dict[str, str] = {}

    for key, value in (meta.get("args") or {}).items():
        factors[key] = str(value)
        sources[key] = "meta.args"

    if manifest is not None:
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            tag = entry.get("env", {}).get("CONFIG_TAG", "")
            if tag and tag in run_name:
                for key, value in entry["env"].items():
                    factors[key] = str(value)
                    sources[key] = f"manifest:{manifest.name}"
                break

    for item in explicit:
        assert "=" in item, f"--factor expects K=V, got {item!r}"
        key, value = item.split("=", 1)
        factors[key] = value
        sources[key] = "cli"

    return factors, sources


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _fallback_run_id(dump_dir: Path | None, slurm_logs: list[Path]) -> str:
    """Name the extract after the config directory, or the first job log."""
    if dump_dir is not None:
        return dump_dir.parent.name
    assert slurm_logs
    return slurm_logs[0].stem


def build_extract(args: argparse.Namespace) -> tuple[RunExtract, dict[str, dict[int, np.ndarray]], dict[str, np.ndarray]]:
    dump_dir = Path(args.dump_details) if args.dump_details else None
    meta = read_meta(dump_dir)
    records = read_metric_records(dump_dir, args.slurm_log)
    wall_by_ts, time_report = log_source.active_elapsed_hours(records, args.max_step_gap_minutes * 60)
    elapsed_of = {id(record): hours for record, hours in zip(records, wall_by_ts, strict=True)}
    eval_recs = eval_records(records)
    assert eval_recs, "the metric stream carries no eval/* records; nothing to measure"

    names = benchmark_names(eval_recs)
    steps = [int(r["metrics"].get(EVAL_STEP_KEY, r.get("step") or 0)) for r in eval_recs]
    stamps = [float(r["ts"]) for r in eval_recs]
    start_ts = log_source.run_start_ts(records)
    # Never `ts - meta.start_ts`: the dashboard rewrites meta.json on every resume
    # while metrics.jsonl appends, so that difference goes negative for every
    # allocation but the last. active_elapsed_hours is resume-safe by construction.
    wall_h = [elapsed_of[id(r)] for r in eval_recs]
    gpus = total_gpus(meta, args.total_gpus)

    notes: list[str] = [
        f"time base: {time_report['active_h']:.2f} active h over {time_report['n_allocations']} allocation(s); "
        f"{time_report['excluded_h']:.2f} h of inter-allocation gaps excluded "
        f"(span {time_report['span_h']:.2f} h)"
    ]
    matrices: dict[str, dict[int, np.ndarray]] = {}
    if dump_dir is not None:
        matrices, notes = extract_eval_matrices(dump_dir, eval_recs, args.n_samples_per_eval_prompt)
    else:
        notes.append(
            "log-only mode: no per-prompt eval rewards, so the bootstrap can only resample training seeds"
        )
    logged = {name: [float(r["metrics"].get(f"eval/{name}", np.nan)) for r in eval_recs] for name in names}

    lag_arrays: dict[str, np.ndarray] = {}
    if args.with_lag and dump_dir is not None:
        lags, lag_steps, lag_notes = realized_lag(dump_dir, args.max_lag_steps)
        lag_arrays = {"lag": lags, "lag_step": lag_steps}
        notes += lag_notes
    else:
        proxy = log_source.lag_from_weight_versions(records)
        if proxy:
            lag_arrays = {
                "lag_step": np.array([r["step"] for r in proxy], dtype=np.int32),
                "lag_bracket_lo": np.array([r["lag_min"] for r in proxy]),
                "lag_bracket_hi": np.array([r["lag_max"] for r in proxy]),
                "avg_staleness": np.array([r["avg_staleness"] for r in proxy]),
                "mixed_version_ratio": np.array([r["mixed_version_ratio"] for r in proxy]),
            }
        notes.append("realized lag is the logged weight-version bracket, not the per-sample distribution")

    factors, sources = resolve_factors(args.factor, args.manifest, meta.get("run_name", ""), meta)

    extract = RunExtract(
        run_id=args.run_id or meta.get("run_name") or _fallback_run_id(dump_dir, args.slurm_log),
        arm=args.arm,
        seed=args.seed,
        run_name=meta.get("run_name", ""),
        start_ts=start_ts,
        total_gpus=gpus,
        benchmarks=names,
        eval_steps=steps,
        eval_wall_clock_h=wall_h,
        eval_gpu_hours=[h * gpus for h in wall_h],
        logged_means=logged,
        prompt_level={name: name in matrices for name in names},
        guards=guard_series_on_eval_grid(records, stamps),
        lag_steps=sorted({int(s) for s in lag_arrays.get("lag_step", [])}),
        time_report=time_report,
        factors=factors,
        factor_sources=sources,
        notes=notes,
    )
    return extract, matrices, lag_arrays


def write_extract(
    out_dir: Path,
    extract: RunExtract,
    matrices: dict[str, dict[int, np.ndarray]],
    lag_arrays: dict[str, np.ndarray],
) -> Path:
    run_dir = out_dir / extract.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(asdict(extract), indent=1))

    for name, by_step in matrices.items():
        steps = sorted(by_step)
        stacked = np.stack([by_step[s] for s in steps]) if steps else np.zeros((0, 0, 0))
        np.savez_compressed(run_dir / f"eval_{name}.npz", steps=np.array(steps), rewards=stacked)
    if lag_arrays:
        np.savez_compressed(run_dir / "lag.npz", **lag_arrays)
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-details", default=None, help="the run's --dump-details directory")
    p.add_argument(
        "--slurm-log",
        type=Path,
        action="append",
        default=[],
        help="experiments/outputs/training/.../<job>.log; repeat once per allocation of a resumed run. "
        "The metric source when there is no dump directory",
    )
    p.add_argument(
        "--max-step-gap-minutes",
        type=float,
        default=log_source.DEFAULT_GAP_SECONDS / 60,
        help="a hole longer than this in the metric stream is an inter-allocation gap, not training time",
    )
    p.add_argument("--out", required=True, help="extract root; one subdirectory per run")
    p.add_argument("--arm", required=True, help="arm label shared by the seeds of one configuration")
    p.add_argument("--seed", type=int, required=True, help="training seed, as passed to --seed")
    p.add_argument("--run-id", default=None, help="defaults to meta.json run_name")
    p.add_argument("--factor", action="append", default=[], help="K=V, repeatable; highest priority")
    p.add_argument("--manifest", type=Path, default=None, help="experiments/outputs/sweeps/<name>.jsonl to join on")
    p.add_argument("--n-samples-per-eval-prompt", type=int, default=16, help="must match the run's setting")
    p.add_argument("--total-gpus", type=int, default=None, help="override the GPU count for the GPU-hours axis")
    p.add_argument("--with-lag", action="store_true", help="also read every training dump for the realized lag P(L)")
    p.add_argument("--max-lag-steps", type=int, default=None, help="cap the training dumps read for the lag")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    extract, matrices, lag_arrays = build_extract(args)
    run_dir = write_extract(Path(args.out), extract, matrices, lag_arrays)

    print(f"run        {extract.run_id}  (arm={extract.arm}, seed={extract.seed})")
    print(f"gpus       {extract.total_gpus}")
    print(f"time       {extract.time_report['active_h']:.2f} active h, "
          f"{extract.time_report['n_allocations']} allocation(s), "
          f"{extract.time_report['excluded_h']:.2f} h excluded")
    print(f"evals      {len(extract.eval_steps)} points, steps {extract.eval_steps[:3]}...{extract.eval_steps[-1:]}")
    print(f"benchmarks {', '.join(f'{n}({"per-prompt" if extract.prompt_level[n] else "mean-only"})' for n in extract.benchmarks)}")
    print(f"guards     {', '.join(sorted(extract.guards)) or 'none logged'}")
    for note in extract.notes:
        print(f"  note: {note}")
    print(f"wrote      {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
