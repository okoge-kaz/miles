"""Run the time-to-on-policy-equivalence protocol over a directory of extracts.

Reads what ``extract_run.py`` wrote, groups runs into arms by their ``arm``
label, and applies ``equivalence.py`` end to end:

    on-policy convergence  ->  Q_on*
    paired bootstrap       ->  LCB[Q_m(t) - Q_on*]  ->  tau_m(delta)
    target ladder q_p      ->  tau_m(q_p), S_m(p)   ->  the speedup profile
    realized lag           ->  P(L) next to the configured bound

and writes ``results.json`` (everything, for ``figures.py``) plus
``summary.csv`` (one tidy row per arm x target, for a paper table).

Numpy only -- no torch, no cluster, no container. Run it on a login node:

    uv run --with numpy python -m experiments.src.offpolicy_acceleration.analyze \\
      --extracts /lustre/.../offpolicy-study/extracts \\
      --benchmark aime24 --reference-arm on-policy \\
      --out /lustre/.../offpolicy-study/results/aime24

The reference arm is named explicitly rather than inferred. "Which run is the
on-policy baseline" is a claim about the experiment design, and a script that
guesses it can silently compare every arm against the wrong thing.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.src.offpolicy_acceleration import equivalence as eq


# --------------------------------------------------------------------------
# extracts -> arms
# --------------------------------------------------------------------------


def load_runs(extract_root: Path) -> list[dict[str, Any]]:
    runs = []
    for run_json in sorted(extract_root.glob("*/run.json")):
        record = json.loads(run_json.read_text())
        record["_dir"] = run_json.parent
        runs.append(record)
    assert runs, f"no run.json under {extract_root}; run extract_run.py first"
    return runs


def rewards_for(run: dict[str, Any], benchmark: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """(steps, rewards, prompt_level) for one run and one benchmark.

    Falls back to the logged scalar mean, shaped ``(T, 1, 1)``, when the eval
    dumps are absent. The fallback is not silent: ``prompt_level`` propagates all
    the way into the results file, and an arm that lacks per-prompt data cannot
    contribute prompt-level uncertainty no matter how many replicates are drawn.
    """
    npz_path = run["_dir"] / f"eval_{benchmark}.npz"
    if npz_path.is_file():
        payload = np.load(npz_path)
        return payload["steps"], payload["rewards"], True
    means = run["logged_means"].get(benchmark)
    assert means, f"{run['run_id']}: benchmark {benchmark!r} not in {list(run['logged_means'])}"
    return np.array(run["eval_steps"]), np.array(means, dtype=float).reshape(-1, 1, 1), False


def pooled_rewards_for(run: dict[str, Any], benchmarks: list[str]) -> tuple[np.ndarray, np.ndarray, bool]:
    """Concatenate several benchmarks along the prompt axis into one held-out set.

    Prompt sampling dominates the uncertainty on a 30-problem benchmark, and it
    shrinks only as 1/sqrt(P), so pooling four AIME years is a 2x tightening that
    nothing else on the menu can match -- most importantly for ``tau(q_p)``, which
    is a threshold crossing on raw Q and gets no help from the paired design.

    Pooling changes the estimand: the score becomes a prompt-count-weighted mean
    over the years, mixing their difficulty. That is legitimate *provided the
    composition is fixed in advance and applied identically to every arm*, which
    concatenating a fixed list of files enforces. Per-benchmark runs remain
    available as a consistency check -- pass one name.

    Mean-only benchmarks cannot be pooled: without prompt counts there is no
    correct weight, and averaging the means would silently assume equal sizes.
    """
    if len(benchmarks) == 1:
        return rewards_for(run, benchmarks[0])

    parts = [rewards_for(run, name) for name in benchmarks]
    for name, (_, _, prompt_level) in zip(benchmarks, parts, strict=True):
        assert prompt_level, (
            f"{run['run_id']}: {name} has no per-prompt rewards, so it cannot be pooled. "
            "Pooling weights by prompt count, which the logged mean does not carry."
        )
    steps = parts[0][0]
    for name, (other, _, _) in zip(benchmarks, parts, strict=True):
        assert np.array_equal(steps, other), f"{run['run_id']}: {name} was evaluated on a different step grid"
    samples = {matrix.shape[2] for _, matrix, _ in parts}
    assert len(samples) == 1, f"{run['run_id']}: benchmarks disagree on samples per prompt {samples}"
    return steps, np.concatenate([matrix for _, matrix, _ in parts], axis=1), True


def build_arm(
    runs: list[dict[str, Any]], benchmarks: list[str], name: str, smooth_window: int = 1
) -> tuple[eq.ArmSamples, dict[str, np.ndarray]]:
    """Stack one arm's seeds on their common step grid.

    Seeds are intersected rather than padded: a seed that died early contributes
    the steps it reached and the arm's curve stops where its shortest seed
    stopped. Padding it forward would extend the arm at a quality it never held
    at that wall-clock time, which is exactly the error a time-to-quality metric
    must not make.
    """
    per_seed = [pooled_rewards_for(run, benchmarks) for run in runs]
    common = sorted(set.intersection(*(set(int(s) for s in steps) for steps, _, _ in per_seed)))
    assert common, f"arm {name}: seeds share no evaluation steps"
    prompt_level = all(flag for _, _, flag in per_seed)

    rewards, wall, gpuh = [], [], []
    for run, (steps, matrix, _) in zip(runs, per_seed, strict=True):
        index = {int(s): i for i, s in enumerate(steps)}
        rewards.append(np.stack([matrix[index[s]] for s in common]))
        eval_index = {int(s): i for i, s in enumerate(run["eval_steps"])}
        wall.append([run["eval_wall_clock_h"][eval_index[s]] for s in common])
        gpuh.append([run["eval_gpu_hours"][eval_index[s]] for s in common])

    shapes = {r.shape[1:] for r in rewards}
    assert len(shapes) == 1, f"arm {name}: seeds disagree on the eval set shape {shapes}"

    guards = _merge_guards(runs, common)
    arm = eq.ArmSamples(
        smooth_window=smooth_window,
        name=name,
        steps=np.array(common),
        wall_clock_h=np.array(wall),
        gpu_hours=np.array(gpuh),
        rewards=np.stack(rewards),
        prompt_level=prompt_level,
        factors=runs[0].get("factors", {}),
        seed_labels=tuple(str(run["seed"]) for run in runs),
    )
    return arm, guards


def _merge_guards(runs: list[dict[str, Any]], common: list[int]) -> dict[str, np.ndarray]:
    """Guard series averaged over seeds, on the arm's common step grid."""
    merged: dict[str, np.ndarray] = {}
    metrics = sorted({m for run in runs for m in run["guards"]})
    for metric in metrics:
        columns = []
        for run in runs:
            values = run["guards"].get(metric)
            if values is None:
                continue
            index = {int(s): i for i, s in enumerate(run["eval_steps"])}
            columns.append([_as_float(values[index[s]]) for s in common])
        if columns:
            merged[metric] = np.nanmean(np.array(columns, dtype=float), axis=0)
    return merged


def _as_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else float("nan")


# --------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------


def analyze(
    arms: dict[str, tuple[eq.ArmSamples, dict[str, np.ndarray]]],
    reference: str,
    spec: eq.BootstrapSpec,
    conv_spec: eq.ConvergenceSpec,
    *,
    delta: float,
    consecutive: int,
    fractions: tuple[float, ...],
    axis: str,
    monotone: bool,
) -> dict[str, Any]:
    assert reference in arms, f"reference arm {reference!r} not among {sorted(arms)}"

    n_prompts = {arm.n_prompts for arm, _ in arms.values()}
    assert len(n_prompts) == 1, (
        f"arms disagree on the held-out prompt count {n_prompts}; a paired bootstrap needs the "
        "same evaluation set everywhere"
    )
    prompt_draws = eq.draw_prompt_indices(n_prompts.pop(), spec)
    boots = {name: eq.bootstrap_arm(arm, prompt_draws, spec) for name, (arm, _) in arms.items()}

    ref_arm, ref_guards = arms[reference]
    convergence = eq.detect_convergence(ref_arm, boots[reference], ref_guards, conv_spec)
    q_star_boot = eq.q_star_bootstrap(boots[reference], convergence)
    q0_boot = _base_quality(arms, boots)
    targets = eq.quality_targets(q0_boot, q_star_boot, fractions)

    equivalences, profiles, speedups, variances, tau_by_target = {}, {}, {}, {}, {}
    for name, (arm, _) in arms.items():
        equivalences[name] = eq.noninferiority_time(
            arm, boots[name], q_star_boot, delta=delta, alpha=spec.alpha, consecutive=consecutive, axis=axis
        )
        variances[name] = eq.variance_components(arm, spec, window=conv_spec.window)
        profiles[name] = []
        tau_by_target[name] = {}
        for fraction, q_target in targets.items():
            result, taus = eq.target_times(
                arm,
                boots[name],
                q_target,
                fraction=fraction,
                alpha=spec.alpha,
                consecutive=consecutive,
                axis=axis,
                monotone=monotone,
            )
            profiles[name].append(result)
            tau_by_target[name][fraction] = taus

    for name in arms:
        speedups[name] = [
            eq.speedup(name, tau_by_target[reference][p], tau_by_target[name][p], fraction=p, alpha=spec.alpha)
            for p in fractions
        ]

    detectable = eq.smallest_detectable_margin(equivalences[reference].lcb, consecutive)

    return {
        "reference": reference,
        "axis": axis,
        "delta": delta,
        "detectable_margin": detectable,
        "smooth_window": ref_arm.smooth_window,
        "alpha": spec.alpha,
        "consecutive": consecutive,
        "monotone": monotone,
        "convergence": asdict(convergence),
        "q_star": float(np.median(q_star_boot)),
        "q_star_ci": [float(np.quantile(q_star_boot, spec.alpha / 2)), float(np.quantile(q_star_boot, 1 - spec.alpha / 2))],
        "q0": float(np.median(q0_boot)),
        "arms": {
            name: {
                "factors": arm.factors,
                "seeds": list(arm.seed_labels),
                "prompt_level": arm.prompt_level,
                "steps": arm.steps.tolist(),
                "quality": arm.observed_quality().tolist(),
                "wall_clock_h": arm.observed_time("wall_clock").tolist(),
                "gpu_hours": arm.observed_time("gpu_hours").tolist(),
                "quality_lo": np.quantile(boots[name].quality, spec.alpha / 2, axis=0).tolist(),
                "quality_hi": np.quantile(boots[name].quality, 1 - spec.alpha / 2, axis=0).tolist(),
                "equivalence": _serialize_equivalence(equivalences[name]),
                "targets": [asdict(t) for t in profiles[name]],
                "speedups": [asdict(s) for s in speedups[name]],
                "variance_sd": variances[name],
                "guards": {k: v.tolist() for k, v in arms[name][1].items()},
            }
            for name, (arm, _) in arms.items()
        },
    }


def _base_quality(arms, boots) -> np.ndarray:
    """Q0, the base model's score, pooled over every arm that evaluated at step 0.

    Every arm starts from the same checkpoint, so their step-0 evaluations are
    repeated measurements of one quantity; pooling them is both more precise and
    a check -- if they disagree beyond sampling noise, the arms are not starting
    from the same policy and nothing downstream is comparable.
    """
    at_zero = [boots[name].quality[:, 0] for name, (arm, _) in arms.items() if int(arm.steps[0]) == 0]
    assert at_zero, "no arm has an evaluation at step 0, so Q0 is unknown; add --eval-interval coverage of step 0"
    return np.mean(np.stack(at_zero), axis=0)


def _serialize_equivalence(result: eq.EquivalenceResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["lcb"] = result.lcb.tolist()
    return payload


# --------------------------------------------------------------------------
# realized lag
# --------------------------------------------------------------------------


def lag_summaries(runs_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """P(L) per arm, alongside the configured bound it is supposed to obey.

    Reported together on purpose: a bound that never binds produces a flat row in
    the results and reads as insensitivity to staleness, when what happened is
    that the run was never stale. ``configured`` comes from the factor record and
    ``realized`` from the data.
    """
    out: dict[str, Any] = {}
    for arm, runs in runs_by_arm.items():
        pooled: list[np.ndarray] = []
        brackets: list[dict[str, float]] = []
        for run in runs:
            path = run["_dir"] / "lag.npz"
            if not path.is_file():
                continue
            payload = np.load(path)
            if "lag" in payload:
                pooled.append(payload["lag"])
            elif "lag_bracket_hi" in payload:
                brackets.append(
                    {
                        "bracket_hi_mean": float(np.nanmean(payload["lag_bracket_hi"])),
                        "bracket_lo_mean": float(np.nanmean(payload["lag_bracket_lo"])),
                        "avg_staleness_mean": float(np.nanmean(payload["avg_staleness"])),
                        "mixed_version_ratio_mean": float(np.nanmean(payload["mixed_version_ratio"])),
                    }
                )
        entry: dict[str, Any] = {"configured": runs[0]["factors"].get("MAX_WEIGHT_STALENESS", "unrecorded")}
        if pooled:
            lags = np.concatenate(pooled)
            entry["realized"] = eq.summarize_lag(lags)
            entry["histogram"] = np.bincount(lags.astype(int), minlength=8)[:32].tolist()
            entry["source"] = "per-sample weight_versions in the rollout dumps"
        elif brackets:
            entry["realized_bracket"] = {k: float(np.mean([b[k] for b in brackets])) for k in brackets[0]}
            entry["source"] = "logged weight-version statistics (bracket only, no distribution)"
        else:
            entry["source"] = "unavailable"
        out[arm] = entry
    return out


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------


def write_summary_csv(path: Path, results: dict[str, Any]) -> None:
    """One tidy row per (arm, target): the table a paper prints."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["arm", "p", "q_target", "tau", "tau_lo", "tau_hi", "reach_frac", "speedup", "speedup_lo", "speedup_hi", "paired_frac"]
        )
        for name, arm in results["arms"].items():
            for target, boost in zip(arm["targets"], arm["speedups"], strict=True):
                writer.writerow(
                    [
                        name,
                        target["fraction"],
                        f"{target['q_target']:.4f}",
                        _fmt(target["tau"]),
                        _fmt(target["tau_lo"]),
                        _fmt(target["tau_hi"]),
                        f"{target['reach_frac']:.3f}",
                        _fmt(boost["speedup"]),
                        _fmt(boost["lo"]),
                        _fmt(boost["hi"]),
                        f"{boost['paired_frac']:.3f}",
                    ]
                )


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def print_report(results: dict[str, Any], lags: dict[str, Any]) -> None:
    convergence = results["convergence"]
    axis = "wall-clock h" if results["axis"] == "wall_clock" else "GPU-hours"
    print(f"reference   {results['reference']}")
    print(f"converged   {convergence['converged']}  (slope UCB {convergence['slope_ucb']:.2e}/step, "
          f"plateau dev {convergence['plateau_max_dev']:.3f}, stable steps {convergence['stable_start']}..{convergence['stable_end']})")
    for failure in convergence["failures"]:
        print(f"  NOT CONVERGED: {failure}")
    for guard in convergence["guards"]:
        if guard["status"] != "pass":
            print(f"  guard {guard['metric']}: {guard['status']} ({guard['observed']} vs {guard['threshold']})")
    print(f"Q_on*       {results['q_star']:.4f} {results['q_star_ci']}   Q0 {results['q0']:.4f}")
    detectable = results.get("detectable_margin")
    if detectable is not None:
        verdict = "OK" if detectable <= results["delta"] else "UNDERPOWERED"
        print(f"power       smallest detectable margin {detectable:.4f} vs delta {results['delta']:.4f}  [{verdict}]")
        if detectable > results["delta"]:
            print("            the reference cannot clear its own margin: every 'never non-inferior' below is a")
            print("            statement about the held-out set size, not about the method. Raise --delta, raise")
            print("            --smooth-window, or evaluate on more prompts / more samples per prompt.")
    print()

    for name, arm in results["arms"].items():
        equivalence = arm["equivalence"]
        tau = f"{equivalence['tau']:.2f} {axis}" if equivalence["reached"] else "never"
        print(f"{name}")
        print(f"  non-inferiority (delta={results['delta']}): tau {tau}, final LCB {equivalence['final_lcb']:+.4f}")
        print("  bootstrap SD: " + ", ".join(f"{k} {v:.4f}" for k, v in arm["variance_sd"].items()))
        for target, boost in zip(arm["targets"], arm["speedups"], strict=True):
            speed = eq.format_ci(boost["speedup"], boost["lo"], boost["hi"])
            print(
                f"  p={target['fraction']:<5} q={target['q_target']:.3f}  "
                f"tau={eq.format_ci(target['tau'], target['tau_lo'], target['tau_hi'])}  "
                f"S={speed}  reached in {target['reach_frac']:.0%} of replicates"
            )
        lag = lags.get(name, {})
        if lag:
            print(f"  lag: configured {lag.get('configured')}, {lag.get('source')}")
            if "realized" in lag:
                stats = lag["realized"]
                print(f"       realized mean {stats['mean']:.2f}, p90 {stats['p90']:.0f}, max {stats['max']:.0f}")
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extracts", required=True, type=Path, help="root written by extract_run.py")
    p.add_argument(
        "--benchmark",
        required=True,
        help="eval dataset name, or a comma-separated list to pool into one held-out set "
        "(e.g. aime23,aime24,aime25,aime26). Pooling needs per-prompt rewards for every member",
    )
    p.add_argument("--reference-arm", required=True, help="the converged on-policy arm")
    p.add_argument("--out", required=True, type=Path, help="output directory for results.json and summary.csv")
    p.add_argument("--axis", choices=["wall_clock", "gpu_hours"], default="wall_clock")
    p.add_argument("--delta", type=float, default=eq.DEFAULT_DELTA, help="practical equivalence margin")
    p.add_argument("--alpha", type=float, default=eq.DEFAULT_ALPHA)
    p.add_argument("--consecutive", type=int, default=eq.DEFAULT_CONSECUTIVE)
    p.add_argument("--replicates", type=int, default=eq.DEFAULT_REPLICATES)
    p.add_argument("--window", type=int, default=eq.DEFAULT_WINDOW, help="plateau window, in evaluations")
    p.add_argument("--eps-slope", type=float, default=eq.DEFAULT_EPS_SLOPE, help="quality per rollout step")
    p.add_argument("--delta-plateau", type=float, default=eq.DEFAULT_DELTA_PLATEAU)
    p.add_argument("--fractions", default=",".join(str(f) for f in eq.DEFAULT_TARGET_FRACTIONS))
    p.add_argument("--monotone", action="store_true", help="cross on the running maximum of Q instead of Q")
    p.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="trailing mean over this many evaluations before differencing against Q_on*, which is "
        "itself a plateau mean; 1 disables it (equivalence.trailing_mean explains the trade)",
    )
    p.add_argument("--rng-seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs = load_runs(args.extracts)

    runs_by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        runs_by_arm.setdefault(run["arm"], []).append(run)

    benchmarks = [name.strip() for name in args.benchmark.split(",") if name.strip()]
    arms = {
        name: build_arm(group, benchmarks, name, args.smooth_window)
        for name, group in sorted(runs_by_arm.items())
    }
    if len(benchmarks) > 1:
        pooled = next(iter(arms.values()))[0].n_prompts
        print(f"pooled {len(benchmarks)} benchmarks into {pooled} held-out prompts: {', '.join(benchmarks)}")
    single_seed = [name for name, (arm, _) in arms.items() if arm.n_seeds < 2]
    if single_seed:
        print(f"WARNING: single-seed arms, seed variance not estimable: {', '.join(single_seed)}")

    spec = eq.BootstrapSpec(n_replicates=args.replicates, alpha=args.alpha, rng_seed=args.rng_seed)
    conv_spec = eq.ConvergenceSpec(
        window=args.window, eps_slope=args.eps_slope, delta_plateau=args.delta_plateau, alpha=args.alpha
    )
    fractions = tuple(float(f) for f in args.fractions.split(","))

    results = analyze(
        arms,
        args.reference_arm,
        spec,
        conv_spec,
        delta=args.delta,
        consecutive=args.consecutive,
        fractions=fractions,
        axis=args.axis,
        monotone=args.monotone,
    )
    results["lag"] = lag_summaries(runs_by_arm)
    results["benchmark"] = "+".join(benchmarks)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    write_summary_csv(args.out / "summary.csv", results)

    print_report(results, results["lag"])
    print(f"wrote {args.out / 'results.json'} and {args.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
