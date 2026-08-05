"""Time-to-on-policy-equivalence: the statistics, with no I/O in this file.

The protocol has four steps, and each one is a function here:

1. ``bootstrap_arm``      -- a hierarchical (prompt / rollout / seed) bootstrap
                             of the quality curve Q_m(t), one row per replicate.
2. ``detect_convergence`` -- the pre-registered definition of "the on-policy
                             reference has converged", and the plateau mean
                             Q_on* that follows from it.
3. ``noninferiority_time``-- tau_m(delta): the first evaluation time at which a
                             one-sided lower confidence bound on
                             Delta_m(t) = Q_m(t) - Q_on* clears -delta, and stays
                             clear for k consecutive evaluations.
4. ``target_times`` / ``speedup``
                          -- the speedup *profile*: tau and S at a ladder of
                             intermediate quality targets q_p, not at one target.

Two decisions in here are worth stating up front, because they are the ones a
reviewer will ask about.

**The plateau window is chosen once, on the observed curve, and then held fixed
across bootstrap replicates.** A window re-selected inside every replicate would
make Q_on*[rep] the mean of a different set of steps in every replicate, and the
resulting interval would not be an interval for any single estimand. The
uncertainty that *is* propagated is the uncertainty of the mean over that fixed
window, which is what Q_on* is.

**tau(delta) gets no confidence interval; tau(q_p) does.** The non-inferiority
time is already defined *through* a confidence bound -- it is the first time a
one-sided LCB clears the margin -- so it is conservative by construction and a
CI on it would be a CI on a CI. The target-profile times are defined on Q
itself, so re-running the crossing rule inside each replicate is a legitimate
bootstrap of tau, and hence of the speedup ratio.

Quality here is whatever the caller put in ``rewards``: for math RL that is
avg@k on a held-out benchmark, in [0, 1]. Every threshold below is in those
units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Quality unit is the benchmark score itself, so these defaults are absolute:
# 0.2 points of avg@k for the plateau band, and 2e-4 points per rollout step of
# residual slope (0.02 points over a 100-step tail).
DEFAULT_WINDOW = 5
DEFAULT_EPS_SLOPE = 2e-4
DEFAULT_DELTA_PLATEAU = 0.02
DEFAULT_DELTA = 0.02
DEFAULT_ALPHA = 0.05
DEFAULT_CONSECUTIVE = 3
DEFAULT_REPLICATES = 2000
DEFAULT_TARGET_FRACTIONS = (0.5, 0.75, 0.9, 0.95)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSamples:
    """Every evaluation of one experimental arm, aligned on a shared step grid.

    ``rewards`` is ``(n_seeds, n_steps, n_prompts, n_samples_per_prompt)`` --
    the per-prompt, per-rollout eval rewards recovered from the ``--dump-details``
    eval dumps. When those dumps are absent the caller degrades to the logged
    scalar ``eval/<benchmark>`` mean and sets ``prompt_level=False``, in which
    case the array is ``(n_seeds, n_steps, 1, 1)`` and only the seed level of the
    bootstrap carries any information -- see ``bootstrap_arm``.

    ``wall_clock_h`` and ``gpu_hours`` are per seed because two seeds of the same
    configuration do not reach step ``t`` at the same wall-clock time, and that
    spread belongs inside the interval on tau.
    """

    name: str
    steps: np.ndarray  # (T,) rollout ids, ascending
    wall_clock_h: np.ndarray  # (S, T)
    gpu_hours: np.ndarray  # (S, T)
    rewards: np.ndarray  # (S, T, P, N)
    prompt_level: bool = True
    factors: dict[str, str] = field(default_factory=dict)
    seed_labels: tuple[str, ...] = ()
    smooth_window: int = 1

    def __post_init__(self) -> None:
        s, t = self.rewards.shape[0], self.rewards.shape[1]
        assert self.rewards.ndim == 4, f"{self.name}: rewards must be (S,T,P,N), got {self.rewards.shape}"
        assert self.steps.shape == (t,), f"{self.name}: steps {self.steps.shape} != T {t}"
        assert self.wall_clock_h.shape == (s, t), f"{self.name}: wall_clock_h {self.wall_clock_h.shape}"
        assert self.gpu_hours.shape == (s, t), f"{self.name}: gpu_hours {self.gpu_hours.shape}"
        assert not np.isnan(self.rewards).any(), f"{self.name}: rewards contain NaN; drop or impute before this point"

    @property
    def n_seeds(self) -> int:
        return self.rewards.shape[0]

    @property
    def n_prompts(self) -> int:
        return self.rewards.shape[2]

    def observed_quality(self) -> np.ndarray:
        """(T,) the point estimate: mean over samples, prompts, then seeds."""
        return trailing_mean(self.rewards.mean(axis=(2, 3)).mean(axis=0), self.smooth_window)

    def observed_time(self, axis: Literal["wall_clock", "gpu_hours"]) -> np.ndarray:
        return (self.wall_clock_h if axis == "wall_clock" else self.gpu_hours).mean(axis=0)


@dataclass(frozen=True)
class BootstrapSpec:
    n_replicates: int = DEFAULT_REPLICATES
    alpha: float = DEFAULT_ALPHA
    resample_prompts: bool = True
    resample_rollouts: bool = True
    resample_seeds: bool = True
    rng_seed: int = 0


@dataclass(frozen=True)
class ArmBootstrap:
    """One arm's bootstrap: quality and both time axes, ``(n_replicates, n_steps)``."""

    name: str
    quality: np.ndarray
    wall_clock_h: np.ndarray
    gpu_hours: np.ndarray

    def time(self, axis: Literal["wall_clock", "gpu_hours"]) -> np.ndarray:
        return self.wall_clock_h if axis == "wall_clock" else self.gpu_hours


# --------------------------------------------------------------------------
# 1. the hierarchical bootstrap
# --------------------------------------------------------------------------


def draw_prompt_indices(n_prompts: int, spec: BootstrapSpec) -> np.ndarray:
    """(R, P) prompt draws, generated once and reused by every arm.

    Sharing the draw is what makes the comparison *paired*: in a given replicate
    every arm is scored on the same resampled held-out set, so the prompt-sampling
    component cancels out of Delta_m(t) instead of adding to it.
    """
    rng = np.random.default_rng(spec.rng_seed)
    if not spec.resample_prompts:
        return np.tile(np.arange(n_prompts), (spec.n_replicates, 1))
    return rng.integers(0, n_prompts, size=(spec.n_replicates, n_prompts))


def bootstrap_arm(arm: ArmSamples, prompt_draws: np.ndarray, spec: BootstrapSpec) -> ArmBootstrap:
    """Resample prompts (shared), rollouts within prompt, and seeds within arm.

    The three levels are the three variance sources the design has to keep
    separate: which held-out prompts were drawn, which continuations the sampler
    happened to produce for them, and which training seed produced the policy.
    ``variance_components`` runs this with one level at a time to report them.
    """
    rng = np.random.default_rng(spec.rng_seed + 1 + _stable_hash(arm.name))
    n_rep = spec.n_replicates
    n_seeds, n_steps, n_prompts, n_samples = arm.rewards.shape

    quality = np.empty((n_rep, n_steps))
    wall = np.empty((n_rep, n_steps))
    gpuh = np.empty((n_rep, n_steps))

    # A mean-only arm has no prompt or rollout axis to resample; forcing the
    # draws through would silently report a prompt-level interval of width zero.
    use_prompts = spec.resample_prompts and arm.prompt_level and n_prompts > 1
    use_rollouts = spec.resample_rollouts and arm.prompt_level and n_samples > 1

    for rep in range(n_rep):
        picked = arm.rewards[:, :, prompt_draws[rep], :] if use_prompts else arm.rewards
        if use_rollouts:
            idx = rng.integers(0, n_samples, size=picked.shape)
            picked = np.take_along_axis(picked, idx, axis=3)
        per_seed = picked.mean(axis=(2, 3))  # (S, T)

        seed_idx = rng.integers(0, n_seeds, size=n_seeds) if spec.resample_seeds else np.arange(n_seeds)
        quality[rep] = trailing_mean(per_seed[seed_idx].mean(axis=0), arm.smooth_window)
        wall[rep] = arm.wall_clock_h[seed_idx].mean(axis=0)
        gpuh[rep] = arm.gpu_hours[seed_idx].mean(axis=0)

    return ArmBootstrap(name=arm.name, quality=quality, wall_clock_h=wall, gpu_hours=gpuh)


def variance_components(arm: ArmSamples, spec: BootstrapSpec, *, window: int = DEFAULT_WINDOW) -> dict[str, float]:
    """Bootstrap SD of Q over the last ``window`` evaluations, one level at a time.

    Reported as standard deviations in quality units rather than as a variance
    decomposition that sums: the levels are nested and the bootstrap draws are
    not orthogonal, so the parts do not add up to the whole and printing them as
    if they did would be wrong. What they do support is the honest statement
    "seed spread dominates prompt spread by 3x", which is the design decision
    they are there to inform.
    """
    levels = {
        "prompt": dict(resample_prompts=True, resample_rollouts=False, resample_seeds=False),
        "rollout": dict(resample_prompts=False, resample_rollouts=True, resample_seeds=False),
        "seed": dict(resample_prompts=False, resample_rollouts=False, resample_seeds=True),
        "total": dict(resample_prompts=True, resample_rollouts=True, resample_seeds=True),
    }
    out: dict[str, float] = {}
    for level, flags in levels.items():
        sub = BootstrapSpec(
            n_replicates=spec.n_replicates,
            alpha=spec.alpha,
            rng_seed=spec.rng_seed,
            **flags,
        )
        boot = bootstrap_arm(arm, draw_prompt_indices(arm.n_prompts, sub), sub)
        tail = boot.quality[:, -window:].mean(axis=1)
        out[level] = float(tail.std(ddof=1))
    if arm.n_seeds < 2:
        out["seed"] = float("nan")  # a one-seed arm cannot estimate seed variance
    return out


# --------------------------------------------------------------------------
# 2. on-policy convergence and Q_on*
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardSpec:
    """One collapse check on a metric series, evaluated over the stable interval.

    ``floor_frac``  -- the stable-interval mean must be at least ``value`` times
                       the run's own maximum (entropy collapse).
    ``ceiling``     -- the stable-interval mean must not exceed ``value``
                       (KL blow-up, truncation rate).
    ``no_decline``  -- the OLS slope over the stable interval must not be below
                       ``-value`` (reward or validation accuracy trending down).
    """

    metric: str
    kind: Literal["floor_frac", "ceiling", "no_decline"]
    value: float


# Metric names are the ones miles actually logs (verified against a job log), not
# plausible ones. train/entropy_loss is identically 0.0 unless the run passes
# --observe-training-entropy, which is why the guard reports "unavailable" rather
# than "pass" when it sees a flat zero -- see _check_guard.
DEFAULT_GUARDS = (
    GuardSpec("train/entropy_loss", "floor_frac", 0.5),
    GuardSpec("train/kl_loss", "ceiling", 0.05),
    GuardSpec("rollout/raw_reward", "no_decline", 5e-4),
    GuardSpec("rollout/truncated_ratio", "ceiling", 0.15),
    GuardSpec("rollout/repetition_frac", "ceiling", 0.15),
)


@dataclass(frozen=True)
class GuardResult:
    metric: str
    status: Literal["pass", "fail", "unavailable"]
    observed: float | None
    threshold: float


@dataclass(frozen=True)
class ConvergenceSpec:
    window: int = DEFAULT_WINDOW
    eps_slope: float = DEFAULT_EPS_SLOPE
    delta_plateau: float = DEFAULT_DELTA_PLATEAU
    alpha: float = DEFAULT_ALPHA


DEFAULT_CONVERGENCE = ConvergenceSpec()


@dataclass(frozen=True)
class ConvergenceReport:
    converged: bool
    slope: float
    slope_ucb: float
    plateau_max_dev: float
    stable_start: int
    stable_end: int
    q_star: float
    guards: tuple[GuardResult, ...]
    failures: tuple[str, ...]


def detect_convergence(
    arm: ArmSamples,
    boot: ArmBootstrap,
    guard_series: dict[str, np.ndarray],
    spec: ConvergenceSpec = DEFAULT_CONVERGENCE,
    guards: tuple[GuardSpec, ...] = DEFAULT_GUARDS,
) -> ConvergenceReport:
    """Apply the pre-registered plateau test to the on-policy reference run.

    The certificate is evaluated on the final ``window`` evaluations; the stable
    interval is then grown backwards from there while the curve stays inside the
    plateau band, and Q_on* is the mean over that interval. Growing it backwards
    rather than fixing it at the last ``window`` points is what stops Q_on* from
    being an average of five noisy numbers when the run has been flat for forty.
    """
    q = arm.observed_quality()
    n = len(q)
    failures: list[str] = []
    if n < spec.window:
        return ConvergenceReport(
            converged=False,
            slope=float("nan"),
            slope_ucb=float("nan"),
            plateau_max_dev=float("nan"),
            stable_start=0,
            stable_end=n,
            q_star=float(q.mean()) if n else float("nan"),
            guards=(),
            failures=(f"only {n} evaluations, window is {spec.window}",),
        )

    tail = slice(n - spec.window, n)
    x = arm.steps[tail].astype(float)
    slope = _ols_slope(x, q[tail])
    slope_boot = np.array([_ols_slope(x, boot.quality[rep, tail]) for rep in range(boot.quality.shape[0])])
    slope_ucb = float(np.quantile(slope_boot, 1.0 - spec.alpha))
    if slope_ucb >= spec.eps_slope:
        failures.append(f"slope UCB {slope_ucb:.2e}/step >= eps_slope {spec.eps_slope:.2e}")

    final_mean = float(q[tail].mean())
    plateau_max_dev = float(np.max(np.abs(q[tail] - final_mean)))
    if plateau_max_dev > spec.delta_plateau:
        failures.append(f"final window deviates {plateau_max_dev:.3f} > delta_plateau {spec.delta_plateau:.3f}")

    start = n - spec.window
    while start > 0 and abs(q[start - 1] - final_mean) <= spec.delta_plateau:
        start -= 1

    guard_results = tuple(_check_guard(g, guard_series, start, n, arm.steps) for g in guards)
    failures += [f"guard {g.metric}: {g.observed} vs {g.threshold}" for g in guard_results if g.status == "fail"]

    return ConvergenceReport(
        converged=not failures,
        slope=float(slope),
        slope_ucb=slope_ucb,
        plateau_max_dev=plateau_max_dev,
        stable_start=int(start),
        stable_end=int(n),
        q_star=float(q[start:].mean()),
        guards=guard_results,
        failures=tuple(failures),
    )


def q_star_bootstrap(boot: ArmBootstrap, report: ConvergenceReport) -> np.ndarray:
    """(R,) Q_on* per replicate, over the window fixed by ``detect_convergence``."""
    return boot.quality[:, report.stable_start : report.stable_end].mean(axis=1)


def _check_guard(
    guard: GuardSpec, series: dict[str, np.ndarray], start: int, end: int, steps: np.ndarray
) -> GuardResult:
    values = series.get(guard.metric)
    if values is None or len(values) < end or not np.isfinite(np.asarray(values[start:end], dtype=float)).any():
        return GuardResult(guard.metric, "unavailable", None, guard.value)
    full = np.asarray(values, dtype=float)
    if guard.kind == "floor_frac" and not np.any(full != 0.0):
        # A metric that is zero everywhere was never computed (train/entropy_loss
        # without --observe-training-entropy). Reporting that as "entropy did not
        # collapse" would certify a check that never ran.
        return GuardResult(guard.metric, "unavailable", 0.0, guard.value)
    window = np.asarray(values[start:end], dtype=float)
    if guard.kind == "floor_frac":
        peak = float(np.nanmax(full))
        observed = float(np.nanmean(window) / peak) if peak else float("nan")
        ok = observed >= guard.value
    elif guard.kind == "ceiling":
        observed = float(np.nanmean(window))
        ok = observed <= guard.value
    else:
        observed = float(_ols_slope(steps[start:end].astype(float), window))
        ok = observed >= -guard.value
    return GuardResult(guard.metric, "pass" if ok else "fail", observed, guard.value)


# --------------------------------------------------------------------------
# 3. tau(delta): time to non-inferiority
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalenceResult:
    arm: str
    delta: float
    tau: float | None  # in the units of the requested time axis
    tau_step: int | None
    lcb: np.ndarray  # (T,) LCB of Delta_m at each evaluation
    final_lcb: float
    reached: bool


def noninferiority_time(
    arm: ArmSamples,
    boot: ArmBootstrap,
    q_star_boot: np.ndarray,
    *,
    delta: float = DEFAULT_DELTA,
    alpha: float = DEFAULT_ALPHA,
    consecutive: int = DEFAULT_CONSECUTIVE,
    axis: Literal["wall_clock", "gpu_hours"] = "wall_clock",
) -> EquivalenceResult:
    """First time the one-sided LCB on Q_m(t) - Q_on* clears -delta and stays clear.

    ``q_star_boot`` must come from the same replicate indices as ``boot`` -- the
    subtraction is done replicate-wise so the prompt draw cancels between the arm
    and the reference, which is the entire point of the shared draw.
    """
    diff = boot.quality - q_star_boot[:, None]
    lcb = np.quantile(diff, alpha, axis=0)
    idx = _first_run_of(lcb > -delta, consecutive)
    times = arm.observed_time(axis)
    return EquivalenceResult(
        arm=arm.name,
        delta=delta,
        tau=float(times[idx]) if idx is not None else None,
        tau_step=int(arm.steps[idx]) if idx is not None else None,
        lcb=lcb,
        final_lcb=float(lcb[-1]),
        reached=idx is not None,
    )


# --------------------------------------------------------------------------
# 4. the speedup profile
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetResult:
    arm: str
    fraction: float
    q_target: float
    tau: float | None
    tau_lo: float | None
    tau_hi: float | None
    reach_frac: float  # share of replicates in which the target was reached


@dataclass(frozen=True)
class SpeedupResult:
    arm: str
    fraction: float
    speedup: float | None
    lo: float | None
    hi: float | None
    paired_frac: float  # share of replicates where both arms reached the target


def quality_targets(
    q0_boot: np.ndarray, q_star_boot: np.ndarray, fractions=DEFAULT_TARGET_FRACTIONS
) -> dict[float, np.ndarray]:
    """q_p = Q0 + p (Q_on* - Q0), per replicate, so the target inherits the
    uncertainty of both endpoints instead of being treated as a known constant."""
    return {p: q0_boot + p * (q_star_boot - q0_boot) for p in fractions}


def target_times(
    arm: ArmSamples,
    boot: ArmBootstrap,
    q_target_boot: np.ndarray,
    *,
    fraction: float,
    alpha: float = DEFAULT_ALPHA,
    consecutive: int = DEFAULT_CONSECUTIVE,
    axis: Literal["wall_clock", "gpu_hours"] = "wall_clock",
    monotone: bool = False,
) -> tuple[TargetResult, np.ndarray]:
    """tau_m(q_p) with a percentile CI, plus the per-replicate tau array.

    ``monotone`` replaces Q by its running maximum before the crossing test. RL
    curves are not monotone, so an unlucky dip can push tau past a later, better
    evaluation; the running max answers "when was this quality first attained"
    instead of "when was it first attained and held". It is off by default
    because it can only ever shorten tau, and reporting the shorter number
    without saying so would flatter every arm equally but not honestly.
    """
    q = _running_max(boot.quality) if monotone else boot.quality
    times = boot.time(axis)
    n_rep = q.shape[0]

    taus = np.full(n_rep, np.nan)
    for rep in range(n_rep):
        idx = _first_run_of(q[rep] >= q_target_boot[rep], consecutive)
        if idx is not None:
            taus[rep] = times[rep, idx]

    reached = np.isfinite(taus)
    observed_q = _running_max(arm.observed_quality()[None, :])[0] if monotone else arm.observed_quality()
    point_idx = _first_run_of(observed_q >= float(np.median(q_target_boot)), consecutive)
    point = float(arm.observed_time(axis)[point_idx]) if point_idx is not None else None

    result = TargetResult(
        arm=arm.name,
        fraction=fraction,
        q_target=float(np.median(q_target_boot)),
        tau=point,
        tau_lo=float(np.quantile(taus[reached], alpha / 2)) if reached.any() else None,
        tau_hi=float(np.quantile(taus[reached], 1 - alpha / 2)) if reached.any() else None,
        reach_frac=float(reached.mean()),
    )
    return result, taus


def speedup(
    arm_name: str,
    tau_reference: np.ndarray,
    tau_arm: np.ndarray,
    *,
    fraction: float,
    alpha: float = DEFAULT_ALPHA,
) -> SpeedupResult:
    """S_m(p) = tau_on(q_p) / tau_m(q_p), paired replicate by replicate.

    Only replicates where *both* arms reached the target contribute. ``paired_frac``
    is reported next to the ratio because a speedup computed on 30% of replicates
    is a statement about the 30%, and a figure that hides that is the standard way
    this metric gets oversold.
    """
    both = np.isfinite(tau_reference) & np.isfinite(tau_arm) & (tau_arm > 0)
    if not both.any():
        return SpeedupResult(arm_name, fraction, None, None, None, 0.0)
    ratio = tau_reference[both] / tau_arm[both]
    return SpeedupResult(
        arm=arm_name,
        fraction=fraction,
        speedup=float(np.median(ratio)),
        lo=float(np.quantile(ratio, alpha / 2)),
        hi=float(np.quantile(ratio, 1 - alpha / 2)),
        paired_frac=float(both.mean()),
    )


# --------------------------------------------------------------------------
# sensitivity: saturating curve fit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SaturatingFit:
    """ScaleRL-style sigmoidal saturation, ``Q(t) = A - (A - Q0) / (1 + (t/t_mid)^B)``.

    Reported as a *sensitivity analysis* only. The asymptote A is an
    extrapolation, and an RL curve that is still rising, or that is non-monotone,
    can support wildly different A at nearly identical fit quality -- which is
    exactly why the plateau definition, not this one, is the primary estimator of
    Q_on*.
    """

    asymptote: float
    q0: float
    t_mid: float
    exponent: float
    rmse: float


def fit_saturating(t: np.ndarray, q: np.ndarray, *, grid: int = 60) -> SaturatingFit:
    """Fit by profiling: for fixed (t_mid, B) the model is linear in (A, Q0).

    A closed-form linear solve inside a 2-D log-spaced grid gives a deterministic
    global-ish optimum with no optimizer and no scipy dependency -- worth more
    here than the last digit of precision, since the point of the fit is to show
    that the conclusion does not hinge on it.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    positive = t > 0
    t, q = t[positive], q[positive]
    assert len(t) >= 4, "saturating fit needs at least 4 positive-time evaluations"

    best = None
    for t_mid in np.geomspace(t.min() + 1e-9, t.max() * 4, grid):
        for exponent in np.geomspace(0.3, 8.0, grid // 2):
            w = 1.0 / (1.0 + (t / t_mid) ** exponent)  # Q = A(1-w) + Q0 w
            design = np.stack([1.0 - w, w], axis=1)
            coef, *_ = np.linalg.lstsq(design, q, rcond=None)
            rmse = float(np.sqrt(np.mean((design @ coef - q) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, coef[0], coef[1], t_mid, exponent)

    rmse, asymptote, q0, t_mid, exponent = best
    return SaturatingFit(float(asymptote), float(q0), float(t_mid), float(exponent), rmse)


# --------------------------------------------------------------------------
# small numerics
# --------------------------------------------------------------------------


def trailing_mean(q: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` evaluations at every position, along the last axis.

    This exists to make the two sides of Delta_m(t) = Q_m(t) - Q_on* symmetric.
    Q_on* is a mean over a plateau of many evaluations, so it is far less noisy
    than a single Q_m(t); differencing them puts the entire single-evaluation
    noise of the arm into the LCB, and on a 30-prompt benchmark that noise
    (~0.05 in avg@k) is larger than any practical equivalence margin. Smoothing
    the arm the same way restores the comparison.

    The cost is honest and one-sided: an arm's tau is delayed by up to
    ``window - 1`` evaluations, because the trailing mean cannot reach a level
    until most of the window is above it. Since every arm including the
    reference pays the same delay, the *ratio* tau_on/tau_m is far less affected
    than either time on its own. The first ``window - 1`` positions use an
    expanding mean rather than dropping out, so the early curve is noisier than
    the late one -- which is where quality is furthest from any target anyway.
    """
    if window <= 1:
        return q
    cumulative = np.cumsum(q, axis=-1)
    padded = np.concatenate([np.zeros(q.shape[:-1] + (1,)), cumulative], axis=-1)
    n = q.shape[-1]
    counts = np.minimum(np.arange(1, n + 1), window)
    lower = np.maximum(np.arange(1, n + 1) - window, 0)
    return (padded[..., 1:] - np.take(padded, lower, axis=-1)) / counts


def smallest_detectable_margin(lcb: np.ndarray, consecutive: int) -> float | None:
    """The smallest delta at which this arm's LCB would ever clear -delta.

    Run on the *reference against itself* it is a power check: if it comes back
    at 0.06 and the study pre-registered delta = 0.02, then no arm can pass, the
    on-policy run included, and every "not non-inferior" in the results is a
    statement about the size of the held-out set rather than about the method.
    Report it next to delta, always.
    """
    best = None
    for start in range(len(lcb) - consecutive + 1):
        floor = float(np.min(lcb[start : start + consecutive]))
        if best is None or floor > best:
            best = floor
    return None if best is None else max(-best, 0.0)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    return float((xc * (y - y.mean())).sum() / denom) if denom else 0.0


def _first_run_of(flags: np.ndarray, k: int) -> int | None:
    """Index of the first position starting a run of ``k`` consecutive True.

    A single lucky evaluation crossing the bar is noise; ``k`` in a row is the
    cheapest defence against reporting it as an arrival time.
    """
    flags = np.asarray(flags, dtype=bool)
    if k <= 0 or len(flags) < k:
        return None
    for i in range(len(flags) - k + 1):
        if flags[i : i + k].all():
            return i
    return None


def _running_max(q: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(q, axis=-1)


def _stable_hash(name: str) -> int:
    """Deterministic across processes, unlike ``hash()`` under PYTHONHASHSEED."""
    return int.from_bytes(name.encode()[:8].ljust(8, b"\0"), "little") % (2**31) if name else 0


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values and their empirical CDF, for the realized-lag panel."""
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1) / max(len(x), 1)


def summarize_lag(lags: np.ndarray) -> dict[str, float]:
    """P(L) reduced to the numbers a table can carry, keeping the tail visible."""
    lags = np.asarray(lags, dtype=float)
    if not len(lags):
        return {}
    return {
        "mean": float(lags.mean()),
        "p50": float(np.percentile(lags, 50)),
        "p90": float(np.percentile(lags, 90)),
        "p99": float(np.percentile(lags, 99)),
        "max": float(lags.max()),
        "frac_zero": float((lags <= 0).mean()),
        "n": int(len(lags)),
    }


def format_ci(value: float | None, lo: float | None, hi: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "not reached"
    if lo is None or hi is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
