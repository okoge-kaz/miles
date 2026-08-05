"""Paper figures from ``analyze.py``'s ``results.json``.

Five panels, each answering one question the study poses:

| figure | question |
|---|---|
| ``quality_vs_time`` | did the off-policy arm get there, and when? |
| ``noninferiority`` | when did it become statistically non-inferior, and did it stay? |
| ``speedup_profile`` | is the speedup uniform, or only early? |
| ``phase_diagram`` | over which slice of the design does the tolerance hold? |
| ``realized_lag`` | was the configured lag the lag the run actually ran at? |

Colour is assigned by the job it does, from a palette validated for
colour-vision deficiency: the on-policy reference is neutral dark grey because it
is the baseline rather than a peer, off-policy arms take the categorical slots in
fixed order, and the phase diagram uses a blue-grey-red diverging ramp because
speedup has a meaningful neutral at S = 1. Every arm also carries a distinct
dash pattern and marker, so the figures survive greyscale printing -- identity is
never colour alone.

    uv run --with numpy --with matplotlib python -m experiments.src.offpolicy_acceleration.figures \\
      --results /lustre/.../results/aime24/results.json --out /lustre/.../figures/aime24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # written to file on a login node, never displayed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# Categorical slots in their validated order; the reference gets ink, not a hue.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
REFERENCE_COLOR = "#3f3f3c"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"
DASHES = ((), (5, 2), (1, 1.5), (6, 2, 1, 2), (3, 1.5, 1, 1.5), (8, 3), (2, 1), (4, 1, 1, 1))
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#8f2020"]


def apply_style() -> None:
    """Recessive grid and axes, ink-coloured text, thin marks."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
        }
    )


def build_palette(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One style per arm, assigned once and reused by every figure.

    Colour follows the entity, not its position in whatever subset a given panel
    happens to draw: the speedup profile omits the reference arm, and if slots
    were handed out per figure the same arm would change hue between panels --
    the single most effective way to make a multi-panel figure unreadable.
    """
    reference = results["reference"]
    palette = {reference: dict(color=REFERENCE_COLOR, dashes=(), marker="o", zorder=5)}
    for slot, name in enumerate(name for name in results["arms"] if name != reference):
        index = slot % len(SERIES)
        palette[name] = dict(color=SERIES[index], dashes=DASHES[index], marker=MARKERS[index], zorder=3)
    return palette


def off_policy_arms(results: dict[str, Any]) -> list[str]:
    return [name for name in results["arms"] if name != results["reference"]]


def axis_label(results: dict[str, Any]) -> str:
    return "wall-clock (h)" if results["axis"] == "wall_clock" else "GPU-hours"


def time_series(arm: dict[str, Any], results: dict[str, Any]) -> np.ndarray:
    return np.array(arm["wall_clock_h" if results["axis"] == "wall_clock" else "gpu_hours"])


# --------------------------------------------------------------------------


def quality_vs_time(results: dict[str, Any], palette: dict[str, dict[str, Any]], out: Path) -> Path:
    """Learning curves against the reference plateau and the equivalence margin.

    The shaded band is the bootstrap interval on Q, not a standard error over
    seeds: it carries prompt, rollout and seed resampling together, which is the
    only version of the band that supports the non-inferiority claim made in the
    next figure.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    q_star, delta = results["q_star"], results["delta"]

    ax.axhspan(q_star - delta, q_star, color=REFERENCE_COLOR, alpha=0.07, lw=0)
    ax.axhline(q_star, color=REFERENCE_COLOR, lw=1.0, ls=(0, (4, 3)))
    ax.annotate(
        f"$Q^{{\\star}}_{{\\mathrm{{on}}}}$ = {q_star:.3f}   (margin $\\delta$ = {delta})",
        xy=(0.01, q_star),
        xycoords=("axes fraction", "data"),
        ha="left",
        va="bottom",
        fontsize=8,
        color=TEXT_SECONDARY,
    )

    for name, arm in results["arms"].items():
        style = palette[name]
        t = time_series(arm, results)
        q = np.array(arm["quality"])
        ax.fill_between(t, arm["quality_lo"], arm["quality_hi"], color=style["color"], alpha=0.12, lw=0)
        ax.plot(t, q, label=name, **style)

    ax.set_xlabel(axis_label(results))
    ax.set_ylabel(f"{results['benchmark']} avg@k")
    ax.set_title("Quality against training time")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    return _save(fig, out / "quality_vs_time.pdf")


def noninferiority(results: dict[str, Any], palette: dict[str, dict[str, Any]], out: Path) -> Path:
    """LCB on Delta_m(t) = Q_m(t) - Q_on*, with tau marked where it clears -delta.

    A single axis, deliberately: the crossing time and the margin live in the
    same units, and the temptation to overlay raw quality on a second scale is
    the one thing that would make this panel unreadable.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    delta = results["delta"]

    ax.axhline(-delta, color="#e34948", lw=1.2, ls=(0, (4, 3)))
    ax.annotate(
        f"$-\\delta$ = {-delta}", xy=(0.01, -delta), xycoords=("axes fraction", "data"),
        va="bottom", fontsize=8, color="#e34948",
    )
    ax.axhline(0.0, color=GRID, lw=1.0)

    for name, arm in results["arms"].items():
        style = palette[name]
        t = time_series(arm, results)
        lcb = np.array(arm["equivalence"]["lcb"])
        ax.plot(t, lcb, label=name, **style)
        tau = arm["equivalence"]["tau"]
        if tau is not None:
            position = int(np.argmin(np.abs(t - tau)))
            ax.plot([tau], [lcb[position]], marker=style["marker"], color=style["color"], ms=9, mfc="white", mew=1.6, zorder=6)
            ax.annotate(f"$\\tau$={tau:.1f}", xy=(tau, lcb[position]), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=7.5, color=style["color"])

    ax.set_xlabel(axis_label(results))
    ax.set_ylabel(r"one-sided LCB of $Q_m(t)-Q^{\star}_{\mathrm{on}}$")
    ax.set_title(f"Time to non-inferiority ({int((1 - results['alpha']) * 100)}% LCB, "
                 f"{results['consecutive']} consecutive evaluations)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    return _save(fig, out / "noninferiority.pdf")


def speedup_profile(results: dict[str, Any], palette: dict[str, dict[str, Any]], out: Path) -> Path:
    """S_m(p) over the target ladder -- the figure that replaces "2.5x faster".

    Targets where fewer than half the bootstrap replicates reached the quality
    are drawn hollow. An unreached target is not a speedup of zero and not a
    missing point; it is a measurement whose denominator barely exists, and the
    figure has to say so rather than let a confident marker imply otherwise.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axhline(1.0, color=REFERENCE_COLOR, lw=1.0, ls=(0, (4, 3)))
    ax.annotate("no acceleration", xy=(0.01, 1.0), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=8, color=TEXT_SECONDARY)

    names = off_policy_arms(results)
    width = 0.7 / max(len(names), 1)
    for index, name in enumerate(names):
        arm = results["arms"][name]
        style = palette[name]
        points = [s for s in arm["speedups"] if s["speedup"] is not None]
        if not points:
            continue
        xs = np.arange(len(points)) + (index - (len(names) - 1) / 2) * width
        values = [s["speedup"] for s in points]
        lows = [max(s["speedup"] - s["lo"], 0) for s in points]
        highs = [max(s["hi"] - s["speedup"], 0) for s in points]
        solid = [s["paired_frac"] >= 0.5 for s in points]
        ax.errorbar(xs, values, yerr=[lows, highs], fmt="none", ecolor=style["color"], elinewidth=1.4, capsize=3)
        for x, value, filled in zip(xs, values, solid, strict=True):
            ax.plot([x], [value], marker=style["marker"], color=style["color"], ms=6,
                    mfc=style["color"] if filled else "white", mew=1.4)
        ax.plot([], [], marker=style["marker"], color=style["color"], label=name, dashes=style["dashes"])

    fractions = [s["fraction"] for s in results["arms"][names[0]]["speedups"]] if names else []
    ax.set_xticks(np.arange(len(fractions)))
    ax.set_xticklabels([f"p={f:g}" for f in fractions])
    ax.set_xlabel(r"target $q_p = Q_0 + p\,(Q^{\star}_{\mathrm{on}} - Q_0)$")
    ax.set_ylabel(r"speedup $S_m(p)=\tau_{\mathrm{on}}/\tau_m$")
    ax.set_title("Speedup profile (hollow: reached in <50% of replicates)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    return _save(fig, out / "speedup_profile.pdf")


def phase_diagram(results: dict[str, Any], out: Path, x_factor: str, y_factor: str, fraction: float) -> Path | None:
    """The robustness surface: speedup at one target across two design axes.

    A diverging ramp centred on S = 1 because the neutral is meaningful -- above
    is acceleration, below is a slowdown, and a sequential ramp would hide which
    side of the line a cell falls on. Cells that never reached non-inferiority
    are hatched rather than coloured, so "fast but never equivalent" cannot be
    mistaken for a win.
    """
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in results["arms"].values():
        x, y = arm["factors"].get(x_factor), arm["factors"].get(y_factor)
        if x is None or y is None:
            continue
        match = next((s for s in arm["speedups"] if abs(s["fraction"] - fraction) < 1e-9), None)
        if match:
            cells[(str(x), str(y))] = {"speedup": match["speedup"], "reached": arm["equivalence"]["reached"]}
    if not cells:
        print(f"phase_diagram: no arm carries both {x_factor!r} and {y_factor!r}; skipping")
        return None

    xs = sorted({x for x, _ in cells}, key=_natural)
    ys = sorted({y for _, y in cells}, key=_natural)
    grid = np.full((len(ys), len(xs)), np.nan)
    for (x, y), cell in cells.items():
        if cell["speedup"] is not None:
            grid[ys.index(y), xs.index(x)] = cell["speedup"]

    finite = grid[np.isfinite(grid)]
    span = max(abs(np.log2(finite)).max() if finite.size else 1.0, 0.2)
    cmap = LinearSegmentedColormap.from_list("speedup", DIVERGING)
    norm = TwoSlopeNorm(vmin=2**-span, vcenter=1.0, vmax=2**span)

    fig, ax = plt.subplots(figsize=(1.1 * len(xs) + 2.2, 0.9 * len(ys) + 1.6))
    image = ax.imshow(grid, cmap=cmap, norm=norm, aspect="equal", origin="lower")
    ax.set_xticks(range(len(xs)), xs)
    ax.set_yticks(range(len(ys)), ys)
    ax.set_xlabel(x_factor)
    ax.set_ylabel(y_factor)
    ax.grid(False)

    for (x, y), cell in cells.items():
        column, row = xs.index(x), ys.index(y)
        if not cell["reached"]:
            ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, hatch="///",
                                       edgecolor=TEXT_SECONDARY, lw=0.0))
        label = "n/a" if cell["speedup"] is None else f"{cell['speedup']:.2f}"
        ax.text(column, row, label, ha="center", va="center", fontsize=8, color=TEXT_PRIMARY)

    ax.set_title(f"Speedup at p={fraction:g}  (hatched: never non-inferior)")
    fig.colorbar(image, ax=ax, shrink=0.85, label=r"$S_m$")
    return _save(fig, out / "phase_diagram.pdf")


def realized_lag(results: dict[str, Any], palette: dict[str, dict[str, Any]], out: Path) -> Path | None:
    """Configured bound against the lag the run actually ran at.

    The single most common way an asynchronous-RL result misleads is a bound that
    never binds: the arm is labelled "staleness 4" and trained at a realized lag
    of 0.3, so the plot reads as robustness to staleness the run never
    experienced. Plotting P(L) beside the label is the check.
    """
    lag = results.get("lag", {})
    with_distribution = {name: entry for name, entry in lag.items() if "realized" in entry}
    if not with_distribution:
        print("realized_lag: no per-sample lag extracted (rerun extract_run.py with --with-lag); skipping")
        return None

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for name, entry in with_distribution.items():
        style = palette[name]
        counts = np.array(entry["histogram"], dtype=float)
        cdf = np.cumsum(counts) / counts.sum()
        ax.step(np.arange(len(cdf)), cdf, where="post", label=f"{name} (configured {entry['configured']})", **style)
        configured = entry.get("configured")
        if str(configured).isdigit() and int(configured) < len(cdf):
            ax.axvline(int(configured), color=style["color"], lw=0.8, alpha=0.4)

    ax.set_xlabel("realized policy lag $L$ (weight versions)")
    ax.set_ylabel(r"$P(L \leq \ell)$")
    ax.set_ylim(0, 1.02)
    ax.set_title("Realized lag distribution vs the configured bound")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out / "realized_lag.pdf")


def _natural(value: str):
    """Sort ``2`` before ``10`` on a factor axis, and fall back to text."""
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {path}")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, type=Path, help="results.json from analyze.py")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--x-factor", default="MAX_WEIGHT_STALENESS", help="phase-diagram x axis, a factor key")
    p.add_argument("--y-factor", default="MODEL_NAME", help="phase-diagram y axis, a factor key")
    p.add_argument("--phase-fraction", type=float, default=0.9, help="which target the phase diagram reports")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results = json.loads(args.results.read_text())
    apply_style()
    args.out.mkdir(parents=True, exist_ok=True)

    palette = build_palette(results)
    quality_vs_time(results, palette, args.out)
    noninferiority(results, palette, args.out)
    speedup_profile(results, palette, args.out)
    phase_diagram(results, args.out, args.x_factor, args.y_factor, args.phase_fraction)
    realized_lag(results, palette, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
