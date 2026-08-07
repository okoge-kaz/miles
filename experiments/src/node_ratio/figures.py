"""Appendix figures for the train/rollout node split in asynchronous RL.

Four panels, each answering one question:

| figure | question |
|---|---|
| ``scaling`` | where does the trainer stop waiting, and what stops step time falling? |
| ``efficiency`` | is the extra rollout node paying for itself? |
| ``decode_rate`` | *why* is rollout scaling sublinear? |
| ``timeline`` | while the trainer was idle, what were the engines doing? |

``scaling`` carries the horizontal line that makes the argument: a sample that
runs to ``--rollout-max-response-len`` needs ``len / uncontended_tok_s`` seconds
of serial decoding, and no number of rollout GPUs moves it, because token n+1
cannot be produced before token n. Rollout time falls towards that line and then
stops. The rate is measured from the logs' own low-concurrency decode lines
rather than assumed.

Colour is assigned by the job it does. Adoptable node counts -- the ones whose
data-parallel size divides the global batch in the colocated arm, so the same
total GPU count can run both arms -- are drawn filled; the appendix-only points
are hollow, because they can be reported but never shipped. Every series also
carries a distinct dash and marker so the figures survive greyscale.

    python -m experiments.src.node_ratio.figures \\
        experiments/outputs/training/math/.../noderatio-r*.log \\
        --out experiments/outputs/figures/node_ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # written to file on a login node, never displayed
import matplotlib.pyplot as plt  # noqa: E402

from experiments.src.node_ratio.parse_logs import (  # noqa: E402
    RunLog,
    parse,
    uncontended_tok_s,
)

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
REFERENCE_COLOR = "#3f3f3c"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"
SURFACE = "#ffffff"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "font.size": 10,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
        }
    )


def adoptable(rollout_nodes: int, global_batch_size: int, tp: int = 2) -> bool:
    """Can the colocated arm run at this total GPU count?

    The comparison is at equal total GPUs, and colocated trains on all of them,
    so ``dp = 8 * (1 + rollout_nodes) / tp`` has to divide the global batch.
    """
    dp = 8 * (1 + rollout_nodes) // tp
    return dp > 0 and global_batch_size % dp == 0


def _finish(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_scaling(rows: list[dict], floor_s: float | None, out: Path) -> None:
    """Step time and its two parts against rollout node count."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = [r["rollout_nodes"] for r in rows]

    for i, (key, label, dash) in enumerate(
        [
            ("step_s", "step (wait + train)", ()),
            ("train_wait", "train_wait — trainer idle, waiting on generation", (5, 2)),
            ("train", "train — actor_train + log_probs", (1, 1.5)),
        ]
    ):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=SERIES[i], dashes=dash, marker="o", markersize=6, label=label, zorder=3)
        # Hollow marker where the colocated arm cannot match the total GPU count.
        for xi, yi, r in zip(x, y, rows, strict=True):
            if not adoptable(r["rollout_nodes"], r["global_batch_size"]):
                ax.plot(xi, yi, marker="o", markersize=6, color=SURFACE, mec=SERIES[i], mew=1.6, zorder=4)

    if floor_s:
        ax.axhline(floor_s, color=REFERENCE_COLOR, linewidth=1.2, dashes=(6, 3), zorder=2)
        ax.annotate(
            f"longest-sample floor {floor_s:.0f} s\n(serial decode; no GPU count moves it)",
            xy=(max(x), floor_s),
            xytext=(-6, 8),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )

    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    _finish(ax, "Rollout capacity moves the wait, not the training", "rollout nodes (8 GPU each)", "seconds per step")
    ax.legend(loc="upper right")
    fig.text(
        0.0,
        -0.04,
        "Hollow markers: data-parallel size does not divide the global batch in the colocated arm, "
        "so this total GPU count cannot run both arms. Reportable, not adoptable.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.savefig(out / "scaling.png")
    plt.close(fig)


def fig_efficiency(rows: list[dict], out: Path) -> None:
    """Throughput per GPU, which is what says whether a node paid for itself."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = [r["rollout_nodes"] for r in rows]
    y = [r["steps_per_h_per_gpu"] for r in rows]
    ax.plot(x, y, color=SERIES[0], marker="o", markersize=6, zorder=3)
    for xi, yi, r in zip(x, y, rows, strict=True):
        if not adoptable(r["rollout_nodes"], r["global_batch_size"]):
            ax.plot(xi, yi, marker="o", markersize=6, color=SURFACE, mec=SERIES[0], mew=1.6, zorder=4)
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5,
                    color=TEXT_SECONDARY)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    # One series: the title names it, so no legend box.
    _finish(ax, "Steps per hour per GPU", "rollout nodes (8 GPU each)", "steps / h / GPU")
    fig.savefig(out / "efficiency.png")
    plt.close(fig)


def fig_decode_rate(runs: list[RunLog], rate: float | None, out: Path) -> None:
    """Per-sequence decode rate against concurrency: the mechanism behind the floor."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, run in enumerate(r for r in runs if r.decodes):
        pts = [(d.running_req, d.per_seq_tok_s) for d in run.decodes if d.running_req > 0]
        if not pts:
            continue
        label = f"{run.rollout_nodes} rollout node" + ("s" if run.rollout_nodes != 1 else "")
        ax.scatter(*zip(*pts, strict=True), s=6, alpha=0.35, color=SERIES[i % len(SERIES)],
                   edgecolors="none", label=label, zorder=3)
    if rate:
        ax.axhline(rate, color=REFERENCE_COLOR, linewidth=1.2, dashes=(6, 3), zorder=2)
        ax.annotate(f"uncontended {rate:.0f} tok/s", xy=(0, rate), xytext=(6, 6),
                    textcoords="offset points", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_xscale("log")
    _finish(
        ax,
        "Adding rollout nodes lowers concurrency, which raises per-sequence speed — up to a ceiling",
        "concurrent requests on the engine",
        "decode rate per sequence (tok/s)",
    )
    leg = ax.legend(loc="upper right", markerscale=2.5)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.savefig(out / "decode_rate.png")
    plt.close(fig)


def fig_timeline(run: RunLog, out: Path) -> None:
    """What the trainer and the engines were doing, on one shared clock."""
    apply_style()
    fig, (ax_t, ax_e) = plt.subplots(
        2, 1, figsize=(9.0, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.4]}
    )
    t0 = run.t0

    # Two tracks are deliberate rather than a twin y-axis: the quantities share a
    # clock and nothing else, and a shared axis would invite reading a crossing
    # that has no meaning.
    bands = {
        "train_wait": (SERIES[1], "train_wait (idle)"),
        "log_probs": (SERIES[2], "log_probs"),
        "actor_train": (SERIES[0], "actor_train"),
        "update_weights_implementation": (SERIES[3], "update_weights"),
    }
    seen: set[str] = set()
    for s in run.spans:
        if s.name not in bands:
            continue
        color, label = bands[s.name]
        ax_t.barh(
            0, s.seconds, left=s.start - t0, height=0.55, color=color,
            edgecolor=SURFACE, linewidth=0.8,
            label=label if label not in seen else None, zorder=3,
        )
        seen.add(label)
    ax_t.set_yticks([])
    ax_t.set_ylabel("trainer")
    fig.suptitle(
        f"{run.name} — {run.train_gpus} train GPU + {run.rollout_gpus} rollout GPU",
        color=TEXT_PRIMARY, x=0.0, ha="left", fontsize=11, y=1.06,
    )
    # Above the track, not on it: the bands are the data and a legend box sitting
    # on them hides exactly the phase boundaries the figure exists to show.
    ax_t.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, borderaxespad=0.0)
    for side in ("top", "right", "left"):
        ax_t.spines[side].set_visible(False)

    for i, eng in enumerate(sorted({d.engine for d in run.decodes})):
        pts = [(d.t - t0, d.running_req) for d in run.decodes if d.engine == eng]
        if pts:
            ax_e.plot(*zip(*pts, strict=True), color=SERIES[i % len(SERIES)], linewidth=1.0, alpha=0.75, zorder=3)
    ax_e.set_ylabel("concurrent requests\nper engine")
    ax_e.set_xlabel("seconds since the first trainer phase")
    ax_e.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax_e.spines[side].set_visible(False)
    ax_e.annotate(
        "one line per engine",
        xy=(0.995, 0.94), xycoords="axes fraction", ha="right", fontsize=8.5, color=TEXT_SECONDARY,
    )

    fig.savefig(out / f"timeline-{run.name}.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--uncontended-tok-s",
        type=float,
        default=None,
        help="single-stream decode rate; measured from the logs when they contain low-concurrency lines",
    )
    ap.add_argument("--timeline-for", default=None, help="log stem to draw the timeline for (default: the first)")
    ap.add_argument(
        "--allow-mixed",
        action="store_true",
        help="plot logs with different batch shapes together; their step times are not comparable",
    )
    args = ap.parse_args()

    runs = [parse(p) for p in args.logs]
    rows = [s for r in runs if (s := r.summary())]
    if not rows:
        raise SystemExit("no steady-state steps in any log")
    rows.sort(key=lambda r: r["rollout_nodes"])

    # Step time is proportional to the samples generated per step, so a run with
    # a different batch shape or response budget plots as a scaling effect it did
    # not have. An 8x smaller rollout batch looks exactly like an 8x speedup.
    shapes = {(r["global_batch_size"], r["max_response_len"]) for r in rows}
    if len(shapes) > 1 and not args.allow_mixed:
        detail = "\n".join(
            f"    {r['name']:<32} gbs {r['global_batch_size']:>6}  max_response_len {r['max_response_len']:>6}"
            for r in rows
        )
        raise SystemExit(
            "these logs do not share a batch shape, so their step times are not comparable:\n"
            f"{detail}\n"
            "Re-run the sweep at one shape, or pass --allow-mixed if you know why they differ."
        )

    rate = args.uncontended_tok_s or uncontended_tok_s(runs)
    max_len = max(r["max_response_len"] for r in rows)
    floor_s = max_len / rate if rate and max_len else None

    args.out.mkdir(parents=True, exist_ok=True)
    fig_scaling(rows, floor_s, args.out)
    fig_efficiency(rows, args.out)
    fig_decode_rate(runs, rate, args.out)

    target = next((r for r in runs if r.name == args.timeline_for), None) or next(
        (r for r in runs if r.spans and r.decodes), None
    )
    if target:
        fig_timeline(target, args.out)

    print(f"rollout nodes measured: {[r['rollout_nodes'] for r in rows]}")
    if rate:
        print(f"uncontended decode {rate:.1f} tok/s -> longest-sample floor {floor_s:.0f} s at {max_len} tokens")
    print(f"wrote {sorted(p.name for p in args.out.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
