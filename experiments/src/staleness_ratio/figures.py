"""Figures for the train:rollout split x staleness bound sweep.

Seven figures, each answering one question:

| figure | question |
|---|---|
| ``overview`` | which split is fastest, where does the time go, and what lag did the loss see? |
| ``rho`` | which side paces the run, in groups per second? |
| ``rates`` | the same capacities per staleness bound, in tokens per second |
| ``lag_trace_*_staleness`` | how each staleness distribution develops over a run |
| ``staleness_components`` | how pre-queue, in-queue, and total staleness compare |

Every number comes from one window, ``--window 10 36`` by default. That matters:
step time is read from the ``Timer`` lines and the token and lag counts from the
metrics dict, and the two are only comparable over the same rollouts. The window
is also chosen so every point of the sweep reached it, which matches response
length across the arms -- step time scales with length, so an unmatched window
ranks the runs that got further as slower.

``rho`` reports two readings of the producer rate, because they answer different
questions. Counting only the groups that reached the loss gives
``rho == T_train / step``, so it is bounded by 1 and is really the trainer's
utilisation. Counting everything the engines completed -- including what the
bound then discarded -- is what makes ``rho > 1`` mean "the engines can outrun
the trainer".

Colour is assigned by the job it does. The staleness bound, the realized lag and
the parts of one total are all ordered magnitudes, so each is one hue stepped
light to dark rather than a categorical set; every line also carries a distinct
dash and marker so the figures survive greyscale.

    python -m experiments.src.staleness_ratio.figures \\
        experiments/outputs/training/math/.../s[0-9]*-t*r*-*.log \\
        --out experiments/outputs/figures/staleness_ratio \\
        --include-pre-queue
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6"
BLUE = {1: "#86b6ef", 2: "#5598e7", 4: "#2a78d6", 8: "#184f95"}
RAMP = [
    "#ddebfc",
    "#c5dcf9",
    "#a4caf5",
    "#7eb4ef",
    "#5598e7",
    "#3981d9",
    "#256abf",
    "#184f95",
    "#0d366b",
]
C_FWD, C_LOGP, C_IDLE, C_ROLL, C_WASTE = "#1c5cab", "#86b6ef", "#e4e2db", "#0b0b0b", "#a8391a"
DASH = {1: (None, None), 2: (5, 2), 4: (1, 2), 8: (7, 2, 1, 2)}
MARK = {1: "o", 2: "s", 4: "^", 8: "D"}

TIMER = re.compile(r"Timer (\w+) end \(elapsed: ([\d.]+)s\)")
NAME = re.compile(r"s(\d+)-t(\d+)r(\d+)")
KMAX = 9
LAG_LABELS = [str(k) for k in range(KMAX - 1)] + [f"{KMAX - 1}+"]
FS_TICK, FS_LAB, FS_NUM, FS_LEG = 15, 17, 13, 15

# The output queue caps at OUTPUT_QUEUE_MAX_GROUPS (fully_async_rollout.py). A run
# sitting against it is backpressured, so its generation rate is the trainer's
# speed rather than the engines' ceiling, and is not a capacity measurement.
QUEUE_THROTTLED = 800


def staleness_histogram(record: dict, prefix: str) -> list[float]:
    """Return exact bins through lag 7 and one overflow bin for lag 8+."""
    counts = [record.get(f"{prefix}/count_{k}", 0.0) for k in range(KMAX)]
    counts[-1] = sum(
        value
        for key, value in record.items()
        if key.startswith(f"{prefix}/count_") and int(key.rsplit("_", 1)[-1]) >= KMAX - 1
    )
    return counts


def parse(path: Path, window: slice, staleness_prefix: str) -> dict | None:
    m = NAME.match(path.name)
    if not m:
        return None
    bound, train, rollout = (int(x) for x in m.group(1, 2, 3))
    try:
        job_id = int(path.stem.rsplit("-", 1)[-1])
    except ValueError:
        job_id = -1
    text = path.read_text(errors="ignore")

    steps, cur = [], {}
    for line in text.splitlines():
        t = TIMER.search(line)
        if t:
            cur[t.group(1)] = float(t.group(2))
            if t.group(1) == "train":
                steps.append(cur)
                cur = {}
    recs = [
        ast.literal_eval(re.search(r"\{.*\}", line).group(0))
        for line in text.splitlines()
        if "fully_async/queue_size" in line
    ]
    w, r = steps[window], recs[window]
    if len(w) < 10 or len(r) < 10:
        return dict(bound=bound, train=train, rollout=rollout, job_id=job_id, ran=False, trace=[])

    def avg(key):
        return st.mean([x.get(key, 0.0) for x in w])

    def met(key):
        vals = [x[key] for x in r if key in x]
        return st.mean(vals) if vals else float("nan")

    kept = met("rollout/fully_async/kept_tokens")
    gross = (
        kept
        + met("rollout/fully_async/stale_tokens")
        + met("rollout/fully_async/aborted_tokens")
        + met("rollout/fully_async/dynamic_filter_tokens")
    )
    t_train = avg("log_probs") + avg("actor_train")
    step = avg("train_wait") + avg("train") + avg("update_weights")

    trace = []
    trace_mean = []
    component_trace = {name: [] for name in ("pre_queue", "in_queue", "total")}
    component_histogram_trace = {name: [] for name in component_trace}
    for rec in recs:
        h = staleness_histogram(rec, staleness_prefix)
        tot = sum(h)
        trace.append([100 * x / tot for x in h] if tot else [float("nan")] * KMAX)
        trace_mean.append(rec.get(f"{staleness_prefix}/mean", float("nan")))
        for name in component_trace:
            component_trace[name].append(rec.get(f"staleness/{name}/mean", float("nan")))
            component_histogram = staleness_histogram(rec, f"staleness/{name}")
            component_total = sum(component_histogram)
            component_histogram_trace[name].append(
                [100 * x / component_total for x in component_histogram]
                if component_total
                else [float("nan")] * KMAX
            )

    return dict(
        bound=bound,
        train=train,
        rollout=rollout,
        job_id=job_id,
        ran=True,
        step=step,
        wait=avg("train_wait"),
        logp=avg("log_probs"),
        fwdbwd=avg("actor_train"),
        t_train=t_train,
        # Time to generate one batch at the rate the engines actually sustained.
        t_roll=kept / (gross / step),
        waste=1 - kept / gross,
        queue=met("rollout/fully_async/queue_size"),
        hist=[sum(staleness_histogram(x, staleness_prefix)[k] for x in r) for k in range(KMAX)],
        trace=trace,
        trace_mean=trace_mean,
        component_trace=component_trace,
        component_histogram_trace=component_histogram_trace,
        pre_queue=met("staleness/pre_queue/mean"),
        in_queue=met("staleness/in_queue/mean"),
        total=met("staleness/total/mean"),
        kept=kept,
        gross=gross,
    )


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=FS_TICK)


def absent(ax, x, y, text="not run"):
    ax.annotate(text, (x, y), color=INK2, fontsize=FS_NUM, ha="center", rotation=90, alpha=0.75)


def draw_step(ax, data, splits, bounds, labels):
    xs = list(range(len(splits)))
    for b in bounds:
        ys = [data[(b, t)]["step"] if data.get((b, t), {}).get("ran") else float("nan") for t, _ in splits]
        ax.plot(xs, ys, color=BLUE[b], marker=MARK[b], markersize=9, linewidth=2.4, dashes=DASH[b],
                label=f"s = {b}", clip_on=False, zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.6)
        for i, v in enumerate(ys):
            if v != v:
                absent(ax, xs[i], 300)
    style(ax)
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.2, len(splits) - 0.8)
    ax.set_xlabel("train : rollout", color=INK2, fontsize=FS_LAB)
    ax.set_ylabel("seconds per rollout", color=INK2, fontsize=FS_LAB)


def draw_breakdown(axes, data, splits, bounds, labels):
    xs = list(range(len(splits)))
    top = max(d["step"] for d in data.values() if d.get("ran")) * 1.12
    for ax, b in zip(axes, bounds):
        for i, (t, _) in enumerate(splits):
            d = data.get((b, t))
            if not d or not d["ran"]:
                absent(ax, i, top * 0.29)
                continue
            f, l, w = d["fwdbwd"], d["logp"], d["wait"]
            ax.bar([i], [f], 0.66, color=C_FWD, zorder=3)
            ax.bar([i], [l], 0.66, bottom=[f + 4], color=C_LOGP, zorder=3)
            ax.bar([i], [w], 0.66, bottom=[f + l + 8], color=C_IDLE, zorder=3)
            ax.plot([i - 0.33, i + 0.33], [d["t_roll"]] * 2, color=C_ROLL, linewidth=2.4, zorder=6)
            ax.annotate(f"{d['t_roll']:.0f}", (i + 0.36, d["t_roll"]), color=INK, fontsize=FS_NUM,
                        ha="left", va="center", zorder=6)
            if d["waste"] > 0.005:
                ax.annotate(f"{d['waste']:.0%}", (i, f + l + w + 12), color=C_WASTE, fontsize=FS_NUM + 1,
                            ha="center", va="bottom", zorder=4, fontweight="bold")
        style(ax)
        ax.set_xticks(xs, labels)
        ax.set_ylim(0, top)
        ax.set_xlabel("train : rollout", color=INK2, fontsize=FS_LAB)
        ax.text(0.5, 1.04, f"s = {b}", transform=ax.transAxes, color=INK, fontsize=FS_LAB, ha="center")
    axes[0].set_ylabel("seconds per rollout", color=INK2, fontsize=FS_LAB)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def draw_lag(axes, data, splits, bounds, labels):
    xs = list(range(len(splits)))
    for ax, b in zip(axes, bounds):
        bottoms = [0.0] * len(splits)
        for k in range(KMAX):
            vals = []
            for t, _ in splits:
                h = data.get((b, t), {}).get("hist")
                vals.append(100 * h[k] / sum(h) if h and sum(h) else float("nan"))
            ax.bar(xs, vals, 0.66, bottom=bottoms, color=RAMP[k], zorder=3)
            for i, v in enumerate(vals):
                if v == v and v >= 8:
                    ax.annotate(f"{LAG_LABELS[k]}\n{v:.0f}%", (i, bottoms[i] + v / 2),
                                color="#ffffff" if k >= 4 else INK, fontsize=FS_NUM,
                                ha="center", va="center", zorder=5, linespacing=1.25)
            bottoms = [x + (v if v == v else 0) for x, v in zip(bottoms, vals)]
        for i, (t, _) in enumerate(splits):
            if not data.get((b, t), {}).get("ran"):
                absent(ax, i, 48)
        style(ax)
        ax.set_xticks(xs, labels)
        ax.set_ylim(0, 100)
        ax.set_xlabel("train : rollout", color=INK2, fontsize=FS_LAB)
        ax.text(0.5, 1.04, f"s = {b}", transform=ax.transAxes, color=INK, fontsize=FS_LAB, ha="center")
    axes[0].set_ylabel("trained groups  [%]", color=INK2, fontsize=FS_LAB)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def fig_overview(data, splits, bounds, labels, out: Path, suffix: str, lag_title: str):
    fig = plt.figure(figsize=(14.4, 15.6), facecolor=SURFACE)
    outer = GridSpec(3, 1, figure=fig, height_ratios=[0.92, 1.0, 1.0], hspace=0.34,
                     left=0.06, right=0.985, top=0.975, bottom=0.045)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.05)
    draw_step(fig.add_subplot(top[0]), data, splits, bounds, labels)

    key = fig.add_subplot(top[1])
    key.axis("off")
    key.legend(
        handles=[Line2D([], [], color=BLUE[b], marker=MARK[b], markersize=9, linewidth=2.4,
                        dashes=DASH[b], markeredgecolor=SURFACE, markeredgewidth=1.6, label=f"s = {b}")
                 for b in bounds]
        + [Patch(facecolor=C_FWD, label="forward + backward + optimizer"),
           Patch(facecolor=C_LOGP, label="log-prob pass"),
           Patch(facecolor=C_IDLE, label="trainer idle"),
           Line2D([], [], color=C_ROLL, linewidth=2.4, label="time to generate one batch"),
           Patch(facecolor=SURFACE, edgecolor=SURFACE, label="bold % = tokens discarded")],
        frameon=False, labelcolor=INK2, fontsize=FS_LEG, loc="center left", ncol=1,
        handlelength=2.6, labelspacing=0.75, borderpad=0,
    )

    mid = outer[1].subgridspec(1, len(bounds), wspace=0.10)
    draw_breakdown([fig.add_subplot(mid[i]) for i in range(len(bounds))], data, splits, bounds, labels)

    bot = outer[2].subgridspec(1, len(bounds), wspace=0.10)
    ax_bot = [fig.add_subplot(bot[i]) for i in range(len(bounds))]
    draw_lag(ax_bot, data, splits, bounds, labels)
    ax_bot[0].legend(handles=[Patch(facecolor=RAMP[k], label=LAG_LABELS[k]) for k in range(KMAX)],
                     frameon=False, labelcolor=INK2, fontsize=FS_LEG, ncol=KMAX, loc="upper center",
                     bbox_to_anchor=(len(bounds) / 2 + 0.28, -0.16),
                     title=lag_title, title_fontproperties={"size": FS_LEG})
    fig.savefig(out / f"overview{suffix}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")


def fig_rho(data, splits, bounds, labels, batch: int, out: Path, suffix: str):
    xs = list(range(len(splits)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.2), facecolor=SURFACE)

    lam_t = [st.mean([batch / data[(b, t)]["t_train"] for b in bounds if data.get((b, t), {}).get("ran")])
             for t, _ in splits]
    ax1.plot(xs, lam_t, color=INK, linewidth=2.6, marker="o", markersize=9, zorder=4,
             markeredgecolor=SURFACE, markeredgewidth=1.6, clip_on=False)
    for i, v in enumerate(lam_t):
        ax1.annotate(f"{v:.2f}", (i, v * 1.04), color=INK, fontsize=FS_NUM, ha="center")
    for b in bounds:
        ys = [batch * (data[(b, t)]["gross"] / data[(b, t)]["kept"]) / data[(b, t)]["step"]
              if data.get((b, t), {}).get("ran") else float("nan") for t, _ in splits]
        ax1.plot(xs, ys, color=BLUE[b], marker=MARK[b], markersize=8, linewidth=2.2, dashes=DASH[b],
                 zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False)
    style(ax1)
    ax1.set_xticks(xs, labels)
    ax1.set_xlim(-0.15, len(splits) - 0.85)
    ax1.set_ylim(0, max(lam_t) * 1.15)
    ax1.set_xlabel("train : rollout", color=INK2, fontsize=FS_LAB)
    ax1.set_ylabel("groups per second", color=INK2, fontsize=FS_LAB)
    ax1.annotate(r"$\lambda_T$", (len(splits) - 0.98, lam_t[-1] * 1.08), color=INK, fontsize=FS_LAB + 2)
    ax1.annotate(r"$\lambda_R$", (len(splits) - 0.98, max(lam_t) * 0.24), color=BLUE[4], fontsize=FS_LAB + 2)

    for b in bounds:
        gross, adm = [], []
        for t, _ in splits:
            d = data.get((b, t))
            if not d or not d["ran"]:
                gross.append(float("nan"))
                adm.append(float("nan"))
                continue
            gross.append(d["t_train"] / d["step"] * (d["gross"] / d["kept"]))
            adm.append(d["t_train"] / d["step"])
        ax2.plot(xs, gross, color=BLUE[b], marker=MARK[b], markersize=8, linewidth=2.2, dashes=DASH[b],
                 zorder=3, markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False)
        ax2.plot(xs, adm, linestyle="none", marker=MARK[b], markersize=8, zorder=4,
                 markerfacecolor="none", markeredgecolor=BLUE[b], markeredgewidth=1.8, clip_on=False)
    ax2.axhline(1.0, color=INK, linewidth=1.4, dashes=(4, 3), zorder=2)
    ax2.axhspan(1.0, 1.75, color="#f0efe9", zorder=1)
    ax2.annotate("train bound", (0.06, 1.30), color=INK2, fontsize=FS_LAB)
    ax2.annotate("rollout bound", (0.06, 0.30), color=INK2, fontsize=FS_LAB)
    ax2.annotate(r"$\rho = 1$", (len(splits) - 0.98, 1.02), color=INK, fontsize=FS_LAB)
    style(ax2)
    ax2.set_xticks(xs, labels)
    ax2.set_xlim(-0.15, len(splits) - 0.85)
    ax2.set_ylim(0, 1.75)
    ax2.set_xlabel("train : rollout", color=INK2, fontsize=FS_LAB)
    ax2.set_ylabel(r"$\rho\;=\;\lambda_R\,/\,\lambda_T$", color=INK2, fontsize=FS_LAB)

    handles = [Line2D([], [], color=BLUE[b], marker=MARK[b], markersize=8, linewidth=2.2, dashes=DASH[b],
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"s = {b}") for b in bounds]
    handles += [
        Line2D([], [], color=INK, linewidth=2.6, marker="o", markersize=9, markeredgecolor=SURFACE,
               label=r"$\lambda_T$  trainer consumption capacity"),
        Line2D([], [], color=INK2, marker="o", markersize=8, linestyle="none", markerfacecolor="none",
               markeredgewidth=1.8, label=r"open: $\lambda_R$ counting admissible groups only"),
    ]
    fig.legend(handles=handles, frameon=False, labelcolor=INK2, fontsize=FS_LEG, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), handlelength=2.8)
    fig.tight_layout(rect=(0, 0.115, 1, 1))
    fig.savefig(out / f"rho{suffix}.png", dpi=200, facecolor=SURFACE)


def fig_rates(data, splits, bounds, labels, out: Path, suffix: str):
    xs = list(range(len(splits)))
    fig, axes_grid = plt.subplots(
        1,
        len(bounds),
        figsize=(14.2, 6.0),
        sharey=True,
        facecolor=SURFACE,
        squeeze=False,
    )
    axes = axes_grid[0]
    top = max(d["kept"] / d["t_train"] for d in data.values() if d.get("ran")) / 1000 * 1.18
    for ax, b in zip(axes, bounds):
        for i, (t, _) in enumerate(splits):
            d = data.get((b, t))
            if not d or not d["ran"]:
                absent(ax, i, top * 0.4)
                continue
            tcap, rcap = d["kept"] / d["t_train"] / 1000, d["gross"] / d["step"] / 1000
            throttled = d["queue"] >= QUEUE_THROTTLED
            ax.bar([i - 0.19], [tcap], 0.34, color=C_FWD, zorder=3,
                   label="trainer: batch / (log-prob + fwd/bwd)" if (b, i) == (bounds[0], 0) else None)
            ax.bar([i + 0.19], [rcap], 0.34, zorder=3, color="none" if throttled else C_LOGP,
                   edgecolor=C_LOGP, linewidth=1.8 if throttled else 0, hatch="///" if throttled else None,
                   label="engines: tokens generated / s" if (b, i) == (bounds[0], 0) else None)
            ax.annotate(f"{tcap:.0f}", (i - 0.19, tcap + top * 0.025), color=INK, fontsize=FS_NUM,
                        ha="center", zorder=6)
            ax.annotate(f"{rcap:.0f}", (i + 0.19, rcap + top * 0.025), color=INK, fontsize=FS_NUM,
                        ha="center", zorder=6)
            if throttled:
                ax.annotate("throttled by\nbackpressure", (i + 0.19, rcap + top * 0.08), color=C_WASTE,
                            fontsize=FS_NUM, ha="center", zorder=6, linespacing=1.2)
            elif d["waste"] > 0.005:
                ax.annotate(f"{d['waste']:.0%}", (i + 0.19, rcap / 2), color="#ffffff", fontsize=FS_NUM + 1,
                            ha="center", va="center", zorder=6, fontweight="bold")
        style(ax)
        ax.set_xticks(xs, labels)
        ax.set_ylim(0, top)
        ax.set_xlabel("train : rollout", color=INK2, fontsize=FS_LAB)
        ax.text(0.5, 1.03, f"s = {b}", transform=ax.transAxes, color=INK, fontsize=FS_LAB, ha="center")
    axes[0].set_ylabel("thousand tokens per second", color=INK2, fontsize=FS_LAB)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, labelcolor=INK2, fontsize=FS_LEG, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.005))
    fig.text(0.008, 0.955, "white % inside the right bar = share of that generation discarded",
             color=INK2, fontsize=FS_NUM)
    fig.tight_layout(rect=(0, 0.11, 1, 0.935))
    fig.savefig(out / f"rates{suffix}.png", dpi=200, facecolor=SURFACE)


def fig_lag_trace(
    data,
    splits,
    bounds,
    window: slice,
    out: Path,
    filename: str,
    lag_title: str,
    component: str | None = None,
):
    def distribution(d):
        if component is None:
            return d.get("trace", [])
        return d.get("component_histogram_trace", {}).get(component, [])

    def means(d):
        if component is None:
            return d.get("trace_mean", [])
        return d.get("component_trace", {}).get(component, [])

    nmax = max((len(distribution(d)) for d in data.values()), default=1)
    fig, axes = plt.subplots(
        len(splits),
        len(bounds),
        figsize=(14.6, 12.6),
        sharey=True,
        sharex=True,
        facecolor=SURFACE,
        squeeze=False,
    )
    for row, (t, r) in enumerate(splits):
        for col, b in enumerate(bounds):
            ax = axes[row][col]
            style(ax)
            ax.set_xlim(0, nmax - 1)
            ax.set_ylim(0, 100)
            if row == 0:
                ax.text(0.5, 1.06, f"s = {b}", transform=ax.transAxes, color=INK, fontsize=FS_LAB,
                        ha="center")
            if col == 0:
                ax.set_ylabel(f"{t}:{r}\n\ntrained groups  [%]", color=INK2, fontsize=FS_LAB)
            if row == len(splits) - 1:
                ax.set_xlabel("rollout", color=INK2, fontsize=FS_LAB)
            d = data.get((b, t))
            rows = distribution(d) if d else []
            if not rows:
                ax.annotate("not run", (0.5, 0.5), xycoords="axes fraction", color=INK2,
                            fontsize=FS_LAB, ha="center", va="center", alpha=0.8)
                continue
            ax.stackplot(range(len(rows)), *[[x[k] for x in rows] for k in range(KMAX)],
                         colors=RAMP, zorder=3, edgecolor="none")
            w = rows[window]
            if w:
                window_means = [x for x in means(d)[window] if not math.isnan(x)]
                if window_means:
                    mean = st.mean(window_means)
                    ax.annotate(f"mean {mean:.2f}", (0.97, 0.06), xycoords="axes fraction", color=INK,
                                fontsize=FS_NUM, ha="right", zorder=6)
            if len(rows) < nmax:
                ax.annotate("run ended", (len(rows) - 1, 50), color=INK2, fontsize=FS_NUM, rotation=90,
                            ha="right", va="center", alpha=0.8, zorder=6)
    fig.legend(handles=[Patch(facecolor=RAMP[k], label=LAG_LABELS[k]) for k in range(KMAX)], frameon=False,
               labelcolor=INK2, fontsize=FS_LEG, ncol=KMAX, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               title=lag_title, title_fontproperties={"size": FS_LEG})
    fig.tight_layout(rect=(0, 0.062, 1, 1))
    fig.savefig(out / filename, dpi=200, facecolor=SURFACE)


def fig_staleness_components(data, splits, bounds, labels, window: slice, out: Path, suffix: str):
    components = (
        ("pre_queue", "pre-queue staleness\nQ - R"),
        ("in_queue", "in-queue staleness\nC - Q"),
        ("total", "total staleness\nC - R"),
    )
    matrices = []
    for key, _ in components:
        matrices.append(
            [
                [
                    data[(bound, train)][key]
                    if data.get((bound, train), {}).get("ran")
                    else float("nan")
                    for train, _ in splits
                ]
                for bound in bounds
            ]
        )

    finite = [value for matrix in matrices for row in matrix for value in row if not math.isnan(value)]
    vmax = max(finite, default=1.0)
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("#eceae4")

    fig, axes = plt.subplots(1, len(components), figsize=(15.2, 6.8), facecolor=SURFACE,
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.11, top=0.68, wspace=0.04)
    image = None
    for ax, (_, title), matrix in zip(axes, components, matrices):
        image = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_title(title, color=INK, fontsize=FS_LAB, linespacing=1.35, pad=14)
        ax.set_xticks(range(len(splits)), labels)
        ax.set_yticks(range(len(bounds)), [f"s = {bound}" for bound in bounds])
        ax.set_xlabel("train : rollout", color=INK2, fontsize=FS_LAB)
        ax.tick_params(colors=INK2, labelsize=FS_TICK, length=0)
        ax.set_xticks([x - 0.5 for x in range(len(splits) + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(len(bounds) + 1)], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for side in ax.spines.values():
            side.set_visible(False)

        for row, values in enumerate(matrix):
            for col, value in enumerate(values):
                if math.isnan(value):
                    label, color = "not run", INK2
                else:
                    label = f"{value:.2f}"
                    color = "#ffffff" if value > vmax * 0.52 else INK
                ax.text(col, row, label, ha="center", va="center", color=color,
                        fontsize=FS_NUM + 1, fontweight="bold" if not math.isnan(value) else "normal")

    axes[0].set_ylabel("max weight staleness", color=INK2, fontsize=FS_LAB)
    colorbar_ax = fig.add_axes([0.915, 0.16, 0.018, 0.48])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("mean weight-version gap", color=INK2, fontsize=FS_LAB)
    colorbar.ax.tick_params(colors=INK2, labelsize=FS_TICK)
    colorbar.outline.set_visible(False)
    first = window.start if window.start is not None else 0
    last = window.stop - 1 if window.stop is not None else "end"
    fig.suptitle(
        f"Mean trained-group staleness, rollouts {first}-{last}\n"
        "R = selected bound reference, Q = group ready, C = training drain",
        color=INK,
        fontsize=FS_LAB + 2,
        y=0.97,
    )
    fig.savefig(out / f"staleness_components{suffix}.png", dpi=200, facecolor=SURFACE,
                bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", nargs=2, type=int, default=(10, 36), metavar=("FIRST", "LAST"),
                    help="rollouts to average over; must be a range every point of the sweep reached")
    ap.add_argument("--batch", type=int, default=192,
                    help="--rollout-batch-size, the groups one training step consumes")
    ap.add_argument(
        "--include-pre-queue",
        action="store_true",
        help="plot selected-reference total staleness C-R and write explicit component figures",
    )
    args = ap.parse_args()

    window = slice(*args.window)
    if args.include_pre_queue:
        staleness_prefix = "staleness/total"
        suffix = "_selected_reference_total_staleness"
        lag_title = "realized total lag C-R (includes pre-queue)  [weight updates]"
    else:
        staleness_prefix = "staleness/bound/train"
        suffix = ""
        lag_title = "realized selected-reference lag C-R  [weight updates]"
    data = {}
    for path in args.logs:
        parsed = parse(path, window, staleness_prefix)
        if parsed:
            key = (parsed["bound"], parsed["train"])
            previous = data.get(key)
            # A dependency chain leaves several logs per point. Prefer the newest
            # segment that reached the analysis window; surplus post-completion
            # jobs can be newer but contain no training records.
            if previous is None or (parsed["ran"], parsed["job_id"]) > (
                previous["ran"],
                previous["job_id"],
            ):
                data[key] = parsed
    if not data:
        raise SystemExit("no logs matched s<bound>-t<train>r<rollout>-<jobid>.log")

    bounds = sorted({k[0] for k in data})
    splits = sorted({(d["train"], d["rollout"]) for d in data.values()})
    labels = [f"{t}:{r}" for t, r in splits]

    args.out.mkdir(parents=True, exist_ok=True)
    fig_overview(data, splits, bounds, labels, args.out, suffix, lag_title)
    fig_rho(data, splits, bounds, labels, args.batch, args.out, suffix)
    fig_rates(data, splits, bounds, labels, args.out, suffix)
    if args.include_pre_queue:
        for component, filename, title in (
            (
                "pre_queue",
                "lag_trace_pre_queue_staleness.png",
                "realized pre-queue lag Q-R  [weight updates]",
            ),
            (
                "in_queue",
                "lag_trace_in_queue_staleness.png",
                "realized in-queue lag C-Q  [weight updates]",
            ),
            (
                "total",
                "lag_trace_total_staleness.png",
                "realized total lag C-R  [weight updates]",
            ),
        ):
            fig_lag_trace(
                data,
                splits,
                bounds,
                window,
                args.out,
                filename,
                title,
                component,
            )
        fig_staleness_components(data, splits, bounds, labels, window, args.out, suffix)
    else:
        fig_lag_trace(
            data,
            splits,
            bounds,
            window,
            args.out,
            "lag_trace.png",
            lag_title,
        )

    missing = [(b, t) for b in bounds for t, _ in splits if not data.get((b, t), {}).get("ran")]
    if missing:
        print(f"no steady state for: {missing}")
    print(f"window rollouts {args.window[0]}-{args.window[1] - 1}, {len(data)} points")
    print(f"output directory: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
