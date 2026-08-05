#!/usr/bin/env python3
"""Turn measure_pass_rate output into reportable accuracies.

    experiments/src/offline_eval/report.py <dir> [<dir> ...]

Each ``<dir>/<benchmark>.jsonl`` is one benchmark's per-prompt record, as written
by measure_pass_rate.py. Reports the per-benchmark accuracy and the pooled number
over all of them.

Both are printed on purpose. The pooled figure has the smaller standard error —
30 AIME problems put it near 9 points, 120 near 4.5 — and is the one to compare
runs on. The per-year figures are what published numbers are stated in, and they
are also the only place a contamination gap can show: AIME 2024 and 2025 may sit
in pretraining data, AIME 2026 was held after every checkpoint here was trained,
so a model that is memorising shows a gap that the pooled mean hides.

The standard error reported is over problems, treating each problem's avg@k as
the observation. That is the component that dominates at k=16: raising k shrinks
the within-problem term and does nothing to this one.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarise(rows: list[dict]) -> dict:
    rates = [r["pass_rate"] for r in rows if "pass_rate" in r]
    if not rates:
        return {}
    n = len(rates)
    mean = st.mean(rates)
    # SE over problems; with n=1 there is no spread to report.
    sd = st.stdev(rates) if n > 1 else 0.0
    trunc = [r.get("truncated_frac") for r in rows if r.get("truncated_frac") is not None]
    return {
        "n": n,
        "acc": mean,
        "se": sd / math.sqrt(n) if n > 1 else float("nan"),
        "truncated": st.mean(trunc) if trunc else float("nan"),
        "resp_len": st.mean(r["response_len_mean"] for r in rows if "response_len_mean" in r)
        if any("response_len_mean" in r for r in rows)
        else float("nan"),
    }


def report(directory: Path) -> None:
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        print(f"{directory}: no benchmark files")
        return

    print(f"\n{directory}")
    print(f"  {'benchmark':<12}{'n':>5}{'acc':>9}{'±SE':>8}{'trunc':>8}{'resp_len':>10}")
    print("  " + "-" * 52)

    pooled: list[float] = []
    for f in files:
        rows = load(f)
        s = summarise(rows)
        if not s:
            continue
        pooled.extend(r["pass_rate"] for r in rows if "pass_rate" in r)
        print(
            f"  {f.stem:<12}{s['n']:>5}{100 * s['acc']:>8.1f}%{100 * s['se']:>7.1f}"
            f"{100 * s['truncated']:>7.1f}%{s['resp_len']:>10.0f}"
        )

    if len(files) > 1 and pooled:
        n = len(pooled)
        se = st.stdev(pooled) / math.sqrt(n)
        print("  " + "-" * 52)
        print(f"  {'pooled':<12}{n:>5}{100 * st.mean(pooled):>8.1f}%{100 * se:>7.1f}")
        print()
        print("  Compare runs on the pooled number; compare against published")
        print("  results per benchmark. A gap between the pre- and post-cutoff")
        print("  years is a contamination signal the pooled mean would hide.")


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]] or [Path(".")]
    for d in dirs:
        report(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
