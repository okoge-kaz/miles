#!/usr/bin/env python3
"""Compare the throughput of finished runs from their job logs.

    experiments/analyze_throughput.py <log> [<log> ...]
    experiments/analyze_throughput.py experiments/outputs/training/math/*/*/tp-*.log

Reports, per run, the steady-state step: what the trainer spent its time on,
how long it waited for rollout, and what the engines produced. The first
training step is dropped everywhere — it carries the cold-start cost of the
first weight sync and the first reference/logprob pass, which is several times
the steady value and would swamp a mean over a short run.

The two numbers to compare across configurations are `step_s` (wall clock per
optimizer step) and `tok/s/gpu` (generation throughput per GPU in the whole
allocation), because they answer "how fast" and "how efficiently" separately.
"""

from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

TIMER = re.compile(r"Timer (\w+) end \(elapsed: ([\d.]+)s\)")
DECODE = re.compile(
    r"#running-req: (\d+), #token: (\d+), token usage: ([\d.]+).*gen throughput \(token/s\): ([\d.]+)"
)
PLACEMENT = re.compile(r"placement (\d+)x(\d+) train \((\d+) GPU\) \+ (\d+) rollout, tp(\d+) cp(\d+) -> dp(\d+)")
METRIC = re.compile(r"'(rollout/[a-z_/]+)': ([\d.]+)")


def analyse(path: Path) -> dict | None:
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    m = PLACEMENT.search(text)
    if not m:
        return None
    train_gpus, rollout_gpus = int(m.group(3)), int(m.group(4))
    total_gpus = train_gpus + rollout_gpus

    steps, cur = [], {}
    for line in lines:
        t = TIMER.search(line)
        if not t:
            continue
        cur[t.group(1)] = float(t.group(2))
        if t.group(1) == "train":
            steps.append(cur)
            cur = {}
    steady = steps[1:]
    if not steady:
        return None

    def mean(key: str) -> float:
        return st.mean(s.get(key, 0.0) for s in steady)

    # Rollout side, over the same steady window.
    first_train = next((i for i, l in enumerate(lines) if "Timer train end" in l), 0)
    dec = [tuple(map(float, d.groups())) for l in lines[first_train:] if (d := DECODE.search(l))]

    metrics = {}
    for line in lines:
        for k, v in METRIC.findall(line):
            metrics[k] = float(v)

    step_s = mean("train_wait") + mean("train")
    resp = metrics.get("rollout/response_len/mean", 0.0)
    # Samples per step is fixed by the batch shape; recover it from the log's own
    # global batch rather than assuming, so a swept batch shape stays correct.
    gbs = None
    g = re.search(r"--global-batch-size\D+(\d+)", text) or re.search(r"global_batch_size \.+ (\d+)", text)
    if g:
        gbs = int(g.group(1))

    return {
        "name": path.stem,
        "train_gpus": train_gpus,
        "rollout_gpus": rollout_gpus,
        "steps": len(steady),
        "step_s": step_s,
        "train_wait": mean("train_wait"),
        "actor_train": mean("actor_train"),
        "log_probs": mean("log_probs"),
        "ref_log_probs": mean("ref_log_probs"),
        "update_weights": mean("update_weights"),
        "run_req": st.mean(d[0] for d in dec) if dec else 0.0,
        "kv_use": st.mean(d[2] for d in dec) if dec else 0.0,
        "tok_s_engine": st.mean(d[3] for d in dec) if dec else 0.0,
        "resp_len": resp,
        "gbs": gbs,
        "total_gpus": total_gpus,
    }


def main() -> int:
    rows = []
    for arg in sys.argv[1:]:
        try:
            r = analyse(Path(arg))
        except Exception as exc:  # noqa: BLE001
            print(f"  {arg}: {exc}", file=sys.stderr)
            continue
        if r:
            rows.append(r)
        else:
            print(f"  {Path(arg).stem}: no steady-state step yet", file=sys.stderr)
    if not rows:
        return 1

    rows.sort(key=lambda r: r["step_s"])
    hdr = (
        f"{'run':<22}{'train':>6}{'roll':>5}{'n':>3}{'step_s':>8}{'wait':>7}"
        f"{'actor':>7}{'logp':>6}{'ref':>6}{'req/eng':>8}{'kv':>6}{'tok/s/eng':>10}{'tok/s/gpu':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        agg = r["tok_s_engine"] * r["rollout_gpus"] if r["rollout_gpus"] else r["tok_s_engine"] * r["train_gpus"]
        print(
            f"{r['name'][:22]:<22}{r['train_gpus']:>6}{r['rollout_gpus']:>5}{r['steps']:>3}"
            f"{r['step_s']:>8.1f}{r['train_wait']:>7.1f}{r['actor_train']:>7.1f}"
            f"{r['log_probs']:>6.1f}{r['ref_log_probs']:>6.1f}"
            f"{r['run_req']:>8.1f}{r['kv_use']:>6.2f}{r['tok_s_engine']:>10.0f}{agg / r['total_gpus']:>10.0f}"
        )
    print()
    print("step_s = train_wait + train.  wait>0 means the trainer is starved (rollout-bound);")
    print("wait~0 with the rollout node saturated means the trainer is the pacer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
