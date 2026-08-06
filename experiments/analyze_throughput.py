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
    # Two prefixes have to come off, not one.
    #
    # Step 1 carries the cold-start cost. Everyone drops that. But the several
    # minutes of startup are minutes the rollout engines spend generating, so by
    # the time the first optimizer step runs there is already a buffer of ready
    # groups. The next few steps drain it at train_wait ~= 0.4s, which is not a
    # throughput measurement -- it is the trainer running unobstructed against
    # work that was produced while it was still booting. Steady state begins when
    # the buffer is gone and train_wait rises to the rate the engines can
    # actually sustain.
    #
    # Measured on six replicates: 3n drained for four steps at 0.4-0.5s and then
    # jumped to 33s; 4n drained for one and settled at 15-19s. Averaging across
    # the transition reported 65-77s per step where the sustained value is
    # 74-90s, and inverted the 3n/4n ranking.
    DRAIN_WAIT_S = 2.0
    tail = steps[1:]
    n_drained = 0
    while n_drained < len(tail) and tail[n_drained].get("train_wait", 0.0) < DRAIN_WAIT_S:
        n_drained += 1
    steady = tail[n_drained:]
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
        "drained": n_drained,
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
        f"{'run':<22}{'train':>6}{'roll':>5}{'n':>3}{'drn':>4}{'step_s':>8}{'wait':>7}"
        f"{'actor':>7}{'logp':>6}{'ref':>6}{'req/eng':>8}{'kv':>6}{'tok/s/eng':>10}{'steps/h/gpu':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    thin = False
    for r in rows:
        thin = thin or r["steps"] < 3
        print(
            f"{r['name'][:22]:<22}{r['train_gpus']:>6}{r['rollout_gpus']:>5}{r['steps']:>3}{r['drained']:>4}"
            f"{r['step_s']:>8.1f}{r['train_wait']:>7.1f}{r['actor_train']:>7.1f}"
            f"{r['log_probs']:>6.1f}{r['ref_log_probs']:>6.1f}"
            f"{r['run_req']:>8.1f}{r['kv_use']:>6.2f}{r['tok_s_engine']:>10.0f}"
            f"{3600 / r['step_s'] / r['total_gpus']:>12.2f}"
        )
    print()
    print("n   = steady steps kept.  drn = leading steps dropped as buffer drain")
    print("      (train_wait < 2s: the trainer eating groups the engines produced during startup).")
    print("steps/h/gpu is the number to rank configurations on -- wall clock alone rewards")
    print("      throwing GPUs at the problem.")
    print("wait>0 at steady state means the trainer is starved and rollout is the pacer;")
    print("      making training faster then buys nothing until the balance is changed.")
    if thin:
        print()
        print("WARNING: a run has fewer than 3 steady steps. The job was too short to")
        print("         measure sustained throughput -- treat its numbers as indicative only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
