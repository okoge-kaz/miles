"""Pull the train/rollout time structure out of a job log.

Both of the lines this needs carry a wall clock, which is what makes a real
timeline possible rather than a bar chart of averages:

    [2026-08-06 07:47:45.630 actor_cell0_rank0] timer.py:42 - Timer actor_train end (elapsed: 173.1s)
    (SGLangEngine pid=3725811, ip=...) [2026-08-06 07:47:53] Decode batch, #running-req: 129, ... token usage: 0.22, ... gen throughput (token/s): 12405.28, #queue-req: 0

A timer line gives an end instant and a duration, so the span is
``[end - elapsed, end]``. A decode line gives one engine's concurrency, KV
occupancy and throughput at an instant. Together they answer the question the
appendix asks: *while the trainer was idle, what were the engines doing?*

Stdlib only, so it runs on a login node without an environment.

    python -m experiments.src.node_ratio.parse_logs <log> [<log> ...]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Ray prefixes every forwarded line, so anchor on the bracketed timestamp rather
# than the start of the line.
TIMER = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) [^\]]*\] timer\.py:\d+ - Timer (\w+) end \(elapsed: ([\d.]+)s\)"
)
DECODE = re.compile(
    # An engine with TP>1 stamps its rank inside the bracket ("... 23:33:39 TP0").
    r"\(SGLangEngine pid=(\d+)[^)]*\) \[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] Decode batch, "
    r"#running-req: (\d+), #token: (\d+), token usage: ([\d.]+).*?"
    # Ray splices "[repeated Nx across cluster]" in before #queue-req on some
    # lines, so the tail is optional rather than required.
    r"gen throughput \(token/s\): ([\d.]+)(?:.*?#queue-req: (\d+))?"
)
PLACEMENT = re.compile(r"placement (\d+)x(\d+) train \((\d+) GPU\) \+ (\d+) rollout, tp(\d+) cp(\d+) -> dp(\d+)")
# Ray colours the prefix it adds to every forwarded line, and the escapes land
# between the engine id and the timestamp, so they have to come off first.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ARG = re.compile(r"^ +(\w+) \.+ (\S+)$", re.M)

# A leading step whose train_wait is under this is draining the buffer the
# engines filled during startup, not measuring steady state.
DRAIN_WAIT_S = 2.0

# Phases drawn on the trainer track, in the order they occur within a step.
TRAINER_PHASES = ("train_wait", "log_probs", "ref_log_probs", "actor_train", "update_weights")


def _ts(text: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in text else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(text, fmt).timestamp()


@dataclass(frozen=True)
class Span:
    """One trainer phase, reconstructed as ``[end - elapsed, end]``."""

    name: str
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Decode:
    t: float
    engine: str
    running_req: int
    kv_use: float
    throughput: float
    queue_req: int

    @property
    def per_seq_tok_s(self) -> float:
        """Throughput divided over the sequences sharing the engine.

        This is the quantity that makes rollout scaling sublinear: it rises as
        concurrency falls, so adding rollout nodes speeds up each *sequence*, but
        only up to the uncontended rate.
        """
        return self.throughput / self.running_req if self.running_req else 0.0


@dataclass
class RunLog:
    name: str
    train_gpus: int = 0
    rollout_gpus: int = 0
    max_response_len: int = 0
    global_batch_size: int = 0
    spans: list[Span] = field(default_factory=list)
    decodes: list[Decode] = field(default_factory=list)

    @property
    def rollout_nodes(self) -> int:
        return self.rollout_gpus // 8

    @property
    def t0(self) -> float:
        """Wall clock of the first trainer phase, so runs can be overlaid."""
        return min((s.start for s in self.spans), default=0.0)

    def steps(self) -> list[dict[str, float]]:
        """Group spans into steps, closing each one on ``train``."""
        out, cur = [], {}
        for s in self.spans:
            cur[s.name] = s.seconds
            if s.name == "train":
                out.append(cur)
                cur = {}
        return out

    def steady(self) -> list[dict[str, float]]:
        """Steps after the cold start and after the startup buffer has drained."""
        tail = self.steps()[1:]
        # The drain is a prefix that ENDS when train_wait rises. If it never
        # rises, there was no buffer to drain -- rollout simply keeps up and a
        # low wait is the steady state. Dropping those steps would discard
        # exactly the well-provisioned configuration the sweep is looking for.
        if not any(s.get("train_wait", 0.0) >= DRAIN_WAIT_S for s in tail):
            return tail
        n = 0
        while n < len(tail) and tail[n].get("train_wait", 0.0) < DRAIN_WAIT_S:
            n += 1
        return tail[n:]

    def summary(self) -> dict[str, float] | None:
        steady = self.steady()
        if not steady:
            return None

        def mean(key: str) -> float:
            return st.mean(s.get(key, 0.0) for s in steady)

        step_s = mean("train_wait") + mean("train")
        total_gpus = self.train_gpus + self.rollout_gpus
        return {
            "name": self.name,
            "rollout_nodes": self.rollout_nodes,
            "train_gpus": self.train_gpus,
            "rollout_gpus": self.rollout_gpus,
            "total_gpus": total_gpus,
            "steps": len(steady),
            "step_s": step_s,
            "train_wait": mean("train_wait"),
            "train": mean("train"),
            "actor_train": mean("actor_train"),
            "log_probs": mean("log_probs"),
            "steps_per_h_per_gpu": 3600.0 / step_s / total_gpus if step_s and total_gpus else 0.0,
            "kv_use": st.mean(d.kv_use for d in self.decodes) if self.decodes else 0.0,
            "running_req": st.mean(d.running_req for d in self.decodes) if self.decodes else 0.0,
            "per_seq_tok_s": st.mean(d.per_seq_tok_s for d in self.decodes) if self.decodes else 0.0,
            "max_response_len": self.max_response_len,
            "global_batch_size": self.global_batch_size,
        }


def parse(path: Path) -> RunLog:
    text = ANSI.sub("", path.read_text(errors="ignore"))
    run = RunLog(name=path.stem)

    if m := PLACEMENT.search(text):
        run.train_gpus, run.rollout_gpus = int(m.group(3)), int(m.group(4))

    args = dict(ARG.findall(text))
    run.max_response_len = int(args.get("rollout_max_response_len", 0) or 0)
    run.global_batch_size = int(args.get("global_batch_size", 0) or 0)

    for ts, name, elapsed in TIMER.findall(text):
        end, secs = _ts(ts), float(elapsed)
        run.spans.append(Span(name=name, start=end - secs, end=end))

    for pid, ts, req, _tok, kv, thr, q in DECODE.findall(text):
        run.decodes.append(
            Decode(
                t=_ts(ts),
                engine=pid,
                running_req=int(req),
                kv_use=float(kv),
                throughput=float(thr),
                queue_req=int(q) if q else 0,
            )
        )

    run.spans.sort(key=lambda s: s.end)
    run.decodes.sort(key=lambda d: d.t)
    return run


def uncontended_tok_s(runs: list[RunLog], max_req: int = 4) -> float | None:
    """Single-stream decode rate, from decode lines with almost nothing in flight.

    This sets the hard floor on a rollout step: the longest sample needs
    ``max_response_len / this`` seconds no matter how many GPUs are added,
    because token n+1 cannot be produced before token n.
    """
    rates = [d.per_seq_tok_s for r in runs for d in r.decodes if 0 < d.running_req <= max_req]
    return st.median(rates) if rates else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="write the per-run summaries here")
    args = ap.parse_args()

    runs = [parse(p) for p in args.logs]
    rows = [s for r in runs if (s := r.summary())]
    rows.sort(key=lambda r: r["rollout_nodes"])

    hdr = f"{'run':<26}{'roll':>5}{'steps':>6}{'step_s':>9}{'wait':>8}{'train':>8}{'req/eng':>9}{'kv':>6}{'tok/s/seq':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['name'][:25]:<26}{r['rollout_nodes']:>5}{r['steps']:>6}{r['step_s']:>9.1f}"
            f"{r['train_wait']:>8.1f}{r['train']:>8.1f}{r['running_req']:>9.1f}{r['kv_use']:>6.2f}"
            f"{r['per_seq_tok_s']:>11.1f}"
        )

    if (rate := uncontended_tok_s(runs)) is not None:
        longest = rows[0]["max_response_len"] / rate if rows else 0.0
        print(f"\nuncontended decode {rate:.1f} tok/s -> longest sample floor {longest:.0f} s")
    else:
        print("\nno low-concurrency decode lines; pass --uncontended-tok-s to the figure script")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
