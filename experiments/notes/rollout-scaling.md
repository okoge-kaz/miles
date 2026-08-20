# Why rollout time stops falling when you add rollout GPUs

Two separate floors hold rollout time up, and they need separate experiments
because they have different fixes and only one of them is a property of the
method rather than of the configuration.

## Floor 1: the longest sample's latency

Measured directly, not modelled. Taking every SGLang decode line at
`#running-req: 1` from job 15150857 (26 observations), a single sequence decodes
at **203.6 tok/s**. At `--rollout-max-response-len 24576`, one sample that runs
to the cap therefore takes

    24576 / 203.6 = 120.7 s

and `rollout/response_len/max` is exactly 24576 on **every** step of every run
measured, with `rollout/truncated_ratio` between 0.4% and 4.3% -- so 1 to 11 of
the 256 samples hit the cap every single step. This is not a rare tail event, it
is the steady state.

If a rollout step has to contain the whole generation of its longest sample,
120.7 s is a hard floor. No amount of rollout GPUs moves it: the sequence is
serial, and a second GPU cannot decode token *n+1* before token *n* exists.

**The floor does not bind here.** Observed steady-state `perf/rollout_time` is
65-78 s, well under 120.7 s. That is only possible if long samples span more than
one rollout step, which is exactly what `--fully-async` with
`--max-weight-staleness 2` permits. So adding rollout capacity is not futile in
this configuration -- floor 2 is what is holding it up.

### The experiment worth publishing

The interesting claim is the conditional one: *where the floor binds, rollout
scaling saturates, and removing the floor is what async is actually for.* It
isolates the mechanism instead of reporting that async is faster.

    x-axis   rollout GPUs: 8, 16, 24, 32, 48
    series   (a) colocated/sync
             (b) fully-async queue-max, --max-weight-staleness 0
             (c) fully-async queue-recycle, --max-weight-staleness 2
    y-axis   perf/rollout_time, steady state
    marker   a horizontal line at max_response_len / 203.6 = 120.7 s

`queue-recycle` cannot represent the zero endpoint: its strict admission rule
`D-F < M` admits no nonnegative gap at `M=0`. Series (b) therefore uses
`queue-max`, whose dequeue rule accepts equality. This queue-type change must
remain explicit in the legend.

Prediction: (a) and (b) flatten onto the 120.7 s line and stay there while GPUs
keep being added -- both forbid a sample from crossing a step boundary, so both
inherit the floor. (c) passes under the line and keeps falling until floor 2
catches it. Staleness 0 is the control that matters: it separates "async removes
the latency floor" from "async happens to come with other differences".

A second panel makes the floor's origin explicit by sweeping
`--rollout-max-response-len` (8k, 16k, 24k, 32k) in sync mode and showing that
the plateau tracks `len / 203.6` -- the floor is the generation length divided by
single-stream decode speed, and nothing else.

Cheap to run: rollout time is visible within a few steps, so these do not need
convergence, only steady state.

## Floor 2: concurrency is capped by the training batch

`--async-max-concurrent-samples` defaults to `None`, which miles resolves to
`rollout_batch_size * n_samples_per_prompt` = 32 * 8 = **256** trajectories in
flight (`arguments.py:629-639`). That is a training-batch quantity being used as
a generation-concurrency quantity, and the two have no reason to be equal.

Spread over N single-GPU engines, each engine gets 256/N samples. Per-engine
decode throughput measured against concurrency in the same job:

| `#running-req` | tok/s per engine | tok/s per sequence |
|---:|---:|---:|
| 1 | 204 | 204 |
| 8 | 1418 | 177 |
| 16 | 2420 | 151 |
| 24 | 2004 | 84 |
| 40 | 4301 | 108 |
| 54 | 3684 | 68 |

An engine at 40 concurrent requests does ~20x the aggregate work of one at 1.
So adding rollout GPUs at a fixed 256 samples moves every engine left along this
table, and the dilution nearly cancels the added capacity:

    16 engines: 256/16 = 16 each -> ~2400 tok/s each -> ~38k tok/s total
    24 engines: 256/24 = 10.7    -> ~1700 tok/s each -> ~41k tok/s total

50% more GPUs for ~6% more throughput. That is the measured 3n-vs-4n result
(steps/h/GPU 1.66-1.88 against 1.37-1.55), and it is a configuration artifact,
not a property of async.

KV cache is not the constraint: `token usage` sits at 0.09-0.14, so there is
7-10x headroom before memory binds.

**In flight** (jobs 15161704-6): `--async-max-concurrent-samples` at default,
512, and 1024, at a fixed 3 nodes (1 train node + 2 rollout), 90 minutes each.
90 rather than 40 because the first 1-4 steps are buffer drain, not steady state
-- see the drain handling in `analyze_throughput.py`.

## What the answer changes

The target is `rollout_time` slightly below `actor_train + log_probs` ~= 57 s, so
the trainer paces and never starves. From 65-78 s at 24 rollout GPUs, the gap is
about 1.3x. Floor-2 arithmetic says raising concurrency should get there without
buying nodes; adding nodes at fixed concurrency should not.

## SGLang static memory, measured (2026-08-06)

**Adopted: async 0.70, colocated 0.80.** Each recipe carries the value that was
measured on that recipe, at `MAX_RESPONSE_LEN=32768` and the production batch
shape (rbs 256, n 8, gbs 2048).

| config | KV tokens | free after graph | `token usage` p50 | retractions | `train_wait` |
|---|---|---|---|---|---|
| async 0.70 | 343,207 | 19.50 GB | 0.92 | 820 | 446 s |
| async 0.80 | 400,368 | 11.77 GB | 0.92 | 910 | 403 s |
| async 0.85 | 428,948 | 7.83 GB | — | — | **hangs** |
| colocated 0.80 | 843,351 | 12.59 GB | 0.47 | 245 | 547 s |

Jobs 15197854 (a070), 15200997 (a080), 15190819 (a085), 15192282 (colo080).
Retraction counts are over 5 rollout steps except colocated, which is 4.

### The previous revision of this section was wrong

It argued: 19.5 GB sits idle on dedicated rollout GPUs, so raise the fraction;
a bigger KV cache means fewer retractions, and retraction inflates realized lag,
which is a headline output of the study. The reasoning was checked against a
measurement of *idle memory* and against nothing else.

**Both halves failed.**

*0.85 does not run at all.* Engines came up, captured CUDA graphs, and then
served only `/health` for a full hour — zero generation requests, zero rollout
steps, no OOM and no traceback. `flush_cache` appears three times and then the
main process logs nothing, so it is stuck in the initial `update_weights`
(`train_async.py:49`), before generation. It is a silent hang, not a clean
failure, which is the worst way for a memory setting to be wrong. The 0.70
control on identical code, batch shape and node count completed five rollouts —
that comparison is what isolates the fraction as the cause, and cancelling it
"because the KV arithmetic is already settled" was a mistake that cost a rerun.

*Raising the fraction does not relieve KV pressure.* 0.70 -> 0.80 buys 17% more
cache and leaves `token usage` p50 at 0.92 in both, with retractions going **up**
(820 -> 910). SGLang admits concurrent requests until the cache is full, so a
bigger pool is spent on more in-flight sequences rather than on slack. The
utilisation is a property of the scheduler, not of the pool size, and
`train_wait` moved 446 s -> 403 s, which is inside the run-to-run spread.

So the retraction-contaminates-lag concern is real but **is not addressable by
this knob**. If it needs fixing, the lever is concurrency
(`--async-max-concurrent-samples`) or rollout capacity, not memory.

### Why the two arms differ

colocated sits at `token usage` p50 0.47 against async's 0.92 on the same batch
shape, because `ROLLOUT_NUM_GPUS_PER_ENGINE=2` there gives 843k tokens per engine
against async's 343k at 1 GPU/engine. colocated is simply not under the same
pressure, and 0.80 was measured working on it with healthy headroom.

async stays at 0.70: 0.80 showed no measured benefit and sits closer to the
boundary where 0.85 hangs. Taking risk without measured benefit is not a trade.


### Step cost at the production batch shape, and why it is generation-bound

Same jobs, steady-state steps only (step 1 is cold start and is dropped).

| | `train_wait` | `actor_train` | `log_probs` | `train` | step |
|---|---|---|---|---|---|
| async 0.70, 1 rollout node | 344 / 640 / 426 / 372 s | ~178 s | ~43 s | ~226 s | **~620 s** |
| async 0.80, 1 rollout node | 409 / 482 / 332 / 388 s | ~175 s | ~44 s | ~220 s | **~620 s** |
| colocated 0.80, 2 nodes | 489 / 539 / 612 s | ~93 s | ~24 s | ~117 s | **~660 s** |

`train_wait` is 1.7-2.9x `train` on the async arm at one rollout node, so
generation is the bottleneck by a wide margin and the trainer idles most of the
step. That ratio is the reason the node-ratio sweep exists, and it predicts the
crossing near 3 rollout nodes.

The cause is response length, which the move to a 32768 budget changed
substantially:

```
rollout/response_len/p90 = 9,431 - 9,966
rollout/response_len/p99 = 32,768      <- ~1% of samples run to the cap
```

Every earlier estimate in these notes used the 6,880-token mean measured at a
24576 budget. That number is superseded: the tail is far heavier at 32768, a
percent of samples run the full length, and generation cost scales with it. Any
throughput arithmetic written before 2026-08-06 is low.
