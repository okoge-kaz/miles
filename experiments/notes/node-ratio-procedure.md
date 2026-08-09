# Choosing the train:rollout node split

The procedure for fixing one train:rollout split for the whole study, and the
reason each step is in the order it is. Nothing here has been run yet — every
measurement taken before 2026-08-06 is discarded, because the staleness cap was
never enforced (`notes/off-policy-variables.md`) and the throughput analysis
silently dropped well-provisioned configurations (`analyze_throughput.py` drain
rule).

## What makes this circular

```
node split  ->  buffer depth  ->  realized staleness
     ^                                    |
     |                                    v
rollout demand  <-  recycling forced by the cap
```

A tighter cap recycles more groups, each recycle costs another full generation,
and that raises rollout demand — which is what the node split has to satisfy. So
the split cannot be chosen without knowing the staleness distribution, and the
staleness distribution depends on the split.

The way out is that the two can be measured in one pass and combined
arithmetically, instead of sweeping the split once per cap.

## Step 1 — measure the natural staleness, uncapped

**The cap must not be enforced.** Under a cap, a group that exceeds it is
recycled and regenerated from scratch, so the system is never allowed to reach
the lag it would have reached, and recycling consumes generation capacity, which
changes buffer depth, which changes everyone else's lag. The steady state under a
cap is a different dynamical system, not a censored view of the same one.
Whatever an s=2 run reports as its lag distribution, it is not the natural one.

Pass `MAX_WEIGHT_STALENESS=64`, not nothing. `fully_async_rollout.py:306` guards
the whole block on `max_weight_staleness is not None`, so leaving it unset skips
the measurement along with the enforcement. 64 is far above anything observed and
never binds, so it measures without acting.

Run it at every candidate split, because the answer is per-split: more rollout
capacity means a deeper buffer means more lag. That relationship is itself a
result — it is the mechanism by which over-provisioning manufactures staleness,
which is the same criticism this study makes of prior work that imposes staleness
artificially.

Record per split R, from the steady steps only (after the buffer has filled —
observed at step 7 of 12 in the discarded runs, so 12 rollouts is the minimum):

| symbol | from |
|---|---|
| `tau_train(R)` | `actor_train + log_probs`, which does not depend on R |
| `tau_roll(R)` | `perf/rollout_time` |
| `P(L=k)` | `rollout/fully_async/staleness_count_<k>` |

## Step 2 — compute the split for a given cap, do not sweep for it

The fraction a cap `s` sends back:

    p(s) = P(L > s) = sum over k > s of P(L=k)

A recycled group is generated again, so the expected generations per delivered
group is geometric:

    m(s) = 1 / (1 - p(s))

`m(s)` is an **upper bound**: a recycled group restarts at lag 0 and is more
likely to survive its second attempt than its first, so the true multiplier is
smaller. Using the bound errs towards more rollout capacity, which is the safe
direction — under-provisioning would slow every capped arm and be read as an
algorithmic result rather than a provisioning one.

Effective generation time and the resulting wait:

    tau_roll(R, s)  = tau_roll(R) * m(s)
    train_wait(R,s) = max(0, tau_roll(R, s) - tau_train)
    step(R, s)      = max(tau_train, tau_roll(R, s))

## Step 3 — pick R, two-sided

Adoptable R is only {1, 3, 7} **for Qwen3-4B-Instruct-2507**, which trains on
one node. The colocated arm runs at the same total GPU count with every GPU
training, so

    dp = 8 * (T + R) / (TP * CP * PP)      must divide the global batch

with T the train node count. `T + R` must be a power of two, so the set moves
with both T, which grows with the model, and the total node budget, which goes to
16 for the larger models:

| T | adoptable R, `T + R <= 16` |
|---|---|
| 1 | 1, 3, 7, 15 |
| 2 | 2, 6, 14 |
| 4 | 4, 12 |
| 8 | 8 |

Re-derive it for every model. R ∈ {2, 4} are measured here for the appendix and
cannot be shipped.

Choose the **smallest** adoptable R satisfying both:

    (a) train_wait(R, s) <= 0.10 * tau_train       rollout keeps up
    (b) tau_roll(R, s)   >= 0.50 * tau_train       rollout is not idle

(a) is the user-chosen criterion: minimise rollout nodes, tolerate a little
waiting. (b) is the other side, and it is not merely about wasted GPUs — a
rollout that finishes in half the training time leaves generated groups sitting
in the buffer until the trainer catches up, and that wait *is* the lag. An
over-provisioned split manufactures the very quantity the study measures.

With only three adoptable points, (a) and (b) may not both hold. If they cannot,
(a) wins — a starved trainer corrupts the wall-clock metric, while an idle
rollout only wastes GPUs and inflates lag, and the inflated lag is at least
measured and reported rather than hidden. State which constraint was relaxed.

## Step 4 — verify, once

Run the chosen R at `MAX_WEIGHT_STALENESS=2` and check the prediction:

| predicted | measured |
|---|---|
| `train_wait(R, 2)` | `train_wait` timer |
| `p(2)` | `stale_groups_recycled / staleness_num_groups` |
| `1 - 1/m(2)` | `rollout/fully_async/wasted_token_frac` |

Agreement adopts the split. Disagreement means `m(s)` is the wrong model — most
likely because recycled groups do not face the original lag distribution — and
the fix is to measure `p(s)` directly at that split rather than to widen the
sweep.

## Step 5 — re-measure the staleness at the adopted split

The P(L) from step 1 was taken at every candidate split. Only the adopted one is
the study's operating point, so that is the histogram the paper reports. It also
fixes the staleness axis: a bound above the observed maximum never binds and its
arm is identical to the uncapped run, so those levels are dropped rather than
run.

## What this feeds

The output is a frozen training environment: node split, `SGLANG_MEM_FRACTION`,
batch shape, and the staleness levels that are worth running. Every algorithm
arm is then run inside it, unchanged, so that a difference between arms is a
difference between algorithms. See `notes/algorithm-ablation.md`.

## The bound feeds back on the thing it is supposed to measure (2026-08-08)

The node-ratio sweep ran at `max_weight_staleness=64`, where nothing is ever
recycled. Re-reading it next to the tier-1/2 arms, which run at bounds 1/2/4:

| run | rollout_time | train | wait | resp_len | avg_staleness |
|---|---|---|---|---|---|
| sweep R=1 | 960.0 | 262 | 612.3 | 6395 | 0.43 |
| sweep R=2 | 462.4 | 262 | 143.9 | 6493 | 0.64 |
| **sweep R=3** | **110.0** | 262 | **1.6** | 6477 | **1.38** |
| sweep R=5 | 38.8 | 269 | 1.9 | 6550 | 2.98 |
| now, bound 4 | 259-381 | 238-306 | 36-89 | 6-8.5k | **0.70** |
| now, bound 2 | 314-364 | 250-290 | 55-84 | 6-8.5k | 0.69 |
| now, bound 1 | 343-520 | 222-318 | 136-215 | 6-8.5k | **0.45** |

**`rollout_time` under `--fully-async` is drain time, not generation time.** The
module docstring says it: "each training step only drains already-completed
groups from the worker's output queue". A deep queue drains instantly, so
`rollout_time` falls as staleness *rises* -- the sweep's 960 -> 39 is that
inverse, not generation getting faster. It is an observable for queue depth.

So the ordering is: production rate > consumption rate -> queue fills -> samples
wait -> lag. The ceiling is `OUTPUT_QUEUE_MAX_GROUPS / rollout_batch_size` =
1000/192 ~ 5, which is where R=5's 2.98 sits.

**The bound closes a negative feedback loop around this.** A recycled group is
regenerated from scratch (`_recycle` -> `reset_for_retry` -> back to the buffer),
so tightening the bound slows production, which drains the queue, which lowers
the natural lag -- on top of the truncation the bound applies directly. Bound 1
recycled 1046-2085 groups and lands at 0.45; bound 4 recycled zero and lands at
0.70.

`max_weight_staleness` therefore **cannot be a clean independent variable**: it
moves realized lag and generation throughput together, in the same direction, and
the arms differ in throughput for a reason that has nothing to do with
off-policy learning. Bound 2 and bound 4 came out at 0.69 and 0.70 -- the same
experiment twice, since neither binds.

`--num-steps-per-rollout` has no such feedback. Splitting one rollout into k
gradient steps makes minibatch j train on data j-1 updates old, with no
regeneration and no change to production rate. It is also the mechanism
DeepSeek-V3.2 names as the origin of off-policyness in practice ("split into
multiple mini-batches for several gradient update steps ... inherently
introduces off-policy behavior"). We run it at 1, the most on-policy value there
is.

### Choosing R

R is a throughput decision, not a staleness knob. Pick the smallest R whose
`train_wait_time` is near zero and report whatever lag results.

At 6.5k responses that was R=3 (wait 1.6 s). **At the 8k responses lr 5e-6
produces it no longer is** -- wait is 13-39% of step time again, so R=3 is now
under-provisioned and the trainer is idling. Re-measure the crossover whenever
response length moves materially; it is not a constant of the recipe.

### The queue ceiling, measured (2026-08-08)

The `1000/192 ~ 5` above is not just an arithmetic bound; at R=7 the pipeline
reaches it and pins there. `noderatio-s64-t1r7-rs42` (job 15254233), bound 64 so
nothing truncates:

| rollout | `queue_size` | `avg_staleness` | `max_staleness` |
|---|---|---|---|
| 0 | 3 | 0.00 | 0 |
| 2 | 217 | 1.00 | 1 |
| 4 | 490 | 2.44 | 3 |
| 6 | 568 | 3.29 | 5 |
| 8 | 779 | 3.91 | 5 |
| 9 | **1000** | 4.10 | **6** |
| 10 | 922 | 4.62 | **6** |
| 11 | 980 | 5.05 | **6** |

`OUTPUT_QUEUE_MAX_GROUPS` (1000, `fully_async_rollout.py:34`) plus the 192
in-flight groups, over 192 consumed per step, is 6.2 — and `max_staleness` stops
at 6 while `queue_size` sits against the cap. R=5 reaches 660-830 and max 5-6;
R=1..3 never leave `queue_size` 0-1 and are ordered by node ratio instead.

Two consequences:

- **A bound above ~6 is unrealizable at k=1**, whatever is passed to
  `--max-weight-staleness`. The s=64 arms are s~6 arms; their
  `stale_groups_recycled` is 0 because the bound never gets the chance to bite.
  The ceiling scales with `--num-steps-per-rollout` (~6.2k), not with the bound.
  Reaching further up the staleness axis means editing the constant — it is not
  a CLI flag — or shrinking `rollout_batch_size`.
- **A full queue is idle rollout GPUs.** `await self._output.put(...)`
  (`fully_async_rollout.py:268`) blocks the producer, so past R=5 the extra
  rollout nodes are throttled by backpressure rather than generating. That is
  the same ceiling "Choosing R" reaches from the `train_wait_time` side.

### Second pass: the balance per bound, at 8 nodes (2026-08-08)

`experiments/staleness_ratio_sweep.sh`. The first sweep held the bound at 64 and
swept both node axes to find the *natural* lag; this one fixes the allocation at
8 nodes and asks a different question — given a bound the study will actually
run, which split does that bound want?

- 4 splits x 4 bounds: T:R in {1:7, 2:6, 3:5, 4:4}, `max_weight_staleness` in
  {1, 2, 4, 8}. lr 1e-6, TIS 2.0, actor denominator, production batch shape.
- **The bound is enforced here, not parked at 64.** Recycling is the mechanism
  under test, not noise to be excluded: a tight bound slows production, which
  drains the queue, which lowers the lag — the feedback loop described above. The
  readout is therefore the pair (realized lag, throughput), never lag alone.
- 8 nodes, `batch`, 4 h, one job per point. **`NUM_ROLLOUT` is not overridden**:
  the recipe's 300 stands, the wall stops each job, and the points are compared
  per step rather than by time-to-completion. Leaving it at the production value
  is also what keeps a point resumable — `--num-rollout` feeds `train_iters` and
  so `lr_decay_steps`, which `OptimizerParamScheduler.load_state_dict` asserts
  against the checkpoint, so a probe budget would freeze the run at that budget
  forever (`notes/cluster.md`).
- **Checkpoints are written on the recipe's cadence**, the same one the
  convergence sweep runs at: `--save-interval 10`, `--save-retain-interval 100`,
  HF export every 5. A point that turns out to be the right balance is then a
  run that can be continued rather than repeated.
- dp is the trainer's alone under `--fully-async`, so 3:5 is legal: 24 train GPUs
  at tp2 is dp12, and 3072/12 = 256. The script derives this from the recipe,
  prints `gbs/dp` per point, and drops any split megatron would reject, at
  submission. Across the four splits dp is 4/8/12/16 and gbs/dp is
  768/384/256/192 — all four divide.
- **`--submit` deletes the checkpoint directory of each point it submits**,
  unconditionally and without a flag — the script is meant to be handed to
  someone else to run. A point is a fresh measurement; resuming would report a
  warm queue and an already-moved policy as a cold start. Three guards, because
  the operator is not necessarily the author: the path comes from
  `run_identity.sh` rather than from a template, so it is the directory the job
  will actually write to; the delete refuses anything outside `TRAIN_CKPT_DIR`;
  and it refuses to touch a point whose job name is already in `squeue`, since
  deleting under a running job corrupts that run and measures neither. The dry
  run marks which directories would go.
- `--check` reports **one log per point**, the highest job id, and says how many
  it skipped. Re-running a point is now routine, and two runs of one
  configuration in the table read as two configurations.
- wandb project `async-rl-dapo-math-node-ratio`, separate from the convergence
  study's `off-policy-<dataset>`, because these runs are a throughput
  measurement and do not belong on the same board as the quality curves.
  `train.sh` honours `WANDB_PROJECT` with the old value as its default, so
  nothing else moves.
- The wandb group is `s<S>-t<T>r<R>`: the two swept axes, nothing else.
  `run_identity.sh`'s derivation is longer, not shorter, so it is overridden
  rather than inherited. A rejection from `run_identity.sh` is fatal at
  submission: a command substitution would otherwise swallow it and the job would
  fail 100 s into an 8-node allocation instead.

Read out `step_s` and `tok/s/gpu` from `--check` (`analyze_throughput.py`)
against `staleness/mean`, `staleness/frac_at_bound`,
`stale_groups_recycled` and `wasted_token_frac` from the same table. A split that
wins on `step_s` while recycling a third of its generation has not won.

**The s=8 row is not a fourth bound level.** The output queue caps realized lag
at `(1000 + 192)/192 ~ 6.2` at k=1 (see the queue-ceiling section above), so a
bound of 8 can never bite: that row measures the natural lag of each split and is
the unbounded reference, not a point on the bound axis. Expect
`stale_groups_recycled` 0 and `frac_at_bound` 0 there; if either is non-zero,
the ceiling arithmetic or the batch shape has changed and both notes need
revisiting.

One rollout seed, by decision rather than by default: 16 jobs, 512 node-hours.
That orders the bounds. It does **not** separate two adjacent splits — generation
order is the largest source of run-to-run spread here — so read a 2:6 vs 3:5
difference as a direction, not a measurement, unless it is large next to the
spread the first sweep recorded at fixed settings.

The checkpoint cost is small and its bias runs the other way from what
`notes/checkpoints.md` estimates. Measured from the `save_model` timer on jobs
15319376 and 15319392, at the recipe's cadence:

| artifact | every | elapsed |
|---|---|---|
| HF only | 5 rollouts | 7.8-8.0 s |
| HF + torch_dist | 10 rollouts | 13.6-19.0 s |

against a 280 s `actor_train`, which is **0.55% and 0.72%** of accounted wall on
those two runs — not the ~2% the sizing argument assumed.

The save is serial in the trainer loop, but `generate.remote(rollout_id + 1)` is
dispatched before `train` and `_worker_loop` runs continuously on the rollout
engines' own GPUs, so **generation does not stop while the trainer saves**. On a
rollout-bound split the save time comes straight out of the `train_wait` that
follows it and costs nothing; on a trainer-bound split it is fully on the
critical path. So it penalises the *trainer-heavy* splits, not the fast ones. At
0.7% it cannot move a split comparison, and `HF_SAVE_INTERVAL=20` would buy back
0.4 points at the cost of the `Q(t)` resolution — not worth it.
