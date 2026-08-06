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

with T the train node count. A larger model needs a larger T, and the adoptable
set moves with it: at T=1 it is R ∈ {1, 3, 7}, at T=2 it is R ∈ {0, 2, 6}.
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
