# What the runs record, and what the analysis needs

Audited against a live async run (`v-3n-8t16r-r1`, job 15150858) plus the one
earlier job that reached an eval (15113756). Every key below was observed in a
log, not read off the source.

wandb namespaces and their step axes are registered in
`miles/utils/tracking_utils/wandb_utils.py:161-169`:

| namespace | step metric |
|---|---|
| `train/*` | `train/step` |
| `rollout/*`, `multi_turn/*`, `passrate/*`, `perf/*` | `rollout/step` |
| `eval/*` | `eval/step` |

`train/step` and `rollout/step` are separate axes on purpose: with
`--num-steps-per-rollout > 1` they advance at different rates, and the off-policy
question is asked in both currencies.

## Covered

**Realized staleness** — the thing the study is about, and it is measured rather
than assumed:

    rollout/weight_version/{min,max,mean,median,mixed_version_ratio}
    staleness/{total,pre_queue,in_queue}/*      # the decomposition
    staleness/bound/{rollout,train}/*           # what --max-weight-staleness tests

`--max-weight-staleness` is a bound; these are what actually happened. Any claim
about off-policy degradation has to be plotted against these, not against the
flag.

Two cautions on this block, both from "Staleness is measured from the completion
version" below. `mixed_version_ratio` is **structurally 0** for every run in this
study, so it is not evidence of anything. And `rollout/weight_version/*` is the
version each sample *finished* under, which is why the `staleness/` families
exist alongside it.

**Train/rollout mismatch and importance sampling**:

    train/train_rollout_kl, train/train_rollout_logprob_abs_diff, train/ppo_kl
    train/tis, train/tis_abs, train/tis_clipfrac
    train/ess_ratio, train/rollout_ess_ratio, train/ois

`train_rollout_logprob_abs_diff` is the direct numerical-mismatch probe --
non-zero even at zero staleness, because it also picks up the Megatron/SGLang
kernel difference. Separating that floor from the staleness-induced part is what
makes the `--true-on-policy-mode` parity run worth having as a baseline.

**Waste accounting**, which is how the pause modes get compared:

    rollout/fully_async/{kept_tokens, stale_tokens, aborted_tokens,
                         dynamic_filter_tokens, wasted_token_frac,
                         stale_groups_recycled, aborted_groups_recycled,
                         queue_size}

`abort` discards tokens, `retract` recomputes KV, `in_place` reuses it. The first
shows up in `aborted_tokens`, the second only in wall clock. Both are visible.

**Dynamic sampling**: `rollout/dynamic_filter/drop_zero_std_*`,
`rollout/zero_std/{all_zero,all_one}_percentage`. The all-zero percentage is the
livelock indicator -- when a too-short generation budget made every group
unanimously wrong, this went to 1.0 and the drain never filled.

**Timing**: the full `perf/*` breakdown, including `step_time`, `train_wait_time`,
`actor_train_time`, `log_probs_time`, `rollout_time`, `wait_time_ratio`.

**Eval**, per benchmark, with the truncation rate alongside the score:

    eval/aime25, eval/aime25-truncated_ratio, eval/aime25/response_len/{mean,max},
    eval/aime25/repetition_frac, eval/aime25-none_reward_ratio

The truncation rate is not a nicety. AIME-2024 truncates 13.75% at a 24576
budget and a truncated sample scores zero under every rule-based verifier, so a
score moving because the model got more verbose is distinguishable from a score
moving because it got better.

## Gaps, and why they are tolerable

**`perf/actor_train_tflops` undercounts.** `train_metric_utils.py:41` computes
`3 * total_fwd_flops / actor_train_time`. Under `--recompute-granularity full`
the real cost is 4x forward, so the logged number is 3/4 of the truth. Multiply
by 4/3 before quoting an MFU. This is a reporting correction, not a data loss.

**No cumulative sample or token counter.** The plan calls for reporting sample
efficiency, and there is no running total. It is recoverable: samples consumed
through step *n* is `n * rollout_batch_size * n_samples_per_prompt`, and tokens
is the running sum of the `rollout/fully_async/*_tokens` family, all of which are
per-step on the `rollout/step` axis. Summation after the fact is exact as long as
no step is missing from the export.

**Wall clock does not survive a resume.** Every chained job is a new wandb run
with its own `_runtime` starting at zero. `--wandb-group` puts them in one group
and `train/step` continues correctly across the boundary, so the step-axis curves
concatenate; the time axis does not. Since the plan's primary x-axis is wall
clock, reconstruct it by cumulating `perf/step_time` rather than reading
`_runtime`. That is also the more honest number -- it excludes the requeue gap,
which is a property of the queue and not of the method.

**GPU memory is host-level only.** wandb's system metrics come from the node
running the wandb process, which is the Ray head, which is a training node. In
the async layout train and rollout are on different nodes, so host-level HBM is
attributable to training without further work -- which is the side the TP-sizing
question is about. In the colocated layout it is not separable.

## Not yet observed

`multi_turn/*` and `passrate/*` are registered but this task family never emits
them.

## Flags the recipes carry for this study, and why

Recorded here rather than in the recipes: the scripts state what runs, this file
states why.

**`--observe-training-entropy`.** Without it `train/entropy_loss` is emitted but
identically 0. `calculate_entropy` is `entropy_coef != 0 or
observe_training_entropy` (`loss_hub/losses.py:99`) and the recipes set
`--entropy-coef 0.00`, so the metric is a constant. The convergence definition
names entropy, so the criterion could not be evaluated at all. The flag is
forward-only and detached at coefficient 0, so it adds no backward cost.

**`--no-dump-policy-loss-debug`.** `policy_loss_debug/` writes one file per
micro-batch per rank and scales with training calls rather than rollout steps.
Measured on a 12-step run: 1.17 GB in 2512 files, against 287 MB for the rollout
dumps over the same steps -- 76% of the dump. Nothing in this study reads it.
Turning it off takes a run from ~41 GB to ~11 GB.

**`--adam-beta2 0.999`, `--adam-eps 1e-8`, `--weight-decay 0.0`.** From the one
production RL config this study could verify from primary source: Nemotron 3
Super `stage1_rlvr.yaml` (`adam_beta2: 0.999`, `adam_eps: 1e-8`,
`weight_decay: 0.0`). An earlier revision of this file justified `0.01` from
M2PO and GAC; that was wrong twice over -- those are algorithm papers, not
model-building groups, and this study runs their methods as part of the scaling
sweep rather than competing with them, so their optimizer choices carry no
authority here. The 0.1 the recipes carried before that is a *pretraining* value
from the Nemotron 3 Nano/Super reports.

A survey of other model-building groups produced nothing usable: Qwen3's
technical report does not state RL optimizer settings, and GLM, MiniMax, Kimi and
DeepSeek do not publish theirs (Kimi K2 states Muon for *pretraining*, which does
not transfer). Search results that appear to give "Qwen3 GRPO hyperparameters"
are third-party papers using Qwen3, not Qwen's own recipe. Nemotron 3 Super is
therefore a single data point, which is why the LR is left at `1e-6` and no
warmup was added: one config is enough to pick a decay of 0.0 (the alternatives
had worse provenance) but not enough to move the LR. Its `3.0e-6` belongs in the
LR axis as a level, not as the default.

**`--partial-rollout` removed from the `math_sync` recipes.** It recycles
in-flight generations into the data buffer at abort time, and those samples
resume against a newer policy in a later rollout step -- so any sample it
collects is off-policy by construction. The colocated arm is the on-policy
reference, so the flag contradicts the arm's definition.

It was in fact already inert. `generate_rollout_async` (`sglang_rollout.py:502`)
exits its loop only when `len(data) >= rollout_batch_size`; with
`over_sampling_batch_size == rollout_batch_size` (the default,
`arguments.py:3101`) and no dynamic filter, every submitted group must complete
before that count is reached, so `state.pendings` is empty when `abort()` runs
and it collects nothing. But inertness that depends on two *other* settings
staying put is not a property to rely on in the reference arm: enabling dynamic
sampling or over-sampling later would silently make the on-policy baseline
off-policy. The flag is now absent rather than merely ineffective.

**`--balance-data`.** Repartitions each rollout batch across data-parallel ranks
by total token count (Karmarkar-Karp) instead of the default strided
`range(i, n, dp_size)` (`train_data_conversion.py:208`). It is a throughput knob:
with a heavy-tailed response length distribution the strided split leaves the
rank holding the longest samples as the straggler for the whole step.

It does not change the loss under this study's settings. `sum_of_sample_mean`
(`cp_utils.py:107`) is a *sum over samples of per-sample means*, so it is
additive across samples; `get_seqlen_balanced_partitions` is called with
`equal_size=True`, so every rank holds the same sample count and the DP gradient
average is invariant to which rank holds which sample. This would stop being true
under `--calculate-per-token-loss`, where the denominator is a token count and
per-rank token totals differ by construction -- the recipes do not set it.

Floating-point summation order does change, so `--balance-data` is not
bitwise-reproducible against a run without it. That matters only for the
deterministic-kernel check, which is a separate axis.

**`EVAL_MAX_RESPONSE_LEN` is fixed at 32768 and is not swept with
`MAX_RESPONSE_LEN`.** The response-length axis applies to training generation
only. If the evaluation budget tracked it, the 4k arm would be scored under a 4k
budget and its `Q(t)` would be depressed by the evaluation budget rather than by
what training did to the policy -- conflating "trained short" with "evaluated
short". It matches the offline evaluation budget in
`src/offline_eval/run_eval.sbatch`.

**`MAX_RESPONSE_LEN=32768` against `--rollout-max-context-len 32768`.** The
response cap and the context are equal, so the effective cap is
`32768 - prompt_len`. Measured over all 5865 training prompts: mean 393 tokens,
p90 573, p99 890, max 1547. The effective cap is therefore 32375 on average and
31221 in the worst case, 1.2-4.7% below nominal. This is a cap reduction, not an
error: `rollout_max_prompt_len` defaults to `context - 1` (`arguments.py:3149`)
and generation is truncated at the remaining budget. Raising the context would
force `MAX_TOKENS_PER_GPU` up with it (`mtpg * cp >= context`), and that is a
frozen throughput parameter -- moving it for one arm would break wall-clock
comparability across arms.

## `max_staleness` is the offered lag, not the trained lag (2026-08-07)

`rollout/fully_async/{avg,max}_staleness` and the `staleness_count_k` histogram
count every group **as it is offered**, including the ones the bound then throws
away. `fully_async_rollout.py:407-419` appends before it filters:

```python
staleness = current - oldest
staleness_values.append(staleness)          # recorded here
if staleness > args.max_weight_staleness:
    self._recycle(prompt_group)             # discarded here
    continue
```

So a run with `--max-weight-staleness 1` legitimately logs `max_staleness = 3`.
The bound is not violated; the metric simply is not measuring what its name
suggests.

Measured on job 15288337 (bound 1) and 15288347 (bound 2), rollout 4:

| | L=0 | L=1 | L=2 | L=3 | offered | recycled | trained on |
|---|---|---|---|---|---|---|---|
| bound 1 | 96 | 96 | 14 | 1 | 207 | 15 | 192 |
| bound 2 | 88 | 66 | 38 | 2 | 194 | 2 | 192 |

The accounting closes exactly: offered − recycled = 192 = `rollout_batch_size`,
and nothing above the bound survives. `staleness_num_groups` exceeds 192 because
a recycled prompt is re-offered and counted again.

Consequences for the analysis:

- **The realized lag P(L) must be taken from `staleness_count_k` truncated at
  `k <= bound`**, not from `avg_staleness`. At bound 1 the logged mean was 0.75
  against a trained mean of 0.495 -- a 50% overstatement, and it is worst exactly
  where the bound bites hardest.
- A wandb chart of `max_staleness` across arms compares *offered* lag, which is a
  property of the node ratio and is nearly identical across bounds. It is not
  the study's independent variable.
- `wasted_token_frac` is the honest companion metric: it is the cost of the gap
  between offered and trained lag.

## Reported training time excludes evaluation (2026-08-07)

**The paper reports training wall-clock with evaluation removed.** Evaluation is
instrumentation, not training: it does not change the policy, its cost is a
choice of `--eval-interval` and `--n-samples-per-eval-prompt`, and charging it to
the arms would put a fixed instrument cost inside the quantity under test.

It is not a rounding error. The first in-run eval on job 15288347 took ~20 min
against a 344 s rollout, and at `EVAL_INTERVAL=20` there are 15 of them in a
300-rollout run -- about 5 h, or 10-17% of an arm.

### Subtraction does not work, and the two placements differ

Under `--colocate` the same GPUs evaluate, then generate, then train, so eval is
a clean additive span. Under `--fully-async` the eval runs on the rollout engines
**concurrently** with training generation: it does not stop the trainer, it slows
the rollout. There is no eval span to subtract -- the cost appears as inflated
`perf/rollout_time` in the rollouts around it.

`timer("eval_rollout")` exists (`ray/rollout/rollout_manager.py:156`) but its
value is never logged, so there is no recorded eval duration to subtract even
where subtraction would be valid.

### The method

Work from the per-rollout timestamps of `metrics.py:79 - perf <id>` and **drop
the intervals that overlap an eval**, then take the mean of the rest. An eval
completes at the `metrics.py:53 - eval <id>` line and is triggered at
`rollout_id % eval_interval == 0`, so the affected window is bounded and
identifiable; how many rollouts it spans is measured, not assumed.

Two rules that go with it:

- **Never change `--eval-interval` mid-tier.** The comparison is between arms on
  wall-clock; giving one arm fewer evals changes the measured quantity, even
  though it changes nothing about the learning.
- Report the discarded fraction alongside the result, the same way
  `active_elapsed_hours` reports `excluded_h` for inter-allocation gaps
  (`experiments/src/offpolicy_acceleration/log_source.py:133`). The two
  exclusions compose: queue gaps between jobs, and eval windows within a job.

### The alternative worth taking for later tiers

`HF_SAVE_INTERVAL=5` already exports an HF checkpoint every 5 rollouts, and
`experiments/src/offline_eval/run_eval.sbatch` can score them off the training
critical path. That removes the exclusion entirely and buys a better eval
(more samples, more benchmarks) than 30 prompts x 8 affords -- `eval/aime25` at
n=8 on 30 prompts has se ~0.032, which is wider than the effects being chased.
Do not switch tier 1 mid-run; switch a whole tier at once or not at all.

### The eval also perturbs the independent variable (2026-08-07)

Excluding eval from the *time* axis is not sufficient. An in-run eval injects a
backlog that drives the realized lag up and triggers a recycling storm, and it
does so unequally across the staleness arms.

Job 15288347 (bound 2), around the eval that completed after rollout 19:

| rollout | L=0 | L=1 | L=2 | L=3 | offered | recycled | meanL | queue | `rollout_time` | `train_wait` | waste |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 19 | 88 | 72 | 32 | 0 | 192 | 0 | 0.71 | 0 | 377.5 | 27.7 | 0.0000 |
| 20 | 82 | 70 | 40 | 0 | 192 | 0 | 0.78 | 0 | 332.1 | **1028.1** | 0.0000 |
| 21 | 66 | 97 | 29 | 2 | 194 | 2 | 0.83 | **427** | 20.2 | 1.8 | 0.0329 |
| 22 | 0 | 151 | 41 | 1 | 193 | 1 | 1.22 | 433 | 14.8 | 2.1 | 0.0167 |
| 23 | 0 | 0 | 192 | 4 | 196 | 4 | 2.02 | 431 | 22.0 | -- | 0.0378 |
| 24 | 0 | 39 | 153 | **359** | 552 | **360** | 2.58 | 99 | 22.1 | -- | **0.7398** |

The mechanism is structural, not incidental. `_worker_loop` is a background
`asyncio.create_task` (`fully_async_rollout.py:268`), so while the manager awaits
the eval the trainer gets no batch -- `train_wait` 1028 s -- but generation keeps
running and fills the output queue to 427 groups, about 2.2 batches. Draining
that backlog costs no generation time (`rollout_time` collapses to ~20 s), so the
trainer advances one weight version per step against samples that were all
produced before the stall. The queue ages one version per step: L=0 disappears
entirely by rollout 22, and by rollout 24 the offered lag has walked past the
bound and **74% of generated tokens are discarded**, against 3% in steady state.

**This biases the arms unequally and in the direction of the effect under study.**
A tight bound recycles the aged backlog; a loose one absorbs it. The bound-4 arm
would take a lag-2.5 backlog without discarding anything, while the bound-1 arm
would discard nearly all of it. So the instrument penalises exactly the arms the
study is asking about.

Consequences:

- The exclusion window applies to `staleness_count_*`, `avg_staleness` and
  `wasted_token_frac` as well as to wall-clock. Drop rollouts from the eval until
  `queue_size` returns to its steady value -- 5 rollouts in the case above.
- Moving evaluation offline (`experiments/src/offline_eval/run_eval.sbatch`, over
  the `HF_SAVE_INTERVAL=5` exports) removes the perturbation rather than
  correcting for it, and is the right configuration for tier 2 onward. It is
  still not something to change inside a running tier.

### In-run evaluation is off by default from 2026-08-07

`EVAL_INTERVAL` defaults to `0` in both recipes, and `train.sh` then passes no
`--eval-interval` at all, which leaves `args.eval_interval` None and turns off
both call sites (`train.py:98` before-train, `train.py:144` periodic). Quality is
read by scoring the `HF_SAVE_INTERVAL=5` exports offline.

Three reasons, in order of how much they distort the measurement:

1. **It perturbs the independent variable.** The backlog an eval injects walks the
   realized lag past the bound and produced a 74% recycling storm on the bound-2
   arm -- see the table above -- and it hurts a tight bound more than a loose one.
2. **It perturbs the reported time**, ~20 min a call, 15 calls, 10-17% of an arm,
   and unequally: sequential under `--colocate`, concurrent under `--fully-async`.
3. **It was the weaker measurement anyway.** 30 prompts at n=8 on one year gives
   se ~0.061; aime24/25/26 at n=16 gives ~0.033, and against a fixed prompt set
   the step-to-step term falls from 0.032 to 0.013.

`validate.py` enforces both halves: the default is 0, and `train.sh` honours 0 by
emitting nothing. A recipe that passes `--eval-interval` unconditionally fails,
because there the default would be decorative.

**Tier 1 keeps its in-run eval.** Those chains were submitted against the old
default, and an arm that stops evaluating halfway is no longer comparable with one
that did not -- the wall-clock being compared would change under it. Do not
`git pull` into a clone with a running tier. The switch applies from tier 2.

## Open option, not adopted: separating policy lag from engine mismatch

Every off-policyness number we currently log is built on

    delta = log pi_Megatron(theta_t) - log pi_SGLang(theta_{t-k})

which is **two effects added together**: the policy moved between the rollout and
the update, and the two engines disagree numerically about the same
distribution. Measured, the second dominates -- `train_rollout_kl` is 4.82e-04
on the *colocated on-policy* arm, against 5.03e-04 at staleness 4, and the four
arms are not even ordered by staleness (s1 is the lowest at 4.68e-04). Any
statistic built on `delta` inherits that.

It splits exactly:

    log pi_M(theta_t) - log pi_M(theta_{t-k})     <- policy lag only
  + log pi_M(theta_{t-k}) - log pi_S(theta_{t-k}) <- engine mismatch only

The first term evaluates both policies **on the same engine**, so the numerical
floor cancels rather than being estimated and subtracted.

miles already has the mechanism: `--keep-old-actor`
(`RATIO_DENOMINATOR=old-actor`) recomputes the denominator with the weights the
rollout engines used. As a *metric* rather than a loss it does not need the
whole batch -- 64 of 3072 sequences give the per-token lag KL to within a factor
of 8 in standard error, for roughly 2% extra compute.

**Not adopted.** Under fully-async the groups in one batch carry different lags,
so "the old weights" is up to `max_weight_staleness + 1` distinct weight sets,
and the bookkeeping for that is not designed. Revisit if the lag-stratified
metric below turns out not to separate the arms.

## What the staleness metrics currently mean (checked 2026-08-07)

`af90e72e` split the two: unprefixed keys became the lag of groups the loss
actually consumed, and `rollout/fully_async/offered/*` kept the pre-filter lag.

**The running tier-1 arms do not have it.** hiso's checkout contains neither
`trained_staleness` nor `offered_staleness` in `fully_async_rollout.py`, and no
`offered` key appears in any arm's `dump/dashboard/metrics.jsonl`. So for every
number reported off the current runs:

    rollout/fully_async/avg_staleness      <- OFFERED, before the bound check
    rollout/fully_async/staleness_count_*  <- OFFERED
    rollout/fully_async/staleness_p50/90/99 <- OFFERED

This is why `staleness_count_3` is non-zero in a `max_weight_staleness=1` run:
those groups were offered at lag 3 and discarded. The accounting closes exactly
against `staleness_num_groups`.

It takes effect on the next job launched from a clone that has the commit --
`miles/**` is read from disk at launch, so a `git pull` is enough; no
resubmission is needed beyond the normal chain boundary. Runs that straddle the
boundary will have a discontinuity in the series, which is the reason not to
pull mid-chain without noting the rollout index where it happened.

## The staleness keys, renamed for the side they describe (2026-08-08)

`staleness/*` and `staleness/offered/*` were not readable as a pair: the trained
lag sat unprefixed, which reads like a total rather than one of two populations,
and "offered" describes the transaction rather than the producer. Both are now
subgroups:

| key | population |
|---|---|
| `staleness/bound/rollout/*` | every group the pipeline handed over, counted **before** the bound check — the natural lag of this node ratio |
| `staleness/bound/train/*` | what survived into the batch — what the loss actually saw |

Sub-keys, identical under both: `mean max p50 p90 p99 frac_zero num_groups
frac_at_bound count_0 … count_16 count_ge_17`. `frac_at_bound` is a `>=` test, so
it is "how often the cap was reached", not the rejection rate.

Scalars that belong to neither population stay at the root: `bound_exceeded_{groups,tokens}`,
`retry_count_{mean,max}`, `retry_frac_nonzero`.

Three things the pair does not say on its own:

- **The gap is not all bound.** A group is dropped between the two populations
  either by the bound *or* by the dynamic filter, which fires for a reason
  unrelated to staleness. Attribute the bound's share with
  `staleness/bound_exceeded_groups`, not by differencing `num_groups`.
- **Neither population contains aborted groups**, nor groups whose weight
  version could not be read. Those are recycled before the count.
- `rollout/fully_async/{avg,max}_staleness` are upstream's keys and still mean
  the **offered** lag. They were not renamed, because two miles versions must not
  plot different quantities under one name.

Old runs keep the old keys. A chart or a script that spans the rename has to
accept both spellings; nothing back-fills.

## Staleness reference and queue policy (2026-08-09; updated 2026-08-13)

`Sample.weight_versions` alone cannot identify the policy at the start of a
request. SGLang stamps it when it builds a reply, so single-turn generation
usually has one completion version. With `PAUSE_GENERATION_MODE=in_place`, a
request can span an update and still finish with one version and zero
retractions. The old `weight_version/mixed_version_ratio` therefore remains
structurally blind to that crossing.

The patched SGLang image also returns scheduler-authoritative
`first_prefill_weight_versions`, `min_forward_weight_versions`,
`max_forward_weight_versions`, and `last_forward_weight_versions`. The
`prefill` reference uses the minimum first-prefill version over every sample and
turn in the group. Unlike the submission-side HTTP snapshot, it does not charge
router or scheduler waiting before the first actual forward.

### The decomposition

The scalar families follow Applied Compute's PQS/IQS split
([staleness in fully-async RL](https://www.appliedcompute.com/research/staleness-in-fully-async-rl)).
Let **R** be the configured reference, **Q** the applied version when the whole
group became ready, and **C** the applied version at queue selection:

| key | quantity | gated on the bound? |
|---|---|---|
| `staleness/pre_queue/*` | `Q - R` — updates crossed before the whole group became ready | no |
| `staleness/in_queue/*` | `C - Q` — updates crossed while waiting for selection | no |
| `staleness/total/*` | `C - R` = `pre_queue + in_queue` | no |

Q is stamped only after the group's slowest sample lands. A group is one
concurrent request per sample joined by `asyncio.gather`; using an earlier sample
completion would incorrectly charge the straggler interval to `in_queue`.

The `submission` reference is a TTL-cached HTTP snapshot written immediately
before generation and carried in `Sample.metadata` under
`submission_weight_version`. The `prefill` reference comes from the engine
provenance above. Missing provenance stays missing rather than becoming zero;
`prefill` enforcement fails fast if those fields are absent or misaligned.

### What the bound tests

`staleness/bound/{rollout,train}/*` records the exact quantity tested by
`--max-weight-staleness`:

| `--staleness-reference` | bound tests | relation to the components |
|---|---|---|
| `completion` (library default) | `C - oldest completion` | includes in-queue lag plus the group's internal completion spread; not `total` |
| `submission` | `C - S` | exactly `total` |
| `prefill` (math-async recipe default) | `C - F` | exactly `total` |

The choice is also logged as `staleness/bound_reference_is_submission` and
`staleness/bound_reference_is_prefill` and included in the run path. `rollout`
means every group offered before the check; `train` means the groups that
survived selection.

Under `submission` or `prefill`, `s=N` accepts a group exactly when the tested
version gap is at most N; equality is accepted. In particular, queue-max with
the prefill reference treats `s=0` as on-policy and `s=1` as permitting one
policy-version gap. Its selection after the preceding update does not shift or
re-index that definition. Queue-recycle uses the same inequality at its
historical drain point. `queue/consumption/selection_to_train_gap` is a separate
scheduling diagnostic, not an offset to apply to the configured queue-max
bound.

A bound tighter than the pipeline can meet collapses overlap rather than
deadlocking. While drain waits, training cannot publish a new version, so fresh
groups eventually pass under the frozen version. The signatures are a high
`wasted_token_frac`, many bound-exceeded groups, and rising retry counts.

The staleness histograms resolve `count_0` through `count_16` plus
`count_ge_17`. Under `completion`, an arm label is not an upper bound on `total`;
under `submission` or `prefill`, it is. Dumps carry the submission stamp, all
four forward-provenance arrays, completion versions, and the queue lifecycle's
ready/enqueue/dequeue/decision fields for offline reconstruction.

Each new lifecycle record also carries scalar `reward_values` aligned with its
`sample_indices` and `response_lengths`. Reward evaluation has already completed
before the group becomes ready, so recording the values does not call the RM or
send anything through the trainer object store. The offline analyzer reports
sample and group reward distributions, all-zero/all-one/mixed group fractions,
and sample-length/reward and group-max-length/group-mean-reward correlations for
every terminal disposition. Older dumps remain readable and report missing
reward coverage rather than inventing zero rewards.

### Queue policies and the formula boundary

- `queue-recycle` is the compatibility default: completion FIFO, immediate
  next-batch prefetch by the driver, safety backpressure at 1000 completed
  groups, and over-age groups returned to the prompt buffer. With
  `--staleness-reference prefill`, this is the former
  FIFO-prefill-age-cutoff configuration.
- `queue-max` waits until the trainer requests a complete batch, takes the oldest
  completed groups, and permanently drops groups whose first-prefill age exceeds
  the required `--max-weight-staleness`. It does not reserve the next training
  batch before the preceding weight update.
- `queue-drop` keeps at most `q * rollout_batch_size` completed groups and evicts
  the oldest completed group on overflow. It has no age cutoff and likewise
  selects only when the trainer is ready after the preceding weight update.

The Applied Compute IQS closed form applies only to `queue-drop`. The offline
analyzer implements `IQS = rho` for `rho < 1` and
`(2q + rho - 1) / (2rho)` for `rho > 1`, adds the measured group-tailness PQS
term, and intentionally rejects the exact `rho = 1` boundary. Do not apply the
formula to `queue-recycle` or `queue-max`.

Do not infer a queue-max versus queue-recycle length-bias ordering by shifting
one policy's configured bound because of driver prefetch. The robust claims are
that queue-max's permanent age rejection preferentially removes long responses,
whereas queue-drop is approximately length-unbiased in steady state.
Queue-recycle retries an over-age prompt under fresh weights, so where it falls
between those policies is an empirical question involving retries, startup and
final censoring, queue utilization, group tailness, and dynamic filtering. A
single-seed run can validate the queue mechanism and compare the queue-drop mean
staleness with the closed form, but it cannot establish a seed-general
downstream-task significance claim.

The submission-side version is TTL-cached at 1 s; engine-reported prefill
provenance is not. Full rollout checkpointing currently supports only
`queue-recycle`; argument validation rejects the two policy-deque layouts rather
than silently writing an incomplete snapshot.

Runs before 2026-08-09 have none of these keys, and nothing back-fills.
