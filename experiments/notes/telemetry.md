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
    train/ess_ratio, train/rollout_token_level_ess,
    train/rollout_sequence_level_ess, train/ois
    train/policy_rollout_abs_diff, train/policy_rollout_kl,
    train/policy_rollout_token_ess, train/policy_rollout_sequence_ess

`train_rollout_logprob_abs_diff` is the direct numerical-mismatch probe --
non-zero even at zero staleness, because it also picks up the Megatron/SGLang
kernel difference. Separating that floor from the staleness-induced part is what
makes the `--true-on-policy-mode` parity run worth having as a baseline.

The historical token-level ESS is Kish's ratio over tokens *within each
response*, averaged with the training reducer. It remains for compatibility but
is not the population ESS normally meant by an off-policy sequence diagnostic.
`rollout_sequence_level_ess` is the standard
`(sum_i w_i)^2 / (B sum_i w_i^2)` over response weights. The `policy_rollout_*`
family always uses the current actor forward, independently of the PPO ratio
denominator; use it when `--use-rollout-logprobs` would otherwise make the
historical family identically zero/one. Both ESS families reuse tensors already
computed by the loss and require no additional model forward.

**Waste accounting**, which is how the pause modes get compared:

    rollout/fully_async/{kept_tokens, stale_tokens, aborted_tokens,
                         dynamic_filter_tokens, wasted_token_frac,
                         stale_groups_recycled, aborted_groups_recycled,
                         queue_size}

`abort` discards tokens, `retract` recomputes KV, `in_place` reuses it. The first
shows up in `aborted_tokens`, the second only in wall clock. Both are visible.

**Dynamic sampling**: `rollout/dynamic_filter/drop_zero_std_*`, historical
`rollout/zero_std/count_{0.0,1.0}`, and normalized
`rollout/zero_std/{all_zero,all_one}_percentage`. With binary math reward these
are respectively all-wrong and all-correct group counts; mixed groups are the
total prompt-group count minus the two. The all-zero percentage is the
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
and normally continues `train/step`, but a restart from an older published
checkpoint can replay step numbers that were already logged. Therefore the
step-axis curves must not be blindly concatenated either. In the audited
2026-08-15 extension, 35 overlapping run/step rows across 11 runs changed 807
values across the selected major metrics when the latest checkpoint lineage
was selected. Reparse the
complete timestamp-ordered lineage, keep the latest valid incarnation of a
replayed step, and record the replacements; appending only new step numbers
mixes abandoned and current attempts. Since the plan's primary x-axis is wall
clock, reconstruct it by cumulating `perf/step_time` on that selected lineage
rather than reading `_runtime`. That is also the more honest number -- it
excludes the requeue gap, which is a property of the scheduler and not of the
method. Future replay-buffer schemas should persist a run-incarnation ID, global update ID,
logging cursor, and resume-source checkpoint ID so this selection is explicit.

**GPU memory is host-level only.** wandb's system metrics come from the node
running the wandb process, which is the Ray head, which is a training node. In
the async layout train and rollout are on different nodes, so host-level HBM is
attributable to training without further work -- which is the side the TP-sizing
question is about. In the colocated layout it is not separable.

**Checkpoint presence is not checkpoint access.** A visible index, nonzero
shard size, and complete weight map do not prove that the analysis identity can
read the shard. Later checkpoints in this study reverted to mode 600: 50/175
structurally complete checkpoint sets were unreadable, including all 18 added
after the later training cutoff. Every artifact manifest must separately record
`checkpoint_complete` and `checkpoint_readable`, with the latter verified by an
actual one-byte open before a weight job is submitted. A permission failure is
missing parameter evidence, never a zero displacement.

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
provenance is not. Replay-buffer persistence supports all three queue policies.
The replay buffer records policy plus effective capacity and rejects a cross-policy or
cross-capacity restore. For `queue-drop`, snapshot-time completed-task promotion
applies oldest-first overflow eviction before serialization, including the same
length, reward-lifecycle, token, and group accounting as live eviction.

Runs before 2026-08-09 have none of these keys, and nothing back-fills.

## Additive recycle, contribution, and pipeline telemetry (2026-08-13)

This instrumentation is additive.  The historical meanings of
`rollout/fully_async/{avg,max}_staleness`, `staleness/bound/*`, and
`staleness/{pre_queue,in_queue,total}/*` above are unchanged.  New analyses use
new namespaces so a dashboard spanning old and new jobs cannot silently change
its estimand.

### The generated-token ledger

For one drained cohort, the strict accounting boundary is:

```
generated response tokens
  = recycle/abort/filter/age-cutoff/queue-eviction discards
  + admitted response tokens

admitted response tokens
  = postprocess-trimmed tokens
  + selected response tokens

selected response tokens
  = final loss-input tokens
  + final loss-masked tokens
```

The corresponding keys are under
`rollout/fully_async/useful_rollout/`.  In particular,

```
efficiency = loss_input_tokens / generated_tokens
```

is the cohort definition of useful rollout efficiency.  It is bounded by one,
and `accounting_error_tokens` must be exactly zero.  A custom sample converter
can change the final boundary outside the built-in path, so the metric reports
`available=0` instead of guessing.

The same-window pipeline view is deliberately named differently:
`throughput/window_useful_efficiency = accepted tokens in this trainer window /
generated tokens completed in this wall window`.  Queue inventory can cross a
window boundary, so this ratio can temporarily exceed one.  For comparisons use
`throughput/cohort_useful_efficiency`; use the window ratio only to diagnose
transients.

Discarded compute is a vector, never one heterogeneous scalar:

| component | unit | implementation boundary |
|---|---|---|
| `decode_tokens` | response tokens | generated response length |
| `prefill_uncached_tokens` | prompt tokens | SGLang prompt tokens minus cached tokens |
| `tool_env_seconds` | wall seconds | `Sample.non_generation_time` |
| `reward_seconds` | wall seconds | measured reward/verifier calls, apportioned once across scored samples |

Keys are `rollout/fully_async/waste/<reason>/<component>` and
`.../all_discarded/<component>`.  Adding the four values together is invalid;
turn them into cost or joules only after applying an explicitly reported hardware
cost model.

The token components are workload-unit proxies, not direct GPU FLOPs.
`decode_tokens` counts returned response tokens and therefore does not charge
speculative draft/rejected tokens; `tool_env_seconds` is summed sample resource
time and can exceed critical-path wall time when environments overlap.

`prefill_uncached_tokens` is exact only for generation paths that populate
SGLang's prompt/cache metadata. A custom generator that leaves both prompt
counters at zero makes this component unavailable in practice; zero must not be
interpreted as proof of a fully cached prefill without checking that provenance.

Reason keys have fixed cardinality, including zero-valued series:

- `stale_at_generation_completion`
- `stale_during_reward_finalize`
- `stale_during_queue_backpressure`
- `stale_in_output_queue`
- `stale_stage_unknown`
- `generation_aborted`
- `actor_weight_sync_overlap`

`rollout/fully_async/recycle_aux/group_straggler_collateral/*` is an auxiliary
label: it counts samples that would have passed the same bound independently but
were recycled with their stale group.  It is not added again to total waste.
Likewise, `staleness/late_stale_trained/forward_handoff_*` is a diagnostic for a
group that crossed the bound after queue selection but before the trainer
forward; it was trained, not recycled.

### Pre-queue critical-path split

For each sample, four lifecycle boundaries are stamped by the rollout event loop:

```
A = trajectory generation/environment work starts
G_i = this sample finishes generation/environment interaction
G = the last generation task in its prompt group finishes
Q = group reward/postprocess finishes and the group is trainable
```

Both applied-weight-version and wall-second views are logged under
`staleness/pre_queue_phase/{version,wall_seconds}/`:

```
active      = G_i - A
group_wait  = G - G_i
postprocess = Q - G
total       = Q - A
```

`identity_max_abs_error` must be zero.  This is an additive *critical-path*
partition: reward work for an early sample may overlap a straggler's generation;
that overlap is assigned to `group_wait`, not double-counted as postprocess.
`exact_sample_frac` distinguishes the built-in lifecycle callback from a coarse
fallback for custom generation functions.

The mitigation mapping is now testable rather than inferred from one total:

| dominant component | first intervention to test |
|---|---|
| active | decode/prefill/tool/environment acceleration |
| group wait | group scheduling, partial admission, straggler handling |
| postprocess | verifier/reward/serialization pipeline |
| in queue | train/rollout allocation and backpressure |
| selection-to-forward gap | updates-per-rollout or trainer scheduling |

### Generated, consumed, recycled, and dropped populations

The queue-policy branch already owns response-length telemetry, so it remains
the canonical source rather than being duplicated under a second namespace:

- `queue/selection/generated/*` is the non-aborted producer-completion window;
- `queue/selection/{offered,trained}/*` is the selection cohort before/after
  admission controls;
- `queue/selection/{stale_recycled,age_cutoff_dropped,dynamic_filter_dropped,
  aborted_recycled,queue_evicted}/*` gives each terminal mechanism.

Each population has sample-length and group-max-length count, sum, mean, std,
p50, p90, p99, and max. `selection_bias/consumed/response_length/*` is the one
additional response-length view: it is after rollout postprocessing and final
loss-input selection, a boundary the queue metrics do not observe.

`selection_bias/<population>/<field>/{mean,max,p50,p90,p99}` supplies the
non-length marginals for `generated`, `admitted`, `consumed`, `recycled`, and
`dropped`: generation duration, reward, group reward mean/variance, numeric
difficulty when supplied, tool/reward time, the three pre-queue phases,
in-queue staleness, and queue wait. The `generated` accumulator is stamped at
producer completion, includes attempts later evicted by queue-drop, and is
checkpointed with the existing queue telemetry.

The scalar `dropped` population is producer-side dynamic-filter removal. Rows
trimmed later by sample filtering or rollout postprocessing remain visible in
`generated` versus `consumed`; the offline reconciler labels them explicitly as
`postprocess_trimmed`.

Marginals answer whether the trained distribution shifted, but not a joint
conditional.  With `--dump-details`, schema-v2 primitive records contain the
group/prompt index, sample index, generation attempt id, disposition and reason,
versions, lengths, durations, rewards, difficulty, phase splits, waste vector,
and final training step/loss-input tokens.  Run:

```
python -m experiments.src.offpolicy_acceleration.analyze_staleness_telemetry \
  --dump-details <run>/dump \
  --out <run>/selection-summary.json \
  --rows-out <run>/selection-rows.jsonl
```

The tool reconciles an admitted attempt with the final postprocessed loss input,
labels admitted rows that disappeared as `postprocess_trimmed`, and reports
`P(consumed | length)`, `P(consumed | reward)`, and the non-parametric joint
`P(consumed | length, reward, difficulty)`.  It can join trainer debug rows by
`(training_step, generation_attempt_id, sample_index)` to add clip fraction,
mask fraction, sequence log-ratio, and absolute policy-objective contribution.
Tensor-parallel duplicates are ignored, context-parallel token parts are summed,
and repeated optimizer updates are kept separate before their diagnostics are
aggregated; `optimizer_updates_observed` records that multiplicity.
Queue-drop evictions use the existing canonical queue lifecycle record; the
analyzer adapts that row rather than writing a duplicate response-length record.
Fields that the older compact lifecycle schema cannot identify, such as exact
per-sample generation duration, remain missing instead of being imputed.
The full records are debug-only because their storage and CPU transfer are not a
reasonable default for production training.

### Consumed versus effective-contribution staleness

Pass `--log-staleness-gradient-metrics` to enable fixed bins `s_0` through
`s_<max>` plus one overflow bin (default max 16).  No additional model forward is
performed.  The implementation reuses the final tokenwise policy objective after
PPO/dual clipping, OPSM, TIS/MIS correction, rejection masks, and the actual loss
reducer.  Its contribution proxy is therefore

```
abs(final tokenwise policy-gradient objective) * final reducer weight
```

This is used rather than reconstructing
`mask * abs(advantage) * abs(ratio)` from earlier intermediates.  It is closer
to the loss that was differentiated, but remains an objective-contribution
proxy, not a gradient norm: it does not contain parameter Jacobians or
cross-token cancellation.

The bin value is the consumed sample's own `drain_version - selected_reference`
(`prefill`, `submission`, or `completion` according to the run). It is not the
historical group-weighted scalar and does not silently include a later trainer
handoff; crossings after drain are reported separately under
`staleness/late_stale_trained/`.

For each lag bin, the log contains:

- consumed sequence, response-token, and pre-loss-token mass;
- effective-contribution mass and absolute contribution per pre-loss token;
- initial, correction, and final mask fractions;
- PPO and importance-correction clip fractions;
- mean absolute policy/rollout and PPO-objective (`current/old_actor`) log-ratios;
- current/rollout token and sequence ESS;
- non-zero contribution fraction.

The pre-loss-token masses are also the sufficient statistics for the age seen
by the loss.  When the overflow bin is empty,

```
K_loss_token = sum_k(k * pre_loss_tokens_k) / sum_k(pre_loss_tokens_k)
```

is exact; the corresponding first and second moments recover its mean,
variance, and tail mass without retaining token ids.  The historical 2026-08-14
sample dumps show why this measure should not be replaced by group mean age:
over 957 resume-eligible updates, loss-token age is 0.159 update older on
average, with a 0.064--0.337 update shift across the common configuration
window.  Response length and age are positively associated in every common
configuration.

For future runs that do not enable all fixed bins, retain the equivalent compact
per-update moments alongside the ordinary group histogram:

- `sum(loss_tokens)`, `sum(K * loss_tokens)`, and
  `sum(K^2 * loss_tokens)`;
- loss-token counts above each configured age threshold;
- the same first moments weighted by absolute advantage or reward contrast
  when those values are already present during batch construction.

These are detached scalar reductions over values already needed by training;
they require neither response text nor an additional model evaluation.  The
absolute-advantage/reward moments remain exposure proxies, not exact parameter
influence, because clipping, importance ratios, parameter Jacobians, and signed
cancellation act later.

The fixed age bins support a stronger time-aligned diagnostic than
`mean(K) * current_grad`.  With per-update scale
`q_j = lr_j * min(grad_norm_j, clip_norm)`, reconstruct

```
D_t = sum_h q_(t-h) * P_loss_token(K >= h)
```

which is algebraically identical to averaging `sum(q_(t-K), ..., q_(t-1))`
over the loss-token age distribution.  This needs the tail counts already in
the fixed bins and the historical LR/gradient scalars, but no token ids and no
additional training compute.  It is an SGD-style path-scale proxy, not an Adam
parameter displacement: exact displacement additionally requires the applied
preconditioned update norm and direction/cosine.

This path consists of detached `torch.bincount` reductions over tensors already
resident on the training GPU, followed by the existing distributed metric
reduction.  It does not copy token arrays to CPU or touch the autograd graph.
In deterministic training mode, the deterministic-algorithm guard is disabled
only while dispatching these detached CUDA bincounts and is restored before the
loss returns. Training kernels and gradients remain deterministic; the new
diagnostic sums themselves may differ in their last floating-point bits because
CUDA bincount uses atomic additions.
`--log-staleness-gradient-ratio-histogram` adds a fixed 15-bin signed log-ratio
histogram and a capped approximate p95; it is separate because it emits many
more scalar series.  Custom policy-gradient reducers are rejected for this
metric rather than reported with the wrong normalization.

### Exact token versions and throughput

The SGLang flag `--sglang-enable-response-weight-version-segments` returns compact
`[start, end, applied_weight_version]` runs for response tokens.  It is off by
default.  Consecutive forwards under one version merge, speculative accepted
tokens form one segment, stop-trimmed tails are clipped before returning, and the
normal disabled path adds no response metadata.  Miles reports coverage and the
exact token-weighted lag distribution under `staleness/token_lag/exact/*`.
Malformed, overlapping, gapped, or future-version segments are excluded rather
than repaired, with explicit invalid segment/turn/sample counters.

The pipeline telemetry closes one non-overlapping wall window immediately after
each successful actor train call. Generated tokens, accepted loss-input tokens,
and completed optimizer updates therefore refer to the same elapsed interval;
the first and final batches are not shifted into adjacent windows.

It reports:

- `throughput/{generated,accepted,useful}_tokens_per_second`;
- `throughput/optimizer_updates_per_second`;
- time-weighted `queue/depth_time_mean` and instantaneous `depth_current`;
- trainer starvation and rollout backpressure seconds;
- equivalent full-capacity rollout idle seconds, time-mean utilization, and
  instantaneous active rollout capacity fraction;
- queue wait mean/p50/p90/p99;
- existing `perf/step_time`, `perf/log_probs_time`, actor-train, and weight-sync
  timers.

Accepted/useful token rates are marked unavailable for a custom train-data
converter, because Miles cannot prove which response tokens that converter
actually sends to the loss. Generated-token rate and queue timings remain
available.

`throughput/useful_tokens_per_second` is intentionally the reported form of
`eta_useful * generated_tokens_per_second`; algebraically it equals
`accepted_tokens_per_second`. Both names are kept because the former exposes
the efficiency decomposition and the latter is the direct service rate.

The pre-existing `perf/optimizer_updates_per_second` is not renamed or replaced:
it divides updates by the trainer's local `perf/step_time` boundary.
`throughput/optimizer_updates_per_second` closes on the rollout event loop after
successful actor training and shares its wall window with generated and accepted
tokens.  They should be close in steady state, but only the latter supports the
same-window service-rate decomposition.  Keeping the namespaces separate avoids
silently changing the historical `perf/*` estimand.

The queue-depth mean integrates depth over time; it is not the mean of samples
taken only when a training step finishes.

### Validation envelope (2026-08-13)

Job 15728682 compared this branch with a detached clean checkout of
`experiments/cw-dfw-math-rl` on one 8-H100 interactive node.  All arms used the
same fixed 128-sample dump with non-zero policy loss and lags 0, 1, 2, 4, 8, and
17.  The deterministic three-step check compared 99 common training scalars per
step and three saved gradient-norm files.  Clean versus always-on telemetry, and
telemetry-off versus gradient bins plus histogram, were bitwise equal as logged;
the maximum absolute difference was zero in both comparisons.

The performance check discarded two warmup steps and measured eight steps:

| increment | median `perf/step_time` | median `perf/actor_train_time` | peak HBM |
|---|---:|---:|---:|
| always-on telemetry minus clean target | +0.071% (+22.3 ms) | -0.008% (-2.4 ms) | +0 MiB |
| gradient bins minus telemetry off | +0.668% (+209.8 ms) | +0.661% (+204.4 ms) | +4 MiB |
| ratio histogram minus gradient bins | +0.109% (+34.5 ms) | +0.087% (+27.1 ms) | +0 MiB |

These are short-run operational measurements, not confidence intervals.  They
bound the observed cost in the production training path and show why the bins
and histogram remain opt-in.  The always-on Python hot-path benchmark processed
3072 samples / 3.21M generated tokens in 61.1 ms for final accounting; the
individual generated/consumed population and pre-queue passes were 29.5 ms and
33.9 ms.  One discarded group's waste vector took 8.4 us.

The SGLang opt-in scheduler stamp added at most 5.6 ns per request through batch
size 128 in the microbenchmark (the batch-512 delta was below timing noise).
Response-segment recording cost about 0.20 us per generated token when a weight
version lasted 32 or 512 tokens, and 1.69 us in the adversarial case where every
token changed version.  It remains disabled unless exact token provenance is
requested.

The separate real-engine smoke, job 15727740, completed three optimizer updates
with exact response-token segment coverage 1.0, zero useful-token accounting
error, and the consumption/throughput streams present.  Its short 1024-token
responses all truncated and had zero reward variance, so it validates live
integration rather than non-zero contribution; job 15728682 supplies the latter.

### What can be causal

The useful causal skeleton for interpretation is:

```
node ratio / queue policy / update cadence
  -> generation-to-training service ratio
  -> queue depth, wait, starvation, backpressure
  -> realized staleness and recycle reason
  -> generated-to-consumed selection
  -> trained length/reward/difficulty distribution
  -> objective contribution by staleness
  -> learning curve

prompt difficulty / tools / response length
  -> active, group-wait, and reward latency
  -> both staleness and probability of consumption
```

Within one observational run, correlations and conditional acceptance rates can
locate a mechanism but cannot identify its causal effect: prompt difficulty and
response length are common causes of both latency and reward.  A causal statement
requires changing one upstream control while holding the rest fixed—for example
randomizing the max-staleness rule, queue capacity/policy, train:rollout node
ratio, or update cadence—and then checking the predicted mediator chain.  An
algorithm comparison can causally attribute a change in effective-contribution
staleness to the algorithm only when both arms receive the same generated batch
or are paired on the same attempt records; otherwise selection changed first.
