# What the runs record, and what the analysis needs

The historical inventory was audited against a live async run
(`v-3n-8t16r-r1`, job 15150858) plus an earlier job that reached an eval
(15113756).  Later sections identify newer instrumentation and its validation
envelope explicitly rather than implying that it existed in those old runs.

## W&B namespace and producer map

W&B namespace-to-axis assignment has one source of truth,
`_STEP_METRIC_PREFIXES` in `miles/utils/tracking_utils/wandb_utils.py`:

| namespace | step metric |
|---|---|
| `train/*` | `train/step` |
| `sample_staleness/*` | `train/step` |
| `rollout/*`, `fully_async/*`, `resume/*`, `multi_turn/*`, `passrate/*`, `perf/*` | `rollout/step` |
| `staleness/*`, `selection_bias/*`, `throughput/*`, `queue/*` | `rollout/step` |
| `eval/*` | `eval/step` |

`train/step` and `rollout/step` are separate axes on purpose: with
`--num-steps-per-rollout > 1` they advance at different rates, and the off-policy
question is asked in both currencies.

| producer | principal namespaces | contents |
|---|---|---|
| Ray rollout manager | `rollout/*`, `passrate/*`, `perf/*` | generated sample reward, length, truncation, version, pass rate, and generation timing |
| fully-async rollout manager | `staleness/*`, `selection_bias/*`, `queue/*`, `throughput/*`, `fully_async/*`, `resume/*`, `rollout/fully_async/*` | queue lifecycle, exact/aggregate lag, filtering/recycling, useful-token ledger, batch train version, pipeline rates, and replay restore |
| trainer rollout summary | `rollout/*`, `multi_turn/*`, `perf/*` | actor/ref/rollout log-prob, entropy/advantage summaries, multi-turn summaries, and trainer timers |
| trainer optimizer step | `train/*`, `sample_staleness/*` | loss, gradient, LR, batching, TIS/PPO, mismatch/ESS, and sample-staleness-conditioned objective diagnostics |
| evaluator | `eval/*` | benchmark score, reward failures, truncation, length, repetition, and optional pass rate |

Thus a namespace identifies both meaning and plotting axis; several writers may
contribute rows at one rollout step.  W&B automatic host telemetry and optional
forwarded SGLang OpenMetrics (`sgl_engine.*`) are outside this Miles namespace
registry.  Multi-LoRA adapter names are dynamic and register their own
`<adapter>/step` axes on demand.

## Covered

**Realized staleness** — the thing the study is about, and it is measured rather
than assumed:

    rollout/weight_version/{min,max,mean,median,mixed_version_ratio}
    staleness/rollout/*                         # before the max-staleness check
    staleness/{total,pre_queue,in_queue}/*      # accepted training population
    staleness/bound_exceeded_{samples,groups,tokens}
    staleness/bound_exceeded_sample_frac        # rejected samples / evaluated samples
    fully_async/train_weight_version             # absolute T_b used by those train-time gaps

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

`train_rollout_logprob_abs_diff` combines policy lag with the Megatron/SGLang
scoring mismatch.  It can be non-zero at zero staleness, so the zero-lag or
`--true-on-policy-mode` value is the numerical floor; only the excess above a
matched floor can be associated with staleness.  The signed log-ratio is also
needed to distinguish a directional drift from symmetric numerical noise.

For vanilla TIS, `tis_clipfrac` is the fraction of valid pre-rejection loss
tokens whose importance weight was changed by `clamp(tis, low, high)`.  Those
tokens are not removed: they remain in the loss with the capped weight.  It is
therefore neither response truncation nor a discarded-sample fraction.  A
custom rejection-style correction such as IcePop can instead turn out-of-range
weights into zero, so the correction mode must accompany this metric when runs
are compared.  `pg_clipfrac` separately measures PPO objective clipping.

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

**Cumulative accepted loss tokens are recorded for new replay-buffer
checkpoints.** `throughput/cumulative_accepted_loss_tokens` is the running sum
of postprocessed response tokens whose built-in loss mask is one.  It excludes
prompt, padding, postprocess-trimmed, and initially masked response tokens.  It
is a pre-correction eligibility count: vanilla TIS clipping retains these
tokens, while a rejection-style custom correction can subsequently set some
final masks to zero.  The companion
`throughput/cumulative_accepted_loss_tokens_available` must be one before the
counter is used.  The counter is persisted in the replay state, so it continues
across a replay-enabled resume without
turning the pre-resume total into a first-window rate spike.  A legacy replay
checkpoint or custom sample converter reports unavailable instead of silently
restarting at zero.  Cumulative samples remain recoverable from the batch
shape, and historical token totals can still be reconstructed by summing the
per-step cohort counter after selecting the valid checkpoint lineage.
Without replay-buffer restoration, a new process starts this counter at zero;
the value is then cumulative only within that job, just as the applied numeric
weight-version label is.  The current async recipe uses replay restoration.

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

**`--partial-rollout` removed from the `math/sync` recipes.** It recycles
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

The runs analyzed in this section used `HF_SAVE_INTERVAL=5`; the current math
recipes use `HF_SAVE_INTERVAL=10`. In either case,
`experiments/src/offline_eval/run_eval.sbatch` can score the exports off the
training critical path. That removes the exclusion entirely and buys a better eval
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
  the then-current `HF_SAVE_INTERVAL=5` exports) removes the perturbation rather than
  correcting for it, and is the right configuration for tier 2 onward. It is
  still not something to change inside a running tier.

### In-run evaluation is off by default from 2026-08-07

`EVAL_INTERVAL` defaults to `0` in both recipes, and `train.sh` then passes no
`--eval-interval` at all, which leaves `args.eval_interval` None and turns off
both call sites (`train.py:98` before-train, `train.py:144` periodic). Quality is
read by scoring the current `HF_SAVE_INTERVAL=10` exports offline; the historical
measurements above used an interval of 5.

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

## The two staleness populations (updated 2026-08-19)

| key | population |
|---|---|
| `staleness/rollout/*` | would-be train staleness of every group examined before the max-staleness check |
| `staleness/total/*` | train staleness of groups accepted by the max-staleness check and dynamic filter |

Both distributions contain `mean`, `variance`, `std`, `max`, `p50`, `p90`,
`p99`, `frac_zero`, `num_groups`, and the fixed histogram `count_0` through
`count_16` plus `count_ge_17`. There is no `frac_at_bound`: the admission check
uses dequeue staleness while these distributions use scheduled train
staleness, and the exact boundary also depends on queue policy. Rejection is
therefore recorded directly rather than inferred from a histogram bin.

Actual rejection by `--max-weight-staleness` is recorded directly as
`staleness/bound_exceeded_samples`, `staleness/bound_exceeded_sample_frac`,
`staleness/bound_exceeded_groups`, and `staleness/bound_exceeded_tokens`. The
sample fraction is

```
rejected samples / samples whose reference version was available at dequeue
```

Do not infer the rejection count by subtracting the two distributions: the
dynamic filter can also remove a group after it passes the bound. Neither
distribution contains aborted groups or groups whose reference version could
not be read.

## Staleness reference and queue policy (2026-08-09; updated 2026-08-19)

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

### The decomposition with `--staleness-reference prefill`

The scalar families follow Applied Compute's PQS/IQS split
([staleness in fully-async RL](https://www.appliedcompute.com/research/staleness-in-fully-async-rl)).
For group `g`, let **F_g** be the minimum first-prefill version across its
samples and turns, **Q_g** the applied version when the whole group became
ready, **P_g** its actual queue-put version, **D_g** the applied version at
dequeue/selection, and **T_b** the version used to train batch `b`:

| key | quantity | gated on the bound? |
|---|---|---|
| `staleness/pre_queue/*` | `Q_g - F_g` — updates crossed before the whole group became ready | accepted groups |
| `staleness/in_queue/*` | `T_b - Q_g` — updates crossed from ready to training | accepted groups |
| `staleness/total/*` | `T_b - F_g` = `pre_queue + in_queue` | accepted groups |
| `staleness/rollout/*` | `T*_b - F_g`, where `T*_b` is the scheduled train version if offered | before the bound check |

Q is stamped only after the whole group has completed generation, reward, and
postprocessing and is ready for the queue. The actual queue-put version is a
separate boundary `P_g`, stamped after any output-capacity backpressure. A group
is one concurrent request per sample joined by `asyncio.gather`; using an earlier
sample completion would incorrectly charge the straggler interval to `in_queue`.
`G_g`, the last model-forward boundary, equals `Q_g` only when no weight update
is applied during the post-forward reward/finalization interval. Likewise,
`Q_g = P_g` only when no update lands while the ready group waits for an output
slot. The lifecycle order is `G_g <= Q_g <= P_g <= D_g <= T_b`, although adjacent
boundaries often have the same numeric version. The current `in_queue` metric
starts at Q and therefore contains both ready-to-put backpressure and
put-to-train queue residence.

The `prefill` reference comes from scheduler-authoritative engine provenance.
Missing provenance stays missing rather than becoming zero; `prefill`
enforcement fails fast if those fields are absent or misaligned.

### What the bound tests

For `queue-recycle`, dequeue admission is strict:

```
D_g - F_g < --max-weight-staleness
```

With the normal prefetched schedule and `--update-weights-interval 1`, `T_b=D_g+1`,
so this is equivalent to `T_b-F_g <= max`. Equality at dequeue is rejected only
for `queue-recycle`; `queue-max` retains its inclusive dequeue condition.
`staleness/rollout/*` reports the would-be train-time quantity and
`staleness/total/*` reports the accepted train-time population. Actual rejection is counted directly by the
`staleness/bound_exceeded_*` metrics.

The minimum usable queue-recycle bound is 1 because `D_g-F_g` is nonnegative:
no group can satisfy the strict rule when `max=0`. This has nothing to do with
the startup offset. The startup batch has `updates_before_train=0` and therefore
`T_b=D_g`; the same strict check is deliberately conservative for that one
batch. Later prefetched batches normally have `updates_before_train=1` and
`T_b=D_g+1`.

This dequeue-time definition is not always the version gap at the trainer
forward. `queue-recycle` reserves the next batch while the current batch is
training. With the recipe's `--update-weights-interval 1`, after the first batch
the complete order is

```
time ---------------------------------------------------------------------->
next batch: prefill/decode(F...) -> ready(Q) -> dequeue/prefetch(D) ---------+
trainer:                                         train current batch        |
                                                 -> publish D + 1            |
                                                 -> train next batch at T=D+1
```

Therefore its train-time group gap is `D_g + 1 - F_g`; the strict dequeue check
keeps that gap at or below `max`. For `queue-max` and `queue-drop`, the
driver does not prefetch a training batch: selection occurs after the preceding
weight update, so no corresponding `+1` is introduced. The deterministic
offset is reflected in `T_b` and the train-time staleness metrics; it is not
reported as a separate handoff metric.

A bound tighter than the pipeline can meet collapses overlap rather than
deadlocking. While drain waits, training cannot publish a new version, so fresh
groups eventually pass under the frozen version. The signatures are a high
`wasted_token_frac`, many bound-exceeded groups, and rising retry counts.

The staleness histograms resolve `count_0` through `count_16` plus
`count_ge_17`. Dumps carry all four forward-provenance arrays, completion
versions, and the queue lifecycle's ready/enqueue/dequeue/decision fields for
offline reconstruction.

Each new lifecycle record also carries scalar `reward_values` aligned with its
`sample_indices` and `response_lengths`. Reward evaluation has already completed
before the group becomes ready, so recording the values does not call the RM or
send anything through the trainer object store. The offline analyzer reports
sample and group reward distributions, all-zero/all-one/mixed group fractions,
and sample-length/reward and group-max-length/group-mean-reward correlations for
every terminal disposition. Older dumps remain readable and report missing
reward coverage rather than inventing zero rewards.

### Queue policies and the formula boundary

- `queue-recycle` is the default: completion FIFO, immediate
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

The 2026-08-19 logging revision moves `staleness/rollout`,
`staleness/{in_queue,total}`, lifecycle `bound_staleness`, sample lag, and exact
token lag to the scheduled train version. It also removes the former
selection-to-train and train-side bound diagnostics. Do not merge pre-revision
and post-revision series as if their estimands were identical.

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

### Pre-queue critical-path split

For each sample, four lifecycle boundaries are stamped by the rollout event loop:

```
A = trajectory generation/environment work starts
C_i = this sample's generation/environment task returns
C_g = the last generation task in its prompt group returns
Q = group reward/postprocess finishes and the group is trainable
```

Both applied-weight-version and wall-second views are logged under
`staleness/pre_queue_phase/{version,wall_seconds}/`:

```
active      = C_i - A
group_wait  = C_g - C_i
postprocess = Q - C_g
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
conditional.  With `--dump-details`, schema-v3 primitive records contain the
group/prompt index, sample index, generation attempt id, disposition and reason,
dequeue and scheduled-train versions, lengths, durations, rewards, difficulty,
phase splits, waste vector, and final training step/loss-input tokens.  Run:

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

### Trainer diagnostics conditioned on each sample's staleness

Pass `--log-sample-staleness-metrics` to enable fixed bins `s_0` through
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

These metrics use the top-level `sample_staleness/*` namespace and the
`train/step` axis. The name deliberately does not say "gradient": the logged
quantity is an objective-contribution proxy grouped by staleness, not a gradient
norm.

With the prefill reference, sample `i` uses

```
s_i = T_b - F_i
```

where `F_i` is that sample's own minimum first-prefill version and `T_b` is the
batch train version. This differs from the group-control reference `F_g`, which
is the minimum across the whole group.

The rollout-side `staleness/sample_lag/*` decomposition uses two additional
scheduler-provenance boundaries:

```
G_i = max(sample_i.last_forward_weight_versions)
G_g = max_i(G_i)

generation            = G_i - F_i
group_sync            = G_g - G_i
last_forward_to_train = T_b - G_g
total                 = T_b - F_i
```

`G_i` is the applied SGLang weight version of sample `i`'s last model forward,
not a group id, gradient, or wall-clock timestamp. `G_g` is the latest such
version among samples in the prompt group. Consequently `group_sync` measures
weight updates crossed while an early sample waited for the group's last model
forward; environment/reward/postprocess work after that forward is included in
`last_forward_to_train`.

`last_forward_to_train` is not the in-queue staleness. Let `Q_g` be the
group-ready version and `P_g` the later actual queue-put version. Then

```
in_queue             = T_b - Q_g
last_forward_to_train = T_b - G_g
                      = (Q_g - G_g) + (T_b - Q_g)
in_queue              = (P_g - Q_g) + (T_b - P_g)
```

`Q_g - G_g` covers work after the group's last model forward but before it is
queue-ready. `P_g - Q_g` is output-capacity backpressure, and `T_b - P_g` covers
the interval after the actual put through training. Calling all of
`T_b - G_g` an in-queue or handoff lag would therefore hide real boundaries.

With `--staleness-reference prefill`, `sample_staleness` and
`staleness/sample_lag/total` therefore use exactly the same per-sample scalar,
`T_b - F_i`. They are not duplicate log products: `sample_staleness/*` bins
trainer-side loss/objective diagnostics, while `sample_lag/total` reports
rollout-side sequence-, generated-token-, and loss-token-weighted summaries and
correlations. `sample_lag` additionally requires complete last-forward
provenance for the group, so its `provenance_sample_frac` can be below one even
when the simpler sample-staleness scalar is available.

This assignment is per sample.  If one response was decoded across several
weight versions, every token in that response currently enters the same
`sample_staleness/s_*` bin.  The section is consequently exact for the
selected sample-reference definition, but it is not yet an exact-token-version
breakdown for mixed-version responses.

For example, suppose training uses version 12, the selected prefill
reference is version 10, but 100 response tokens were actually decoded under
version 10 and 900 under version 11.  The current sample-level implementation
puts all 1,000 tokens in `sample_staleness/s_2`, because
`12 - selected_reference = 2`.  An exact-token implementation would put 100
tokens in lag 2 and 900 in lag 1.  This is what it means that compact mixed
response segments are not yet reflected in `sample_staleness`: the segments
are retained and summarized on the rollout side, but are not carried into the
trainer's per-token bin ids.

For each lag bin, the log contains:

- consumed sequence, response-token, and pre-loss-token mass;
- effective-contribution mass and absolute contribution per pre-loss token;
- initial, correction, and final mask fractions;
- PPO and importance-correction clip fractions;
- mean absolute policy/rollout and PPO-objective (`current/old_actor`) log-ratios;
- current/rollout token and sequence ESS;
- non-zero contribution fraction.

The main conditionals have the following meanings:

- `mean_abs_policy_rollout_log_ratio` averages
  `abs(log pi_train(token) - log pi_rollout(token))` over pre-loss tokens in the
  bin.  It measures behavior/current disagreement, including both true policy
  lag and the Megatron/SGLang numerical floor.
- `importance_clip_fraction` is the fraction of those tokens whose TIS/MIS
  importance correction was clipped or truncated.  Vanilla TIS keeps a clipped
  token at the capped weight; this is not response truncation.
- `ppo_clip_fraction` is clipping of the current/old-actor proximal objective,
  not TIS.  In the current one-step fused-actor recipe the old-actor anchor is a
  detach of the same forward, so its ratio is exactly one and this metric should
  be zero.  TIS, not PPO clipping, is the relevant stale-policy correction in
  that recipe.
- `policy_rollout_ratio_token_ess` is normalized Kish ESS of token importance
  weights, `(sum w)^2 / (N sum w^2)`, within the lag bin.  One means uniform
  weights; values near zero mean that a small token subset dominates.
- `policy_rollout_ratio_sequence_ess` applies the same formula to per-response
  weights `exp(sum_t log(pi_train/pi_rollout))`.  It exposes response-level
  concentration that a within-response token ESS can hide.
- `effective_contribution_mass` is the bin's share of
  `abs(final tokenwise policy objective) * reducer weight` after masks,
  PPO/dual clipping, OPSM, and importance correction.  It is a scalar-objective
  attribution proxy, not a parameter-gradient norm or direction.

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
`--log-sample-staleness-ratio-histogram` adds a fixed 15-bin signed log-ratio
histogram and a capped approximate p95; it is separate because it emits many
more scalar series.  Custom policy-gradient reducers are rejected for this
metric rather than reported with the wrong normalization.

### Low-overhead optimizer-update diagnostics (2026-08-19)

The async DAPO-Math recipe enables `--log-update-diagnostics`. It adds the
following train-step scalars without another model forward or parameter scan:

- `train/optimizer_step_applied`: one only when the optimizer step ran;
- `train/grad_norm_pre_clip`: the pre-clip norm already returned by Megatron's
  optimizer (`train/grad_norm` remains as the compatibility name);
- `train/grad_clip_coefficient`: `min(1, clip_grad / (grad_norm + 1e-6))`;
- `train/final_loss_tokens`: tokens surviving the final loss mask after
  importance correction or rejection;
- `train/advantage_std`, `train/advantage_rms`, and
  `train/advantage_abs_mean`, weighted over those final loss tokens.

The advantage sums, squared sums, absolute sums, and token count join the
existing loss-metric reduction, so they add elementwise operations over tensors
already resident in policy loss but no new distributed collective.
`train/num_zeros_in_grad` is emitted only if the optimizer has already computed
that value; the recipe does not enable Megatron's extra gradient-zero scan.

`train/update_norm`, `train/parameter_norm`, `train/relative_update_norm`, and
`train/cumulative_update_path_norm` are deliberately not produced. Exact values
would require traversing all parameters and, for update norms, retaining or
reconstructing the preconditioned parameter delta. That cost is not justified
for an always-on diagnostic, and an LR-scaled gradient norm is not labeled as an
Adam parameter displacement.

### Exact token versions and throughput

The SGLang flag `--sglang-enable-response-weight-version-segments` returns compact
`[start, end, applied_weight_version]` runs for response tokens.  It is off by
default.  Consecutive forwards under one version merge, speculative accepted
tokens form one segment, stop-trimmed tails are clipped before returning, and the
normal disabled path adds no response metadata.  Miles reports coverage and the
exact token-weighted lag distribution under `staleness/token_lag/exact/*`.
Malformed, overlapping, gapped, or future-version segments are excluded rather
than repaired, with explicit invalid segment/turn/sample counters.

Here **exact** means that each covered response token is assigned the applied
SGLang scheduler weight version that produced it.  It does not mean request
submission version, first/last/min/max forward version, or one version inferred
for a whole sample. For covered response token `t`, lag is evaluated at training as

```
L_t = T_b - V_t
```

where `V_t` is the scheduler-applied version that actually produced the token.
Coverage and invalid counters must accompany the distribution; uncovered tokens
are not imputed.

The response-token distribution remains at the root.  The
`staleness/token_lag/exact/loss_token/*` child intersects complete ordered
segments with the built-in pre-correction response loss mask and reports
coverage, mean, population variance/std, max, p50/p90/p99, count, and tail
fractions at lags 1, 2, 4, 8, and 16.  It expands no object-store token-version
array: all-one/all-zero masks and single-version responses reduce directly, and
only a mixed mask spanning multiple version segments builds one response-local
prefix sum.  The sibling
`loss_sequence/*` distribution gives each covered response one
loss-token-weighted effective lag, then weights responses equally.

The async DAPO-Math Qwen3-4B recipe enables this flag by default.  Other recipes
retain their existing defaults.  Group-, trained-sequence-, lifecycle-, and exact
token-weighted staleness distributions report population variance and standard
deviation (`ddof=0`) alongside their mean and percentiles.  The fixed
`staleness/version_mix/train/*` section summarizes the always-on min/max forward
provenance: coverage, mixed-sample fraction, response-token mass belonging to
mixed samples, and the distribution of each sample's forward-version span.  It
does not create absolute-version keys, so metric cardinality stays fixed.

For mixed-version responses, one mean should not replace the distribution.  Four
weightings answer different questions:

- sequence weighting gives every sampled response equal mass and describes the
  sampling/queue population;
- response-token weighting describes generated compute and is the meaning of
  the current `staleness/token_lag/exact/*` distribution;
- loss-token weighting intersects the exact token-version segments with the
  pre-correction loss mask and is the preferred measure of staleness exposure
  seen by training;
- effective-contribution weighting describes how much of the final scalar
  policy objective came from each lag, after masks, clipping, and importance
  correction.

The rollout-side exact loss-token distribution and tail fractions are now
present.  A later training-facing mixed-version extension should retain the
fixed sample-reference bins for queue-control compatibility and add fixed
relative lag bins (not absolute version-number keys) for exact final
objective-contribution mass.  Useful per-lag conditionals are signed and
absolute train/rollout log-ratio, TIS and PPO clip fractions, token/sequence ESS,
and non-zero objective contribution.  Carry the existing compact segments to
the trainer and expand only local lag-bin ids on GPU; do not materialize an
object-store int64 version array for every token.  Those exact-lag conditionals
are not emitted by the current `sample_staleness/*` implementation.

### Applied SGLang weight version across resume

The version label continues only when replay-buffer state is restored.  Miles
captures the rollout manager's committed `applied_weight_version = N`, restores
both that tracker and the trainer weight-updater counter to N, then performs the
normal startup weight push.  The updater tags that push N+1, and the rollout
manager commits N+1 only after every SGLang engine has finalized the update.
The current async DAPO-Math recipe has `USE_REPLAY_BUFFER=1`, so its version
sequence is continuous across checkpoint jobs.
`resume/replay_buffer/applied_weight_version_restored` records N.  A restored
prepared batch also records
`resume/replay_buffer/current_applied_weight_version`; otherwise compare it with
`fully_async/train_weight_version`.  The first post-startup value must
be N+1, providing a direct W&B continuity check.

SGLang does not independently recover the numeric label from the Megatron model
checkpoint.  With replay disabled (or with no usable replay state), new
processes initialize the counter at zero and the startup push is version 1 even
though the model parameters themselves resume correctly.  In that case the
label is locally consistent within the new job but not a cross-job global
version axis.

Even that is an objective-level attribution, not a parameter-gradient
attribution.  Exact per-lag gradient norms or gradient cosines require separate
backward/per-sample-gradient work and are unsuitable as an always-on metric.
For instability and convergence, pair the lag conditionals with global gradient
norm and optimizer-clipping incidence, applied update norm divided by parameter
norm, entropy, reward/eval score, response truncation/repetition, accepted loss
tokens, and wall time.  Plot convergence against cumulative accepted loss tokens
as well as optimizer step; logging alone establishes association, so causal
claims still require matched runs that vary staleness while holding inference
engine and token budget fixed.

On a 3,072-sample synthetic batch, the `version_mix` reduction took a median
1.24 ms and emitted 11 scalars.  The SGLang CPU hot-path microbenchmark measured
approximately 222--223 ns per generated token for compact segment maintenance
when a version remained stable for 32--512 tokens; changing version every token
cost approximately 1.9 microseconds per token.  These are isolated CPU costs,
not an end-to-end GPU throughput measurement.

The pipeline telemetry closes one non-overlapping wall window immediately after
each successful actor train call. Generated tokens, accepted loss-input tokens,
and completed optimizer updates therefore refer to the same elapsed interval;
the first and final batches are not shifted into adjacent windows.

It reports:

- `throughput/{generated,accepted,useful}_tokens_per_second`;
- `throughput/cumulative_accepted_loss_tokens`, with its availability flag;
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
bound the observed cost in the production training path.  The implementation
remains flag-gated; the later Qwen3-4B recipe decision is recorded in the
2026-08-19 validation below.  The always-on Python hot-path benchmark processed
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

### Current recipe validation (2026-08-19)

Job 16109706 reran the fixed Qwen3-4B workload on one 8-H100 interactive node
after adding the loss-token exact-lag and resume-stable cumulative-token
telemetry.  A single trainer node is the relevant isolation boundary here:
`sample_staleness/*` consists of trainer-local detached CUDA reductions and
does not add a rollout-engine or cross-node communication path.  A two-node live
rollout would add queue and generation noise without exercising a different
implementation path.

The deterministic telemetry-off versus base-bin check compared 99 common
training scalars at each of three steps plus three saved gradient-norm files.
They were bitwise equal as logged, with maximum absolute difference zero.  The
performance check discarded two warmup steps and measured eight steps:

| increment | median `perf/step_time` | median `perf/actor_train_time` | peak HBM |
|---|---:|---:|---:|
| base bins minus telemetry off | +0.537% (+168.7 ms) | +0.673% (+208.1 ms) | +4 MiB |
| ratio histogram minus base bins | +0.273% (+86.3 ms) | +0.330% (+102.7 ms) | +0 MiB |

The corresponding end-to-end wall deltas were within short-run noise: -0.022%
for base bins versus off and +0.222% for the histogram versus base bins.  These
are operational point measurements, not confidence intervals.  On this evidence
the async DAPO-Math Qwen3-4B recipe sets
`LOG_SAMPLE_STALENESS_METRICS=1`; the high-cardinality ratio histogram remains
off by default.

The same job measured rollout-side exact-lag accounting over 3,072 samples and
3.21M generated tokens.  Compared with the 60.71 ms no-segment finalization
path, one exact version segment per response added 7.14 ms per whole batch
(2.32 us/sample), while the synthetic two-segment responses with a loss mask
crossing the version boundary added 57.92 ms (18.85 us/sample).  The all-loss-
token, single-segment case added 7.27 ms.  These costs occur once during rollout
batch finalization, not per training token or model forward.

The isolated SGLang benchmark found approximately 0.20 us per generated token
for compact segment maintenance when a version remained stable for 32--512
tokens and 1.70 us/token in the adversarial every-token-version-change case.
Scheduler version stamping was at most 2.9 ns/request for tested batch sizes
8--512 (19 ns at batch size one).  The production pattern is the stable-version
case, since versions change only on weight updates.  The recipe therefore keeps
`SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1` so exact response- and loss-token lag
remain observable.

The result bundle is under
`experiments/outputs/staleness-telemetry-gpu-16109706/`.  Its analyzer also has
a nominal `clean_head` arm, but this invocation pointed that arm at the same
working tree; only the telemetry-off versus enabled deltas above are used as a
code-path comparison.

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
