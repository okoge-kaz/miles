# offpolicy_acceleration

Does an off-policy step actually **accelerate** training, or does it only reach a
selected final checkpoint faster on paper? This directory measures that, with a
pre-registered convergence definition and confidence intervals.

`experiments/notes/off-policy-variables.md` catalogues the variables and says the
measurement design is a separate exercise. This is that exercise.

## What it computes

| symbol | meaning | where |
|---|---|---|
| `Q_on*` | on-policy plateau quality, from a pre-registered convergence test | `equivalence.detect_convergence` |
| `Δ_m(t)` | `Q_m(t) − Q_on*`, paired over held-out prompts | `equivalence.noninferiority_time` |
| `τ_m(δ)` | first time the one-sided LCB of `Δ_m(t)` clears `−δ`, `k` evaluations running | `equivalence.noninferiority_time` |
| `q_p` | `Q_0 + p (Q_on* − Q_0)`, the intermediate target ladder | `equivalence.quality_targets` |
| `τ_m(q_p)`, `S_m(p)` | time to each target, and `τ_on(q_p) / τ_m(q_p)` — the **speedup profile** | `equivalence.target_times`, `.speedup` |
| `P(L)` | realized policy-lag distribution, next to the configured bound | `extract_run.realized_lag` |

The primary time axis is **wall-clock at a fixed total GPU budget**; GPU-hours is
reported alongside as the secondary axis, so calendar-time reduction and compute
efficiency never get conflated. `--axis gpu_hours` switches which one is primary.

Uncertainty is a three-level bootstrap — held-out **prompts** (shared across arms,
so the comparison is paired), **rollouts** within a prompt, **training seeds**
within an arm — reported per level by `equivalence.variance_components`.

## Files

| file | role | needs |
|---|---|---|
| `equivalence.py` | the statistics; no I/O | numpy |
| `extract_run.py` | one run's dump dir (or job log) → a compact extract | numpy, torch (only for the dumps) |
| `log_source.py` | recovers the metric stream from a plain Slurm log | stdlib |
| `analyze.py` | extracts → `results.json` + `summary.csv` | numpy |
| `figures.py` | `results.json` → five paper figures | numpy, matplotlib |
| `check_logging.py` | audits a run against what the protocol needs | stdlib |

Split the same way `difficulty_filter` splits measurement from filtering: the one
step that needs torch and the training image runs once per run, and every
question asked of the numbers afterwards runs on a login node.

## Usage

```bash
# 0. before spending GPU time: does this configuration log what the protocol needs?
python -m experiments.src.offpolicy_acceleration.check_logging \
  --slurm-log experiments/outputs/training/.../<job>.log

# 1. once per run, inside the container (srun --container-image=... or on the CPU partition)
python -m experiments.src.offpolicy_acceleration.extract_run \
  --dump-details /ckpt/training/math/dapo-math/Qwen3-4B/<config-tag>/dump \
  --out $WS/offpolicy-study/extracts \
  --arm stale2-retract --seed 0 --with-lag \
  --factor MAX_WEIGHT_STALENESS=2 --factor PAUSE_GENERATION_MODE=retract

# 2. anywhere
uv run --with numpy python -m experiments.src.offpolicy_acceleration.analyze \
  --extracts $WS/offpolicy-study/extracts \
  --benchmark aime24 --reference-arm on-policy \
  --out $WS/offpolicy-study/results/aime24

uv run --with numpy --with matplotlib python -m experiments.src.offpolicy_acceleration.figures \
  --results $WS/offpolicy-study/results/aime24/results.json \
  --out $WS/offpolicy-study/figures/aime24 \
  --x-factor MAX_WEIGHT_STALENESS --y-factor MODEL_NAME
```

`extract_run.py` also accepts `--slurm-log` instead of `--dump-details`, for runs
whose dump directory is gone. That path loses the per-prompt eval rewards; see
the audit below. Repeat `--slurm-log` once per allocation of a resumed run.

## Is the current logging enough?

Audited against the source and against a real job log
(`experiments/outputs/training/math/dapo-math-p10-80/qwen3-4b-instruct-2507/…-15113756.log`).

**The Slurm log and wandb carry the same thing.** `tracking.log` hands one payload
to every enabled backend, and miles prints that payload to stdout first
(`ray/rollout/metrics.py:53,79`, `backends/training_utils/log_utils.py:460`). So
the job log *is* a timestamped copy of the wandb run, and `log_source.py` parses
it back into the same records. wandb adds convenience, not information; the two
are insufficient in exactly the same ways.

### Already covered — no change needed

| what | metric | why it matters here |
|---|---|---|
| quality vs time | `eval/<bench>` + the line's own timestamp | `Q(t)`, both time axes |
| collapse guards | `rollout/raw_reward`, `rollout/truncated_ratio`, `rollout/repetition_frac`, `train/kl_loss`, `train/grad_norm` | the convergence definition |
| **train/rollout mismatch** | `train/train_rollout_logprob_abs_diff`, `train/train_rollout_kl` | the numerical floor, and drift above it. Measured **0.0100** at zero staleness on the audited run — that is the constant to subtract before attributing drift to policy lag |
| **staleness-induced drift** | `train/tis` = `exp(train_lp − rollout_lp)`, `train/tis_abs` = `|tis − 1|`, `train/tis_clipfrac` | the actual importance weight π_train/π_rollout. Measured `tis_abs` **0.0100**, `tis_clipfrac` **4.9e-6** on the audited run |
| async pipeline shape | `perf/rollout_time`, `perf/train_wait_time`, `perf/wait_time_ratio`, `perf/tokens_per_gpu_per_sec` | *why* an arm is faster. The audited run sits at `wait_time_ratio` **0.83** — training idles 83% of the step, so rollout is the bottleneck and every group arrives fresh |
| lag bracket | `rollout/weight_version/{min,mean,median,max}`, `…/mixed_version_ratio` | a bracket on the lag when the exact mean is missing |

That last row is worth reading twice. On the audited run `weight_version/min ==
max == 1` and `mixed_version_ratio == 0.0` for every step: **the configured
staleness never bound.** Whatever `MAX_WEIGHT_STALENESS` said, that run trained
on-policy. A results table that reported it as a staleness arm would be reporting
robustness to a lag the run never experienced.

**`train/ess_ratio` does not measure staleness.** `ppo_kl = old_log_probs −
log_probs` with `old_log_probs = batch["log_probs"]` (`losses.py:94,158`) — both
are Megatron logprobs, so `ppo_kl` is the *PPO inner-loop* ratio across the
`NUM_STEPS_PER_ROLLOUT` minibatch updates. At `NUM_STEPS_PER_ROLLOUT=1` the two
forwards use identical weights, so `ppo_kl ≡ 0`, `pg_clipfrac ≡ 0` and
`ess_ratio ≡ 1.0` **by construction, at any staleness**. The audited log shows
exactly that triple. `--use-rollout-logprobs` would repoint it at the rollout
policy but is mutually exclusive with `--use-tis` (`arguments.py:2823`). The ESS
the off-policy literature means is gap 8 below.

### Eight gaps, and what each one blocks

**1. Per-prompt eval rewards are not logged.** `log_eval_rollout_data` reduces
each benchmark to one mean before logging (`ray/rollout/metrics.py:38`). The
paired-over-prompts bootstrap, and separating prompt variance from rollout
variance, both need the per-sample rewards — which exist only in the
`--dump-details` eval dumps (`rollout_data/eval_<rid>.pt`). The `math_async` and
`math_sync` recipes already pass `--dump-details`, so this costs nothing new; it
does mean **the dumps must be kept**, not cleaned up with the checkpoints.

A power consequence to plan for, not to discover later: with AIME24's 30 prompts
at `n_samples_per_eval_prompt=16`, one evaluation's bootstrap half-width is
≈ 0.05 in avg@k. Differencing a single `Q_m(t)` against the many-evaluation mean
`Q_on*` therefore cannot resolve any margin below ≈ 0.05. `analyze.py` reports
the **smallest detectable margin** next to `δ` and prints `UNDERPOWERED` when
`δ` is below it. Three ways out, in order of cost: `--smooth-window` (a trailing
mean, symmetric with how `Q_on*` is formed — the default, 3); a larger
`--n-samples-per-eval-prompt`; a bigger held-out set than a 30-problem benchmark.

**2. Exact realized staleness is conditional on a bound being set.**
`fully_async_rollout.py:202` only measures staleness when
`args.max_weight_staleness is not None`, so the "unbounded" arm would be the one
arm with no staleness measurement. Confirmed empirically: the audited log
contains **zero** `rollout/fully_async/avg_staleness` records. Run the unbounded
arm with `MAX_WEIGHT_STALENESS=1000000` rather than unset.

**3. `P(L)` is never logged as a distribution** — only `avg_staleness` and
`max_staleness`, per step. The full per-sample distribution is recoverable from
`weight_versions` in the rollout dumps (`extract_run.realized_lag`), with two
stated censorings: groups recycled for exceeding the bound never reach the dump,
so the dumped distribution is truncated at the bound; and the reference "current
version" is proxied by the newest version in the same batch. If percentiles
matter enough to want them exactly, it is a five-line addition next to
`fully_async_rollout.py:251`.

**4. Entropy is not actually measured.** `train/entropy_loss` appears in the log
but is identically `0.0`: `calculate_entropy = args.entropy_coef != 0 or
args.observe_training_entropy` (`loss_hub/losses.py:99`), and the recipes set
`--entropy-coef 0.00` without `--observe-training-entropy`. The convergence
definition names entropy explicitly, so **as configured, that criterion cannot be
evaluated**. `equivalence._check_guard` reports an all-zero guard as
`unavailable` rather than `pass`, so this fails loudly instead of certifying a
check that never ran. Fix: add `--observe-training-entropy` to the recipes.
(`--use-rollout-entropy` is a different flag — per-token entropy into the train
dumps — and the recipes only set it when `DUMP_TRAIN_DATA != 0`.)

**5. Neither the factors nor the seed are recorded with the run.**
`dashboard/args.py:13` snapshots 16 keys into `meta.json`; none of them are
`max_weight_staleness`, `num_steps_per_rollout`, `pause_generation_mode`,
`advantage_estimator`, `lr`, `rollout_max_response_len`, `use_tis` or `seed`. So
a run cannot be attributed to a cell of the factorial design from its own
artifacts. `extract_run.py` works around it with `--factor K=V` and
`--manifest` (joining `experiments/sweep.py`'s manifest, which does record the
env per job); the durable fix is adding those keys to `_SNAPSHOT_KEYS`.

**Seed replication is currently not runnable, which matters more.**
`common/run_identity.sh` builds `CONFIG_TAG` from rollout mode, steps, length and
LR — no seed — and `CKPT_PATH` is derived from `CONFIG_TAG`. Two seeds of one
configuration would therefore share a checkpoint directory and resume from each
other's optimizer state. Since seed variance is one of the three levels the
protocol requires, this needs fixing before the study starts: add `SEED` to the
recipe env, pass `--seed`/`--rollout-seed`, and include it in `CONFIG_TAG`.

**6. Wall-clock is not continuous across Slurm allocations.**
`submit_training.sh` documents the normal mode as "resumable across three 4 h
jobs", and a converged on-policy math run on this cluster *is* several
allocations. Two consequences, both of which hit the study's primary metric:

* the dashboard **overwrites** `meta.json` on every resume (`collector.py:153`)
  while `metrics.jsonl` appends, so `ts - meta.start_ts` is **negative** for
  every allocation but the last;
* the queue wait between allocations is unbounded and has nothing to do with the
  method under test.

`log_source.active_elapsed_hours` is the fix and is now the default time base:
sum per-record deltas, drop any delta longer than `--max-step-gap-minutes`
(default 30), and report how much was excluded. It never reads `meta.start_ts`.
Pass every allocation's log with a repeated `--slurm-log` so replayed steps
dedupe to the surviving trajectory. The residual bias — per-allocation startup
is excluded — is identical in shape across arms, so `tau_on/tau_m` is nearly
unaffected, but the absolute times are lower bounds.

**8. ESS of the rollout-vs-train importance weights is not logged.** `train/tis`
and `train/tis_abs` give the *mean* of `π_train/π_rollout` and of `|ratio − 1|`,
but ESS is a sequence-level nonlinear functional: a handful of catastrophic
tokens can crater it while barely moving a mean. That is precisely why the
literature reports ESS and not the mean, and precisely why "ESS collapses on long
sequences" cannot be tested from `tis_abs`. The fix is small —
`compute_ess_ratio_contribution` already exists and takes a `ppo_kl`-shaped
tensor, so feeding it `rollout_log_probs − train_log_probs` yields the wanted
quantity in about three lines next to `corrections.py:23`.

**A ninth, smaller one:** wasted generation is counted, not measured.
`stale_groups_recycled` / `aborted_groups_recycled` are group counts per drain
(`fully_async_rollout.py:246`); the tokens thrown away are not recorded, so
sample- and token-efficiency claims stay coarser than the wall-clock ones.

### What to take from the sglang side

Realized lag is produced by generation latency, so the engine side is where the
*mechanism* behind a lag distribution lives — and it is the weakest part of the
log.

**The job log is not a usable engine source.** sglang's `Decode batch,
#running-req: …, gen throughput (token/s): …, #queue-req: …` lines are per-engine
and frequently identical, so Ray collapses them (`[repeated 126x across
cluster]`). What survives is a sample of one engine at unpredictable times.

The reliable source is `--use-miles-dashboard`, whose scraper polls every
engine's `/metrics` every 2 s into `{dump}/dashboard/engine_series`
(`dashboard/sglang_scraper.py:42`): `sglang_num_running_reqs`,
`sglang_num_queue_reqs`, `sglang_gen_throughput`, `sglang_token_usage`,
`sglang_cache_hit_rate`, plus TTFT / inter-token-latency / e2e-latency
sum+count pairs and `sglang_num_aborted_requests_total`. The async recipes
already pass it. Widen with `--dashboard-sglang-metrics` if needed.

Two things it still will not give, worth deciding on up front:

* **Per-request latency and length distributions.** Only histogram `_sum` /
  `_count` pairs are scraped, so the engine side yields means, not tails. The
  per-sample `response_length` in the rollout dumps covers the output-length
  distribution properly; use that for the length axis and reserve the engine
  series for queueing and throughput.
* **Per-request start/finish timestamps**, which would let a lag be attributed to
  a specific slow generation. Nothing collects them today. If the study wants to
  claim "long-tail generations cause the stale groups" rather than observe it
  correlationally, that is a new instrumentation ask — the cheapest version is
  recording each sample's generation start and end in `Sample.metadata`, which
  the dumps then carry for free.

### Recommended deltas before the study starts

| where | change | unlocks |
|---|---|---|
| every recipe | `--observe-training-entropy` | the entropy collapse criterion (gap 4) |
| `loss_hub/corrections.py` | ESS of the TIS weights, beside `tis_abs` | gap 8, the long-sequence claim |
| every async recipe | `MAX_WEIGHT_STALENESS=1000000` for the unbounded arm | realized staleness on every arm (gap 2) |
| every recipe | raise `--n-samples-per-eval-prompt`, or add a larger held-out set | a `δ` small enough to be interesting (gap 1) |
| `common/run_identity.sh` | `SEED` in `CONFIG_TAG`, `--seed` / `--rollout-seed` in `train.sh` | seed replication at all |
| `miles/dashboard/args.py` | the study's factors in `_SNAPSHOT_KEYS` | self-describing runs (gap 5) |
| retention | keep `{dump}/rollout_data` and `{dump}/dashboard` | gaps 1 and 3, permanently |
| retention | keep **every** allocation's job log, not just the last | gap 6 |

Run `check_logging.py` on the first job of every arm. It exits non-zero on a
`FAIL`, so it can gate a sweep submission rather than being remembered.

## Design decisions a reviewer will ask about

* **The plateau window is chosen once, on the observed curve, then held fixed
  across bootstrap replicates.** Re-selecting it per replicate would make
  `Q_on*[rep]` a mean over a different set of steps each time, and the resulting
  interval would not be an interval for any single estimand.
* **`τ(δ)` gets no confidence interval; `τ(q_p)` does.** The non-inferiority time
  is defined *through* a confidence bound, so it is conservative by construction
  and a CI on it would be a CI on a CI. The target-ladder times are defined on
  `Q` itself, so re-running the crossing rule inside each replicate is a
  legitimate bootstrap of `τ`, and hence of `S`.
* **`k` consecutive evaluations, not one.** A single lucky evaluation crossing the
  bar is noise. Default 3.
* **Saturating-curve fitting is a sensitivity analysis, never the primary.**
  `equivalence.fit_saturating` implements the ScaleRL-style form; RL curves are
  non-monotone and a still-rising curve supports wildly different asymptotes at
  nearly identical fit quality, so the plateau definition stays primary.
* **`--monotone` is off by default.** Crossing on the running maximum of `Q` can
  only shorten every `τ`; reporting the shorter number without saying so would
  flatter every arm equally but not honestly.
* **Speedups report `paired_frac`.** A speedup computed on the 30% of replicates
  where both arms reached the target is a statement about that 30%. The figure
  draws those markers hollow.
