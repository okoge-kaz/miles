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
| `extract_offline_eval.py` | `src/offline_eval/` checkpoint evaluations → the same extract | numpy |
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

**The offline evaluation is the intended quality source.** The in-training eval is
one AIME year at `n=8` and exists to show a run is learning;
`src/offline_eval/run_eval.sbatch` evaluates saved checkpoints on four years at
`n=16`, and `extract_offline_eval.py` turns those into the same extract shape:

```bash
uv run --with numpy python -m experiments.src.offpolicy_acceleration.extract_offline_eval \
  --eval-root /data/offline_eval --match <config-tag> \
  --base-eval /data/offline_eval/base_<model> \
  --slurm-log experiments/outputs/training/.../<job>.log \
  --out $WS/offpolicy-study/extracts --arm on-policy --seed 0
```

`--base-eval` is the base model's evaluation, placed at step 0: it is `Q0`, which
anchors the whole `q_p` target ladder. Wall-clock comes from the training log, not
from the evaluation job — checkpoints are evaluated long after they were written.

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

### What was missing, and what was changed

The audit below drove a set of changes to miles and to the recipes. Each row says
what the measurement gap was and how it now stands.

| # | gap | status |
|---|---|---|
| 1 | per-prompt eval rewards exist only in the `--dump-details` eval dumps; `log_eval_rollout_data` logs the mean (`metrics.py:38`) | **open by design** — keep `--dump-details`, now far cheaper (row 9) |
| 2 | `avg_staleness` silently absent even with `--max-weight-staleness` set | **fixed** — root cause found: `_CachedWeightVersion.get` returns `None` when the router `/model_info` query fails, which also makes the cap enforce nothing. Now warns loudly, logs `current_weight_version`, and no longer crashes on a malformed payload |
| 3 | `P(L)` logged as mean/max only | **fixed** — `staleness_p50/p90/p99`, `frac_zero`, `frac_at_bound`, `num_groups` |
| 4 | `train/entropy_loss` identically 0 (`losses.py:99`) | **fixed** — `--observe-training-entropy` added to all 20 recipes |
| 5 | factors and seed absent from `meta.json` | **fixed** — study knobs added to `_SNAPSHOT_KEYS` (`dashboard/args.py`) |
| 6 | wall-clock discontinuous across Slurm allocations; `meta.json` overwritten on resume (`collector.py:153`) | **fixed in this tooling** — `log_source.active_elapsed_hours` never reads `start_ts` |
| 7 | wasted generation counted in groups, not tokens | **fixed** — `aborted_tokens`, `stale_tokens`, `dynamic_filter_tokens`, `kept_tokens`, `wasted_token_frac` |
| 8 | no ESS of the rollout-vs-train importance weights | **fixed** — `train/rollout_ess_ratio` |
| 9 | `policy_loss_debug/` is 76% of the dump and unconditional | **fixed** — `--no-dump-policy-loss-debug`, in all recipes |

**`train/ess_ratio` does not measure staleness**, and this is the one that most
easily misleads. `ppo_kl = old_log_probs − log_probs` with `old_log_probs =
batch["log_probs"]` (`losses.py:94,158`) — both Megatron logprobs, so it is the
*PPO inner-loop* ratio across the `NUM_STEPS_PER_ROLLOUT` minibatch updates. At
one step per rollout the two forwards use identical weights, so `ppo_kl ≡ 0`,
`pg_clipfrac ≡ 0` and `ess_ratio ≡ 1.0` **by construction, at any staleness** —
exactly the triple the audited log shows. Use `train/rollout_ess_ratio`.

**Seed replication remains unavailable and is a deliberate limitation.**
`common/run_identity.sh` builds `CONFIG_TAG` without a seed, so two seeds of one
configuration would share `CKPT_PATH`; `--seed` (1234) and `--rollout-seed` (42)
are the same constants in every run. The study is running single-seed, so the
bootstrap has no seed level and its intervals are **within-run** — they
understate total uncertainty and must be described that way. Note also that
fully-async runs are not deterministic at a fixed seed: group arrival order
follows generation timing, so a fixed seed hides run-to-run variance rather than
removing it.

### Measured cost of `--dump-details`

From a real 12-rollout-step run (`async-on-1step-rollout-length-24k-lr1e-6`):

| component | measured | per rollout step |
|---|---|---|
| `policy_loss_debug/` | 1.17 GB in **2512 files** | 100 MB, 209 files |
| `rollout_data/` train dumps | 287 MB / 12 | 24 MB |
| `dashboard_columns/` | 82 MB | 7 MB |
| `rollout_data/eval_0.pt` | 103 MB | 103 MB per eval |
| `dashboard/` | 29 MB | — |
| **total** | **1.7 GB / 12 steps** | **131 MB** |

Extrapolated to 300 steps and 15 evals that is ≈ 41 GB per run; with
`--no-dump-policy-loss-debug` it drops to ≈ 11 GB. **The cost is disk and inodes,
not time**: 131 MB/step against a measured `perf/train_time` of 216 s is well
under 1% even at conservative Lustre bandwidth. That last figure is an estimate
from write volume, not a measurement — A/B `perf/actor_train_time` to confirm.

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
