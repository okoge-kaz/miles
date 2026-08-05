# Off-policy study: the variable space

What can move **throughput** and **time-to-a-given-downstream-score** when the
off-policy degree is varied. This is the catalogue of variables, not the
measurement design — the definition of "time to reach on-policy performance" and
the analysis scripts are a separate exercise.

Primary axis: **`MAX_WEIGHT_STALENESS`** — how many weight versions a group may
lag the engine before it is recycled instead of trained on
(`fully_async_rollout.py:202-213`). miles' own default is `None`, meaning no
bound at all.

`NUM_STEPS_PER_ROLLOUT` is a *second, different* off-policy quantity: minibatch
reuse inside one rollout batch, where the lag is deterministic (`0..N-1`
gradient steps) rather than a distribution over generation latency. Both are
swept; they are not the same axis.

## What miles actually implements

Checked against the source, because several combinations are rejected at startup.

### Hard constraints — these decide the shape of the study

| constraint | source |
|---|---|
| `--fully-async` **rejects** `--partial-rollout` | `arguments.py:54` |
| `--fully-async` **rejects** `--recompute-logprobs-via-prefill` | `arguments.py:57` |
| `--recompute-logprobs-via-prefill` **requires** `--true-on-policy-mode` | `arguments.py:2606` |
| `--fully-async` **rejects** `--colocate` | `arguments.py:53` |
| `--use-miles-dashboard` **requires** `--dump-details` | `dashboard/args.py:69` |

### Two different interruptions, two different flags

`--partial-rollout` and `--pause-generation-mode` are the same idea at two
different boundaries, and they belong to two mutually exclusive execution modes.

**`--partial-rollout` acts at the rollout-loop boundary, not at a weight
update.** Once the colocated rollout loop has collected `rollout_batch_size`
groups that passed the dynamic-sampling filter, it calls `abort()`
(`inference_rollout_train.py:143`), which posts `/abort_request {abort_all:
true}` to every engine and kills *all* remaining in-flight generation. What
happens to those unfinished groups is the flag:

* off — `continue`, the group is dropped on the floor
  (`inference_rollout_train.py:39`)
* on — `start_rollout_id` is stamped on every sample that has a response and the
  group goes back into the data buffer (`:44-46`). Next rollout it is
  resubmitted, and `generate_and_rm` continues from the tokens it already has,
  because `sample.tokens` still contains the partial response

Its trigger is *the rollout batch being satisfied*. It is cross-rollout-step
carryover, and the KV cache is always gone by then (`abort_all`, plus
`flush_cache` at the weight update), so a resumed sample always re-prefills.
There is no `in_place` analogue in the colocated path.

**`--pause-generation-mode` acts at the weight update, inside SGLang**, and is
what decides the fate of in-flight generation there:

| mode | in-flight requests | KV cache | tokens already generated |
|---|---|---|---|
| `retract` (**default**) | returned to SGLang's waiting queue | flushed, recomputed by prefill under the new weights | kept; the sample spans weight versions |
| `in_place` | frozen, then resumed | **kept**, so the continuation attends to KV built by the *old* weights | kept |
| `abort` | terminated | flushed | **discarded** — `_drain` recycles the group and `reset_for_retry()` clears `tokens`, `response` and `rollout_log_probs` (`types.py:236`), so it is regenerated from scratch |

`_pause_and_prepare_engines` calls `flush_cache` for every mode except
`in_place` (`update_weight/.../mixin.py:308-316`).

Why the two never coexist:

| | colocated (`math_sync`) | fully-async (`math_async`) |
|---|---|---|
| end of a rollout batch | `abort()` kills everything in flight; `--partial-rollout` decides carry-over vs discard | no such boundary — the worker generates continuously and nothing calls `abort()` |
| weight update | nothing is generating by then, so the pause mode is moot | the *only* interruption; `--pause-generation-mode` decides |
| flag available | `--partial-rollout` | `--pause-generation-mode` (`--partial-rollout` is rejected, `arguments.py:54`) |

**Pipeline-RL-style continuation across a weight update is
`--pause-generation-mode in_place`**: the request is frozen and resumed on the
KV cache built by the previous weights, with no re-prefill. `retract` is the
same continuation but pays a full KV recompute so the cache matches the new
weights; `abort` throws the tokens away and regenerates. Those three are the
arm, and they are **async-only** — the colocated path cannot express `in_place`
at all, since its cache is flushed before any resumption.

The "eliminate the train/rollout mismatch entirely" arm remains colocated only.

### Advantage estimator / algorithm

`--advantage-estimator` accepts exactly: `grpo`, `gspo`, `reinforce_plus_plus`,
`reinforce_plus_plus_baseline`, `ppo`.

**CISPO and CTPO are not implemented in miles.** Including them means writing
them; `gspo` (sequence-level importance ratio) is the only alternative to `grpo`
in the same family that is available today.

### The clip family

| flag | default | meaning |
|---|---|---|
| `--eps-clip` | 0.2 | PPO lower clip |
| `--eps-clip-high` | = `--eps-clip` | upper clip. DAPO's clip-higher is 0.28 |
| `--eps-clip-c` | None | Dual-clip PPO lower bound, [arXiv:1912.09729](https://arxiv.org/abs/1912.09729). Off by default |

Clipping *is* an off-policy correction, which is why the algorithm axis and the
clip axis are one axis and not two: GRPO+clip-higher (DAPO) and a sequence-level
ratio (GSPO) disagree about what to do with a drifted ratio, and comparing them
at a fixed `--eps-clip-high` compares neither faithfully.

### Importance-sampling correction (MIS)

`--use-tis` is the on/off in the recipe, but the real surface is the YAML passed
through `--custom-config-path` together with
`--custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp`
(the pattern in `scripts/run_qwen3_30b_a3b.py:--enable-mis`):

| key | values | note |
|---|---|---|
| `use_tis` | true / false | truncated importance sampling |
| `tis_level` | `token` / `sequence` | where the ratio is aggregated |
| `tis_mode` | `truncate` / `clip` / `mask` | **behaviourally different**, not a tuning detail: truncate caps the weight, clip bounds it two-sided, mask drops the token entirely (`mis.py:227-238`) |
| `tis_upper_bound` | e.g. 2.0 | the tolerance to lag |
| `tis_lower_bound` | e.g. 0.5, default `1/upper` | `mis.py:175` |
| `tis_batch_normalize` | true / false | `mis.py:274` |
| `use_rs` | true / false | rejection sampling on top |
| `rs_level` | `token` / `sequence` | |
| `rs_veto_threshold` | e.g. 1e-4 | drops a whole sample on one catastrophic token |

The bounds are the direct knob on lag tolerance: a wider `tis_upper_bound`
accepts more drift into the update.

### Train/rollout mismatch

The importance ratio is not 1 even at zero staleness, because Megatron and
SGLang are different implementations. Measured drift therefore contains a
constant numerical term plus the policy-lag term.

| level | how | note |
|---|---|---|
| as-is | default | the floor is whatever the two engines disagree by |
| measured | read `dump/mean_abs_lp_diff` on an on-policy run | the number to subtract |
| eliminated | `--true-on-policy-mode` | parity contract; for Qwen3 dense: `qwen3_dense_true_on_policy_v1` — deterministic inference and training, SGLang attention `fa3`, Megatron sequence-parallel disabled (`true_on_policy/schema.py:26`) |
| eliminated, prefill path | `+ --recompute-logprobs-via-prefill` | for models whose prefill and decode kernels are not numerically identical. Colocated only |

**MoE adds a second mismatch source**: which experts ran. R3
(`--use-rollout-routing-replay`) replays the rollout's routing in the training
forward pass and is **on unconditionally** for `qwen3-30b-a3b`, so it is a fixed
part of the configuration rather than a variable.

## The variables

### Swept

| variable | flag / env | values | affects |
|---|---|---|---|
| weight staleness | `MAX_WEIGHT_STALENESS` | 1, 2, 4, 1000000 (= effectively unbounded, see below) | both |
| minibatch reuse | `NUM_STEPS_PER_ROLLOUT` | 1, 2, 4 | both |
| generation concurrency | `ASYNC_MAX_CONCURRENT_SAMPLES` | 1×, 2×, 4× `rollout_batch × n` | both |
| in-flight fate at weight update | `PAUSE_GENERATION_MODE` | `retract`, `in_place`, `abort` | both |
| learning rate | `LR` | 5e-7, 1e-6, 2e-6 | quality |
| rollout length | `MAX_RESPONSE_LEN` | 8192, 16384, 24576 | both |
| algorithm + clip | `--advantage-estimator`, `--eps-clip-high`, `--eps-clip-c` | grpo/gspo × clip setting | quality |
| IS correction | the MIS YAML above | off / truncate / clip / mask × bounds | quality |
| mismatch level | `--true-on-policy-mode` (± prefill) | as-is / eliminated | quality |
| partial rollout (colocated only) | `--partial-rollout` | off / on / on + logprob recompute | both |
| model | recipe directory | 1.7B … 30B-A3B | both |

`LR` × staleness is deliberately a full grid rather than a nested search: if the
optimum LR moves with the off-policy degree, that interaction *is* a result.

Rollout length is on the list for the reason it is easy to cheat with: a shorter
budget mechanically reduces the number of weight syncs a single sample spans, so
off-policy looks better than it is. Compare at equal length, and record
`dump/truncated_frac` so a length that biases the task is visible.

### Held fixed

| variable | value | why |
|---|---|---|
| `GLOBAL_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `ROLLOUT_BATCH_SIZE` | prior-work values | see the dataset README; not a research question here |
| dynamic sampling | on, **without over-sampling** | the filter stays, but `--over-sampling-batch-size` is not passed, so it defaults to `rollout_batch_size` and the rollout loop submits exactly what it needs. Over-sampling would otherwise add a second, uncontrolled source of aborted generation on top of the weight-update one this study is measuring |
| R3 (MoE) | always on | removes routing mismatch; MoE RL is known to collapse without it |
| verifier (`RM_TYPE`) | per checkpoint | correctness, not a knob |
| temperature, KL coefficient | 1.0, 0 | matches DAPO |
| parallelism, engine geometry, `MAX_TOKENS_PER_GPU`, `SGLANG_MEM_FRACTION` | tuned once per model, then frozen | throughput-only; moving them invalidates the wall-clock axis |

### Recorded, not swept

The x-axis is wall-clock, since the question is GPU efficiency. Optimizer steps
are the second axis. **Sample and token consumption have to be recoverable after
the fact**, or sample-efficiency claims cannot be made later without rerunning:

| quantity | where it comes from |
|---|---|
| wall-clock per phase | dashboard `phases` stream (`rollout`, `actor_train`, `update_weights`, `train_wait`) |
| optimizer steps | `rollout_id × num_steps_per_rollout` |
| samples consumed | `rollout_batch × n_samples` per rollout, plus what dynamic sampling discarded — the drop counters in `MetricGatherer` |
| tokens generated | `dump/response_length_mean × samples`, and `perf/*` from `ray/rollout/metrics.py` |
| realised staleness | `dump/mixed_version_frac`, `rollout/fully_async/avg_staleness`, `max_staleness` |
| realised drift | `dump/mean_abs_lp_diff`, per-sample `mean_imp_ratio` |
| wasted generation | `rollout/fully_async/aborted_groups_recycled`, `stale_groups_recycled` |

The staleness bound is a *cap*, not the realised value — always report
`avg_staleness` next to the setting, or a plateau in the results will be
misread as insensitivity when it was actually the bound never binding.

**The realised-staleness metrics are gated on the bound being set.**
`fully_async_rollout.py:202` only measures staleness when
`args.max_weight_staleness is not None`, so the "unbounded" arm would be the one
run with no staleness measurement at all. Run that arm with a bound so large it
never binds (`MAX_WEIGHT_STALENESS=1000000`) rather than unsetting it.

## Future work, deliberately out of scope

- **Rollout quantization** — fp8 / mxfp8 / nvfp4 / int4 are exposed
  (`scripts/run_qwen3_30b_a3b.py`) and change the mismatch floor directly, which
  makes them a natural extension of the mismatch axis. Too aggressive to include
  in the first study.
- **CISPO / CTPO** — not implemented in miles; would need to be written before
  they can be compared.
- **Speculative decoding**, **PD disaggregation** — both change rollout
  throughput and the numerics, on separate axes from staleness.
- **Multi-turn / agentic rollouts** — turn count is a third source of lag;
  `generate_hub/multi_turn.py:32` rejects partial rollout outright.
