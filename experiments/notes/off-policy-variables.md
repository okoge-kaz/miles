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

The first two are the awkward ones. **Partial rollout as an explicit flag exists
only in the colocated path.** Under `--fully-async` the continuation happens
anyway — a weight sync aborts a group, `_recycle()` hands the prompt samples back
to the data source (`fully_async_rollout.py:198`), and because `generate_and_rm`
mutates samples in place those objects still carry the tokens produced so far, so
the next submission resumes from them. It is not switchable there.

So the partial-rollout arm is a **colocated-only** comparison, and the
"eliminate the train/rollout mismatch entirely" arm is likewise **colocated
only**. Neither can be crossed with the async staleness axis in one run.

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
| weight staleness | `MAX_WEIGHT_STALENESS` | 1, 2, 4, unset (= unbounded) | both |
| minibatch reuse | `NUM_STEPS_PER_ROLLOUT` | 1, 2, 4 | both |
| generation concurrency | `ASYNC_MAX_CONCURRENT_SAMPLES` | 1×, 2×, 4× `rollout_batch × n` | both |
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
| dynamic sampling + over-sampling | always on | never switched off in a real workload, so an "unconfounded" arm without it would not describe anything anyone runs. Accepted as part of the environment, not a variable |
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
