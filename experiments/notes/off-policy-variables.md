# Off-policy study: the variable space

What can move **throughput** and **time-to-a-given-downstream-score** when the
off-policy degree is varied. This is the catalogue of variables, not the
measurement design — the definition of "time to reach on-policy performance" and
the analysis scripts are a separate exercise.

> **2026-08-13 compatibility note:** the measurements below describe the original
> completion-FIFO implementation, now named `queue-recycle`. The current code also
> supports scheduler-authoritative `--staleness-reference prefill`, `queue-max`,
> and bounded `queue-drop`. See the current "Staleness reference and queue policy"
> section in `notes/telemetry.md`; do not apply the historical completion-only or
> blocking-queue conclusions below to those two new policies.

Historical primary axis: **`MAX_WEIGHT_STALENESS`** — how many weight versions a
group may lag the engine before it is recycled instead of trained on
(`fully_async_rollout.py:407-419`). miles' own default is `None`, meaning no
bound at all. The lag it bounds is measured from the version a group *finished*
generating under, not the one it started under; see "The bound measures queue
residency" below.

`NUM_STEPS_PER_ROLLOUT` is a *second, different* off-policy quantity: minibatch
reuse inside one rollout batch, where the lag is deterministic (`0..N-1`
gradient steps) rather than a distribution over generation latency. Both are
swept; they are not the same axis.

## The realized lag has a hard ceiling, and a hard-coded constant sets it (2026-08-07)

Over-provisioning rollout does not send the lag off to infinity. Production is
bounded on two sides in `fully_async_rollout.py`:

```python
OUTPUT_QUEUE_MAX_GROUPS = 1000                                  # :34
self._output = asyncio.Queue(maxsize=OUTPUT_QUEUE_MAX_GROUPS)   # :267

def _max_in_flight_groups(self):                                # :290
    if (x := self.args.async_max_concurrent_samples) is not None:
        return max(1, x // self.args.n_samples_per_prompt)
    return self.args.rollout_batch_size

async def _worker_loop(self):                                   # :320
    ...
    await self._output.put(task.result())                       # :329  blocks when full
```

`await put` is backpressure: with the queue full the worker stops submitting, so
the engines stall rather than the queue growing. Run-ahead is therefore at most
`OUTPUT_QUEUE_MAX_GROUPS + rollout_batch_size` groups, and the trainer drains
`rollout_batch_size` per step, so

    ceiling(L) ~ OUTPUT_QUEUE_MAX_GROUPS / rollout_batch_size + O(1)
               = 1000 / 192 + 1 ~ 6

Measured, T=1, gbs 3072, n 16, rbs 192, 12 rollouts, uncapped (`MAX_WEIGHT_STALENESS=64`):

| R | queue_size trajectory | mean L | max L | P(L>2) |
|---|---|---|---|---|
| 1 | 0 throughout | 0.45–0.50 | 3 | 0.001–0.005 |
| 2 | 0 throughout | 0.63–0.64 | 3 | 0.001–0.002 |
| 3 | 0–2 (one excursion to 14) | 0.86–1.41 | 4 | 0.013–0.109 |
| 5 | 10 → 827, still rising | 2.72–2.80 | 6 | 0.60–0.64 |
| 7 | 3 → **1000** (saturated) → 980 | 3.19 | 6 | 0.68 |

R=7 reaches the queue bound at step 10 and `max L` stops at 6, as predicted.
R=1/2/3 keep an empty queue, so their lag comes from generation latency alone and
settles immediately.

Two consequences.

**The "natural" staleness of an over-provisioned run is a property of the
framework, not of the workload.** At `OUTPUT_QUEUE_MAX_GROUPS = 100` the ceiling
would be ~0.5; at 10000, ~52. The constant has to be reported alongside any
realized-lag histogram, and a staleness level above the ceiling is bit-for-bit
the uncapped run.

**Raising the queue is a way to induce lag, and it is not free.** It costs no GPU
time — it converts engine stall into stale samples — but the queue holds whole
`Sample` objects in the rollout manager's CPU memory: `tokens` (`list[int]`),
`rollout_log_probs` (`list[float]`), `response`, `loss_mask`. At the measured
6.4k mean response that is order 8 MB per group of 16, so ~8 GB at 1000 groups,
scaling linearly with the queue and with response length as training lengthens
responses. The larger cost is methodological: enlarging the buffer to reach a
target lag *is* imposing staleness artificially, which is the practice this study
exists to distinguish itself from. `ASYNC_MAX_CONCURRENT_SAMPLES` bounds the
other side and is already a swept variable; `OUTPUT_QUEUE_MAX_GROUPS` is not an
argument and would need a code change.

## The bound measures queue residency, not the whole off-policy distance (2026-08-09)

`staleness = current - oldest` (`fully_async_rollout.py:409`) is a difference
against the version a group *finished* generating under. It is not the version
generation started under, and under this study's pause mode those are not the
same number.

`Sample.weight_versions` gets one entry per generate **call**
(`types.py:285-286`, appended from `sglang_rollout.py:301` and
`generate_endpoint_utils.py:112`), and single-turn generation is exactly one HTTP
call (`generate_hub/single_turn.py:44`). The value of that entry is whatever
SGLang reads at the moment it builds the reply:

```python
# sglang 0.5.17.dev32+g3fe50ed (the build in the image), tokenizer_manager.py:1977-1984
# in the batch-output handler -- i.e. built when the reply is assembled.
meta_info = {
    "id": rid,
    "finish_reason": recv_obj.finished_reasons[i],
    "prompt_tokens": recv_obj.prompt_tokens[i],
    "weight_version": self.server_args.weight_version,   # server-level current value
    "num_retractions": recv_obj.retraction_counts[i],
}
```

`Req` carries no weight version of its own (nothing matches `weight_version` in
`schedule_batch.py` or `scheduler.py`), so there is no arrival-time snapshot to
report. `/model_info` reads the *same* `server_args.weight_version`
(`http_server.py:716`), which is what miles polls for `current` -- so the two ends
of the subtraction are one variable read at two times, and the difference is
exactly the interval between reply and drain.

**`PAUSE_GENERATION_MODE=in_place` is the recipe default** (`run.sbatch:57`, and
every sweep passes it explicitly), and it is the mode that hides the most.
`pause_generation` returns immediately without touching scheduler state
(`sglang/srt/managers/scheduler.py:4465-4475`), so the request is frozen and
resumed on the KV cache the *old* weights built. One reply, and
`retraction_count` is only incremented by `Req.reset_for_retract`
(`schedule_batch.py:1603-1606`), which that path never reaches — so
`num_retractions` is 0 too. A sample that spanned v3→v5 is recorded as v5, with
nothing anywhere in the reply saying otherwise.

Consequences:

- `--max-weight-staleness` bounds **only** the queue residency. Weight updates
  crossed mid-generation are counted as zero and the bound never sees them.
- `weight_version/mixed_version_ratio` and `dump/mixed_version_frac` are
  `len(set(weight_versions)) > 1` (`ray/rollout/metrics.py:101`,
  `dashboard/dump_reader.py:456`). With one call per sample the list has length 1,
  so **both are structurally 0 for every run in this study**. They measure
  multi-turn and partial-rollout resume, neither of which fully-async does
  (`arguments.py:54` rejects partial rollout).
- `in_place` also means the continuation attends to a KV cache built by the
  previous weights — the mismatch VCPO and PipelineRL are about — and that is
  filed under staleness 0.

Since 2026-08-09 the missing interval is measured, and the logged staleness is a
decomposition rather than one number. `_generate_group` reads the current version
before generating and stamps it on the group (`fully_async_rollout.py:64-90`, the
same idea as multi-LoRA's `metadata["slot_version"]`,
`multi_lora/async_rollout.py:141-146`). With **S** as the selected reference,
**Q** at group-ready, **D** at dequeue, and **T** at training:

| key | quantity |
|---|---|
| `staleness/pre_queue/*` | `Q - S` — updates crossed before generation/reward/finalization made the group ready |
| `staleness/in_queue/*` | `T - Q` — updates crossed from ready to training |
| `staleness/total/*` | `T - S` = `pre_queue + in_queue` |
| `staleness/rollout/*` | `T* - S` — would-be train staleness before admission |

The names are Applied Compute's PQS/IQS
([staleness in fully-async RL](https://www.appliedcompute.com/research/staleness-in-fully-async-rl)).
Q is stamped only after the complete group has finished generation, reward, and
finalization. It can exceed the group's last-forward version if a weight update
lands during post-forward work. See `notes/telemetry.md` for the exact boundaries.
The accepted-group decomposition is emitted whether or not a
bound is configured, but a group rejected by the bound or dynamic filter does
not enter it. The metrics are absent when their provenance is unavailable; a
zero there would incorrectly mean that no update was crossed.

For `queue-recycle`, `--max-weight-staleness` admits `D-S < max`; with one
scheduled update before training this is equivalent to `T-S <= max`.
The bound must be at least 1 because the nonnegative dequeue gap cannot satisfy
the strict rule at 0, not because of startup. Startup has `T=D`; normal
prefetched batches have `T=D+1`.
`--staleness-reference` selects `S`. The flag is in the checkpoint path
(`max-weight-staleness-<s>-from-submission`) because it changes which groups are
recycled. A tight bound under `submission` collapses throughput to the
synchronous condition rather than hanging -- see `notes/telemetry.md`.

**`total` is the train-time number a staleness claim is about, and `pre_queue` is the one to
read first.** If `pre_queue` is materially non-zero the arms are separated by less
off-policy distance than their `MAX_WEIGHT_STALENESS` labels claim, because the
bound never sees it.

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

| | colocated (`math/sync`) | fully-async (`math/async`) |
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
| what the bound measures from | `STALENESS_REFERENCE` | `prefill` (current study), `completion`, `submission` | both |
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
| dynamic sampling | **off**, and no over-sampling | `--dynamic-sampling-filter-path` is not passed, and `--over-sampling-batch-size` is left to default to `rollout_batch_size`. The prompt set is already filtered offline to a 10–80% pass-rate window (`dapo-math-p10-90`, see `tools/difficulty_filter/`), which is where the online filter's value went. Keeping it off also removes the only source of discarded generation from the colocated reference arm: the rollout loop tops up by a whole `over_sampling_batch_size` whenever the filter rejects a group (`inference_rollout_train.py:101-104`), and the surplus is what `abort()` throws away at the batch boundary. With the filter off, `pendings` drains to zero, `abort()` has nothing to discard, and the reference arm's wall-clock contains no wasted generation for the off-policy arms to be compared against |
| R3 (MoE) | always on | removes routing mismatch; MoE RL is known to collapse without it |
| verifier (`RM_TYPE`) | per checkpoint | correctness, not a knob |
| temperature, KL coefficient | 1.0, 0 | matches DAPO |
| parallelism, engine geometry, `MAX_TOKENS_PER_GPU`, `SGLANG_MEM_FRACTION` | tuned once per model, then frozen | throughput-only; moving them invalidates the wall-clock axis |

### Seeds

Run-to-run variance has not been measured, so no difference in this study can yet
be called real. The **on-policy (colocated) runs carry the seed replicates**:
several seeds at one configuration, from which the spread of the metric being
compared is estimated once and applied to the whole sweep. Until that number
exists, treat every ranking here as provisional — the throughput measurements in
this session were five steps per configuration and already showed differences of
that order between nominally identical shapes.

`--seed` and `--rollout-seed` are the two knobs.

### Recorded, not swept

The x-axis is wall-clock, since the question is GPU efficiency. Optimizer steps
are the second axis. **Sample and token consumption have to be recoverable after
the fact**, or sample-efficiency claims cannot be made later without rerunning:

| quantity | where it comes from |
|---|---|
| wall-clock per phase | dashboard `phases` stream (`rollout`, `actor_train`, `update_weights`, `train_wait`) |
| optimizer steps | `rollout_id × num_steps_per_rollout` |
| samples consumed | `rollout_batch × n_samples` per rollout. With dynamic sampling off there is nothing else to add on the colocated side; on the fully-async side, generation is still discarded by the staleness bound and by weight-update aborts, counted in tokens by `rollout/fully_async/{stale,aborted,dynamic_filter}_tokens` and `wasted_token_frac` |
| tokens generated | `dump/response_length_mean × samples`, and `perf/*` from `ray/rollout/metrics.py` |
| realised staleness | `staleness/total/*` (the accepted train-time distance), split into `staleness/pre_queue/*` and `staleness/in_queue/*`; `staleness/rollout/*` is the would-be train-time distribution before admission, and `staleness/bound_exceeded_*` directly counts max-staleness rejection. **Not** `dump/mixed_version_frac` — structurally 0 here, see "The bound measures queue residency" |
| realised drift | `dump/mean_abs_lp_diff`, per-sample `mean_imp_ratio` |
| wasted generation | `rollout/fully_async/aborted_groups_recycled`, `stale_groups_recycled` |

The staleness bound is a *cap*, not the realised value — always report
`staleness/total/mean` and the direct `staleness/bound_exceeded_sample_frac`
next to the setting, or a plateau can be misread as insensitivity when the bound
simply never binds. `staleness/rollout/*` is emitted even when the bound is
unset; the direct rejection counters are then zero.

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

## Run identity: what makes a checkpoint path unique (2026-08-05)

`experiments/common/run_identity.sh` builds the only thing that separates two
runs on disk. `--load` and `--save` are the same directory, so **two runs that
produce the same `CKPT_PATH` do not collide loudly -- the second one resumes the
first.** Any swept knob missing from the path is a silent data corruption.

```
/ckpt/training/math/<DATASET_TAG>/<MODEL_NAME>/<RL_ALGORITHM>/<PLACEMENT>/
    <POLICY_REGIME>/max-weight-staleness-<S>[-from-submission]/<CONFIG_TAG>

CONFIG_TAG = rollout-length-<N>k-lr<LR>-rbs<RBS>-gbs<GBS>
             -tseed<TRAIN_SEED>-rseed<ROLLOUT_SEED>
```

The `-from-submission` suffix appears only when `STALENESS_REFERENCE` is not
`completion`, so paths written before the option existed keep their spelling.
It has to be in the path: the reference decides which groups are recycled, so two
runs differing only there train on different data.

Every axis of the main grid appears exactly once. Three are absent on purpose:

* `N_SAMPLES_PER_PROMPT` is fixed at 8 for this study but is written into the tag
  anyway, because it is what makes `NUM_STEPS_PER_ROLLOUT` derivable (below).
  With `n` present, `rbs`, `gbs` and `n` pin the whole batch shape.
* `PAUSE_GENERATION_MODE` is not in the grid -- like the deterministic-kernel
  check it is verified in a separate, targeted experiment. **That experiment must
  pass `CONFIG_TAG` explicitly**, because its three modes are otherwise identical
  configurations and would share one directory.
* `NUM_STEPS_PER_ROLLOUT` is not independent. miles asserts
  `rollout_batch * n_samples == global_batch * num_steps` at startup, so
  `rbs`, `gbs` and `n` -- all three in the tag -- determine it exactly. Writing it
  as well would let a typo produce two names for one configuration. It still does
  work: it is one of the two inputs to `POLICY_REGIME`. It just is not
  identifying information.

### `RL_ALGORITHM` is derived, not declared

```
grpo-clip0.2-0.28-tis2.0        # the current recipe
grpo-clip0.2-0.28-notis         # USE_TIS=0
grpo-clip0.2-0.28-tis0.5-2.0    # TIS_CLIP_LOW set
```

It is computed from `ADVANTAGE_ESTIMATOR`, `EPS_CLIP`, `EPS_CLIP_HIGH`,
`USE_TIS`, `TIS_CLIP`, `TIS_CLIP_LOW` and `KL_LOSS_COEF` -- the same variables
`train.sh` passes to miles -- so the name cannot drift from the run. A
hand-written label like `dapo` could, and the estimator alone cannot separate
DAPO from plain GRPO (DAPO *is* `grpo` plus clip-higher).

TIS belongs in the name rather than in `CONFIG_TAG` because it is not a nuisance
parameter: it is the off-policy correction itself, so whether an arm survives at
a given staleness is largely its doing. Leaving `--tis-clip` on argparse's
default would have put an unrecorded knob directly on the axis being measured.

**Correction (2026-08-05).** An earlier revision of this section claimed that
`--tis-clip` interacts strongly with the response-length axis, on the grounds
that the ESS of the importance weights decays exponentially in sequence length.
**That argument does not apply to this implementation.** `--use-tis` dispatches
to `vanilla_tis_function` (`loss_hub/corrections.py:7`), which is *token-level*:

```python
tis = torch.exp(old_log_probs - rollout_log_probs)      # elementwise, per token
tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
pg_loss = pg_loss * tis_weights
```

There is no product over the sequence, so `Var(log w) ∝ T` -- the sequence-level
statement -- is simply not what is being clipped. A per-token ratio's
distribution is set by how far training has drifted from rollout, not by how many
tokens follow it. To first order `tis_clipfrac` should be flat in response
length.

Two weaker length effects remain, and their net sign is not obvious a priori:
drift can compound *along* a sequence because a late token is conditioned on a
prefix that is itself off-distribution, which pushes clipfrac up with position;
while `sum_of_sample_mean` averages the per-token weights over more tokens in a
long sample, which pulls sample-level variance down.

So this is an empirical question, not a design constraint, and the instrument
already exists: `train/tis_clipfrac`, `train/tis` and `train/tis_abs` are logged
per step. **Read clipfrac across the response-length arms before deciding
anything.** If it is flat, `TIS_CLIP` stays fixed at 2.0 and there is no
interaction to design around. Crossing `TIS_CLIP` with response length ahead of
that measurement would multiply the grid on a mechanism that has not been shown
to exist.

### The off-policy correction surface miles actually has

Verified against the source, because the checkpoint name only distinguishes what
it encodes. Four families, and they are orthogonal to each other:

**1. Importance-sampling weights on the train/rollout ratio.**

| entry point | level | what it does |
|---|---|---|
| `--use-tis` + `--tis-clip` / `--tis-clip-low` | **token** | `clamp(exp(train-rollout), lo, hi)` multiplied into `pg_loss` (`corrections.py:7`) |
| `icepop_function` (`corrections.py:35`) | token | clip-or-*pop*: zeroes tokens outside the band and passes the in-range ratio through **unweighted** |
| `--custom-tis-function-path .../mis.py:compute_mis_weights_with_cp` | token **or sequence** | the full MIS surface, configured by YAML rather than by flags |

MIS's YAML is where the real variety lives: `tis_level` token/sequence,
`tis_mode` truncate/clip/mask (not a tuning detail -- mask *drops* the token),
`tis_upper_bound` / `tis_lower_bound`, `tis_batch_normalize`, plus rejection
sampling `use_rs` / `rs_level` / `rs_veto_threshold` (one catastrophic token can
veto a whole sample).

**2. Sequence masking.** `--use-opsm` / `--opsm-delta` (default 1e-4) --
Off-Policy Sequence Masking, `math_utils.py:183`. Drops an entire sequence whose
log-prob deviation exceeds the threshold, rather than reweighting it.

**3. What the ratio denominator even is.** `arguments.py:3204` requires exactly
one of these, and they are three different corrections, not three spellings:

* default / `--use-tis` -- denominator recomputed by the current actor
* `--use-rollout-logprobs` -- the engine's own log probs are the denominator
* `--keep-old-actor` -- the rollout-time weights are kept and used to recompute it

**4. Clipping.** `--eps-clip` / `--eps-clip-high` (DAPO clip-higher), and
`--eps-clip-c` for Dual-clip PPO ([arXiv:1912.09729](https://arxiv.org/pdf/1912.09729)),
off by default. `--advantage-estimator gspo` moves the *policy* ratio to sequence
level, which is a different thing from a sequence-level *mismatch* weight.

M2PO, CISPO and VCPO are **not** in miles.

### How `RL_ALGORITHM` encodes all of it

Every correction above is a separate directory, and each part is **omitted at its
default**, so the common case stays short and any deviation is visible:

```
<estimator>-clip<lo>-<hi>[-dualclip<c>][-<denominator>]-<is-correction>[-opsm<delta>][-kl<coef>]
```

| configuration | name |
|---|---|
| the current recipe | `grpo-clip0.2-0.28-tis2.0` |
| no IS correction | `grpo-clip0.2-0.28-nois` |
| IcePop | `grpo-clip0.2-0.28-icepop2.0` |
| MIS, sequence level | `grpo-clip0.2-0.28-mis-seq-truncate-2.0` |
| MIS, token level + mask | `grpo-clip0.2-0.28-mis-token-mask-2.0` |
| two-sided TIS bounds | `grpo-clip0.2-0.28-tis0.5-2.0` |
| Dual-clip PPO | `grpo-clip0.2-0.28-dualclip3.0-tis2.0` |
| OPSM | `grpo-clip0.2-0.28-tis2.0-opsm1e-4` |
| engine log probs as denominator | `grpo-clip0.2-0.28-rolloutlp-nois` |
| rollout-time weights as denominator | `grpo-clip0.2-0.28-oldactor-nois` |
| GSPO | `gspo-clip0.2-0.28-tis2.0` |

Driven by `IS_CORRECTION` (`none｜tis｜icepop｜mis`), `RATIO_DENOMINATOR`
(`actor｜rollout-logprobs｜old-actor`), `EPS_CLIP_C`, `USE_OPSM`/`OPSM_DELTA`,
`TIS_CLIP`/`TIS_CLIP_LOW` and `MIS_PROFILE` -- the same variables `train.sh`
turns into flags, so the name still cannot drift from the run.

**MIS is named by profile, not by parameters.** Its knobs arrive through
`--custom-config-path` as a YAML that `arguments.py:3143` merges into `args`, so
`run_identity.sh` cannot see them. `MIS_PROFILE` names a file under
`experiments/configs/mis/<profile>.yaml`; the file is in git, so the profile name
is a stable identifier for its whole contents. Name profiles after what they do
(`seq-truncate-2.0`, `token-mask-2.0`), because that string is the only record in
the path.

Two impossible combinations are rejected rather than silently mis-named:
`RATIO_DENOMINATOR=rollout-logprobs` with any IS correction (`arguments.py:2851`
rejects it outright), and `IS_CORRECTION=mis` without a `MIS_PROFILE`.

### `POLICY_REGIME`

`on-policy` iff `MAX_WEIGHT_STALENESS == 0` **and** `NUM_STEPS_PER_ROLLOUT == 1`.
Those are the two ways a sample goes off-policy: generated under older weights,
or reused across more than one optimizer step. The test is placement-independent
-- an async `queue-max` run pinned to staleness 0 with one step per rollout is
genuinely on-policy, and a colocated run at 4 steps per rollout genuinely is not.
`queue-recycle` cannot use the zero endpoint because its strict dequeue rule
would admit no group; its minimum bound is 1.

### `PLACEMENT`

`colocated` or `async`. `math/sync` passes `--colocate` and never passes
`--max-weight-staleness` or `--pause-generation-mode`, so `run_identity.sh`
forces those to `0` and `none` and *errors* if the caller set them -- otherwise
a colocated point in a staleness sweep would land in a directory claiming a
staleness the run never had.

### Seeds

`TRAIN_SEED` (`--seed`) and `ROLLOUT_SEED` (`--rollout-seed`) are separate and
both are in the tag. The variance decomposition needs them moved independently:
`ROLLOUT_SEED` drives prompt shuffling and sampling, `TRAIN_SEED` drives
initialisation and data order inside the optimizer step. Tying them to one value
would confound rollout-sampling variance with training-seed variance and leave no
way to attribute a spread in `Q(t)` to either.

### `sweep.py`

Unaffected: it overrides `CONFIG_TAG` with `sweep-<name>-<tag_for(point)>`, which
encodes exactly the knobs that point varies. The directory levels above
`CONFIG_TAG` are still derived per point, so a sweep over staleness or algorithm
still fans out across directories.

## The acceleration is conditional on the node ratio (2026-08-07)

Recycling cost shows up in wall-clock only when rollout capacity is scarce.
Measured at 1 train + 3 rollout, jobs 15288337 (bound 1) and 15288347 (bound 2):

| step | s1 `wasted_token_frac` | s1 `train_wait` | s2 `wasted_token_frac` | s2 `train_wait` |
|---|---|---|---|---|
| 3 | 0.2556 | 177.5 s | 0.0000 | 79.9 s |
| 4 | 0.1162 | 69.1 s | 0.0147 | 13.0 s |
| 5 | 0.2301 | -- | 0.0081 | -- |

The bound-1 arm discards roughly a quarter of the tokens it generates and stays
rollout-bound; the bound-2 arm returns to train-bound. That difference *is* the
mechanism by which a looser bound accelerates training -- but its magnitude is
set by R. With rollout capacity to spare the recycling is absorbed and costs no
wall-clock at all; with R tight it is fully exposed.

So "staleness s buys T seconds per step" is a statement about this ratio, and
the paper has to say so. It is a systems claim, not an algorithmic constant.

R=5 (N=6) is the next ratio the batch shape allows: the colocated arm trains on
all N nodes at `dp = 4N`, and `4N | 3072` needs `N | 768`, so N ∈ {2,3,4,6,8,12,16}
and N=5 is not available. Extrapolating the observed times at `tau_roll x 3/5`,
R=5 is 16-37% faster in wall-clock but costs more node-seconds in three of the
four measured steps: once a step is train-bound, extra rollout nodes are pure
cost. Through rollout 8 the two arms have separated permanently, and the driver is
recycling rather than response length:

| step | s1 `tau_roll` | s1 wait | s1 waste | s2 `tau_roll` | s2 wait | s2 waste |
|---|---|---|---|---|---|---|
| 5 | 421.6 | 83.3 | 0.2301 | 280.2 | 10.7 | 0.0081 |
| 6 | 430.7 | 138.2 | 0.2195 | 208.5 | 1.6 | 0.0038 |
| 7 | 410.7 | -- | 0.2606 | 183.4 | -- | 0.0115 |

s2 settled at `tau_roll` 183-280 s against `tau_train` ~320 s: train-bound, as
the split was designed for. s1 sits at 410-430 s and stays rollout-bound.
`response_len/mean` is flat-to-falling over these steps (6748, 6005, 6426, 5885),
so length growth is not the cause; a recycled group is regenerated from scratch,
so discarding a quarter of the tokens inflates effective `tau_roll` by ~1.33x and
the arm cannot climb out.

This does **not** argue for R=5. The uncapped table above shows R=5 running a
queue of 827 groups and a natural lag of mean 2.7 / P(L>2) = 0.6. A bound-1 arm
fed by that distribution would recycle most of what it generates, so raising R
makes the tight-bound arms worse, not better. The starvation of the low-staleness
arms at a fixed node budget is the phenomenon under study, not an artifact of
under-provisioning.

## Positioning against VCPO (arXiv:2602.17616, 2026-08-07)

"Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs" (MIT Han lab +
NVIDIA) reports "matches the best synchronous accuracy 2.5x faster (~42h vs
~105h)", which reads at first like the question this study asks. It is not, and
the settings are worth recording precisely because the difference is in the
setup rather than in the headline.

| | VCPO | this study |
|---|---|---|
| response length | **2048** for GSM8K / MATH / Countdown; 12,288 only for the tool-use task | 32k primary, 4k as a swept axis |
| model | **Qwen2-1.5B Base**, **Qwen2.5-7B Base** | Qwen3-4B-Instruct-2507 |
| learning rate | 1e-6 fixed; "without learning-rate sweeps" is offered as a *feature* | 1e-7 / 1e-6 / 5e-6 swept |
| convergence | the string "converg" appears **once** in the paper, in an aside | time to non-inferiority against a converged reference is the measurement |
| staleness | configured k in {2, 10, 12, 128}; realized lag never measured | configured bound and realized lag measured separately |
| training steps | 400 (GSM8K/MATH), 200 (tool-use) | 300 at 32k |

The 2.5x number is a **single point**: the tool-use task at k=2, response length
12,288, 200 steps. On math the paper asks whether training collapses at k=10/12/
128, not how long anything takes. There is no statistical definition of
non-inferiority and no check that the synchronous baseline had converged -- the
comparison point is read off the synchronous curve by eye.

So the framing survives: VCPO shows an algorithm that does not collapse at 2k
response length on base models, and this study measures when an off-policy run
becomes non-inferior to a converged reference, as a function of learning rate,
response length and realized lag. Do not weaken the claim; do cite VCPO as the
closest prior wall-clock result and state the three setting differences.
