# Replay-buffer validation

This note records the correctness, resume-latency, batching-efficiency, and
short-horizon convergence checks for the opt-in replay buffer. The earlier runs
cover what is now `--replay-buffer-type rollout`. The 2026-08-19 production-shape
validation below directly compares `rollout` with the newer `inflight`
token-prefix mode. The record is retained so the results remain reproducible
after the Slurm logs have aged out.

## Scope and revisions

- MILES branch: `experiments/cw-dfw-math-rl`
- Integrated revision: `7917d9db`
- Replay buffer lifecycle checkpoint: `3874266d`
- Packed replay serialization: `5134ce9e`
- Streaming tensor parts and incremental checksums: `d0adc6c6`
- Trainer batching metrics: `5a0d2775`
- Container:
  `/lustre/fsw/portfolios/coreai/users/kfujii/container/miles-prefill-weight-version-23aaf6597.sqsh`
- Dataset: real `dapo-math-p10-90` prompts
- Model: `Qwen3-4B-Instruct-2507`

The later Qwen3-4B Base step-4000 comparison was integrated at revision
`5dd7d1a5`; its model, dataset, and run shape are recorded in its own section
below.

The feature is opt-in through `--use-replay-buffer --replay-buffer-type
rollout`. A checkpoint created without the option has no matching replay buffer, so enabling it in
the middle of an existing experimental arm does not retroactively restore the
rollout pipeline. Matched comparisons therefore start at a new experiment
boundary.

## What is preserved

The replay buffer preserves the fully-async prompt lifecycle at a trainer checkpoint:

- pending prompt groups;
- ready groups;
- active groups, which are regenerated after restart;
- partial drains;
- already prepared trainer batches;
- prompt-source state and applied policy version;
- compact tensor payloads plus incremental checksums.

It does not checkpoint live SGLang KV-cache state. Active requests are safely
regenerated. Consequently, a restart is not bit-for-bit continuation of every
in-flight token, but it avoids throwing away pending, ready, and prepared work.

## CPU regression tests

The dedicated batching-metric suite passed 15 tests. The post-merge selected
regression suite passed 282 tests with 17 warnings. Six broader metric-snapshot
failures reproduced at the pre-change baseline `d0adc6c6` and are unrelated
stale expected-value fixtures.

The new per-optimizer-step metrics are:

- `train/useful_tokens`
- `train/scheduled_tokens`
- `train/padding_or_unused_token_frac`
- `train/microbatch_token_min`
- `train/microbatch_token_max`
- `train/microbatch_token_p50`
- `train/dp_token_imbalance`
- `train/packing_efficiency`

Only a small fixed-size statistic is gathered across data-parallel ranks; no
per-token payload is introduced by this telemetry.

## Replay buffer serialization benchmark

CPU job 15669530 replayed a real schema-one replay buffer captured by the 4-node
`full-replay-4n-feature-batch-20260812` run. The input was 97,416,265 bytes and
the packed representation contained 7,040 sample records. Three repetitions
gave the following medians:

| Implementation | Capture | Durable save plus checksum | Total | Size |
|---|---:|---:|---:|---:|
| Legacy Python/per-token state | 4.601 s | 3.301 s | 7.838 s | 97.4 MB input |
| Packed arrays and streamed checksum | 0.778 s | 0.275 s | 1.049 s | 56.8 MB |

The total capture-and-save path improved by 7.47x and the durable save portion
by 12.0x. The benchmark materialized the packed state and loaded the published
replay buffer through the production reader; both comparisons were bit-exact,
including array dtype, shape, and bytes. A second single-repeat run (job
15669990) reproduced the result at 1.043 s versus 7.220 s.

This is a real 4-node queue snapshot, but its 97.4 MB source is not a worst-case
multi-gigabyte 32K queue. The result establishes the benefit of packing and
single-pass checksumming; it should not be extrapolated as a measured upper
bound for every possible 32K queue depth.

## Interactive GPU smoke tests

All jobs used 2 interactive nodes (16 H100 GPUs), real DAPO prompts, and the
same container and model checkpoint.

| Case | Fresh job | Resume job | Result |
|---|---:|---:|---|
| Replay buffer enabled | 15710027 | 15710033 | Passed |
| Replay buffer disabled | 15710428 | 15710458 | Passed |

The enabled resume restored 60 pending groups, 48 ready groups, 8 active groups
for regeneration, and one prepared batch. The trainer logged
`resume/replay_buffer/warm_prepared_batch_hit=1` and immediately reused the
prepared batch.

Resume-side first rollout wait:

| Mode | `perf/rollout_time` |
|---|---:|
| Replay buffer enabled | 0.092381 s |
| Replay buffer disabled | 2.753768 s |

This is a 29.8x reduction, or 2.661 s saved in the deliberately small smoke
configuration. Whole Slurm elapsed time was 3:51 versus 4:10 on resume, a
19-second (7.6%) reduction despite model, Ray, and SGLang initialization being
unchanged.

### Three-policy replay buffer extension

The queue-policy extension was validated on 2026-08-13 after merging remote
revision `b20c5962`. Each case used one node (8 H100 GPUs), split into four
trainer GPUs and four rollout GPUs, with real DAPO prompts, two prompt groups x
two responses per trainer batch, a 256-token response cap, and
`--use-replay-buffer --replay-buffer-type rollout`. The fresh job trained
rollout 0 and published its model plus replay buffer; a separate dependent job
loaded both and trained rollout 1.

| Policy | Fresh job | Resume job | Restored pending / ready / regenerated / prepared | First resumed rollout |
|---|---:|---:|---:|---:|
| `queue-recycle` | 15720395 | 15720412 | 35 / 29 / 4 / 1 | 0.099864 s |
| `queue-max` | 15720396 | 15720411 | 29 / 25 / 4 / 0 | 0.105536 s |
| `queue-drop` (`q=1`, capacity 2) | 15720397 | 15720413 | 6 / 2 / 4 / 0 | 0.106803 s |

All six jobs completed with `ExitCode=0:0`. The policy-specific behavior also
matched the intended contracts:

- `queue-recycle` reported `warm_prepared_batch_hit=1` and reused its prepared
  batch without starting the worker for that batch;
- `queue-max` selected restored ready groups only after the resume-time update.
  With `max_weight_staleness=1`, both trained groups had prefill-bound
  staleness exactly 1 (`staleness/total/count_1=2`) and no rejected samples,
  confirming that one version gap is accepted rather than re-indexed as
  on-policy;
- `queue-drop` restored exactly its two-group capacity plus four active prompt
  leases for regeneration. Its first resumed metrics recovered 30 prior queue
  evictions, 15,360 evicted response tokens, and 60 aligned sample lengths from
  the replay buffer. No evicted prompt was regenerated.

The first fresh replay buffer publication for each policy measured:

| Policy | Ready / active / prepared groups | Capture | Durable write | Total | Stored bytes |
|---|---:|---:|---:|---:|---:|
| `queue-recycle` | 29 / 4 / 1 | 0.006 s | 0.025 s | 0.031 s | 388,161 |
| `queue-max` | 25 / 4 / 0 | 0.005 s | 0.026 s | 0.030 s | 302,274 |
| `queue-drop` | 2 / 4 / 0 | 0.004 s | 0.021 s | 0.025 s | 39,456 |

These small smoke snapshots are latency sanity checks rather than replacements
for the 56.8-MB benchmark above. They show that policy bookkeeping adds no
visible save-boundary spike. `queue-recycle` and `queue-max` continue to use the
same packed schema-three representation. `queue-drop` additionally limits
storage by applying oldest-first overflow at capture and omitting the evicted
trajectories themselves; it retains only their compact counts, token/length
statistics, and optional reward-lifecycle records.

The containerized regression suite passed 234 tests (job 15720283), including
policy storage reconstruction, queue-max staleness rechecks, queue-drop
snapshot overflow, telemetry persistence, legacy replay-buffer compatibility,
and cross-policy/capacity rejection. A real-Ray GPU test also passed (job
15720353).

## Real-data 20-step restart check

These jobs used a 2048-token response cap, 8 prompts x 4 responses, global
batch size 32, and a forced restart after step 9.

| Mode | Steps 0--9 | Steps 10--19 |
|---|---:|---:|
| Replay buffer enabled | 15711550 | 15711552 |
| Replay buffer disabled | 15711554 | 15711562 |

The enabled resume restored 24 pending groups, regenerated 16 active groups,
and reused one prepared batch. Eleven of 20 enabled steps and fourteen of 20
disabled steps had nonzero gradients. This establishes that both paths execute
real optimizer updates; the run is too short to establish convergence.

## Matched 100-step comparison

The definitive short-horizon comparison uses the same initial checkpoint,
training seed 1234, rollout seed 42, data, model, container, batch shape, and
hyperparameters. The only treatment difference is
`USE_REPLAY_BUFFER=1` versus `0`. Each arm is forcibly restarted at
50 steps and continues to 100 steps.

| Mode | Steps 0--49 | Steps 50--99 | Final AIME24 eval |
|---|---:|---:|---:|
| Replay buffer enabled | 15712624 | 15712640 | 15718112 |
| Replay buffer disabled | 15712642 | 15712648 | 15718134 |

Configuration:

```text
MAX_RESPONSE_LEN=2048
ROLLOUT_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=4
GLOBAL_BATCH_SIZE=32
MAX_TOKENS_PER_GPU=8192
ASYNC_MAX_CONCURRENT_SAMPLES=64
SGLANG_MAX_RUNNING_REQUESTS=64
MAX_WEIGHT_STALENESS=2
STALENESS_REFERENCE=prefill
PAUSE_GENERATION_MODE=in_place
FUSE_ONE_STEP_ACTOR_LOGPROBS=1
SAVE_INTERVAL=50
HF_SAVE_INTERVAL=100
```

### Enabled-arm results

Both 50-step segments completed successfully in 9:01 and 9:13. The resume
restored the replay buffer and reused a prepared batch. Its first
`perf/rollout_time` was 0.146476 s.

| Metric | Steps 0--49 | Steps 50--99 |
|---|---:|---:|
| Mean raw reward | 0.091250 | 0.135625 |
| Mean grad norm | 0.213378 | 0.225333 |
| Nonzero-gradient steps | 38/50 | 40/50 |
| Mean packing efficiency | 0.979769 | 0.977763 |
| Mean padding/unused fraction | 0.020231 | 0.022237 |
| Mean total staleness | 1.727941 | 1.807500 |
| Mean pre-queue staleness | 1.654412 | 1.795000 |
| Mean in-queue staleness | 0.073529 | 0.012500 |
| Mean mixed-version fraction | 0.897059 | 0.967500 |

The reward values are generated from different asynchronous batches before
and after restart, so the within-arm increase is descriptive rather than a
paired-sample estimator.

### Disabled-arm results and restart latency

Both disabled segments completed successfully in 8:55 and 9:11. As expected,
the resume found no replay buffer and could not reuse a prepared batch. Its
first `perf/rollout_time` was 10.527783 s.

| Metric | Steps 0--49 | Steps 50--99 |
|---|---:|---:|
| Mean raw reward | 0.096875 | 0.139375 |
| Mean grad norm | 0.255776 | 0.255612 |
| Nonzero-gradient steps | 40/50 | 41/50 |
| Mean packing efficiency | 0.980305 | 0.978152 |
| Mean padding/unused fraction | 0.019695 | 0.021848 |
| Mean total staleness | 1.747549 | 1.730000 |
| Mean pre-queue staleness | 1.688725 | 1.665000 |
| Mean in-queue staleness | 0.058824 | 0.065000 |
| Mean mixed-version fraction | 0.911765 | 0.905000 |

At the forced restart boundary, the replay buffer reduced the first trainer rollout
wait from 10.527783 s to 0.146476 s: 10.381307 s saved, or 71.9x lower. The
initial segments had nearly identical cold-start waits (10.639241 s enabled and
10.611810 s disabled), which confirms that the restart difference came from
restoring prepared work rather than a generally faster treatment job.

The sum of the two Slurm `Elapsed` values was 18:14 enabled and 18:06 disabled.
This deliberately short comparison therefore does not show an end-to-end job
speedup: model/SGLang/Ray startup, final HF export, and asynchronous node noise
swamp a roughly ten-second restart saving. The smaller smoke test above did
show a 19-second (7.6%) whole-resume-job reduction. The stable claim from both
experiments is the reduction in time to the first trainable batch, not a fixed
percentage reduction in total training time.

All four training jobs reported `COMPLETED` with `ExitCode=0:0`. The cluster's
defunct-chain monitor nevertheless held jobs 15712642 and 15712648 because all
segments finished in less than its ten-minute heuristic. They were released
and completed normally; this was a false positive rather than an OOM or
training failure. A matching future false positive can be released with:

```bash
scontrol release <jobid>
```

### Fixed AIME24 evaluation

The final step-99 checkpoints were evaluated on all 30 AIME24 prompts with
eight samples per prompt, a 4096-token response cap, and no generation
failures. The first submissions (15712674 and 15712682) exposed a common HF
export issue: the config declared vocabulary size 151,936 while the padded
embedding tensor had 152,064 rows. This was independent of the replay buffer. The
official `experiments/src/offline_eval/unpad_vocab.py` utility removed only the
128 padded rows in eval-only checkpoint copies. CPU jobs 15718013 and 15718020
verified every retained tensor value against the original before jobs 15718112
and 15718134 performed the successful evaluations. The original training
checkpoints were not modified.

| Metric | Replay buffer enabled | Replay buffer disabled | Enabled - disabled |
|---|---:|---:|---:|
| Correct samples | 83/240 | 75/240 | 8/240 |
| Mean prompt pass rate | 0.345833 | 0.312500 | +0.033333 |
| Prompt-level standard error | 0.077 | 0.073 | -- |
| Truncated response fraction | 0.6417 | 0.6208 | +0.0209 |
| Mean response length | 3422 | 3330 | +92 |

Pairing the 30 prompt pass rates gives an enabled-minus-disabled difference of
0.033333, standard error 0.021554, and a 95% t interval of
[-0.010749, 0.077415]. The interval includes zero. Together with the similar
100-step reward traces, this finds no short-horizon convergence regression
from checkpointing. It is not evidence that the two asynchronous runs are
trajectory-identical or a substitute for a full-scale convergence study.

The retained evaluation results are:

```text
/lustre/fsw/portfolios/coreai/users/kfujii/datasets/offline_eval/sidecar-on-100-v2-20260813/aime24.jsonl
/lustre/fsw/portfolios/coreai/users/kfujii/datasets/offline_eval/sidecar-off-100-v2-20260813/aime24.jsonl
```

## Reproduction command template

Run from the repository root. Use a real W&B credential or `offline`; never put
the credential in this note or a committed script.

```bash
export SLURM_ACCOUNT_NAME=coreai_horizon_dilations
export WS=/lustre/fsw/portfolios/coreai/users/kfujii
export SQSH_IMAGE=/lustre/fsw/portfolios/coreai/users/kfujii/container/miles-prefill-weight-version-23aaf6597.sqsh
export WANDB_MODE=offline
export WANDB_API_KEY=offline

sbatch -A "${SLURM_ACCOUNT_NAME}" -p interactive -N 2 -t 02:00:00 \
  --export=ALL,CONFIG_TAG=<unique-tag>,NUM_ROLLOUT=100,DEBUG_EXIT_AFTER_ROLLOUT=50,SAVE_INTERVAL=50,SAVE_RETAIN_INTERVAL=100,SAVE_HF=1,HF_SAVE_INTERVAL=100,USE_REPLAY_BUFFER=1,REPLAY_BUFFER_TYPE=rollout,MAX_RESPONSE_LEN=2048,ROLLOUT_BATCH_SIZE=8,N_SAMPLES_PER_PROMPT=4,GLOBAL_BATCH_SIZE=32,MAX_TOKENS_PER_GPU=8192,ASYNC_MAX_CONCURRENT_SAMPLES=64,SGLANG_MAX_RUNNING_REQUESTS=64,EVAL_INTERVAL=0,DUMP_TRAIN_DATA=0,FUSE_ONE_STEP_ACTOR_LOGPROBS=1 \
  experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
```

Submit the second invocation with the same `CONFIG_TAG` and
`--dependency=afterok:<first-job-id>` so it resumes from the step-49
checkpoint. For the control, change only
`USE_REPLAY_BUFFER=0` and use a distinct `CONFIG_TAG`.

## Interpretation constraints

- Replay buffer ON and OFF do not consume exactly the same post-restart trajectories:
  preserving queued work is the treatment itself. Fixed-batch unit tests cover
  serialization equivalence, while live runs measure system behavior.
- A 100-step run is a short-horizon non-regression check, not a full convergence
  proof. The paired AIME24 interval includes both a small regression and a
  modest improvement; longer runs and more evaluation samples are needed to
  resolve either.
- Slurm queue delay is excluded from trainer wallclock comparisons; job
  `Elapsed`, per-step timing, and resume-side `perf/rollout_time` are reported
  separately.
- The 2048-token validation cap is intentionally cheaper than the production
  32K recipe. Serialization benchmarks separately cover large compact tensor
  payloads; full 32K training remains the production-scale confirmation.

## Qwen3-4B Base step-4000 `rollout` versus `inflight` validation (2026-08-19)

This validation tests the new Qwen3-4B Base checkpoint and its model-specific
DAPO-Math p10--90 dataset at the production response length and batch shape. It
also corrects an initially misleading comparison of whole-job staleness means:
fresh and resumed jobs contain different startup phases, so their aggregate
means do not measure the discontinuity at the restart boundary.

### Run identity and configuration

- Integrated revision: `5dd7d1a5`
- Validation namespace: `rbtype-step4000-16080496`
- W&B project: `async-rl-miles-replay-buffer`
- Model: `Qwen3-4B-Base`, pretraining step 4000
- Dataset: the corresponding 16-sample, 16K-response, zero-truncation-reward
  DAPO-Math p10--90 filter output
- Resources per training job: two nodes, eight actor GPUs plus eight rollout GPUs
- Response cap: 16,384 tokens; context cap: 32,768 tokens
- Rollout batch: 192 prompt groups x 16 samples = 3,072 samples
- Global batch: 3,072; one optimizer step per rollout
- Queue: `queue-recycle`, prefill staleness reference, maximum staleness 8
- Reward: `deepscaler` with `--zero-reward-on-truncated`
- Validation checkpointing: `SAVE_INTERVAL=1`, `SAVE_HF=0`

The validation deliberately saved a replay buffer at every step and disabled HF
exports to isolate replay-buffer cost. The production recipe instead uses
`SAVE_INTERVAL=10` and `HF_SAVE_INTERVAL=10`; therefore the numbers below do not
measure the complete production checkpoint pause when replay, MCore, and HF are
all due.

| Segment | Slurm job | Trained steps | Result |
|---|---:|---|---|
| `rollout` fresh | 16080545 | 0--3 | passed |
| `rollout` resume | 16080547 | 4--5 | passed |
| `inflight` fresh | 16080548 | 0--3 | passed |
| `inflight` resume | 16080549 | 4--5 | passed |

The generated report is retained at
`experiments/outputs/replay_buffer_validation/rbtype-step4000-16080496.md`, and
the four training logs are under
`experiments/outputs/training/math/dapo-math-p10-90/qwen3-4b/`.

### Correct restart-boundary comparison

The fresh job trained steps 0--3. Before saving, it finished the already
prefetched step-4 batch so that the replay buffer contained one complete
prepared batch. The resume job trained that identical step-4 batch first; step
5 is the first batch containing newly continued or regenerated work after the
restart. A further step-6 metric was prepared after the resumed segment's last
trained step and is likewise excluded from the trained-segment comparison.

Consequently, compare prepared step 4 with newly produced step 5, not the mean
of every metric in the fresh job with the mean of every metric in the resumed
job. Counts in the following table are the number of prompt groups at total
staleness 0, 1, and 2. Distribution distance is total variation,
`0.5 * sum(abs(p_before - p_after))`.

| Buffer | Saved/prepared step 4 mean and counts | First new step 5 mean and counts | Mean change | Distribution distance |
|---|---|---|---:|---:|
| `inflight` | 0.604167 `[77, 114, 1]` | 0.583333 `[82, 108, 2]` | -0.020833 | 0.03125 |
| `rollout` | 0.682292 `[71, 111, 10]` | 0.020833 `[188, 4, 0]` | -0.661459 | 0.609375 |

The conclusion is unchanged if the reference is the last batch actually
trained before the stop rather than the saved prepared batch:

| Buffer | Last pre-stop trained step 3 | First new post-resume step 5 | Mean change | Distribution distance |
|---|---:|---:|---:|---:|
| `inflight` | 0.630208 | 0.583333 | -0.046875 | 0.046875 |
| `rollout` | 0.744792 | 0.020833 | -0.723959 | 0.6875 |

The step-4 mean and histogram were exactly identical in the fresh and resume
logs for each buffer type, and both resume jobs reported
`warm_prepared_batch_hit=1`. This verifies prepared-batch restoration
independently of the step-5 comparison.

The initially reported whole-segment means must not be interpreted as boundary
jumps. In particular, `inflight` fresh included cold-start steps 0 and 1 with
mean staleness zero, while the resume segment started from a warm prepared
batch. This composition effect makes its aggregate `0.3138 -> 0.5938` look like
a large increase. Conversely, the `rollout` aggregates `0.3346 -> 0.3516` hide
the severe step-5 reset. Boundary-aligned batches show the opposite and correct
result: `inflight` preserves the distribution substantially better.

### Restored work and resume latency

| Resume metric | `rollout` | `inflight` |
|---|---:|---:|
| Restored inflight groups | 0 | 191 |
| Restored inflight response tokens | 0 | 7,326,600 |
| Regenerated active groups | 192 | 0 |
| Restored pending groups | 388 | 388 |
| Restored prepared batches | 1 | 1 |
| First resumed rollout wait | 10.645 s | 10.609 s |
| Second resumed rollout wait | 775.393 s | 517.877 s |

The first wait is the same because both modes reuse the complete step-4
prepared batch. The second wait exposes the active-generation treatment:
continuing the saved prefixes reduced it by 257.516 seconds, while `rollout`
regenerated all 192 active groups.

This is one fresh/resume sequence with only two resumed training steps. It is
strong evidence for the persistence mechanism and for restart-distribution
continuity, but it is not a seed-general convergence result.

### Value at a ten-step production interval

| Replay save metric | `rollout` | `inflight` | `inflight - rollout` |
|---|---:|---:|---:|
| Capture median | 1.7945 s | 57.8465 s | +56.0520 s |
| Durable-write median | 1.2630 s | 0.6380 s | -0.6250 s |
| Total median | 2.9340 s | 58.5285 s | +55.5945 s |
| Stored size median | 234.075 MiB | 333.692 MiB | +99.617 MiB |

At `SAVE_INTERVAL=10`, the measured incremental `inflight` cost amortizes to
5.559 seconds per optimizer step. Against the observed roughly 450--700-second
steps, this is about 0.8--1.2%. One observed resume recovered 257.516 seconds on
the second rollout and avoided the much larger staleness-distribution reset.
The ten-step interval therefore gives `inflight` meaningful operational value
for restart-heavy training, even before assigning value to preserving the
experimental staleness distribution itself.

The math-async recipe currently defaults to `REPLAY_BUFFER_TYPE=rollout`.
Production uses `inflight` only when it is selected explicitly (or if the recipe
default is changed); the validation jobs set the type explicitly and keep the
two checkpoint namespaces separate.

### Why `inflight` saving pauses training

The durable file write is not the bottleneck. Across the six `inflight` saves,
capture took 53.610--60.392 seconds while the durable write took only
0.629--0.735 seconds. Each snapshot materialized 191--192 inflight prompt groups
and approximately 6.0--7.3 million partial response tokens.

The current capture path:

1. stops the fully-async producer worker;
2. sends `abort_all` to every SGLang engine;
3. awaits every unfinished group task so roughly three thousand individual
   `/generate` calls return partial token IDs and logprobs;
4. encodes the mutable partial samples into the replay-buffer state;
5. restarts the producer, then publishes the replay buffer.

The SGLang `abort_request` endpoints acknowledged in the same log second in
which capture started, but the `Captured replay buffer` record followed roughly
54--60 seconds later. Packing statistics increased by roughly 4--5 seconds on
later saves; the remaining delay is most plausibly dominated by draining and
materializing the many large HTTP responses. There are not yet subphase timers,
so this split is an evidence-based diagnosis rather than an exact measurement.

Optimizing compression or making the final file write asynchronous can save
less than one second. The semantics-preserving optimization order is:

1. add separate timers for worker stop, abort RPC, active-task drain, encoding,
   and durable write;
2. replace thousands of independent JSON reply drains with an engine-level,
   batched and compact `abort-and-snapshot` result, or continuously stream the
   partial state to Miles so the save barrier only finalizes it;
3. consider overlapping staged MCore/HF work only if the invariant
   `durable replay buffer -> visible model tracker -> replay commit` remains
   intact.

Reducing active concurrency or deliberately draining the pipeline before every
checkpoint would change throughput and the staleness distribution that the
experiment is intended to preserve, so those are not equivalent optimizations.

### Generation resumes before checkpoint saving finishes

Replay capture is a barrier only through the point at which the replay state is
materialized.  The capture path restarts the fully-async producer immediately
after encoding that state; it does not keep generation paused until the replay
file, MCore checkpoint, and optional HF export have all finished.  New rollout
work can therefore run in the background during those later save phases and can
advance under the pre-update weight version until the normal weight-update pause.

The `inflight` validation at step 3 makes this ordering visible.  Replay capture
finished at `03:38:03.270`, the producer restarted at `03:38:03.298`, and replay
publication completed at `03:38:03.931`.  SGLang prefill/decode activity then
overlapped the MCore save from approximately `03:38:03.970` to `03:38:09.510`;
the weight-update pause began around `03:38:11`.  Thus there was an approximately
eight-second post-snapshot generation window even with HF export disabled.

This can affect the staleness distribution of work produced around a checkpoint,
and a production checkpoint that also writes HF weights can make the window
longer.  It does not invalidate the saved replay state: the resumed work is newer
live state and is intentionally absent from the already materialized snapshot.
For the current experiments this is accepted as a known caveat, and the producer
restart behavior is left unchanged.  Comparisons around a restart should remain
aligned to the first newly produced batch, and checkpoint-boundary effects should
not be attributed solely to replay restoration.

### Sixteen-thousand-token truncation reward

Both the async and sync Qwen3-4B recipes set:

```text
MAX_RESPONSE_LEN=16384
RM_TYPE=deepscaler
ZERO_REWARD_ON_TRUNCATED=1
```

Their `train.sh` files translate the last setting into
`--zero-reward-on-truncated`. The validation command line contained all three
settings, and its trained batches had a roughly 5.1--5.6% truncated-response
fraction, so the run did exercise length truncation.

For the built-in single-turn SGLang path, `finish_reason=length` sets
`Sample.Status.TRUNCATED`. The reward dispatcher then returns scalar zero before
calling DeepScaler. The same status is used when the remaining 32K context
budget prevents further generation, so both response-limit and context-limit
truncations receive zero. A normal `finish_reason=stop` remains completed and is
graded by DeepScaler.

This remains an option rather than a global behavior change: argparse defaults
`--zero-reward-on-truncated` to off, preserving the historical behavior of
grading truncated text. The Qwen3-4B async/sync recipes and the model-specific
difficulty-filter jobs enable it by default; setting
`ZERO_REWARD_ON_TRUNCATED=0` disables it for a recipe invocation.
