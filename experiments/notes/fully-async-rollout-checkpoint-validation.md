# Fully-async rollout checkpoint validation

This note records the correctness, resume-latency, batching-efficiency, and
short-horizon convergence checks for the opt-in fully-async rollout checkpoint
sidecar. It is intended to make the measurements reproducible after the Slurm
logs have aged out.

## Scope and revisions

- MILES branch: `experiments/cw-dfw-math-rl`
- Integrated revision: `7917d9db`
- Sidecar lifecycle checkpoint: `3874266d`
- Packed replay serialization: `5134ce9e`
- Streaming tensor parts and incremental checksums: `d0adc6c6`
- Trainer batching metrics: `5a0d2775`
- Container:
  `/lustre/fsw/portfolios/coreai/users/kfujii/container/miles-prefill-weight-version-23aaf6597.sqsh`
- Dataset: real `dapo-math-p10-90` prompts
- Model: `Qwen3-4B-Instruct-2507`

The feature is opt-in through `--fully-async-rollout-checkpoint`. A checkpoint
created without the option has no matching replay sidecar, so enabling it in
the middle of an existing experimental arm does not retroactively restore the
rollout pipeline. Matched comparisons therefore start at a new experiment
boundary.

## What is preserved

The sidecar preserves the fully-async prompt lifecycle at a trainer checkpoint:

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

## Sidecar serialization benchmark

CPU job 15669530 replayed a real schema-one sidecar captured by the 4-node
`full-replay-4n-feature-batch-20260812` run. The input was 97,416,265 bytes and
the packed representation contained 7,040 sample records. Three repetitions
gave the following medians:

| Implementation | Capture | Durable save plus checksum | Total | Size |
|---|---:|---:|---:|---:|
| Legacy Python/per-token state | 4.601 s | 3.301 s | 7.838 s | 97.4 MB input |
| Packed arrays and streamed checksum | 0.778 s | 0.275 s | 1.049 s | 56.8 MB |

The total capture-and-save path improved by 7.47x and the durable save portion
by 12.0x. The benchmark materialized the packed state and loaded the published
sidecar through the production reader; both comparisons were bit-exact,
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
| Sidecar enabled | 15710027 | 15710033 | Passed |
| Sidecar disabled | 15710428 | 15710458 | Passed |

The enabled resume restored 60 pending groups, 48 ready groups, 8 active groups
for regeneration, and one prepared batch. The trainer logged
`resume/fully_async/warm_prepared_batch_hit=1` and immediately reused the
prepared batch.

Resume-side first rollout wait:

| Mode | `perf/rollout_time` |
|---|---:|
| Sidecar enabled | 0.092381 s |
| Sidecar disabled | 2.753768 s |

This is a 29.8x reduction, or 2.661 s saved in the deliberately small smoke
configuration. Whole Slurm elapsed time was 3:51 versus 4:10 on resume, a
19-second (7.6%) reduction despite model, Ray, and SGLang initialization being
unchanged.

## Real-data 20-step restart check

These jobs used a 2048-token response cap, 8 prompts x 4 responses, global
batch size 32, and a forced restart after step 9.

| Mode | Steps 0--9 | Steps 10--19 |
|---|---:|---:|
| Sidecar enabled | 15711550 | 15711552 |
| Sidecar disabled | 15711554 | 15711562 |

The enabled resume restored 24 pending groups, regenerated 16 active groups,
and reused one prepared batch. Eleven of 20 enabled steps and fourteen of 20
disabled steps had nonzero gradients. This establishes that both paths execute
real optimizer updates; the run is too short to establish convergence.

## Matched 100-step comparison

The definitive short-horizon comparison uses the same initial checkpoint,
training seed 1234, rollout seed 42, data, model, container, batch shape, and
hyperparameters. The only treatment difference is
`FULLY_ASYNC_ROLLOUT_CHECKPOINT=1` versus `0`. Each arm is forcibly restarted at
50 steps and continues to 100 steps.

| Mode | Steps 0--49 | Steps 50--99 | Final AIME24 eval |
|---|---:|---:|---:|
| Sidecar enabled | 15712624 | 15712640 | 15718112 |
| Sidecar disabled | 15712642 | 15712648 | 15718134 |

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
restored the sidecar and reused a prepared batch. Its first
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
the resume found no replay sidecar and could not reuse a prepared batch. Its
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

At the forced restart boundary, the sidecar reduced the first trainer rollout
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
embedding tensor had 152,064 rows. This was independent of the sidecar. The
official `experiments/src/offline_eval/unpad_vocab.py` utility removed only the
128 padded rows in eval-only checkpoint copies. CPU jobs 15718013 and 15718020
verified every retained tensor value against the original before jobs 15718112
and 15718134 performed the successful evaluations. The original training
checkpoints were not modified.

| Metric | Sidecar enabled | Sidecar disabled | Enabled - disabled |
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
  --export=ALL,CONFIG_TAG=<unique-tag>,NUM_ROLLOUT=100,DEBUG_EXIT_AFTER_ROLLOUT=50,SAVE_INTERVAL=50,SAVE_RETAIN_INTERVAL=100,SAVE_HF=1,HF_SAVE_INTERVAL=100,FULLY_ASYNC_ROLLOUT_CHECKPOINT=1,MAX_RESPONSE_LEN=2048,ROLLOUT_BATCH_SIZE=8,N_SAMPLES_PER_PROMPT=4,GLOBAL_BATCH_SIZE=32,MAX_TOKENS_PER_GPU=8192,ASYNC_MAX_CONCURRENT_SAMPLES=64,SGLANG_MAX_RUNNING_REQUESTS=64,EVAL_INTERVAL=0,DUMP_TRAIN_DATA=0,FUSE_ONE_STEP_ACTOR_LOGPROBS=1 \
  experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
```

Submit the second invocation with the same `CONFIG_TAG` and
`--dependency=afterok:<first-job-id>` so it resumes from the step-49
checkpoint. For the control, change only
`FULLY_ASYNC_ROLLOUT_CHECKPOINT=0` and use a distinct `CONFIG_TAG`.

## Interpretation constraints

- Sidecar ON and OFF do not consume exactly the same post-restart trajectories:
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
