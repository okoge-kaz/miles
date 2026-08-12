---
name: miles-run-ladder
description: Take a miles RL recipe under experiments/ from first submission to a production run on cw-dfw in three stages — interactive for bring-up, batch_short for parallelism and rollout tuning, batch for the real 4-hour job. Use when asked to run, scale, tune, or debug a training recipe on this cluster, when a run needs more nodes, or when deciding whether a configuration is ready for a production allocation. Enforces that only one class of setting changes per stage, so a regression is attributable.
---

# Miles Run Ladder

Three stages, each answering one question. Do not skip a stage, and do not change
a setting that belongs to a later stage while working in an earlier one — the
whole point is that when something regresses, only one class of thing moved.

| Stage | Partition | Shape | Question |
|---|---|---|---|
| 1. Bring-up | `interactive` | 2 nodes, short | Does it run correctly, multi-node? |
| 2. Tuning | `batch_short` | 2–4 nodes | How fast can one step be made? |
| 3. Production | `batch` | N nodes, `4:00:00` | Does it learn? |

Partition is the only scheduling lever on cw-dfw — it selects the QoS, so never
pass `--qos`. `interactive` caps at 2 nodes and schedules ahead of everything
else; `batch_short` caps at 2 h and 4 nodes; `batch` is the 4 h production lane.

Recipes live at `experiments/<task>/<dataset>/<model>/`, submitted as
`experiments/<task>/<dataset>/<model>/run.sbatch` with `-N <nodes>`. Every knob is a
`: "${VAR:=value}"` line at the top of `run.sbatch`: edit it there for a lasting
change, or override on the command line with `--export=ALL,VAR=…`, which wins
over the `:=` default. `train.sh` defines no defaults, so what `run.sbatch` says
is what runs. Submit from the repo root — the `#SBATCH --output` paths are
relative.

Node counts are never inferred. `-N 4` allocates four nodes; `ACTOR_NUM_NODES`,
`ACTOR_GPUS_PER_NODE` and `ROLLOUT_NUM_GPUS` say how they are used, and
`common/placement.sh` rejects a shape that does not add up before `srun` runs.

## Stage 1 — bring-up on `interactive`

Two nodes, not one. A single node exercises none of the multi-node paths (Ray
worker join, cross-node weight sync, data-parallel sharding across nodes), so a
one-node success says nothing about the run you actually intend to submit.

Cut the work down so a failure surfaces in minutes, not hours:

```bash
sbatch -A coreai_horizon_dilations -N 2 --time=01:00:00 \
  --export=ALL,NUM_ROLLOUT=3,ROLLOUT_BATCH_SIZE=8,N_SAMPLES_PER_PROMPT=8,GLOBAL_BATCH_SIZE=64,MAX_RESPONSE_LEN=1024,EVAL_INTERVAL=1 \
  experiments/math_sync/dapo-math-p10-80/qwen3-1.7b/run.sbatch
```

Keep the four-knob invariant when shrinking:
`rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout`.
The recipes pass `--num-steps-per-rollout`, so miles asserts this at startup
rather than silently reshaping the run.

Check, in this order — each one fails earlier than the next:

1. **Ray cluster formed.** `ray cluster ready: 2/2 nodes` in the log. A run that
   proceeds with `1/2` was never multi-node. The usual cause is a stale done-flag
   under `experiments/outputs/.ray/`, which releases the workers immediately.
2. **Placement line matches intent.** `run.sbatch` echoes
   `placement AxG train (W GPU) + R rollout, tpT cpC -> dpD` before `srun`.
   Confirm `D` is what you expect: data parallelism is whatever `tp × cp` leaves
   over, so it changes whenever the node count or the parallelism does.
3. **Checkpoint path.** The recipe echoes `checkpoints: /ckpt/training/<task>/<dataset>/<model>/<config>`.
   Two different configurations must never print the same path — that would make
   one resume from the other's optimizer state.
4. **Reward is not identically zero.** The first-sample log line shows the
   response and its reward. Reward ≡ 0 with no error is the signature of a
   verifier that does not match the checkpoint: `deepscaler` returns 0 unless the
   response contains `</think>`, so a non-thinking model needs `RM_TYPE=math`.
5. **Eval ran.** With `EVAL_INTERVAL=1`, each configured benchmark reports its
   own series. A benchmark that errors here is a prompt/label shape problem, not
   a model problem.
6. **A checkpoint saved and reloaded.** Resubmit the same command; it must
   `--load` the checkpoint written by the first job. Production runs on this
   cluster span several 4 h jobs, so resume is a correctness requirement.

Only after all six: raise the response length and batch back to production
values and confirm one full step completes without OOM. That is the last thing
stage 1 owns.

## Stage 2 — tuning on `batch_short`

`batch_short` is 2 h and up to 4 nodes, and it schedules ahead of `batch`. Use it
for measurement only, never for a run whose result you intend to keep.

Enable telemetry and read it rather than guessing:

```bash
sbatch -A coreai_horizon_dilations -p batch_short -N 2 --time=02:00:00 \
  --export=ALL,NUM_ROLLOUT=10,CONFIG_TAG=tune-<what-you-changed> \
  experiments/math_sync/dapo-math-p10-80/qwen3-4b/run.sbatch

python -m miles.dashboard.serve --dump-details <dump-dir> --follow   # port 7788
```

Set `CONFIG_TAG` on every tuning submission. Without it, each variant writes to
the same checkpoint directory as the production run and pollutes it.

For more than two or three points, use `experiments/sweep.py` instead of a loop:
it derives a unique `CONFIG_TAG` per point, closes the four-knob invariant in
whichever direction the grid leaves open, refuses to exceed `--max-jobs`, and
records a manifest joining job ids to configurations. It prints the grid and
stops unless `--submit` is passed.

```bash
experiments/sweep.py --sweep experiments/sweeps/offpolicy.txt \
  --recipe math_async/dapo-math-p10-80/qwen3-4b -- -N 2 -p batch_short --time=02:00:00
```

Change exactly one class of setting per job, and record the step time:

**Training side.** `TENSOR_PARALLEL_SIZE`, `CONTEXT_PARALLEL_SIZE`,
`EXPERT_PARALLEL_SIZE`, `MAX_TOKENS_PER_GPU`, `ACTOR_NUM_NODES`,
`ACTOR_GPUS_PER_NODE`. The binding constraint is that a single sample must fit in
`max_tokens_per_gpu × cp_size`, so `max_tokens_per_gpu × cp ≥ rollout_max_context_len`
always. Raising `MAX_TOKENS_PER_GPU` reduces the microbatch count and is usually
the cheapest win until it OOMs. In the dashboard, the Compute Utilization view
shows this per rank: a lane whose `actor_train` starts late is a straggler, and
time in `train_wait` is trainer idle.

**Rollout side.** `ROLLOUT_NUM_GPUS_PER_ENGINE`, `SGLANG_MEM_FRACTION`,
`SGLANG_MAX_RUNNING_REQUESTS`, `SGLANG_CUDA_GRAPH_MAX_BS`, and for async
`ROLLOUT_NUM_GPUS`, `ASYNC_MAX_CONCURRENT_SAMPLES`. The dashboard's advisory
panel compares what the engines did against what they were allowed to do:

- peak `sglang_num_running_reqs` far below `--sglang-max-running-requests` → lower
  it; under `--colocate` that hands the memory back to training
- `sglang_cache_hit_rate` low on a non-colocated run → raise `SGLANG_MEM_FRACTION`
  for a bigger KV cache
- `sglang_token_usage` above 95% → KV cache is the bottleneck; more engine GPUs or
  a smaller rollout batch

`SGLANG_CUDA_GRAPH_MAX_BS` sets the largest batch captured into a CUDA graph.
Capturing costs startup time and memory, and buys nothing above the concurrency
the run actually reaches — size it from the observed peak running requests, not
from the theoretical maximum.

Anything reachable in sglang's `ServerArgs` is already exposed as `--sglang-<field>`
(miles generates them from `ServerArgs.add_cli_args`), so a knob that is not in
the recipe can still be passed without a code change. To see what this image's
sglang actually offers, run inside the container:

```bash
python3 -c "from sglang.srt.server_args import ServerArgs; import dataclasses; print('\n'.join(sorted(f.name for f in dataclasses.fields(ServerArgs))))"
```

Do not tune the learning rate, the batch shape, dynamic sampling, or the
staleness bound here. Those change what is learned; stage 2 only changes how
fast a step is.

## Stage 3 — production on `batch`

```bash
experiments/submit_training.sh math_sync/dapo-math-p10-80/qwen3-8b <run-name> \
  -p batch -N 8 --time=04:00:00 --export=ALL,SAVE_INTERVAL=5
```

Fix the settings that stage 2 chose and stop touching them. From here on, a
change to parallelism invalidates the comparison between this run and the last.

Before submitting:

- **`SAVE_INTERVAL` small enough that a 4 h wall loses little.** The run is
  expected to span several jobs; resubmitting the same command resumes from
  `--load`.
- **Retain HF checkpoints for every dependency chain.** Resolve the values on
  the actual `sbatch` command and require `SAVE_HF=1` plus a positive
  `HF_SAVE_INTERVAL`; do not rely on recipe defaults. A sweep wrapper can
  override a recipe's `SAVE_HF=1` with `SAVE_HF=0`, leaving only resumable
  `torch_dist` checkpoints and making the requested evaluation curve
  unavailable. Confirm that `train.sh` receives both `--save-hf` and
  `--hf-save-interval`, and make the sweep's dry-run output state the retained
  HF cadence before submitting any link.
- **Scale the batch, not just the node count.** More nodes raise data parallelism
  and shrink the per-rank batch; they do not by themselves make each step more
  informative. Raise `ROLLOUT_BATCH_SIZE` with the allocation, and keep the
  four-knob invariant.
- **Leave `DUMP_TRAIN_DATA` off.** The default keeps the collector and the rollout
  dumps, which cost almost nothing. The per-rank train dumps scale with the
  training world size and write inline on the training path, with no retention or
  interval knob — they are for a short diagnostic run, not a production one. If a
  run does need them, estimate the volume from a stage 2 run first
  (`du -sh <dump-dir>` divided by the step count) before multiplying by the
  production step count.
- **`CONFIG_TAG` distinguishes this run from every other.** It names the
  checkpoint directory and is the only thing separating two settings that share a
  model and a dataset.

While running, the questions worth asking are about learning, not speed:
`dump/zero_std_group_frac` climbing means the batch is losing gradient signal;
`staleness/total/*` is the staleness that matters in async runs
(**not** `dump/mixed_version_frac`, which is structurally 0 for single-turn
fully-async — see `notes/telemetry.md`); a gap opening between AIME-2024 and
AIME-2025 means memorisation rather than learning.

## Handing the command to someone else

Before another person runs a submission command, run that command yourself once,
end to end, shrunk to fit `interactive`. Not the recipe — **the command**, by the
path they will invoke it through, with every variable resolved the way it will
resolve for them.

The trap is anything the submission *derives* rather than receives. Tuning
sweeps usually override the derived names with short job names, so the derived
path is the one thing a long tuning campaign never exercises, and it fails on the
first production submission instead:

```bash
# What the sweeps ran: RUN_NAME supplied, ~23 characters.
--export=ALL,RUN_NAME=noderatio-s64-t1r3-rs42,…
# What a production submission runs: RUN_NAME derived, ~124 characters, and
# wandb appends "_" + an 8-character id to the group on top of that.
```

Shrink only what does not feed a derived name or path — `NUM_ROLLOUT`,
`MAX_RESPONSE_LEN`, the batch (keeping the four-knob invariant). Hold the
learning rate, the algorithm knobs, the staleness bound and the placement at
their production values, because those are what `common/run_identity.sh` turns
into `RL_ALGORITHM`, `CONFIG_TAG`, `CKPT_PATH` and `RUN_NAME`.

The smoke run has to get past three points, in order:

1. `init_tracking` — the wandb group is accepted. This is the first thing that
   touches a derived name, and it fails about 100 seconds in.
2. The placement and checkpoint lines echoed by `run.sbatch` name the directory
   the production run will write to, not a sweep's.
3. One save and one resume, because production spans several jobs.

**Do not chain jobs with `--dependency=afterany` until one link is known to
work.** `afterany` releases the next job on failure too, so a command that dies
during startup walks the whole chain in minutes and reports it as scheduling
activity rather than as a failure. Before creating the chain, also inspect the
resolved `sbatch --export` for every arm and refuse submission unless
`SAVE_HF=1` and `HF_SAVE_INTERVAL` is a positive integer. Verify the first link
actually writes the expected `hf/<rollout_id>` directory before allowing the
rest of the chain to proceed.

## Rules

- One stage, one class of change. A run that moves parallelism and learning rate
  together has produced no usable information.
- Never hand out a command that has not been run once at `interactive` scale
  with its derived variables left to derive.
- `interactive` for correctness, `batch_short` for speed, `batch` for results.
  Never tune on `batch` — it is slower to schedule and the result is not the
  point of that lane.
- Every tuning submission gets its own `CONFIG_TAG`.
- Never submit a dependency chain with HF retention disabled: require
  `SAVE_HF=1`, validate `HF_SAVE_INTERVAL`, and verify one HF export on the first
  link.
- Report measured step times and dump sizes, not estimates, once a stage 2 run
  exists to measure.
