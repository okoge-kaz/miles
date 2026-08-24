# Qwen3-4B RL reasoning evaluation

The shell entry points live in `experiments/scripts/reasoning_eval/`; this
directory contains only their Python helpers, result aggregation, figures, and
documentation.

The evaluator follows the pinned NeMo 26.03 reference protocol:

- vLLM 0.20.2 is imported as SquashFS and serves one Qwen3-4B RL Hugging Face
  checkpoint with tensor parallelism 1 and data parallelism 8.
- NeMo Evaluator/NeMo Skills 26.03 is imported as a second SquashFS image and
  prepares and grades AIME24, AIME25, and AIME26.
- Qwen3 thinking is enabled in the chat template and vLLM uses
  `--reasoning-parser qwen3`. A preflight request requires both parser-separated
  reasoning and final content before evaluation starts.
- Full evaluation uses temperature 0.6, top-p 0.95, top-k 20, 64 repeats, a
  32,768-token context, and at most 28,672 generated tokens.

## Setup

From the Miles repository root, submit and wait for these jobs in order:

```bash
experiments/scripts/reasoning_eval/import-evaluator-images.sbatch
experiments/scripts/reasoning_eval/prepare-aime-data.sbatch
```

Both setup jobs use `cpu_interactive`; no GPU allocation is held while images
or benchmark data are downloaded and prepared.

The default shared locations can be changed with
`REASONING_EVAL_CONTAINER_ROOT`, `REASONING_EVAL_DATA_ROOT`, and
`REASONING_EVAL_CACHE_ROOT`.

## Evaluate the 17-arm staleness sweep

The current sweep namespace defaults to `sr-20260819-212906`. Point
`TRAINING_ROOT` at the owner of the training checkpoints when evaluating another
user's readable output:

```bash
TRAINING_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training \
experiments/scripts/reasoning_eval/submit-staleness-sweep.sh

TRAINING_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training \
experiments/scripts/reasoning_eval/submit-staleness-sweep.sh --submit
```

The scan covers 16 async arms (`max weight staleness = 1,2,4,8` crossed with
trainer:rollout nodes `1:7,2:6,3:5,4:4`) plus the colocated arm, at learning
steps 10 through 300. Miles' HF directory number is zero-based at save time, so
learning steps `10,20,...,300` resolve to directories `9,19,...,299`.

One Slurm job loads each checkpoint once and evaluates all three AIME tasks.
Each task has its own response cache and is finalized atomically with `_SUCCESS`.
Re-running the launcher skips completed task suites and queued/running jobs; a
job interrupted between tasks resumes only unfinished task caches. Use
`--max-submissions N` to change the default eight-job submission wave. Re-run
the same command as jobs finish; use `--max-submissions 0` only when the Slurm
association is allowed to queue every pending checkpoint at once.

For a sweep that is still producing or publishing checkpoints, the refill
controller can continuously validate all 510 candidates and submit newly ready
ones. It also keeps an interactive queue populated by moving only pending jobs;
running evaluations are never canceled or migrated:

```bash
sbatch \
  --export=ALL,TRAINING_ROOT=/path/to/checkpoints/training,RUN_NAMESPACE=sr-20260819-212906,POLL_SECONDS=300,DYNAMIC_QUIESCENT_SCANS=12,BATCH_INFLIGHT_TARGET=0,REGULAR_INFLIGHT_TARGET=490,INTERACTIVE_INFLIGHT_TARGET=20,REGULAR_WALL=04:00:00,INTERACTIVE_WALL=04:00:00,SLURM_JOB_USER="$USER",TRUST_PINNED_SNAPSHOT=0 \
  experiments/scripts/reasoning_eval/refill-snapshot.sbatch
```

With no `SNAPSHOT_ARM_MAX_STEPS`, the controller operates dynamically and does
not require all checkpoints to exist at startup. Once every currently available
checkpoint is evaluated, it waits for 12 unchanged scans by default before
declaring the accessible set complete; a newly completed export resets that
window. Supplying an arm-max-step snapshot retains the fixed-snapshot
availability check. Queue targets accept zero, which is useful when
`batch_short` is unavailable or node-limited. The controller requeues itself
five minutes before its two-day wall limit when work remains. Set
`CONTROLLER_DRY_RUN=1` to perform exactly one read-only scan without canceling,
reassigning, or submitting any evaluation job.

## Results and figures

```bash
experiments/scripts/reasoning_eval/show-results.sh sr-20260819-212906
```

This writes detailed task scores, three-task AIME means, JSON, Markdown, and
two dependency-free SVG figures below the study's `analysis/` directory.
`summarize_results.py` reads the NeMo Skills metric exactly from
`metrics.json[task]["pass@1"]["symbolic_correct"]`; an AIME mean is emitted only
after AIME24/25/26 have all completed for a checkpoint.

For the joined staleness, throughput, and wall-clock analysis, use a Python
environment containing `wandb` and run:

```bash
WANDB_PYTHON=/path/to/python-with-wandb \
experiments/scripts/reasoning_eval/show-staleness-analysis.sh sr-20260819-212906
```

The dependency-driven `finalize-staleness-analysis.sbatch` job exports W&B
history inside `SQSH_IMAGE`, then invokes the same analysis with
`SKIP_WANDB_EXPORT=1`. This keeps the finalizer independent of Python packages
installed on the CPU host while still refreshing W&B immediately before the
final figures are generated.

The command resolves resumed W&B runs into one latest-write-wins training
lineage per arm, joins evaluation step `N` to update `N`, and writes the
following under `analysis/<protocol>/full/staleness/`:

- the selected training history, ten-update score intervals, full correlation
  tables, and an analysis summary;
- AIME24, AIME25, AIME26, and AIME mean trajectories for all 17 settings, once by
  training step and once by active wall-clock;
- an AIME mean wall-clock comparison that separates max-weight-staleness panels and
  trainer:rollout ratios;
- a balanced-common-window, per-setting `dQ/dt = (dQ/dU) × (dU/dt)`
  decomposition separating AIME mean points per update, optimizer-update
  throughput, and AIME mean points per hour;
- a correlation heatmap spanning total, pre-queue, and in-queue mean/variance,
  exact token lag, and within-sample forward-version span;
- downstream-correlation rows for mean, variance, standard deviation, p90, and
  maximum total/pre-queue/in-queue staleness versus ten-update AIME improvement;
- a reduced downstream trajectory figure only when a relationship has
  `|r| >= 0.2` and its arm-cluster bootstrap interval excludes zero.

The plotted wall-clock estimates one uninterrupted training run. It cumulatively
sums `perf/step_time`, but at the first update of each resumed W&B segment it caps
`perf/train_wait_time` at the median of the eight nearest non-boundary updates;
the actual `perf/train_time` remains. Raw cumulative active time, removed resume
overhead, resume-boundary markers, and calendar elapsed time are retained in
`training-history.csv`. Scheduler/requeue downtime is excluded. Correlations are
centered within the same update or ending checkpoint and trainer:rollout ratio,
so they compare realized staleness across max-weight-staleness settings without
confounding the nominal training ratio.

`unpad_vocab.py` creates a job-local vLLM view when a Megatron export retains
padded embedding rows beyond `config.json:vocab_size`. The source checkpoint is
never modified. `export_adapter_cache.py` joins NeMo Evaluator request and
response caches into a readable `model-outputs.jsonl` artifact.
`validate_checkpoint.py` checks every indexed tensor mapping and requires each
safetensors shard's byte length to match the end offset in its header, so a
launcher running alongside training does not enqueue a partially written HF
export. The sweep status reports unreadable shards separately from structurally
incomplete exports, so a `0600` permission issue is not mistaken for corruption.
