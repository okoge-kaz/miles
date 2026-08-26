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
- vLLM compilation state lives in each job's container rather than a shared
  writable compilation cache. This avoids concurrent jobs mutating the same
  TorchInductor artifacts. The evaluation server still uses data parallelism
  8 and compiled execution. vLLM's internal engine deadline is 40 minutes and
  the outer health check allows 45 minutes, so a slow compile cannot be killed
  by the default ten-minute engine timeout. The pinned vLLM release includes
  this operational deadline in its compile-cache hash even though it cannot
  affect a compiled graph. A mounted `sitecustomize.py` keeps that one hash
  factor at the image's 600-second default while preserving the effective
  2,400-second deadline, allowing the image's validated compile artifacts to be
  reused.
- The pinned vLLM `.sqsh` is atomically staged under the job's node-local
  temporary root before the server starts. All server startup attempts in that
  allocation reuse the copy; cross-job reuse is not assumed because Slurm may
  assign a private `SLURM_TMPDIR` to each job. Thus eight vLLM workers do not
  repeatedly fault Python, CUDA, and JIT artifacts through the shared
  filesystem. Before the workers spawn, the runner reads the image's Python
  site-packages once into the node page cache. The overlapping
  vLLM Slurm step explicitly requests 512 GiB: without that step-level request,
  the cluster assigns only 2 GiB even though the parent eight-GPU job owns the
  node's full memory allocation, causing cold nodes to swap and repeatedly read
  the same container pages. The NeMo dry-run and evaluation steps likewise
  request 128 GiB and 512 GiB respectively, so their parallel Python workers do
  not fall back to the same 2 GiB step default.

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

## Evaluate a staleness sweep

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

The high-staleness t1r7 cohort uses the same training namespace printed by the
launcher and includes the non-default queue size in checkpoint identity:

```bash
TRAINING_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training \
STALENESS_LEVELS="16 20 24 28" \
RATIOS="1:7" \
INCLUDE_COLOCATED=0 \
TRAINING_BUFFER_QUEUE_SIZE=6000 \
ASYNC_MAX_CONCURRENT_SAMPLES=4096 \
experiments/scripts/reasoning_eval/submit-staleness-sweep.sh \
  --namespace <training-namespace>
```

Add `--submit` only after inspecting the dry-run. The first submitted evaluation
writes `grid.env` under the result study; refill, summarization, and plotting
load it automatically, so later commands only need the namespace. The legacy
17-arm grid above remains the default when no cohort configuration is supplied.

One Slurm job loads each checkpoint once and evaluates all three AIME tasks.
Each task has its own response cache and is finalized atomically with `_SUCCESS`.
Re-running the launcher skips completed task suites and queued/running jobs; a
job interrupted between tasks resumes only unfinished task caches. The
evaluator also retries a transient task-process failure up to three times
inside the same allocation, reusing the partial response cache on each attempt.
Use
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

Exact Adam update norms were not logged during training. To add an offline,
zero-training-overhead proxy, compute the observed net parameter displacement
between adjacent evaluated Hugging Face checkpoints. Submit one CPU array task
per sweep arm, then merge the resumable per-arm files:

```bash
ANALYSIS_ROOT=/path/to/analysis/protocol/full \
STUDY_ROOT=/path/to/grpo-clip0.2-0.28-tis2.0 \
RUN_NAMESPACE=sr-20260819-212906 \
SEED_PARTS_ROOT=/path/to/earlier/staleness/checkpoint-displacements-parts \
sbatch --partition=cpu_interactive --array=0-16%4 \
  --export=ALL,ANALYSIS_ROOT,STUDY_ROOT,RUN_NAMESPACE,SEED_PARTS_ROOT \
  experiments/scripts/reasoning_eval/compute-checkpoint-displacements.sbatch

ANALYSIS_ROOT=/path/to/analysis/protocol/full \
STUDY_ROOT=/path/to/grpo-clip0.2-0.28-tis2.0 \
RUN_NAMESPACE=sr-20260819-212906 \
MERGE_ONLY=1 \
sbatch --dependency=afterok:<array-job-id> \
  --export=ALL,ANALYSIS_ROOT,STUDY_ROOT,RUN_NAMESPACE,MERGE_ONLY \
  experiments/scripts/reasoning_eval/compute-checkpoint-displacements.sbatch
```

`net_parameter_displacement_per_update` is
`||theta_end - theta_start||_2 / 10`. It is a lower-bound net displacement and
must not be interpreted as the mean Adam update norm or cumulative update path;
opposing update directions can cancel between checkpoints. The Hugging Face
weights are BF16, so serialization precision also limits this checkpoint-space
proxy; it is not an exact reconstruction of FP32 optimizer updates. The current
4B checkpoint scan takes roughly two minutes per ten-update interval on the CPU
filesystem, but it does not touch the training process. `SEED_PARTS_ROOT` is
optional; when supplied, a row is reused only if its study root, namespace,
checkpoint paths, interval, and required numeric outputs match the current
request. Older seed files without that provenance are safely recomputed. The
analysis consumes the merged CSV only after the merge job atomically publishes
`checkpoint-displacements._SUCCESS`.

The command resolves resumed W&B runs into one latest-write-wins training
lineage per arm, joins evaluation step `N` to update `N`, and writes the
following under `analysis/<protocol>/full/staleness/`:

The result study's `grid.env` controls the arm set used by summarization and
plotting. Async W&B groups with a `cN` concurrency identity are normalized to
their corresponding `sN-tNrN` arm. Partial colocated variants such as
`s0-colocated-partial-o256` remain intentionally rejected: they must not be
collapsed into the fully colocated baseline with structural staleness zero.

- the selected training history, ten-update score intervals, full correlation
  tables, and an analysis summary;
- AIME24, AIME25, AIME26, and AIME mean trajectories for every configured setting, once by
  training step and once by active wall-clock;
- an AIME mean wall-clock comparison that separates max-weight-staleness panels and
  trainer:rollout ratios;
- a configured-grid W&B heatmap of the realized late-window
  `staleness/total/mean`, with max weight staleness as rows and
  train:rollout node ratios as columns. Each cell reports the mean and
  population standard deviation over the trailing 50 contiguous optimizer
  updates (normally steps 251--300), and marks a setting as still changing
  when its fitted drift across that window exceeds the larger of one standard
  deviation, 10% of the mean, and 0.05. The exact window, slope, tolerance,
  and status are written to `steady-state-staleness.csv`;
- per-update W&B `staleness/total/mean` trajectories in one panel per configured
  train:rollout ratio, showing both the raw value and a trailing 10-update mean
  for every configured max weight staleness;
- a W&B time-series matrix for training metrics whose strongest correlation
  with a realized-staleness predictor has `|r| >= 0.25`. Signed `train/tis` is
  included as a reference even when its mean stays near one, so cancellation
  in signed TIS can be compared with `train/tis_abs` and
  `train/tis_clipfrac`. The exact metric selection and strongest predictor are
  recorded in `staleness-sensitive-metrics.csv`;
- a balanced-common-window, per-setting `dQ/dt = (dQ/dU) × (dU/dt)`
  decomposition separating the AIME mean linear trend per update,
  optimizer-update throughput, AIME mean points per hour, and training-data
  staleness mean. The learning-effect term is an OLS slope over every evaluated
  checkpoint in the common window rather than a noise-sensitive endpoint
  difference. The colocated on-policy baseline is shown as zero even though it
  does not log the async staleness namespace;
- a standalone `optimizer-update-throughput-by-setting.svg` comparison of
  `dU/dt` for all configured settings. Each row reports both optimizer updates per
  active hour and the reciprocal active seconds per update, using the same
  balanced common window and resume-adjusted active clock as the decomposition;
- a correlation heatmap spanning total, pre-queue, and in-queue mean/variance,
  exact token lag, and within-sample forward-version span. When
  `checkpoint-displacements.csv` is present, it also includes the observed net
  parameter displacement between adjacent 10-step Hugging Face checkpoints;
- the heatmap excludes cohort useful efficiency, wasted-token fraction, step
  time, and useful tokens per second because scheduling and the configured
  staleness bound jointly determine them;
- downstream-correlation rows for mean, variance, standard deviation, p90, and
  maximum total/pre-queue/in-queue staleness, plus exact token lag and
  within-sample forward-version span, versus ten-update AIME improvement. Every
  predictor is averaged over the same ten-update interval as its score change;
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
never modified. All shards in that runtime view are materialized on job-local
storage before the eight data-parallel vLLM workers start, avoiding repeated
shared-filesystem reads from each worker. `export_adapter_cache.py` joins NeMo
Evaluator request and response caches into a readable `model-outputs.jsonl`
artifact.
`validate_checkpoint.py` checks every indexed tensor mapping and requires each
safetensors shard's byte length to match the end offset in its header, so a
launcher running alongside training does not enqueue a partially written HF
export. The sweep status reports unreadable shards separately from structurally
incomplete exports, so a `0600` permission issue is not mistaken for corruption.
