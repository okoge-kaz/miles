# Qwen3-4B RL checkpoint evaluation

The shell entry points live in `experiments/scripts/reasoning_eval/`; this
directory contains only their Python helpers, result aggregation, figures, and
documentation.

There are two benchmark-specific entry points in the same hierarchy:

- `run-evaluation.sbatch` is the reportable AIME24/25/26 path and follows the
  pinned NeMo 26.03 reference protocol described below.
- `run-suite.sbatch` serves the checkpoint once, then generates and scores
  MATH-500, GPQA, LiveCodeBench, and IFBench (and can run lightweight AIME
  diagnostics). Its scorer is `suite.py`; code execution is isolated with an
  unroutable user namespace and Bubblewrap. `score-suite.sbatch` can rescore
  existing candidates without starting a GPU server.

They share the reasoning-evaluation location and checkpoint validation helpers,
but do not pretend that the NeMo AIME metric and the benchmark-specific suite
scorers are interchangeable.

The reportable AIME evaluator follows the pinned NeMo 26.03 reference protocol:

- vLLM 0.20.2 is imported as a Singularity SIF and serves one Qwen3-4B RL
  Hugging Face checkpoint with tensor parallelism 1 and data parallelism 8.
- NeMo Evaluator/NeMo Skills 26.03 is imported as a second SIF and
  prepares and grades AIME24, AIME25, and AIME26.
- The checked-in default targets `Qwen3-4B-Base-LR2e-5-Step4000` and uses
  `ENABLE_THINKING=true`. vLLM uses `--reasoning-parser qwen3`, and the preflight
  requires parser-separated reasoning plus non-empty final content. Set
  `ENABLE_THINKING=false` explicitly only when evaluating a non-thinking
  checkpoint; that mode still requires non-empty final content.
- Full evaluation uses temperature 0.6, top-p 0.95, top-k 20, 64 repeats, a
  32,768-token context, and at most 28,672 generated tokens.

## Setup

From the Miles repository root, submit and wait for these jobs in order:

```bash
source experiments/env.sh
source experiments/common/pbs.sh

images_job="$(pbs_submit --parsable --profile=cpu \
  --time="${PBS_CONTAINER_WALLTIME}" \
  experiments/scripts/reasoning_eval/import-evaluator-images.sbatch)"
pbs_submit --profile=cpu --time="${PBS_PREP_WALLTIME}" \
  --dependency="afterok:${images_job}" \
  experiments/scripts/reasoning_eval/prepare-aime-data.sbatch
```

Both setup jobs use the CPU-only reservation profile: they request CPUs but no
GPUs while images or benchmark data are downloaded and prepared.
An existing preparation marker may contain additional benchmarks, but all three
AIME datasets, their 30 unique records, evaluator image, and any recorded
checksums must validate before it is reused.

The default shared locations can be changed with
`REASONING_EVAL_CONTAINER_ROOT`, `REASONING_EVAL_DATA_ROOT`, and
`REASONING_EVAL_CACHE_ROOT`.

## Evaluate the 17-arm staleness sweep

The current sweep namespace defaults to `sr-20260819-212906`. Point
`TRAINING_ROOT` at the owner of the training checkpoints when evaluating another
user's readable output:

```bash
TRAINING_ROOT="${TRAIN_CKPT_DIR}" \
experiments/scripts/reasoning_eval/submit-staleness-sweep.sh

TRAINING_ROOT="${TRAIN_CKPT_DIR}" \
experiments/scripts/reasoning_eval/submit-staleness-sweep.sh --submit
```

The scan covers 16 async arms (`max weight staleness = 1,2,4,8` crossed with
trainer:rollout nodes `1:7,2:6,3:5,4:4`) plus the colocated arm, at learning
steps 10 through 300. Miles' HF directory number is zero-based at save time, so
learning steps `10,20,...,300` resolve to directories `9,19,...,299`.

One PBS job loads each checkpoint once and evaluates all three AIME tasks.
Each task has its own response cache and is finalized atomically with `_SUCCESS`.
Re-running the launcher skips completed task suites and queued/running jobs; a
job interrupted between tasks resumes only unfinished task caches. Use
`--max-submissions N` to change the default eight-job submission wave. Re-run
the same command as jobs finish; use `--max-submissions 0` only when PBS queue
policy allows every pending checkpoint to be queued at once.

The protocol name is derived from the effective repeat count. For example,
`AIME_REPEATS=8` produces an `aime8` protocol rather than reusing the default
`aime64` identity; smoke evaluation uses `aime1`. An explicit `PROTOCOL_NAME`
must contain the same repeat tag. `evaluation-contract.env` prevents a result
root from being reused with a different checkpoint, dataset, image, or sampling
configuration, and each completed task has a checked `artifact-manifest.sha256`.

For a post-training evaluation, submit `run-after-training.sbatch` with
`TRAINING_HF_ROOT` and a new `RESULT_ROOT`. It selects the newest structurally
complete numeric Hugging Face export and records that choice atomically before
running the ordinary evaluator.

Submit `refill-snapshot.sbatch` through `pbs_submit --profile=cpu`. Set
`REFILL_ONCE=1` to execute one controller iteration, which is useful for a
controlled validation of queue routing.

## Results and figures

```bash
experiments/scripts/reasoning_eval/show-results.sh sr-20260819-212906
```

This writes detailed task scores, three-task macro means, JSON, Markdown, and
two dependency-free SVG figures below the study's `analysis/` directory.
`summarize_results.py` reads the NeMo Skills metric exactly from
`metrics.json[task]["pass@1"]["symbolic_correct"]`; a macro mean is emitted only
after AIME24/25/26 have all completed for a checkpoint.

`unpad_vocab.py` creates a job-local vLLM view when a Megatron export retains
padded embedding rows beyond `config.json:vocab_size`. The source checkpoint is
never modified. `export_adapter_cache.py` joins NeMo Evaluator request and
response caches into a readable `model-outputs.jsonl` artifact.
`validate_checkpoint.py` checks every indexed tensor mapping and requires each
safetensors shard's byte length to match the end offset in its header, so a
launcher running alongside training does not enqueue a partially written HF
export. The evaluator also records a SHA-256 manifest over every indexed shard
and required tokenizer/configuration file.
