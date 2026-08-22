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
export.
