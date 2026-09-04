# Checkpoints

## Layout

Host root: `/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints`

| Directory | In container | Format | Who reads it |
|---|---|---|---|
| `huggingface/` | `/ckpt/hf` | HuggingFace (safetensors + config + tokenizer) | SGLang engines (`--hf-checkpoint`), tokenizer loading, the converter |
| `megatron/` | `/ckpt/megatron` | Megatron `torch_dist` | trainer (`--ref-load`, and `--load` on a cold start) |
| `training/` | `/ckpt/training` | Megatron `torch_dist` + optimizer state | written by `--save`, read back by `--load` |

Runs land under `training/<task>/<dataset>/<model>/<rl-algorithm>/<placement>/
<policy-regime>/<staleness-tag>/<config>/`. For `queue-recycle`, the tag is
`max-weight-staleness-<s>` with `-from-<reference>` appended when the reference
is not `completion`; `queue-max` uses
`queue-max/max-weight-staleness-<s>-from-prefill`. See
[off-policy-variables.md](off-policy-variables.md) for what each level encodes
and why.

## Why two formats

Megatron cannot consume a raw HuggingFace directory, and SGLang cannot consume a
`torch_dist` one, so both exist at once:

```
huggingface/Qwen3-4B  ──convert_hf_to_torch_dist.py──►  megatron/Qwen3-4B_torch_dist
     │                                                  │
     └──► SGLang engines (rollout)                       └──► Megatron actor + reference
```

Keep the HF directory after conversion — the launch scripts still point the
engines at it.

## SFT baselines staged on aws-pdx

The three SFT baselines used for DAPO-Math difficulty measurement were converted
successfully on 2026-08-21. Each directory has a
`latest_checkpointed_iteration.txt` containing `release`.

| Model | Megatron directory | Size |
|---|---|---:|
| Qwen3 4B | `megatron/Qwen3-4B-Base-LR2e-5-Step4000_torch_dist` | 7.5 GiB |
| Qwen3 8B | `megatron/Qwen3-8B-Base-LR1.5e-5-Step4000_torch_dist` | 16 GiB |
| Qwen3 30B-A3B | `megatron/Qwen3-30B-A3B-Base-LR2e-5-Step4000_torch_dist` | 57 GiB |

The exact HF source roots and model-argument scripts are recorded in
`experiments/setup/manifests/sft_checkpoints.txt`; use
`experiments/setup/models/stage_sft_checkpoints.sh` to validate or restage them.

## Conversion

`experiments/setup/models/convert_checkpoint.sbatch` wraps:

```bash
source scripts/models/qwen3-4B.sh          # MODEL_ARGS: layers, hidden size, rotary base, …
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
    tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} \
    --hf-checkpoint /ckpt/hf/Qwen3-4B \
    --save /ckpt/megatron/Qwen3-4B_torch_dist
```

- The `MODEL_ARGS` file must match the model. `scripts/models/` holds one per
  architecture; a mismatch produces wrong-shaped weights rather than a clean error.
- Success is recorded in `latest_checkpointed_iteration.txt` containing `release`.
  miles' own helper (`convert_checkpoint` in
  `miles/utils/external_utils/command_utils.py:33`) skips the conversion when it
  sees that marker, so re-running is cheap.
- Other converters in `tools/`: `convert_torch_dist_to_hf.py` (export a trained
  policy back to HF), `convert_fsdp_to_hf.py`, and quantization variants
  (`convert_hf_to_fp8.py`, `_nvfp4`, `_int4`).

## Resume semantics

`--load` and `--save` point at the same directory in our scripts. Relaunching the
same job resumes from the last saved iteration; there is no separate resume flag.
Consequences worth remembering:

- Changing hyperparameters and relaunching with the same `RUN_NAME` **continues**
  the old run only when they still derive the same `CKPT_PATH`. The maintained
  recipes derive a deterministic `RUN_NAME` and checkpoint path from the
  training identity; the Slurm job id is deliberately not part of either one.
  Re-submit the same recipe with the same overrides and leave
  `CLEAN_CHECKPOINT=0` to resume. Changing an identity-bearing setting, including
  the maintained recipes' `NUM_ROLLOUT`, selects a new checkpoint directory.
- `--save-interval` is in rollouts, not optimizer steps.
- Optimizer state is saved alongside the weights, so these directories are
  several times the size of the model. Watch quota when sweeping.

## Two checkpoint cadences: `--save-interval` and `--hf-save-interval`

The two artifacts serve different consumers and were previously forced onto one
cadence -- `save_hf_model` was called inside `save_model` after the distributed
save, and the whole path sat behind a single `should_run_periodic_action` on
`--save-interval`. `--hf-save-interval` (added 2026-08-05) separates them:

| Artifact | Consumer | Size, measured | Pruned by `--save-retain-interval` |
|---|---|---|---|
| torch_dist (`--save`) | resume after a 4h preemption | 54 GB | yes |
| HF (`--save-hf`) | offline eval, i.e. the `Q(t)` series | 7.6 GB | **no** -- it lands outside `--save` |

`checkpoint_artifacts_due` (`miles/utils/misc.py`) returns `(write_dist,
write_hf)`; with `--hf-save-interval` unset the two are identical, so the old
behaviour is the default. An HF-only save skips the rollout-buffer save and the
post-save hook, both of which only mean something next to a resumable checkpoint.
The external save sentinel forces the *distributed* checkpoint only -- it exists
to make a run resumable on demand, not to export weights.

### Why 10 for this study

`--hf-save-interval` sets the time resolution of `tau_m(delta)`, because the
offline eval of the HF exports *is* the `Q(t)` series. `tau` is quantised to
multiples of the export interval `h`, so `S = tau_on / tau_m` carries a relative
error of about `h/tau_on + h/tau_m`. At 400 rollouts, with a converged on-policy
reference near 300 and a 2x-accelerated arm near 150:

| `h` | error on `S` | HF exports/run | dist writes/run |
|---|---|---|---|
| 20 | ~20% | 20 | 20 |
| **10** | **~10%** | **40** | **20** |
| 5 | ~5% | 80 | 20 |

At `h=20` the quantisation dominates the bootstrap CI, which defeats the point of
computing one. `h=10` puts it below the CI. `h=5` buys little more and doubles a
cost that is *not* disk: every export is synchronous, so it lands on training
wall-clock, and wall-clock is this study's primary metric. Worse, a fixed
per-export cost is a larger fraction of a fast arm's step time than a slow one's,
so it biases `S` in the direction of the effect being measured -- roughly 1% on a
32k-response arm against 3% on a 4k arm at `h=10`, and double that at `h=5`.

`--save-interval` stays at 20. It is sized by preemption, not by analysis: 4h of
wall-clock is about 98 rollouts, so 20 caps redone work at ~50 min.

The current `math/async/.../run.sbatch` defaults to `SAVE_INTERVAL=10`,
`SAVE_RETAIN_INTERVAL=100`, and `HF_SAVE_INTERVAL=10`, matching the sizing
argument above. Historical convergence-sweep jobs used `HF_SAVE_INTERVAL=5`
(verified on job 15319206's command line), so do not reinterpret those runs as
h=10. Before this
flag, getting HF every 10 would have meant `--save-interval 10` and 40 torch_dist
writes -- 2,160 GB/run against 1,080 GB now, or ~311 TB across the 288-run grid.

Do not use a free-space number copied into this note for capacity planning; it
becomes stale immediately. Check the filesystem quota and free space at
submission time. Offline-evaluation GPU time and synchronous export time also
remain part of the experiment cost, so export coarsely by default and refine
around the crossing point from already exported checkpoints.

## Exporting a trained policy

```bash
PYTHONPATH=/root/Megatron-LM python3 tools/convert_torch_dist_to_hf.py \
    --input-dir  /ckpt/training/<run>/iter_XXXXXXX \
    --output-dir /ckpt/hf/<run>-hf
```

Check the exact flags with `--help`; `convert_torch_dist_to_hf_ray.py` is the
multi-node variant for large models.

## What counts as resume validation

There is no maintained `experiments/verify_resume.sh`; older notes that named it
described a deleted one-off launcher. A current recipe is considered validated
for chained four-hour jobs only after this two-job check succeeds:

1. A fresh `batch`/`interactive`-QoS job runs at least one real
   forward/backward optimizer update and writes a distributed checkpoint. It
   need not consume the production
   `NUM_ROLLOUT` schedule or run for four hours.
2. A second job uses the same identity and `CLEAN_CHECKPOINT=0`, loads the saved
   training checkpoint, restores optimizer and RNG state, and advances the
   iteration again. Loading iteration 0 is valid when the first job performed
   optimizer step 0 and then saved it.
3. When replay is enabled, the second job must also restore a matching replay
   artifact and report its restore telemetry. A model checkpoint alone does not
   validate replay resume.

The checked-in recipe contract is covered by
`tests/fast/experiments/test_domain_training_recipes.py`: load and save share one
path, the identity is stable across Slurm job ids, and a changed rollout schedule
gets a different identity. That static test prevents path regressions but does
not replace the two GPU jobs above. Historical fresh/resume jobs may still be
useful evidence for Miles' checkpoint machinery; if their custom reward or
generator import path has since been removed, they are not evidence that the
current environment recipe resumes.

The current IFEvalG recipe has passed this gate. Job 306686 performed optimizer
step 0 and published the iteration-0 checkpoint plus `replay_buffer_0`; job
306687 reused the same identity, restored iteration 0, performed optimizer step
1, and published iteration 1 plus `replay_buffer_1`. Both four-node jobs used
16K responses, 16 samples per prompt, EFA, and exited successfully.
`BrokenPipeError` messages printed during final process teardown only after each
job's checkpoint had been published; they are shutdown noise, not a failed
resume.

The current Code recipe has also passed the same gate at revision `a6dcaaf1`.
Job 306787 performed optimizer step 0 and published iteration 0 plus
`replay_buffer_0`. Job 306788 reused the identity, loaded model iteration 0, and
restored that replay artifact in 0.459 seconds (six pending groups, four
inflight groups / 397,991 inflight tokens, and one prepared batch). It then
performed optimizer step 1 and published iteration 1 plus `replay_buffer_1`.
Both jobs completed with exit code 0. Teardown `BrokenPipeError` messages came
after durable publication and do not invalidate the resume result.

The current STEM recipe at revision `82bfd482` has passed the gate as well. Job
306790 exited 0 after optimizer step 0 and publication of iteration 0 plus
`replay_buffer_0`. Same-identity job 306792 loaded model iteration 0, restored
that replay state, performed optimizer step 1, and published iteration 1 plus
`replay_buffer_1` before exiting 0.

The current Math recipe passed a reduced-batch resume smoke in jobs
307062/307063. The first trained steps 0 and 1; the second loaded iteration 1,
restored 8 pending and 4 inflight groups plus one prepared batch in 0.098
seconds, and advanced through iteration 3. It exercised the 16K response cap and
current checkpoint identity but reduced `n` and the rollout/global batch sizes;
the checked-in production defaults remain n=16.

The current Math+Code+STEM recipe passed the production-shaped n=16 replay gate
in jobs 306793/306796. The second loaded iteration 0, restored 15 pending, 3
ready, and 6 inflight groups plus one prepared batch in 0.868 seconds, performed
optimizer step 1, and published iteration 1 plus `replay_buffer_1`.

Tau local-policy runtime jobs 307433/307434 are historical replay evidence only.
The maintained agentic recipe now uses conversational tool-use Pivot training
and holds Tau three out exclusively for downstream evaluation.
