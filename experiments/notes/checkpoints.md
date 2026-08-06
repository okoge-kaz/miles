# Checkpoints

## Layout

Host root: `/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints`

| Directory | In container | Format | Who reads it |
|---|---|---|---|
| `hf/` | `/ckpt/hf` | HuggingFace (safetensors + config + tokenizer) | SGLang engines (`--hf-checkpoint`), tokenizer loading, the converter |
| `megatron/` | `/ckpt/megatron` | Megatron `torch_dist` | trainer (`--ref-load`, and `--load` on a cold start) |
| `training/` | `/ckpt/training` | Megatron `torch_dist` + optimizer state | written by `--save`, read back by `--load` |

Runs land under `training/<task>/<dataset>/<model>/<rl-algorithm>/<placement>/
<policy-regime>/max-weight-staleness-<s>/<config>/` — see
`notes/off-policy-variables.md` for what each level encodes and why.

## Why two formats

Megatron cannot consume a raw HuggingFace directory, and SGLang cannot consume a
`torch_dist` one, so both exist at once:

```
hf/Qwen3-4B  ──convert_hf_to_torch_dist.py──►  megatron/Qwen3-4B_torch_dist
     │                                                  │
     └──► SGLang engines (rollout)                       └──► Megatron actor + reference
```

Keep the HF directory after conversion — the launch scripts still point the
engines at it.

## Conversion

`experiments/setup/convert_checkpoint.sbatch` wraps:

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
  the old run rather than starting a new one. `run.sbatch` defaults `RUN_NAME` to
  include `$SLURM_JOB_ID` so each submission is fresh; set `RUN_NAME` explicitly
  when you *want* to resume.
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
wall-clock is about 98 rollouts, so 20 caps redone work at ~50 min. Before this
flag, getting HF every 10 would have meant `--save-interval 10` and 40 torch_dist
writes -- 2,160 GB/run against 1,080 GB now, or ~311 TB across the 288-run grid.

Disk is not the binding constraint (`/lustre/fsw` has 7.6 PB free). **Offline eval
GPU time is**: ~0.35 node-hours per checkpoint (90 AIME prompts x n=16 at a 32768
budget on one 8-GPU node), so evaluating all 40 exports costs ~14 node-hours
against ~33 node-hours of training. Export at 10 and evaluate coarsely by default,
then refine around the crossing point -- the exports exist so that refinement
never requires retraining.

## Exporting a trained policy

```bash
PYTHONPATH=/root/Megatron-LM python3 tools/convert_torch_dist_to_hf.py \
    --input-dir  /ckpt/training/<run>/iter_XXXXXXX \
    --output-dir /ckpt/hf/<run>-hf
```

Check the exact flags with `--help`; `convert_torch_dist_to_hf_ray.py` is the
multi-node variant for large models.

## Resume verified end to end (2026-08-06)

`experiments/verify_resume.sh` — two chained jobs, phase A cut by the wall clock
at 25 min, phase B resuming into the same `CKPT_PATH`. Jobs 15194552 / 15194553.
This path had never executed before: the throughput probes are all too short to
reach a save, and the 400-rollout reference run is 3-8 chained jobs where every
one of them resumes.

| | phase A (fresh) | phase B (resume) |
|---|---|---|
| `load` | `/ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist` | the run's own `CKPT_PATH` |
| `finetune` | True | **False** |
| `no_load_optim` / `no_load_rng` | True / True | **None / None** |
| resumed from | — | **iteration 1** (A's only save) |
| iterations saved | 1 | **3, 5, 7, 9** |

The optimizer and RNG line is the one that mattered. On the fallback path
(`arguments.py:2758`) miles sets `finetune` / `no_load_optim` / `no_load_rng`,
which is right for a fresh start and would be silently wrong on a resume — Adam
momentum would reset at every 4-hour boundary and nothing would report it. Phase
B shows all three off, so the optimizer state is genuinely restored.

Phase B saving 3,5,7,9 rather than 1,3,5,7 is what proves it continued rather
than restarting; `start_rollout_id` in the argument dump is still `None` at parse
time because miles resolves it from the checkpoint afterwards, so the dump is not
the thing to read.

`--save-retain-interval 4` left only `iter_0000009`, i.e. it pruned every
non-retained iteration as the next one landed and never removed the one the
tracker points at.

**`--hf-save-interval` was exercised here for the first time**: with
`--save-interval 2` and the recipe's `--hf-save-interval 10`, the distributed
checkpoints landed at 3,5,7,9 while `hf/` got a single export. Two cadences from
one run, which is what the flag was added for.

The verbose `CKPT_PATH` also survives resume — phase B loaded
`.../grpo-clip0.2-0.28-tis2.0/async/off-policy/max-weight-staleness-2/resume-test`
without help, so the added directory levels cost nothing at resume time.
