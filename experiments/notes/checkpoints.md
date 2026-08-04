# Checkpoints

## Layout

Host root: `/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints`

| Directory | In container | Format | Who reads it |
|---|---|---|---|
| `hf/` | `/ckpt/hf` | HuggingFace (safetensors + config + tokenizer) | SGLang engines (`--hf-checkpoint`), tokenizer loading, the converter |
| `megatron/` | `/ckpt/megatron` | Megatron `torch_dist` | trainer (`--ref-load`, and `--load` on a cold start) |
| `training/` | `/ckpt/training` | Megatron `torch_dist` + optimizer state | written by `--save`, read back by `--load` |

One subdirectory per run under `training/`, named by `RUN_NAME`
(`training/math-grpo-qwen3-4b-<jobid>/`).

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

## Exporting a trained policy

```bash
PYTHONPATH=/root/Megatron-LM python3 tools/convert_torch_dist_to_hf.py \
    --input-dir  /ckpt/training/<run>/iter_XXXXXXX \
    --output-dir /ckpt/hf/<run>-hf
```

Check the exact flags with `--help`; `convert_torch_dist_to_hf_ray.py` is the
multi-node variant for large models.
