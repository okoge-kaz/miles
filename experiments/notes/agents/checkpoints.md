# Work log — checkpoints

## 2026-08-03 — layout decision and conversion mechanics

### Layout

Requested split under `/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints`:

```
hf/         HuggingFace weights          → /ckpt/hf
megatron/   torch_dist (Megatron)        → /ckpt/megatron
training/   checkpoints written by runs  → /ckpt/training
```

This differs from the upstream examples, which keep everything under `/root` and
place `<model>_torch_dist` next to the HF directory. Because of that, our recipes
pass paths explicitly rather than reusing `examples/retool_v2/run_retool_multi_turn.py`
— that launcher marks `hf_checkpoint` / `ref_load` as `init=False`
(`run_retool_multi_turn.py:29-38`), so they cannot be set from the CLI and are
hard-wired to `/root/models/...`. Writing our own `train.sh` avoided a symlink
hack inside the checkpoint tree.

### Conversion

`tools/convert_hf_to_torch_dist.py` with the architecture description expanded
from `scripts/models/<type>.py` into `MODEL_ARGS`. The helper
`miles/utils/external_utils/command_utils.py:33` shows the canonical invocation
and the skip condition:

```python
tracker = Path(path_dst) / "latest_checkpointed_iteration.txt"
if tracker.exists() and tracker.read_text().strip() == "release":
    return   # already converted
```

So re-running `convert_checkpoint.sbatch` is cheap and idempotent. Conversion is
launched with `torchrun --nproc-per-node 8`, hence the GPU job.

The HF directory stays in use after conversion: SGLang serves from it
(`--hf-checkpoint`), and the tokenizer/processor are loaded from it
(`data_source.py:60`).

### Resume

`--load` and `--save` point at the same `training/<RUN_NAME>` directory, which is
how the upstream scripts work too — relaunching resumes. `run.sbatch` therefore
defaults `RUN_NAME` to include `$SLURM_JOB_ID`, so a resubmission does not
silently continue a previous run with different hyperparameters. Set `RUN_NAME`
explicitly to resume on purpose.

`--save-interval` counts rollouts, not optimizer steps.

### Not verified

- No conversion has been run yet; disk footprint of
  `Qwen3-4B_torch_dist` and of a `training/` directory (weights + optimizer
  state) is unmeasured. Check before sweeping many runs — quota is shared.
- `tools/convert_torch_dist_to_hf.py` flags were not confirmed against `--help`;
  `notes/checkpoints.md` says as much.
