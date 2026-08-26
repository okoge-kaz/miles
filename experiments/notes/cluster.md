# aws-pdx cluster notes

Measured on 2026-08-21 from `aws-pdx-slurm-1-vscode-02`. Re-run the commands
below after a cluster maintenance window instead of carrying assumptions from
cw-dfw.

## Account, partitions, and QoS

Slurm account: `coreai_horizon_dilations`.

The important migration detail is that scheduling class is split across two
fields on aws-pdx. `interactive` and `cpu-interactive` are **QoS names**, not
partitions.

| workload | partition | QoS | tested wall time |
|---|---|---|---|
| GPU smoke / conversion | `batch` | `interactive` | 1--4 h |
| GPU production | `batch` | `normal` | up to 4 h |
| GPU long pass-rate sweep | `batch_long` | `normal` | up to 7 d |
| CPU setup fast lane | `cpu` | `cpu-interactive` | setup-job scale |
| CPU regular | `cpu` | `cpu-normal` | up to 7 d |
| CPU long QoS | `cpu` | `cpu-long` | use only when needed |

All six combinations used by the recipes were accepted by `sbatch
--test-only` on 2026-08-21. In particular, do not write `-p interactive` or
`-p cpu_interactive`; Slurm rejects those because no such partitions exist.

Current partitions from `sinfo` / `scontrol show partition`:

| partition | max time | nodes | advertised GRES |
|---|---:|---:|---|
| `batch` | 4 h | 207 | `gpu:8` |
| `batch_long` | 7 d | 207 | `gpu:8` |
| `cpu` | 7 d | 350 | none |
| `cpu_datamover` | unlimited | 350 | none |

The account association currently allows these QoS values:

```
cpu-datamover,cpu-free,cpu-interactive,cpu-long,cpu-normal,cpu-short,
free,free-admin,hero-res,interactive,normal,preemptable-res,short
```

Check the live state with:

```bash
sinfo -h -o '%P|%a|%l|%D|%G'
sacctmgr -n -P show assoc user="$USER" \
  format=Cluster,Account,User,Partition,QOS,DefaultQOS
sbatch --test-only -A coreai_horizon_dilations -p batch \
  --qos=interactive --time=01:00:00 --nodes=1 --gres=gpu:8 <script>
```

The GPU nodes advertise 192 CPUs and eight GPUs through the partition defaults.
The interactive preflight measured eight `NVIDIA B300 SXM6 AC` devices with
275040 MiB each, driver 580.126.09, and CUDA compute capability 10.3. Inside the
selected image, PyTorch 2.11.0+cu130 reported CUDA 13.0; a BF16 CUDA matmul and
the FlashAttention, Transformer Engine, SGLang, and Ray imports all passed.
Re-run the preflight after changing the image or after a cluster maintenance
window rather than treating the filename as qualification evidence.

For Megatron training, the aws-pdx recipes default to Transformer Engine's
`fused` attention backend through `TRAINING_ATTENTION_BACKEND=fused`. In the
end-to-end B300 gate, `flash` entered the FlashAttention/CuTe SM100 backward
kernel but made no optimizer progress, while `fused` completed the same actor
update in 13 seconds. An import check therefore does not qualify the `flash`
training backend on this image.

## Filesystem and assets

`/lustre/fsw/portfolios/coreai/users/kfujii` is a symlink to this project's
per-user directory under `/scratch/fsw/portfolios/coreai/projects/`. The recipes
use the stable `/lustre/fsw/...` spelling.

| asset | host path |
|---|---|
| datasets | `/lustre/fsw/portfolios/coreai/users/kfujii/datasets` |
| Hugging Face checkpoints | `.../checkpoints/huggingface` |
| Megatron checkpoints | `.../checkpoints/megatron` |
| training outputs | `.../checkpoints/training` |
| containers | `.../containers` |
| persistent caches | `.../cache` |

The current Miles image is
`containers/miles-search-r1-b300-20260815.sqsh`. `experiments/env.sh` mounts
the active checkout over `/root/miles`, so repository edits are visible without
rebuilding the image.

## Interactive validation convention

An interactive validation allocation means partition `batch` plus QoS
`interactive`. During bring-up, export `WANDB_MODE=offline`; the math recipes
allow this without requiring an API key and pass the mode into Ray workers.

```bash
salloc -A coreai_horizon_dilations -p batch --qos=interactive \
  --nodes=1 --gres=gpu:8 --time=01:00:00
```

Inside the allocation, validate in order: `nvidia-smi`, container GPU visibility
and CUDA imports, a short SGLang generation, checkpoint conversion/load, then a
single Miles optimizer update. A successful import alone is not an end-to-end
Miles validation. The 2026-08-21 qualification used `batch` + `interactive`
with `WANDB_MODE=offline` and completed rollout, one optimizer update,
torch_dist save, and post-update SGLang weight sync with the `fused` backend.
