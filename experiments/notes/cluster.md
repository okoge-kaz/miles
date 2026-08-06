# cw-dfw cluster notes

Everything here was measured on 2026-08-03 (see `notes/agents/cluster.md` for the
raw commands and outputs).

## Account and partitions

Slurm account: `coreai_horizon_dilations` (required — jobs without `-A` are
rejected with "You forgot to specify which account you want to use").

| Partition | Time limit | Nodes | GRES | CPUs | Memory |
|---|---|---|---|---|---|
| `batch` (default) | 4 h | 1850 | `gpu:8` | 128 | 2 TB |
| `batch_short` | 2 h | 1850 | `gpu:8` | 128 | 2 TB |
| `batch_long` | 8 h | 1850 | `gpu:8` | 128 | 2 TB |
| `batch_large_long` | 14 d | 1850 | `gpu:8` | 128 | 2 TB |
| `backfill` | 7 d | 1850 | `gpu:8` | 128 | 2 TB |
| `interactive` | 4 h | 1852 | `gpu:8` | 128 | 2 TB |
| `cpu` | 1 d | 80 | — | 96 | 252 GB |
| `cpu_short` | 4 h | 80 | — | 96 | 252 GB |
| `cpu_long` | 7 d | 80 | — | 96 | 252 GB |
| `cpu_interactive` | 1 d | 38 | — | 96 | 252 GB |
| `cpu_datamover` / `cpu_dataprocessing` | ∞ | 38 / 46 | — | 96 | 252 GB |

GPU nodes are `pool0-*`, feature `H100`, 8 GPUs each. Partition defaults:
`DefCpuPerGPU=16`, `DefMemPerGPU=250000`, default wall time 31 min.

### What this account may actually use

`batch_long`, `batch_large` and `batch_large_long` are **rejected** for
`coreai_horizon_dilations` ("Invalid account or account/partition combination").
Verified with `sbatch --test-only`. Usable partitions, by scheduling priority:

| Partition | PriorityTier | Max time | Use for |
|---|---|---|---|
| `cpu_interactive` | 13 | 1 d | CPU work: image import, downloads |
| `interactive` | 12 | 4 h | GPU verification runs, conversion, smokes |
| `batch_short` / `cpu_short` | 11 | 2 h / 4 h | |
| `batch` / `cpu` | 10 | 4 h / 1 d | long real runs |
| `backfill` | 9 | 7 d | only when >4 h in one allocation is unavoidable |

Prefer the highest tier the limits allow — the difference is large in practice:
on 2026-08-03 a trivial job was scheduled immediately on `cpu_interactive` but
**2 h 45 m later** on `cpu`.

Runs longer than 4 h are handled by resuming rather than by a longer partition:
`--load`/`--save` point at the same directory, so a follow-up job with the same
`RUN_NAME` continues where the previous one stopped.

### Partition QOS, and why the recipes say `--partition=batch` alone

Behind each GPU partition sits a partition QOS with its own caps:

| partition | QOS | GrpTRES (all users) | MaxTRESPU (per user) |
|---|---|---|---|
| `batch` | `p_batch` | unlimited | node=768 |
| `batch_short` | `p_batch_short` | node=20 | **node=4** |

Two reasons the production recipes name only `batch`, the first of which is a
hard failure rather than a slowdown:

1. **The 4-node per-user cap is enforced at submission.**
   `sbatch --partition=batch,batch_short --nodes=5` is *rejected outright* with
   `QOSMaxNodePerUserLimit` — it does not fall back to `batch`. With the async
   node balance heading past 4 nodes, listing `batch_short` in a production
   recipe would stop the sweep before it started.
2. `batch_short` maxes out at 2 h against the recipes' 4 h, so it could never
   take a production job anyway.

For short verification runs at ≤4 nodes, add it back at submit time, where it
does help: `sbatch --partition=batch,batch_short --time=00:40:00 --nodes=3 ...`.

For a multi-partition job the pending reason names whichever partition was
evaluated last, so `QOSGrpNodeLimit`/`QOSMaxNodePerUserLimit` naming
`batch_short` is **not** proof the job is locked out of `batch`. And the per-user
cap serialises verification work: one running 3-node job leaves only 1 node of
your own `batch_short` allowance, so sibling jobs queue behind it.

### Editing recipes while jobs are queued

`sbatch` snapshots `run.sbatch` at submit time, but `train.sh` is read live from
the lustre mount when the job starts, and the snapshotted `run.sbatch` resolves
the *path* to `train.sh` at run time too.

- Adding a variable to `train.sh` breaks every already-queued job that predates
  the matching default in `run.sbatch` (seen as
  `--save-interval: invalid int value: ''`). Guard new flags:
  `if [[ -n "${VAR:-}" ]]; then ARGS+=(--flag "${VAR}"); fi`.
- **Renaming a recipe directory kills every queued and running job.** The
  `dapo-math-p10-80` → `dapo-math-p10-90` rename took out four at once with
  `train.sh: No such file or directory` and exit 127. Drain the queue first.
- **Editing an optimizer setting breaks the resume of any run in flight.**
  Megatron checks the scheduler state against the checkpoint and asserts on a
  mismatch:

      OptimizerParamScheduler: class input value 0.01 and checkpoint value 0.1
      for start weight decay do not match

  The same guard covers the total iteration count, so `--num-rollout` is frozen
  for a run's lifetime too — a run cannot be extended later by raising it. Since
  a production run is 3–4 chained jobs over ~10 h, `--weight-decay`, `--lr`, the
  decay style, warmup and `--num-rollout` must all be settled *before* the first
  job of a sweep point is submitted.

### A wall-clock kill during a save is safe

Megatron writes the checkpoint files, then the tracker, then deletes the
previous iteration. A job killed between the first two leaves the tracker
pointing at the last *complete* iteration, so the resume loads a good checkpoint
and the partial directory is simply ahead of it. Observed directly: phase A of
the resume test was cut mid-save of `iter_0000003`, and

    tracker -> 1
    dist    : iter_0000001 iter_0000003

The orphan is not leaked — the resumed run passes through that iteration again
and overwrites it — but disk is temporarily doubled (114 GB where a single
checkpoint plus its HF copy is ~61 GB), which matters when a sweep has many
points in flight.

## Container runtime

- **`enroot` + pyxis only.** `docker`, `podman`, `apptainer`, `singularity` are
  **not installed**, and `/var/run/docker.sock` does not exist — on GPU and CPU
  nodes alike.
- `/etc/subuid` is **empty**, so nested rootless dockerd and `apptainer
  --fakeroot` are both out. Anything that needs a Docker daemon has to run off
  this cluster or behind an external sandbox service.
- `srun --container-image=... --container-mounts=...` is the supported path.

## Networking

Egress works from both login and compute nodes (HTTP status observed):

| Target | Status |
|---|---|
| `huggingface.co` | 200 |
| `ghcr.io/v2/` | 401 (reachable, unauthenticated) |
| `app.daytona.io/api/health` | 200 |
| internal sandbox service (see Slack `#swdl-nemo-sandbox-support`) | 401 (reachable, key required) |

No proxy variables are set in the environment.

## Filesystem

- `/lustre/fs1` is the backing filesystem; `/lustre/fsw/portfolios/coreai/users/kfujii`
  is a symlink to `/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii`.
  Both names refer to the same bytes.
- 235 TB total, ~68 TB free at the time of writing.
- **`/tmp` is RAM-backed** on these nodes. Never point enroot scratch, TMPDIR or
  a large intermediate file at it — it consumes node memory and has bitten other
  teams (evals silently broke when TMPDIR moved).
- Default `ulimit -n` is 131072, which is comfortable for servers that hold many
  sockets.
