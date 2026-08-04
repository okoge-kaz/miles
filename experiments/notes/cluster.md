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
