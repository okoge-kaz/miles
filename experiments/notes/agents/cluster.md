# Work log — cluster

## 2026-08-03 — cw-dfw survey (Slurm, container runtime, egress)

Context: deciding where agentic-RL sandboxes can run, and what the
`experiments/` scripts may assume.

### Identity

```
$ hostname
cw-dfw-cs-001-vscode-02
```

Slurm account (jobs without `-A` are rejected):

```
$ sacctmgr -nP show assoc where user=$(whoami) format=account,partition
coreai_horizon_dilations|
```

### Partitions

`sinfo -o "%20P %10l %5D %20G %10m %c"` — GPU partitions (`batch`, `batch_short`,
`batch_long`, `batch_large_long`, `backfill`, `interactive`) all serve the same
~1850 `pool0-*` nodes with `gpu:8`, 128 CPUs, 2 TB RAM. CPU partitions (`cpu`,
`cpu_short`, `cpu_long`, `cpu_interactive`, `cpu_datamover`,
`cpu_dataprocessing`) have 80/38/46 nodes, 96 CPUs, 252 GB, no GRES.

```
$ scontrol show node pool0-00001 | grep -iE "ActiveFeatures|Gres=|RealMemory|CPUTot"
   CPUAlloc=128 CPUEfctv=128 CPUTot=128
   ActiveFeatures=location=local,H100,GPU
   Gres=gpu:8
   RealMemory=2058240 Sockets=2
```

`batch` defaults: `DefCpuPerGPU=16`, `DefMemPerGPU=250000`, `DefaultTime=00:31:00`,
`MaxTime=04:00:00`.

### Container runtime — the decisive finding

On a `cpu` compute node (`cpu1-00106`, `cpu1-00103`):

```
docker:none enroot:/usr/bin/enroot
ls: cannot access '/var/run/docker.sock': No such file or directory
apptainer=none singularity=none enroot=/usr/bin/enroot
$ grep -c . /etc/subuid   →  0
$ ulimit -n               →  131072
```

Consequences:

- Docker-daemon-based sandboxes (Harbor, OpenEnv `TB2_MODE=docker`) **cannot run
  on this cluster's nodes**.
- Empty `/etc/subuid` also rules out nested rootless dockerd and
  `apptainer --fakeroot`. The NeMo-Gym Apptainer sandbox provider is therefore
  not usable here either (consistent with reports that internal Slurm clusters
  are Pyxis/Enroot-only).
- `enroot`-in-`enroot` remains viable as a docker replacement; another team runs
  that pattern in production (enroot installed inside the outer image, `.sqsh`
  per task, rlimit for memory).

### Egress

Login node and `cpu` compute node, identical results:

```
huggingface.co        200
ghcr.io/v2/           401   (reachable, unauthenticated)
app.daytona.io/api/health 200
```

No proxy env vars set. Internal sandbox service (endpoint from
`#swdl-nemo-sandbox-support`, not recorded here): TCP/80 open, HTTP **401** from
both the login node and `cpu1-00103` — reachable, awaiting an API key. 443 and
8080 are filtered; the service speaks plain HTTP on 80.

### Filesystem

```
$ readlink -f /lustre/fsw/portfolios/coreai/users/kfujii
/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii
```

The `fsw` and `fs1` paths are the same bytes (same device id). 235 TB total,
~68 TB free. `/tmp` is RAM-backed — a known trap; another team's evals silently
regressed after a TMPDIR change collided with it.

### Not verified

- Egress from **GPU** compute nodes (only `cpu` nodes were probed). The setup
  scripts therefore download on `cpu` and convert on GPU.
- Whether pyxis honours `--container-writable` for these images at scale — only
  the flag's presence in `srun --help` was checked.
