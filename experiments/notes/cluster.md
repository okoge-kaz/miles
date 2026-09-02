# PBS/Singularity cluster notes

Measured on 2026-08-31 on the current ABCI login environment. Re-run these
checks after cluster maintenance instead of carrying assumptions from the old
Slurm/Enroot deployment.

## Scheduler and queues

The scheduler is PBS Pro 2022.1.6. Maintained jobs use:

| Workload | Queue | Resource type | Per-node request | Default walltime |
| --- | --- | --- | --- | ---: |
| GPU training/evaluation | `R9920261300` | `RTYPE=rt_HF` | 192 CPUs, 8 GPUs | `24:00:00` for training |
| CPU build/download/preparation | `R9920261300` | `RTYPE=rt_HC` | 32 CPUs, 0 GPUs | task-specific |

The reservation has no old four-hour Slurm limit. Request the measured workload
duration: the Miles SIF build normally uses 30 minutes, short validation jobs use minutes,
checkpoint/data preparation is normally one to eight hours, and network
downloads default to 24 hours. A direct normal-queue submission remains an
explicit override; the maintained CPU profile uses the reservation queue with
the `rt_HC` resource class.

Do not add `#PBS -P` to reusable job files. The shared `pbs_submit` helper is
deliberately project-neutral. For a manual direct allocation, pass the project
at invocation time when needed, for example `qsub -P gai51740 ...`; the
`miles-run-ladder` skill contains complete interactive and batch examples.

Inspect live state with:

```bash
qstat -q
qstat -Qf R9920261300
qstat -Qf rt_HC
pbsnodes -aS
qfree
```

PBS provides `PBS_O_WORKDIR` and `PBS_NODEFILE`. `pbs_submit` starts the payload
in the requested submit directory; `miles_container_exec_all` creates a unique
hostfile and launches one Singularity process per node with HPC-X Open MPI:

```text
-map-by ppr:1:node -bind-to none
```

The explicit unbound policy prevents Ray and torch workers from inheriting a
single-core MPI mask. The launcher logs `Cpus_allowed_list` on every node and
checks it against `--cpus-per-task` when that value is supplied.

## Filesystems and scratch

`/groups` is Lustre. Persistent experiment state is rooted at:

```text
/groups/gai51740/kazuki_fujii/
  checkpoints/{hf,megatron,training}/
  containers/
  datasets/{pre-train,rl,sft}/
  cache/
  src/
```

Set `MILES_WORKSPACE_ROOT` to move this complete layout. All purpose-specific
paths in `experiments/env.sh` derive from it and remain individually
overridable for exceptional reads.

Both `/tmp` and `/local` are node-local XFS on the measured host, while `/local`
itself is not generally user-writable. Container builds therefore use
`PBS_LOCALDIR`, then a writable scheduler-created `/local/<job-id>`, then
`TMPDIR`/`/tmp`, and create a private job directory with `mktemp`. They never
try to create a missing job directory directly under the protected `/local`.

The build job changes into that private directory and points all four build-time
variables there:

```text
SINGULARITY_TMPDIR
APPTAINER_TMPDIR
SINGULARITY_CACHEDIR
APPTAINER_CACHEDIR
```

Only the verified SIF, checksum, and provenance are moved back to
`$MILES_WORKSPACE_ROOT/containers`.

## Container runtime

The measured runtime is SingularityCE 4.4.1. Normal Miles jobs run with the
submitting UID, `--no-home`, and `--writable-tmpfs`; they do not use fakeroot.
The one exception in the standard workflow is the unprivileged image build,
which uses `singularity build --fakeroot --fix-perms`.

The definition file makes `/root` traversable, makes image-resident Miles,
Megatron, SGLang, and Tau code readable, and pre-creates every maintained bind
destination. Before publication, the build job runs the candidate as the
ordinary UID with the real repository, dataset, checkpoint, cache, nested
`.env`, and Bubblewrap binds.

AWS EFA is not installed or configured by this repository. Multi-node jobs use
the cluster-provided NCCL and InfiniBand stack. TCP is retained only as an
explicit comparison mode in `validate_nccl_collective.sbatch`.

## Bring-up sequence

```bash
export MILES_WORKSPACE_ROOT=/groups/gai51740/kazuki_fujii
source experiments/env.sh

experiments/setup.sh init
experiments/setup.sh status
experiments/setup.sh container
experiments/setup.sh container --submit
```

After the image build completes, run the short cluster and container validation
jobs before a full training allocation. A successful import alone is not an
end-to-end qualification; check GPU visibility, a short generation, checkpoint
load/conversion, and at least one optimizer update.
