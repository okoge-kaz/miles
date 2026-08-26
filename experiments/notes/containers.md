# Containers on aws-pdx (enroot + pyxis)

## Image workflow

```
docker://radixark/miles:latest
        │  enroot import   (experiments/container/import_image.sbatch, CPU node)
        ▼
$CONTAINER_DIR/miles-latest-YYYYMMDD.sqsh
        └── optional stable link: $CONTAINER_DIR/miles-latest.sqsh

$CONTAINER_DIR/miles-search-r1-b300-20260815.sqsh  ← `env.sh` default
```

The generic importer writes a dated file and can repoint its own stable link; it
does not rewrite the separately pinned image selected by `env.sh`. A failed pull
therefore cannot destroy the image used by running jobs. Pin an immutable file
with `SQSH_IMAGE` when a measurement has to stay reproducible against one image.

The import's extracted-layer overlay must use node-local ext4/XFS rather than
Lustre. The layer cache may remain persistent:

```bash
export ENROOT_CACHE_PATH=$CACHE_DIR/enroot/cache
export ENROOT_LOCAL_SCRATCH_ROOT=/raid/enroot/tmp
```

Docker Hub rate-limits anonymous pulls. On HTTP 429, add credentials to
`$ENROOT_CONFIG_PATH/.credentials`:

```
machine auth.docker.io login <user> password <token>
```

## Running

```bash
WANDB_MODE=offline sbatch -A coreai_horizon_dilations \
  experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch
```

Submit `run.sbatch` from the repository root. Do not invoke a maintained
multi-node recipe's `train.sh` directly: the Slurm wrapper establishes the
checkpoint mounts, transport variables, and one container task per node. Each
task enters `train.sh` and sources `experiments/common/ray_cluster.sh`; worker
tasks join and wait while the head continues to submit training only after the
expected Ray node count is present.

The wrapper's Pyxis launch uses these container controls:

| Flag | Why |
|---|---|
| `--container-writable` | ray, sglang and pip write under `/root` and `/tmp` inside the container |
| `--no-container-mount-home` | keeps `$HOME` dotfiles from leaking in; caches are mounted explicitly instead |
| environment export | carry only the names the job needs; never place a secret in argv or a committed file |

Several maintained training launchers currently use `srun --export=ALL` after
constructing their environment. That is an implementation fact, not recommended
submission guidance: it can forward unrelated login-shell secrets into every
container and Ray worker. Submission wrappers should use fixed-name allowlists
(`USER`, `WANDB_MODE`, explicitly needed credentials and overrides), and future
launcher hardening should replace the inner `ALL` export with a generated
allowlist. Python training/environment code must not discover repository
`.env`; if a task-specific Slurm wrapper supports dotenv input, it must parse a
fixed allowlist at the job boundary without sourcing or evaluating the file.

Pyxis containers share the host network namespace, so ray's dashboard, the
SGLang router and the session server are reachable at the node's address without
extra port plumbing — the equivalent of docker's `--network=host` in the
upstream quick start.

## AWS EFA and NCCL

`docker/install_aws_efa.sh` installs the versioned AWS EFA 1.49.0 userspace
release used by both `docker/Dockerfile` and
`experiments/container/derive_sglang_prefill_version_image.sbatch`. It contains libfabric
2.4.0amzn5.0, AWS OFI NCCL 1.20.0, and rdma-core 63.0. The download is pinned by
SHA-256; the kernel module and EFA devices remain host-owned.

The image intentionally contains only the named plugins:

```
/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi.so
```

It must not contain `/opt/amazon/ofi-nccl/lib/libnccl-net.so`. That generic name
would let NCCL auto-load OFI and could change transport selection on a non-EFA
host. EFA jobs opt in explicitly with:

```bash
export NCCL_NET="AWS Libfabric"
export NCCL_NET_PLUGIN=ofi
export NCCL_TUNER_PLUGIN=ofi
export NCCL_IB_DISABLE=0
export FI_PROVIDER=efa
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:$LD_LIBRARY_PATH
```

`NCCL_NET` forces the network registered by the external plugin; if that network
cannot initialize, NCCL fails instead of falling back to Socket or its built-in
IB transport. Pyxis reconstructs `LD_LIBRARY_PATH` while entering the container,
so exporting the EFA path only on the host is insufficient.
`experiments/common/run_with_efa_env.sh` prepends the path and reasserts all EFA selectors
inside that boundary before it `exec`s the training command.
`experiments/common/check_efa.sh` does the same and is a fail-closed preflight: it checks the
named plugin, dynamic dependencies, and `fi_info -p efa -t FI_EP_RDM` on every
allocated node. EFA-enabled multi-node launchers run it before Ray starts and
carry the same selectors into the Ray runtime environment.

Jobs 294400 (30B-A3B), 294401 (8B), and 297372 (Code) predate this opt-in path
and failed with `NET/IB ... status=IBV_WC_RETRY_EXC_ERR(12) ... vendor_err=129`.
That signature is the built-in NCCL InfiniBand transport, not an AIME dataset or
math-verifier error. Those jobs are failure evidence and do not validate 8B,
30B-A3B, or Code training.

Job 304525 is direct evidence that all 16 ranks across two nodes mapped the OFI
plugin and AWS libfabric and completed a 256 MiB, 20-iteration EFA all-reduce at
183.605 GB/s algorithm bandwidth (344.259 GB/s bus bandwidth). Current IFEvalG
training jobs 306686/306687 also passed the EFA preflight and selected
`NCCL_NET=AWS Libfabric` before successful optimizer updates and checkpointing.
STEM jobs 306790/306792 likewise passed the fail-closed EFA preflight on all four
nodes and selected AWS Libfabric in the Ray runtime before completing the
fresh/resume replay gate through iteration 1.
This proves EFA is active; it does not yet provide a clean EFA-versus-Socket
throughput comparison. Job 304525's EFA phase passed, but its Socket control was
rejected because the then-current validator leaked `FI_PROVIDER=efa`. The
checked-in validator now unsets it, but that corrected comparison has not been
rerun successfully. To validate a new image and obtain that comparison, submit:

```bash
sbatch -A "$ACC" experiments/container/validate_efa_collective.sbatch
```

`NCCL_IB_DISABLE=1` is retained only as the explicit
`MILES_NCCL_TRANSPORT=tcp` diagnostic/control mode, not as the training default.

## Mount layout

Defined once in `experiments/env.sh` as `CONTAINER_MOUNTS`:

| Host | Container | Notes |
|---|---|---|
| `…/src/miles` | `/root/miles` | shadows the copy baked into the image, so local edits take effect immediately |
| `…/datasets` | `/data` | |
| `…/checkpoints/huggingface` | `/ckpt/hf` | |
| `…/checkpoints/megatron` | `/ckpt/megatron` | |
| `…/checkpoints/training` | `/ckpt/training` | |
| `…/cache` | `/root/.cache` | `HF_HOME=/root/.cache/huggingface` |

`/root/Megatron-LM` stays as shipped by the image; `PYTHONPATH` points at it plus
`/root/miles`.

## Gotchas

- The image ships an editable install of miles pointing at `/root/miles`. Because
  the mount replaces that directory, no `pip install -e .` is needed — but if you
  add a **new dependency**, install it inside the job (or rebuild the image);
  changes to site-packages do not persist across jobs.
- `pkill` patterns in the upstream launch scripts (`pkill -9 python`) kill
  everything in the container's PID namespace. Harmless in a dedicated job,
  destructive if you ever run two recipes in one allocation.
- Maintained async recipes are multi-node. Their `run.sbatch` executes one Pyxis
  task per node; `experiments/common/ray_cluster.sh` starts the Ray head on node
  zero, joins each worker, waits for the expected node count, and the head submits
  `train_async.py` through the dashboard. They do not use
  `MILES_SCRIPT_EXTERNAL_RAY`.
