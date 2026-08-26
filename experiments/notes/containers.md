# Containers on cw-dfw (enroot + pyxis)

## Image workflow

```
docker://radixark/miles:latest
        │  enroot import   (experiments/container/import_image.sbatch, CPU node)
        ▼
$CONTAINER_DIR/miles-latest-YYYYMMDD.sqsh
        │  ln -sfn
        ▼
$CONTAINER_DIR/miles-latest.sqsh          ← what run.sbatch uses by default
```

Each import writes a dated file and repoints the symlink, so a failed pull never
destroys the image running jobs are using. Pin a dated file
(`SQSH_IMAGE=.../miles-latest-20260803.sqsh`) when a measurement has to stay
reproducible against one image.

Scratch paths must live on lustre, not `/tmp` (RAM-backed here):

```bash
export ENROOT_CACHE_PATH=$CACHE_DIR/enroot/cache
export ENROOT_TEMP_PATH=$CACHE_DIR/enroot/tmp
export ENROOT_DATA_PATH=$CACHE_DIR/enroot/data
```

Docker Hub rate-limits anonymous pulls. On HTTP 429, add credentials to
`$ENROOT_CONFIG_PATH/.credentials`:

```
machine auth.docker.io login <user> password <token>
```

## Running

```bash
srun --container-image=$SQSH_IMAGE \
     --container-mounts=$CONTAINER_MOUNTS \
     --container-writable \
     --no-container-mount-home \
     bash /root/miles/experiments/<recipe>/train.sh
```

| Flag | Why |
|---|---|
| `--container-writable` | ray, sglang and pip write under `/root` and `/tmp` inside the container |
| `--no-container-mount-home` | keeps `$HOME` dotfiles from leaking in; caches are mounted explicitly instead |
| `--export=ALL` (on `srun`) | carries `RUN_NAME`, `WANDB_API_KEY`, batch-size overrides into the container |

Pyxis containers share the host network namespace, so ray's dashboard, the
SGLang router and the session server are reachable at the node's address without
extra port plumbing — the equivalent of docker's `--network=host` in the
upstream quick start.

## AWS EFA and NCCL

`docker/install_aws_efa.sh` installs the versioned AWS EFA 1.49.0 userspace
release used by both `docker/Dockerfile` and
`container/derive_sglang_prefill_version_image.sbatch`. It contains libfabric
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
`common/run_with_efa_env.sh` prepends the path and reasserts all EFA selectors
inside that boundary before it `exec`s the training command.
`common/check_efa.sh` does the same and is a fail-closed preflight: it checks the
named plugin, dynamic dependencies, and `fi_info -p efa -t FI_EP_RDM` on every
allocated node. EFA-enabled multi-node launchers run it before Ray starts and
carry the same selectors into the Ray runtime environment.
To validate a new image independently and compare its bandwidth with the TCP
fallback, submit:

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
| `…/checkpoints/hf` | `/ckpt/hf` | |
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
- Multi-node runs need ray started on every node and
  `MILES_SCRIPT_EXTERNAL_RAY=1` so miles does not try to start its own head;
  the current recipes are single-node and do not set it.
