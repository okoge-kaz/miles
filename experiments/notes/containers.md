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
