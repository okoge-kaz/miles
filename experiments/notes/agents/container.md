# Work log — container

## 2026-08-03 — image import strategy for pyxis

### Decisions

- Source image `radixark/miles:latest` (the one the quick start uses; ships miles
  at `/root/miles` and Megatron-LM at `/root/Megatron-LM`).
- `enroot import` writes `container/miles-latest-YYYYMMDD.sqsh` and repoints a
  `miles-latest.sqsh` symlink. Rationale: a failed pull must not destroy the
  image running jobs use, every file on disk states when it was pulled, and a
  measurement can be pinned to a dated file.
- The import job runs on the `cpu` partition — registry egress is available there
  (verified) and it needs no GPU.
- enroot scratch (`ENROOT_CACHE_PATH`, `ENROOT_TEMP_PATH`, `ENROOT_DATA_PATH`) is
  forced onto lustre under `$CACHE_DIR/enroot/`, because `/tmp` on these nodes is
  RAM-backed and a multi-GB image would consume node memory.
- The repo checkout is mounted over `/root/miles`, so the image's editable
  install picks up local edits with no rebuild. New *dependencies* still require
  an in-job install or a new image.

### pyxis flags used

`--container-writable` (ray/sglang/pip write inside the container),
`--no-container-mount-home` (keep `$HOME` out; caches are mounted explicitly),
`--export=ALL` on `srun` so `RUN_NAME`, `WANDB_API_KEY` and batch-size overrides
reach the container.

Pyxis shares the host network namespace, which is the equivalent of the upstream
quick start's `docker run --network=host` — no port plumbing needed for the ray
dashboard, the SGLang router or the session server.

### Not verified

- **The import has not been run**, so image size, import duration and whether
  Docker Hub rate-limits this account are all unknown. The script documents the
  429 workaround (`$ENROOT_CONFIG_PATH/.credentials`) but it has not been
  exercised.
- Whether `/dev/shm` inside the pyxis container is large enough for NCCL and ray
  at 8 GPUs was not checked; the upstream docker recipe asks for `--shm-size=32g`.
  If the first GPU run shows NCCL shared-memory errors, this is the first thing
  to look at.
