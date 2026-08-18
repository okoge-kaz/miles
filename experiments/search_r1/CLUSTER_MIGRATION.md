# Search-R1 cluster migration and B300 qualification

This is the hand-off checklist for moving the fixed-difficulty Search-R1 sync
and async experiments to another Slurm cluster. It records the repository and
container audit performed on 2026-08-14. The application recipes are ready to
move, but a container is not considered B300-qualified until the target cluster
passes the GPU smoke sequence below.

## B300 and SQSH verdict

Creating an Enroot `.sqsh` does not require a B300. `enroot import` selects a
Docker manifest for a CPU architecture, extracts its layers, and writes a
SquashFS image. It can run on a CPU node with registry access and sufficiently
large local scratch. GPU compatibility is determined later by the binaries in
the image and the host driver/NVLink stack, not by which GPU was present during
SquashFS creation.

The current CUDA image is a credible B300 candidate:

- `docker/Dockerfile` defaults to CUDA 13 and architecture-specific
  `cu130-x86_64` / `cu130-aarch64` wheels.
- It carries Transformer Engine 2.17 CUDA 13 plus the repository's explicit
  B300/GB300 FlashAttention-2 and backward-override patches.
- `docker/build.py` can publish `cu13-x86`, `cu13-aarch64`, or one multi-arch
  `cu13` manifest.
- The locally deployed
  `miles-staleness-weight-boundaries-f994b9aed.sqsh` was inspected as x86-64
  with PyTorch `2.11.0+cu130`, CUDA `13.0`, Transformer Engine `2.17.0` CUDA 13,
  both B300-related patches, and the required SGLang commit. It should be usable
  as a smoke candidate on an x86-64 DGX B300, but it has not been executed on a
  B300 and is therefore not a qualification result.

Prefer a fresh, immutable CUDA 13 image tag and import it on the destination
cluster. Do not use
`experiments/container/derive_sglang_prefill_version_image.sbatch` as a B300
migration mechanism: that script replaces only the editable SGLang checkout in
an existing root filesystem and cannot upgrade CUDA, PyTorch, Transformer
Engine, NCCL, or compiled wheels.

DGX B300 is x86-64. GB300 systems built around Grace are arm64; an x86 `.sqsh`
cannot be copied to those systems. For a multi-arch tag, Enroot defaults to the
import host's architecture, so import on the same CPU architecture as the GPU
nodes. NVIDIA lists B300/GB300 compute capability as 10.3 (`sm_103`).

## Destination prerequisites

The cluster administrator should confirm all of the following before assets are
copied:

- Release 580 or newer **open** NVIDIA kernel driver. CUDA 13.x minor-version
  compatibility starts at Linux driver 580, and CUDA 13.0 GA lists
  `580.65.06` as its corresponding minimum.
- On an eight-GPU B300 NVSwitch system, a matching driver/Fabric Manager/NVLSM
  stack. NVIDIA requires NVLSM for fifth-generation NVLink systems and warns
  against upgrading these packages independently.
- Slurm with working Pyxis/Enroot and libnvidia-container GPU hooks. Pyxis must
  be built for the installed Slurm release.
- A CPU import node with registry egress, at least 100 GB of free local ext4/XFS
  scratch, `enroot`, `mksquashfs`, and access to the shared container directory.
  The current import script's extracted-layer overlay does not work on Lustre;
  set `ENROOT_LOCAL_SCRATCH_ROOT` to node-local storage.
- Shared storage mounted consistently on every allocated node, routable TCP
  between nodes, and unused Ray/SGLang/retriever ports.
- At least 256 GB host RAM per node for the 61 GB mmap FAISS index, 14 GB corpus,
  E5 encoder, Python objects, and page cache. The retriever is CPU/FAISS based;
  B300 memory does not remove this host-memory requirement.

Primary references:

- [NVIDIA CUDA GPU compute capability list](https://developer.nvidia.com/cuda/gpus)
- [CUDA 13.0 release notes and driver table](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
- [DGX OS 7 driver and NVLink-stack upgrade guidance](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/additional_software.html)
- [DGX OS 7 supported architecture table](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/introduction.html)
- [DGX Station GB300 arm64 software requirements](https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/software-requirements.html)
- [NVIDIA Enroot import documentation](https://github.com/NVIDIA/enroot/blob/main/doc/cmd/import.md)
- [NVIDIA Pyxis requirements and SQSH usage](https://github.com/NVIDIA/pyxis)
- [PyTorch CUDA 13 recommendation for Blackwell](https://pytorch.org/blog/pytorch-2-12-release-blog/)

## Build and import

Use an immutable tag rather than `latest` or `dev`. On a builder with Docker
Buildx and registry credentials:

```bash
# DGX B300 (x86-64). Use --variant cu13 for one amd64+arm64 manifest.
python docker/build.py \
  --variant cu13-x86 \
  --image-tag custom \
  --custom-tag search-r1-b300-<date-or-commit> \
  --push
```

On the destination cluster, create the log directory before submission because
Slurm opens the output file before the job body sources `env.sh`. Override the
site-specific account, CPU partition, local scratch, output image, and stable
link explicitly:

```bash
mkdir -p experiments/outputs/download

export SHARED_WS=/shared/<project>
export WS=/shared/<user>
export DOCKER_IMAGE=radixark/miles:search-r1-b300-<date-or-commit>
export ENROOT_LOCAL_SCRATCH_ROOT=/local/enroot
export IMPORT_OUTPUT_IMAGE=/shared/<project>/container/miles-search-r1-b300-<date-or-commit>.sqsh
export SQSH_LINK=/shared/<project>/container/miles-search-r1-b300.sqsh

sbatch -A <account> -p <cpu-partition> --export=ALL \
  experiments/container/import_image.sbatch
```

Record the Docker tag/digest, `.sqsh` SHA-256, `uname -m`, Enroot/Pyxis
versions, driver version, and this repository commit alongside each run. The
`.sqsh` is immutable, while `/root/miles` is over-mounted from the checkout, so
the repository commit must be recorded separately.

## B300 preflight gate

Run these gates in order. Stop at the first failure; a fallback that silently
changes the attention implementation would invalidate a sync/async throughput
comparison.

1. **Host:** `nvidia-smi` reports eight B300 GPUs and a Release 580+ driver;
   `nvidia-smi topo -m` shows the expected NVLink topology; NVLSM is healthy.
2. **Container visibility:** inside the `.sqsh`, `torch.cuda.is_available()` is
   true, `torch.version.cuda` is 13.x, the device name is B300, and
   `torch.cuda.get_device_capability()` returns `(10, 3)`.
3. **Native imports:** import `torch`, `transformer_engine.pytorch`, `sglang`,
   `flash_attn`, `faiss`, and `sentence_transformers`. Run a BF16 matmul on the
   GPU and synchronize. Any `no kernel image is available`, undefined symbol,
   or architecture error is a hard failure.
4. **Attention backend:** run a short trainer forward and SGLang generation.
   The Docker image also contains a Hopper-only FA3 wheel; B300 must use a
   supported FA2/TE/SGLang path rather than force the `sm_90a` binary. The
   Search-R1 trainer requests `--attention-backend flash`, so inspect startup
   logs for the backend actually selected.
5. **Collectives:** run `all_reduce_perf` over all eight GPUs, then a two-node
   test for the async placement. Resolve NCCL/IB/NVLink errors before Ray.
6. **Search environment:** start the E5/FAISS retriever and require a non-empty
   real `/retrieve` response, not only `/health`.
7. **End to end:** run one fixed-dataset sync update and confirm reward,
   checkpoint, HF export, valid-action/search metrics, and W&B project
   `async-search-r1`.
8. **Async resume:** run one async update, confirm the replay buffer exists,
   stop cleanly, resume the same run identity, and verify that replay-buffer/FIFO state
   restores without a full queue refill.

Functional qualification comes before B300 tuning. The 4B model is not
capacity-limited by B300 HBM, so keep the H100 reference batch/TP shape for the
first gate. Tune tensor parallelism, SGLang memory fraction, CUDA graphs, and
retriever concurrency only after the reference result is correct.

## Search-R1 assets and configuration

Set cluster-specific values before sourcing a recipe:

```bash
export SHARED_WS=/shared/<project>
export WS=/shared/<user>
export SQSH_IMAGE=/shared/<project>/container/miles-search-r1-b300-<tag>.sqsh
export SLURM_ACCOUNT_NAME=<account>
export GPU_PARTITION=<gpu-partition>
export CPU_PARTITION=<cpu-partition>
export GPUS_PER_NODE=8
```

The `#SBATCH` account, partition, GRES spelling, node count, and wall time are
parsed before shell variables exist. Override them on the `sbatch` command line
or maintain thin site-specific wrappers. `GPUS_PER_NODE` changes placement
validation; it does not rewrite `#SBATCH --gres=gpu:8`.

Keep the in-container mounts `/root/miles`, `/data`, and `/ckpt` stable. Transfer
or stage:

- Qwen3-4B-Instruct-2507 HF and Megatron checkpoints.
- E5-base-v2, `e5_Flat.index`, and `wiki-18.jsonl`.
- Search-R1 raw train/eval data.
- The fixed p10-90 JSONL plus its pass-rate and metadata artifacts, only when
  model, tokenizer, retriever/index/corpus, top-k, sampling, max turns, and
  reward definition are identical.

Otherwise run `experiments/setup/stage_all.sh`,
`experiments/setup/prepare_search_r1.sbatch`, and the resumable
`experiments/src/difficulty_filter/run_measure_search_r1.sbatch` on the new
environment. Verify checksums before measuring; the fixed difficulty set is
part of the experiment definition.

The destination is ready for primary experiments only after both placements
pass the gates above. Until then the accurate status is **recipe-ready,
cluster-integration pending**, not B300-validated.
