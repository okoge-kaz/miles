# Containers on OCI (aarch64, Enroot + Pyxis)

Use the aarch64 image variant on GB200. The default import recipe is
`experiments/container/import_image.sbatch`, using account nemotron_sw_post,
partition cpu and QoS cpu-interactive.

```bash
mkdir -p experiments/outputs/download
sbatch -A nemotron_sw_post -p cpu --qos=cpu-interactive \
  experiments/container/import_image.sbatch
```

The recipe imports with `enroot import --arch aarch64`, writes a unique dated
`miles-oci-aarch64-*.sqsh`, and updates `miles-oci-aarch64.sqsh`.
Pin an immutable file via SQSH_IMAGE for measurements. The mutable Docker
`latest` tag is a discovery convenience, not reproducible provenance.

Use local ext4/XFS scratch (`ENROOT_LOCAL_SCRATCH_ROOT=/tmp` on the probed
nodes) for extraction and overlay work. Keep the layer cache and final image on
Lustre. Overlay lower directories on Lustre failed in the historical importer;
do not put ENROOT_DATA_PATH there. Confirm local free space before importing.

## Verification image, 2026-09-09

CPU interactive job 7035384 imported `docker://radixark/miles:latest` explicitly
as aarch64 into `experiments/outputs/miles-oci-aarch64-7035384.sqsh` (38 GB).
The associated import log is in the same outputs directory. The GB200 container
probe reported Python 3.12.3, PyTorch 2.13.0+cu130 / CUDA 13.0, Ray 2.58.0,
SGLang 0.5.20.dev54+ga8e5c63, and a successful CUDA matrix product.
These versions describe this one imported artifact, not future latest pulls.

Mount the checkout on **every** srun, including when reusing a named container:

```bash
srun --jobid=<job-id> -N 1 -n 1 \
  --container-image="$SQSH_IMAGE" \
  --container-mounts="$MILES_REPO:/root/miles" \
  --container-workdir=/root/miles --container-writable --no-container-mount-home \
  bash -lc 'export PYTHONPATH=/root/miles:/root/Megatron-LM; python -m compileall -q miles'
```

A named container does not preserve Pyxis host mounts for subsequent steps.
Training uses CONTAINER_MOUNTS from env.sh to include datasets/checkpoints/caches.
Do not print authentication environment variables into job logs.

## Prefill-provenance qualification

The old cw-dfw image installed an SGLang fork at
f994b9aedfd0b1465dbb8f4e2a02eb789fc76dce. That fork targeted an older SGLang
base; it is **not** the default for current Docker builds or OCI recipes.
`derive_sglang_prefill_version_image.sbatch` now requires an explicit compatible
SGLANG_COMMIT and starts from SQSH_IMAGE. Its output must pass the prefill
weight-version smoke before using `STALENESS_REFERENCE=prefill`.

On 2026-09-10, the pinned `a8e5c63` OCI image plus the opt-in
`OCI_SGLANG_PROVENANCE_PATCH=1` overlay passed the real-GPU prefill/forward-token
probe and four-node async training/save/inflight-resume checks. The helper
`experiments/container/apply_oci_sglang_provenance.sh` verifies the exact base
commit and applies the checked-in patch inside each writable container, before
Ray starts. The base SQSH stays unchanged and still lacks this metadata without
the overlay. See [async qualification](oci-async-bringup.md) for the patch digest,
reproduction command, supported serving scope, and remaining logging limits.

Do not substitute submission or completion versions while labelling results
first-prefill. A functioning upstream GPU image alone does not establish the
extra metadata contract. See [migration status](cluster-migration.md) for
what was actually exercised and what still requires workload qualification.
