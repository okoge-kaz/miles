# Containers with SingularityCE

This is the maintained container contract for the current PBS cluster. The
canonical end-to-end workflow is in
[`../PBS_SINGULARITY.md`](../PBS_SINGULARITY.md).

## Image workflow

```text
docker://radixark/miles:latest
        |  singularity build --fakeroot --fix-perms
        |  experiments/container/miles.def
        v
$CONTAINER_DIR/miles-YYYYMMDD-HHMMSS.sif
        `- stable link: $CONTAINER_DIR/miles.sif
```

Initialize the workspace and preview or submit the build from the repository
root:

```bash
export MILES_WORKSPACE_ROOT=/groups/gai51740/kazuki_fujii
source experiments/env.sh

experiments/setup.sh container
experiments/setup.sh container --submit
```

For direct submission, use the shared PBS helper:

```bash
source experiments/common/pbs.sh
pbs_submit --profile=cpu --time="${PBS_CONTAINER_WALLTIME}" \
  experiments/container/import_image.sbatch
```

The default container build walltime is `00:30:00`. The definition imports the
configured OCI layers, checks out the pinned Miles SGLang policy-provenance
revision, and installs the pinned Tau v1.0.1 runtime that used to require a
separate derived image. Increase the walltime only when measured registry or
build performance requires it.

The job creates a private directory below `PBS_LOCALDIR`, a writable
scheduler-created `/local/<job-id>`, or `TMPDIR`/`/tmp`; it changes into that
directory and points `SINGULARITY_TMPDIR`, `APPTAINER_TMPDIR`,
`SINGULARITY_CACHEDIR`, and `APPTAINER_CACHEDIR` at that local directory. It
builds and tests the SIF there, then moves it to a hidden staging name below
`$CONTAINER_DIR`. The dated SIF becomes visible only after checksum verification;
provenance and the stable link are published with it. A failed build therefore
leaves the image used by existing jobs unchanged.

## Why the definition file is required

Singularity executes the container payload as the submitting UID. A typical OCI
image makes `/root` mode `0700`, but this image contains Miles, Megatron, and
other runtime code below `/root`. A direct OCI pull can consequently produce a
valid SIF that the actual job user cannot traverse.

[`../container/miles.def`](../container/miles.def) is the compatibility layer.
The build job invokes it with both `--fakeroot` and `--fix-perms`. Its `%post`
section:

- changes `/root` and `/root/.cache` to mode `0755`;
- applies `a+rX` to the baked Miles, Megatron, Tau, and SGLang source trees;
- verifies the configured `SGLANG_COMMIT` and its policy-weight-version unit
  test before publishing the image;
- creates every static directory and file target used by maintained bind mounts;
- gives `/tmp` and `/var/tmp` mode `1777` for job-local temporary files.

These changes provide read and traversal access, not general write access to
the baked `/root` tree. Persistent writable state belongs on a host bind mount.
The runtime adds `--writable-tmpfs` for unavoidable ephemeral writes and
`--no-home` so login-node dotfiles are not mounted implicitly.

The cluster must permit configured fakeroot builds for ordinary users. Do not
replace `--fakeroot` with a host-root build. If the option is unavailable, have
the cluster administrator configure the supported unprivileged build path.

## Static bind destinations

`experiments/env.sh` defines the common runtime binds once in
`CONTAINER_MOUNTS`:

| Host source | Container destination | Purpose |
| --- | --- | --- |
| `$MILES_REPO` | `/root/miles` | Current checkout; shadows the baked copy |
| `$RL_DATASET_DIR` | `/data` | RL and evaluation datasets |
| `$PRETRAIN_DATASET_DIR` | `/data/pre-train` | Pre-training datasets |
| `$SFT_DATASET_DIR` | `/data/sft` | SFT datasets |
| `$HF_CKPT_DIR` | `/ckpt/hf` | Hugging Face checkpoints |
| `$MEGATRON_CKPT_DIR` | `/ckpt/megatron` | Converted Megatron checkpoints |
| `$TRAIN_CKPT_DIR` | `/ckpt/training` | Training outputs and resume state |
| `$CACHE_DIR` | `/cache` | Current framework and compiler caches |
| `$CACHE_DIR` | `/root/.cache` | Compatibility path for older tools |

The definition also creates destinations used only by selected evaluation and
SWE jobs: `/checkpoint`, `/search-eval-model`, `/results`, `/dump`,
`/runtime-tokenizer`, `/evaluation-data`, `/evaluation-cache`,
`/evaluation-auth/token`, `/workspace/reasoning_eval`, and
`/usr/local/bin/bwrap`. The token and Bubblewrap executable destinations are
created as file placeholders so a file bind does not depend on an overlay
creating its target.

The repository `.env` mask is different: its destination is below the
`/root/miles` checkout bind, so an image-level placeholder would be shadowed.
The maintained runtime always uses `--writable-tmpfs`; that overlay supplies the
nested target when the checkout has no `.env`, and the post-build smoke test
verifies both the absent-file and masked-file contract.

Every host source must exist and have permissions suitable for the submitting
UID before launch. `experiments/setup.sh init` creates the common workspace
directories. Task-specific launchers remain responsible for their additional
sources and for owner-only modes on credentials.

## Post-build non-root smoke test

The definition's `%test` verifies the image-level read and traversal contract.
`import_image.sbatch` then performs the decisive test after the build:

1. execute the candidate as the ordinary submitting UID without code binds or a
   writable overlay;
2. recursively verify traversal and readability of every baked code tree;
3. execute again with `--no-eval`, `--no-home`, `--writable-tmpfs`, and the real
   `CONTAINER_MOUNTS`;
4. verify writes to every bound repository, dataset, checkpoint, and cache path;
5. verify the nested `.env` mask and import `megatron`, `miles`, `sglang`, and
   Tau with the production paths.

This catches both image permission errors and host-side mount permission errors
before the stable link changes. Do not remove this test when changing the OCI
base or adding a new path below `/root`; add the new code path or bind target to
the definition and smoke test instead.

## Running

Submit maintained recipes from the repository root through `pbs_submit`:

```bash
source experiments/env.sh
source experiments/common/pbs.sh

pbs_submit --profile=gpu --nodes=4 --time="${PBS_DEFAULT_WALLTIME}" \
  --export=ALL,WANDB_MODE=offline \
  experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch
```

The recipe establishes its checkpoint and dataset paths, then uses
`experiments/common/singularity.sh`. Multi-node launchers use host HPC-X
`mpirun` with one container task per allocated host and explicit
`-map-by ppr:1:node -bind-to none`; the Ray worker tasks join the head before
training is submitted. Each node launcher records its CPU affinity before
Singularity starts.

For an interactive preflight within an allocation:

```bash
source experiments/common/singularity.sh
miles_container_exec \
  --image "${CONTAINER_IMAGE}" \
  --bind "${CONTAINER_MOUNTS}" \
  -- python3 -c 'import megatron, miles, sglang'
```

Singularity uses the host network namespace, so Ray, SGLang, and session-server
ports require no container port publishing. The image does not install a
cluster-specific network stack. Multi-node training uses the cluster-provided
NCCL and InfiniBand libraries by default; `MILES_NCCL_TRANSPORT=tcp` is an
explicit diagnostic control for the NCCL validation job.

## Secrets and writable state

`experiments/env.sh` never reads a repository `.env`. Provide credentials in the
submission environment or a scheduler-supported secret mechanism, and keep
fixed-name export allowlists in credential-capable wrappers. Do not place secret
values in command arguments, committed files, or logs.

The SIF is immutable. Writes outside a host bind are temporary because the
runtime uses a writable tmpfs. Framework caches are directed to `/cache`, while
the `/root/.cache` bind preserves compatibility with older tools. A new durable
output or cache needs an explicit host directory, a static destination in
`miles.def`, and a bind in the responsible launcher.

## Operational checks

- The editable Miles install points at `/root/miles`; the checkout bind makes
  local source edits visible without reinstalling the package.
- `/root/Megatron-LM` remains the copy shipped by the image. Rebuild the SIF to
  change it.
- A new dependency installed only into writable tmpfs disappears at job end;
  add durable dependencies to the OCI image and rebuild.
- Record the OCI tag or digest and the dated SIF SHA-256 for reproducible
  measurements. The stable `miles.sif` link is for normal operation, not an
  immutable experiment identity.
