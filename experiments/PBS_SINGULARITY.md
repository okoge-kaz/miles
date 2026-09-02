# PBS Pro + SingularityCE operations

This is the canonical cluster workflow for `experiments/`. Recipe filenames
retain the `.sbatch` suffix for compatibility, but jobs are submitted to PBS
with `pbs_submit` and containers are executed as Singularity SIF images.

## Configure the workspace

All persistent paths derive from one setting. Export it before sourcing the
shared environment:

```bash
cd /groups/gai51740/kazuki_fujii/src/miles-v0.0
export MILES_WORKSPACE_ROOT=/groups/gai51740/kazuki_fujii
source experiments/env.sh
```

Changing `MILES_WORKSPACE_ROOT` is sufficient when the workspace moves. The
default layout is:

```text
$MILES_WORKSPACE_ROOT/
  checkpoints/
    hf/
    megatron/
    training/
  containers/
  datasets/
    pre-train/
    rl/
    sft/
  cache/
  src/
```

The stable image path is `$CONTAINER_DIR/miles.sif`. Dataset and checkpoint
directories are mounted at `/data`, `/ckpt/hf`, `/ckpt/megatron`, and
`/ckpt/training`. Host caches live below `$CACHE_DIR`; Hugging Face, SGLang,
Triton, Torch, vLLM, and related tools use the persistent `/cache` mount inside
the container. With host home mounting disabled, container `HOME` is the
writable persistent path `/cache/home`. Singularity's general host cache is
`$CACHE_DIR/singularity`; temporary files use `PBS_LOCALDIR` or `TMPDIR`, and
the SIF build also keeps its disposable OCI layer cache in that local directory.

Initialize or inspect the layout with:

```bash
experiments/setup.sh init
experiments/setup.sh status
```

## Build the image for non-root execution

Do not convert the OCI image with a bare `singularity pull`. OCI images often
make `/root` mode `0700`, while Miles and Megatron are installed below `/root`
and Singularity runs the payload as the submitting user. The maintained build
job wraps the OCI image with
[`container/miles.def`](container/miles.def) so the resulting SIF has the
required non-root access and bind destinations.

Preview or submit the 30-minute CPU build through the unified setup command:

```bash
experiments/setup.sh container
experiments/setup.sh container --submit
```

The equivalent direct submission is:

```bash
source experiments/env.sh
source experiments/common/pbs.sh
pbs_submit --profile=cpu --time="${PBS_CONTAINER_WALLTIME}" \
  experiments/container/import_image.sbatch
```

For a PBS build, `import_image.sbatch` requires `/local/$PBS_JOBID` (then its
short-ID form), accepts `PBS_LOCALDIR` only when it is also below `/local`, and
creates the full-ID directory when `/local` itself is writable. It fails rather
than building on shared storage when no writable `/local` directory is
available. A manual, non-PBS invocation may use `PBS_LOCALDIR` or
`TMPDIR`/`/tmp`. The job points both Singularity/Apptainer cache and temp
variables at a private directory below that root before building with
`--fakeroot --fix-perms`. It copies the definition into that directory with
mode `0644` and verifies its checksum before invoking fakeroot, so the builder
never needs to reopen a definition below the shared checkout. The definition
checks out and reinstalls the configured `SGLANG_REPO`, `SGLANG_BRANCH`, and
full `SGLANG_COMMIT`, then runs its policy-weight-version unit test. It also
installs the pinned Tau v1.0.1 layer, creates every static bind target, changes
`/root` to mode `0755`, and makes the baked Miles, Megatron, Tau, and SGLang
trees readable and traversable. Because Tau is installed non-editably, the
image fixes `TAU2_DATA_DIR=/opt/tau3/data` and verifies the bundled domain task
files. It does not grant arbitrary write access to the image's `/root` tree.

Before publishing the image, the job first starts it as the ordinary submitting
UID without code or cache binds and recursively checks the baked `/root` code
permissions. It then starts it with `--no-eval`, `--no-home`,
`--writable-tmpfs`, and the real `CONTAINER_MOUNTS`. That second smoke test
checks code imports, nested file binds, and write access to every bound
repository, dataset, checkpoint, and cache directory. The candidate is then
moved from node-local storage to a hidden staging name below
`$CONTAINER_DIR`; checksum and provenance are published before its final atomic
rename and stable-link update. A failed build leaves the current stable image
untouched.

The exact SGLang repository, branch, and commit are recorded alongside the OCI
reference in the provenance sidecar. Changing one makes the managed image stale,
so the next `experiments/setup.sh container --submit` rebuilds it.

The build node must support configured Singularity fakeroot builds. Do not work
around a fakeroot configuration error by building as host root: report it to
the cluster administrator so the resulting image can be reproduced by an
ordinary user.

For a reproducible image, set `DOCKER_IMAGE` to an immutable OCI digest before
submission, for example `registry.example/miles@sha256:<digest>`. The default
`radixark/miles:latest` remains convenient for development but can resolve to a
different image later; the build records the exact reference supplied in the
SIF provenance sidecar and warns when that reference is mutable.

## Prepare assets

`experiments/setup.sh` is the entrypoint for containers, model downloads,
dataset downloads, and SFT checkpoint conversion. Asset actions are dry-runs by
default:

```bash
experiments/setup.sh list
experiments/setup.sh all
experiments/setup.sh container
experiments/setup.sh models
experiments/setup.sh datasets
experiments/setup.sh sft
```

Review the printed jobs, then add `--submit` to enqueue them:

```bash
experiments/setup.sh all --submit
```

For a CPU-only preparation pass that builds the missing SIF first and then
downloads the training datasets, use:

```bash
experiments/setup.sh datasets --submit
```

The container and download submitters explicitly select the CPU-only
reservation profile (`R9920261300`, `RTYPE=rt_HC`, no requested GPUs). The
`models`, `sft`, and `all` actions can additionally enqueue HF-to-Megatron
checkpoint conversion, which uses CUDA/NCCL and therefore requests GPUs.

The `all` action preserves the dependency from container preparation to jobs
that use the image. Completed assets are detected and skipped where supported.
See [setup/README.md](setup/README.md) for the manifests and selective staging
entrypoints.

## Walltime policy

Defaults are sized by workload rather than by an old scheduler limit:

| Workload | Default | Environment variable |
| --- | ---: | --- |
| OCI to SIF container preparation | `00:30:00` | `PBS_CONTAINER_WALLTIME` |
| Deterministic dataset/environment preparation | `08:00:00` | `PBS_PREP_WALLTIME` |
| GPU checkpoint conversion | `08:00:00` | `SETUP_CONVERT_WALLTIME` |
| Network downloads | `24:00:00` | `PBS_DOWNLOAD_WALLTIME` |
| Training | `24:00:00` | `PBS_DEFAULT_WALLTIME` |

Request only the time the job is expected to need. Override a class before
submission, or one job with `pbs_submit --time=HH:MM:SS`. Setup-only overrides
are `SETUP_CONTAINER_WALLTIME`, `SETUP_PREP_WALLTIME`,
`SETUP_DOWNLOAD_WALLTIME`, and `SETUP_CONVERT_WALLTIME`.

Reusable recipes and `pbs_submit` are project-neutral: do not add `#PBS -P` to
job files, and `pbs_submit` never emits a project option. For a manual direct
allocation, the operator may pass the project on the `qsub` command line, for
example `-P gai51740`; the repository skill
`.claude/skills/miles-run-ladder/SKILL.md` records the interactive and batch
forms. This keeps account selection outside portable scripts.

## Submit training

Source both the environment and PBS helper in the shell that submits jobs:

```bash
source experiments/env.sh
source experiments/common/pbs.sh

pbs_submit \
  --profile=gpu \
  --nodes=4 \
  --time="${PBS_DEFAULT_WALLTIME}" \
  --export=ALL,ACTOR_NUM_NODES=2,ACTOR_GPUS_PER_NODE=8,ROLLOUT_NUM_GPUS=16 \
  experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch
```

`pbs_submit` translates the selected resource profile to PBS resources, creates
the log directory, and returns the full PBS job ID. Both profiles default to the
reservation queue: `gpu` passes `RTYPE=rt_HF` and requests all eight GPUs plus
192 CPUs per node, while `cpu` passes `RTYPE=rt_HC` and requests 32 CPUs with no
GPUs. The helper still does not pass a project. Recipe launchers call the shared
Singularity helper; multi-node commands load `ABCI_HPCX_MODULE` (default
`hpcx/2.20`) and run one MPI task per unique `$PBS_NODEFILE` host with
`-map-by ppr:1:node -bind-to none`. The node wrapper logs its allowed CPU set
before starting Singularity and validates it against `--cpus-per-task`.

For a local container preflight on a node with the required resources:

```bash
source experiments/env.sh
source experiments/common/singularity.sh
miles_container_exec \
  --image "${CONTAINER_IMAGE}" \
  --bind "${CONTAINER_MOUNTS}" \
  -- python -c 'import torch; print(torch.cuda.is_available())'
```

`miles_container_exec` uses `--no-eval`, `--no-home`, and `--writable-tmpfs` by
default. Persistent writes must target one of the mounts declared in
`CONTAINER_MOUNTS`; changes elsewhere in the tmpfs disappear with the job.

## Inspect and cancel jobs

Keep the full job ID returned by `pbs_submit`, including its server suffix when
present:

```bash
qstat -u "${USER}"
qstat -f <job-id>
qdel <job-id>
```

PBS combines stdout and stderr for these jobs. Logs are written below
`experiments/outputs/`; wrappers may select a recipe-specific subdirectory.
When `pbs_submit` receives a directory output target, PBS names the file
`<full-job-id>.OU`.
