# Experiment setup

`experiments/setup` contains reproducible asset staging and preparation jobs.
Runtime environments and rewards do not live here: their implementations are
under `experiments/src/environments`, dataset adapters under
`experiments/src/datasets`, and persistent cluster tests under `tests/slurm`.

## Unified entrypoint

Use `experiments/setup.sh` for the common one-time workspace operations. Asset
actions run in preview mode by default. The standalone `sft` preview validates
that its source checkpoints exist; the `all` preview reports missing external
SFT exports without stopping so a fresh workspace can display the complete
plan. Add `--submit` only when the commands should be submitted to PBS.

```bash
# Create directories and inspect what is already present.
experiments/setup.sh init
experiments/setup.sh status
experiments/setup.sh list

# Preview one group or the complete setup.
experiments/setup.sh container
experiments/setup.sh models
experiments/setup.sh datasets
experiments/setup.sh sft
experiments/setup.sh all

# Submit the complete PBS dependency graph.
experiments/setup.sh all --submit

# Build the SIF and download training datasets using CPU jobs only.
experiments/setup.sh datasets --submit
```

`all --submit` validates every SFT checkpoint source before queueing its first
job, so a missing external SFT export cannot leave a partially submitted setup.
It also includes CUDA/NCCL checkpoint conversion. Use `datasets --submit` when
only the SIF and dataset assets should be prepared: the container build and all
download/materialization jobs explicitly select the CPU-only reservation
profile (`R9920261300`, `RTYPE=rt_HC`, no requested GPUs).

| Group | Delegated worker or inventory |
| --- | --- |
| `container` | `experiments/container/import_image.sbatch` |
| `models` | `download/stage_all.sh models` and `manifests/models.txt` |
| `datasets` | `download/stage_all.sh datasets` and `manifests/datasets.txt` |
| `sft` | `models/stage_sft_checkpoints.sh` and `manifests/sft_checkpoints.txt` |
| `all` | Submit the container first, then make model, dataset, and SFT jobs depend on it. |

Change `MILES_WORKSPACE_ROOT` in the environment, or its default in
`experiments/env.sh`, to move the complete persistent layout. The derived
checkpoint, container, dataset, and cache paths can still be overridden
individually for exceptional runs. The default layout is:

```text
$MILES_WORKSPACE_ROOT/
  checkpoints/{hf,megatron,training}/
  containers/
  datasets/{pre-train,rl,sft}/
  cache/
  src/
```

PBS project flags are deliberately omitted from reusable scripts and
`pbs_submit`; an operator can pass `-P gai51740` to a manual direct `qsub`
command. Default walltimes are 30 minutes for container imports, eight hours for
preparation and checkpoint conversion, and 24 hours for downloads. Override `PBS_CONTAINER_WALLTIME`,
`PBS_PREP_WALLTIME`, or `PBS_DOWNLOAD_WALLTIME` before submitting. The
corresponding `SETUP_*_WALLTIME` variables provide a narrower override for
setup commands only.

The generic `datasets` group covers `manifests/datasets.txt`. Pinned or
workflow-specific staging remains selectable through the entrypoints printed by
`experiments/setup.sh list`, including Nemotron RL, SWE/tool-use RL, and AReaL
Tau2 assets. Bulk manifest submission waits one second between `qsub` calls to
avoid scheduler bursts; override `SETUP_SUBMIT_DELAY_SECONDS` only when the
site's submission limits are known.

A dataset is complete only when it has both a nonempty
`MILES_SOURCE_PROVENANCE` marker and a recursively discovered payload file.
Interrupted Hugging Face downloads are resumed with two workers and up to five
attempts by default. Tune `HF_DOWNLOAD_MAX_WORKERS`, `HF_DOWNLOAD_ATTEMPTS`, and
`HF_DOWNLOAD_RETRY_DELAY_SECONDS` when the Hub or site policy requires it.

## Container build and permissions

The `container` action submits `experiments/container/import_image.sbatch` as a
CPU-only reservation job with the 30-minute container walltime. The job builds the configured
OCI source through `experiments/container/miles.def` using Singularity
`--fakeroot --fix-perms`; it does not use a bare OCI-to-SIF pull.

For PBS builds, build output, unpack temp files, and the build-time OCI layer
cache must live below `/local/<job-id>`; a `PBS_LOCALDIR` value is accepted only
when it is itself below `/local`. The job fails instead of building on shared
storage when that node-local path is unavailable. Manual non-PBS builds may
fall back to `TMPDIR`/`/tmp`. After validation, the job moves the SIF via a
hidden staging name into `$CONTAINER_DIR` and publishes it. The Singularity
definition is copied to the local build directory with mode `0644` and checksum
validated before fakeroot starts, avoiding shared-checkout traversal inside the
builder. Publication writes checksum and provenance sidecars before updating
`miles.sif`.

This wrapper is required because the runtime preserves the submitting UID, but
the OCI image installs Miles, Megatron, Tau, and SGLang below paths that are
normally reachable only through `/root`. The definition file:

- makes `/root` traversable without making the baked tree generally writable;
- makes the baked code trees readable and traversable by the runtime UID;
- checks out and tests the pinned Miles SGLang revision that reports
  scheduler-authoritative policy provenance for asynchronous training;
- installs and verifies the pinned Tau v1.0.1 runtime formerly carried by a
  separate derived image;
- creates static dataset, checkpoint, cache, evaluation, and file bind targets;
- keeps `/tmp` and `/var/tmp` suitable for job-local writes.

After the build, the same job first executes the candidate without code binds
or a writable overlay and recursively checks the baked `/root` code permissions
as the ordinary user. It then runs with `--no-eval`, `--no-home`,
`--writable-tmpfs`, and the actual `CONTAINER_MOUNTS` to check imports, the
nested `.env` mask, and read/write access. Only then is the candidate moved into
its dated final path and `$CONTAINER_DIR/miles.sif` updated. A failed test cannot
replace the stable image.

To submit only this job without the unified entrypoint:

```bash
source experiments/env.sh
source experiments/common/pbs.sh
pbs_submit --profile=cpu --time="${PBS_CONTAINER_WALLTIME}" \
  experiments/container/import_image.sbatch
```

If `--fakeroot` is unavailable on a build node, treat that as a cluster
configuration problem. Do not build the shared SIF as host root.

Set `DOCKER_IMAGE` to `repository@sha256:<digest>` when the SIF must be exactly
reproducible. Mutable tags such as the development default `latest` are allowed,
but the build warns about them and records the supplied OCI reference in the
provenance sidecar.

## Layout

```text
experiments/setup/
  download/       reusable transfer jobs and staging orchestration
  models/         checkpoint conversion and model-only orchestration
  manifests/      declarative model and dataset inventories
  datasets/       deterministic dataset conversion and blend construction
  environments/   pinned environment assets and external dependencies
tests/slurm/       persistent cluster/container integration tests
experiments/setup.sh  unified dry-run/PBS submission entrypoint
```

### Download and staging

| File | Purpose |
| --- | --- |
| `download/download_dataset.sbatch` | Download one optionally revision-pinned Hugging Face dataset, record provenance, and reject README-only gated downloads. |
| `download/download_model.sbatch` | Download one Hugging Face model and write a completion marker. |
| `download/download_git_dependency.sbatch` | Clone and optionally pin an external source checkout. |
| `download/stage_all.sh` | Stage the generic model and dataset manifests. |
| `download/stage_model.sh` | Download and convert one model from the model manifest. |
| `download/stage_nemotron_rl_datasets.sh` | Stage the role-annotated Nemotron dataset manifest. |
| `download/download_areal_tau2.sbatch` | Download only the pinned AReaL Tau2 RL tasks and nine DB snapshots. |
| `download/stage_areal_tau2.sh` | Stage and validate the complete AReaL Tau2 user-simulator training input. |
| `download/download_swe_datasets.sbatch` | Selectively fetch pinned Super SWE2, Ultra SWE, R2E-Gym, SWE-rebench V2, SWE-Gym, or SWE-bench Verified without downloading complete Nemotron blends. |
| `download/stage_swe_tool_rl_datasets.sh` | Verify or stage the exact SWE-ReBench V2, SWE-Gym, and NVIDIA conversational tool-use Pivot revisions used by the RL recipes. |
| `download/download_swe_verifiers.sbatch` | Fetch and checksum-pin the official R2E-Gym and SWE-rebench V2 parser modules. |

The old `download_assets.sbatch` was removed. It hard-coded one model and two
datasets and was fully superseded by these manifest-driven jobs.

### Models and manifests

| File | Purpose |
| --- | --- |
| `models/convert_checkpoint.sbatch` | Convert Hugging Face weights to a Megatron `torch_dist` checkpoint. |
| `models/stage_sft_checkpoints.sh` | Validate or submit serialized conversion of existing SFT checkpoints. |
| `manifests/models.txt` | Model source, Megatron type, conversion overrides, and node count. |
| `manifests/datasets.txt` | General Hugging Face dataset inventory. |
| `manifests/nemotron_rl_datasets.tsv` | Nemotron inputs annotated by training/evaluation/environment role. |
| `manifests/sft_checkpoints.txt` | Existing Hugging Face-format SFT checkpoints to convert; source directories are relative to `HF_CKPT_DIR`. |

### Dataset preparation

| File | Output or purpose |
| --- | --- |
| `datasets/prepare_nemotron_rl_math.sbatch` | Restored Nemotron Math v2 Miles JSONL. |
| `datasets/prepare_nemotron_static_components.sbatch` | Deterministic-reward Nano components as per-dataset Miles JSONL. |
| `datasets/prepare_nemotron_nano.sbatch` | Restored Nano blend partitioned by verifier availability. |
| `datasets/prepare_gpqa.sbatch` | GPQA Diamond/Main/Extended evaluation JSONL. |
| `datasets/prepare_livecodebench.sbatch` | LiveCodeBench evaluation prompts with public/private tests. |
| `datasets/prepare_competitive_code_validation.sbatch` | Sampled competitive-programming validation split. |
| `datasets/prepare_performance_transfer_blends.sbatch` | Benchmark-safe competitive-code and STEM training blends. |
| `datasets/prepare_math_code_stem_blend.sbatch` | Balanced Math/Code/STEM multi-environment blend. |
| `datasets/prepare_math500.sbatch` | Convert the held-out MATH-500 split to canonical Miles JSONL. |
| `datasets/prepare_tool_call_pivot_env.sbatch` | Disjoint function-call-only Pivot train/evaluation splits. |
| `datasets/prepare_agentic_tool_use_pivot.sbatch` | Convert the pinned conversational tool-use Pivot source, reserve 2,000 exact-action rows, and use all remaining eligible calls for RL. |
| `datasets/build_math_jsonl.py` | Legacy raw-math JSONL adapter used by the math preparation job. |
| `datasets/prepare_swe_rl.sbatch` | Split pinned SWE sources by executable schema into gold-free rows plus owner-only Harbor manifests; unsupported R2E/SWE-Gym rows remain quarantined. |

These jobs only perform deterministic conversion, validation, sampling, or
merging. New conversion logic belongs in `experiments/src/datasets`; the
explicitly marked legacy math adapter is the remaining migration target.

### Environment assets and dependencies

| File | Purpose |
| --- | --- |
| `environments/prepare_calendar_env.sbatch` | Convert Calendar v2 rows and preflight the local constraint verifier. |
| `environments/prepare_workplace_local_env.sbatch` | Convert Workplace rows against pinned resource fixtures. |
| `environments/prepare_tau_bench.sbatch` | Validate Tau three v1.0.1 splits and materialize only the held-out test tasks for evaluation. |
| `environments/prepare_areal_tau2.sbatch` | Validate all 1,982 external Tau2 RL tasks/DBs and precompute gold terminal states. |
| `environments/prepare_ifbench.sbatch` | Pin IFBench dependencies and build held-out evaluation data. |
| `environments/prepare_search_r1.sbatch` | Assemble Search-R1 index, corpus, encoder, and benchmark data. |
| `environments/prepare_ifeval_dependencies.sbatch` | Pin the IFEvalG dependency closure. |
| `environments/prepare_reasoning_gym_dependencies.sbatch` | Pin the official Reasoning Gym scorer. |
| `environments/materialize_harbor_swe_tasks.sbatch` | Materialize fresh-verifier Harbor task trees; immutable images and live R2E admission are mandatory outside explicit dry-run mode. |

The one-time `inspect_nemotron_schemas.sbatch` probe was removed after its
schema findings were encoded in the canonical adapters and regression tests.
Durable data/container checks are documented in `tests/slurm/README.md`; no
ad-hoc smoke or test launchers remain in setup.
