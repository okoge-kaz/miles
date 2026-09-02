---
name: miles-run-ladder
description: Take a Miles RL recipe from PBS/Singularity bring-up to a production run on the R9920261300 GPU queue. Use when preparing, reviewing, sharing, or executing qsub or pbs_submit commands; requesting an interactive GPU allocation; scaling or tuning a recipe; or deciding whether a configuration is ready for a durable run. Preserve checkpoint/export guarantees and change only one class of settings at each stage.
---

# Miles Run Ladder on PBS

Use three stages, each answering one question:

| Stage | Execution mode | Typical shape | Question |
| --- | --- | --- | --- |
| 1. Bring-up | Interactive PBS allocation | 2 nodes, reduced workload | Does the real multi-node path work? |
| 2. Tuning | Short batch jobs | 2 or more nodes | Which systems setting improves step time? |
| 3. Production | Batch job | Intended full shape | Does it learn and resume correctly? |

Do not carry over the former Slurm partition names, Enroot options, AWS EFA
settings, or four-hour walltime assumption. Runtime jobs use the maintained
Singularity SIF and the system InfiniBand/NCCL stack.

## Request an allocation

For a direct interactive allocation, replace `<nodes>` with a positive node
count:

```bash
qsub -I -P gai51740 -q R9920261300 \
  -l select=<nodes> -l walltime=12:0:0 -v RTYPE=rt_HF
```

For a directly submitted batch script:

```bash
qsub -P gai51740 -q R9920261300 \
  -l select=<nodes> -l walltime=12:0:0 -v RTYPE=rt_HF \
  job_script.sh
```

These are operator-facing `qsub` examples. Pass the project with the command;
do not hard-code `#PBS -P gai51740` into reusable job scripts. Maintained Miles
recipes use `pbs_submit` and project-neutral `#PBS` headers so the checkout can
move between accounts. For GPU jobs, `pbs_submit` pairs the reservation queue
with its configured `RTYPE=rt_HF`. Do not add `-P` to those scripts unless the
user explicitly asks to change that repository-wide policy.

The 12-hour value is a normal allocation example, not a cluster maximum. Size
walltime to the task. Container preparation normally uses a 30-minute CPU-only
job in the reservation with `RTYPE=rt_HC`; it does not request GPUs. GPU
training keeps the `RTYPE=rt_HF` allocation shown above.

Writing or reviewing a command does not authorize a live allocation. Run
`qsub` or use a submitting wrapper only when the user explicitly asks to submit
or execute the job.

After an interactive allocation starts:

```bash
cd /groups/gai51740/kazuki_fujii/src/miles-v0.0
export MILES_WORKSPACE_ROOT=/groups/gai51740/kazuki_fujii
source experiments/env.sh
source experiments/common/pbs.sh
source experiments/common/singularity.sh
```

Change only `MILES_WORKSPACE_ROOT` when the workspace moves. All checkpoint,
container, dataset, and cache paths derive from it.

## Gate 0: Prepare and verify assets

Before reserving GPUs, inspect the workspace and prepare the SIF and required
assets:

```bash
experiments/setup.sh init
experiments/setup.sh status
experiments/setup.sh container
experiments/setup.sh container --submit
```

The container job builds in scheduler-provided node-local storage, validates
ordinary-user access to image paths below `/root`, then publishes a verified SIF
under `$MILES_WORKSPACE_ROOT/containers`. Do not replace it with a bare
`singularity pull` or build a shared image as host root.

Use `experiments/setup.sh models`, `datasets`, `sft`, or `all` to preview the
remaining asset graph; add `--submit` only after reviewing it. Keep the full PBS
job ID, including its server suffix, when inspecting dependencies or status.

## Checkpoint gate for durable jobs

Treat a job as disposable only when it is explicitly a validation or smoke
run. Before sharing or submitting any other command, resolve the actual wrapper,
recipe, and `train.sh` values and require:

1. A unique writable training checkpoint path.
2. Resumable Megatron `torch_dist` saving with a positive `SAVE_INTERVAL`.
3. `SAVE_HF=1` and a positive `HF_SAVE_INTERVAL` when downstream evaluation
   needs Hugging Face checkpoints.
4. A retention cadence that preserves the checkpoints the experiment depends
   on.

Confirm that `train.sh` receives both `--save-hf` and `--hf-save-interval`.
Defaults in one wrapper are not proof if another layer can override them. State
explicitly when a disposable validation job intentionally skips HF export.

## Stage 1: Multi-node bring-up

Use at least two nodes. A one-node run does not exercise PBS node discovery,
the host MPI launcher, Ray worker join, cross-node weight transfer, or
multi-node data parallelism.

ABCI node-level launchers use the host HPC-X Open MPI module (default
`ABCI_HPCX_MODULE=hpcx/2.20`) with one process per PBS node. Every such launch
must include the following options before Singularity starts:

```text
-hostfile <unique-PBS-hostfile> -np <nodes> -map-by ppr:1:node -bind-to none
```

Do not omit `-bind-to none`. Open MPI otherwise binds the node launcher to a
small CPU set, and Ray/torch children inherit that mask. The shared launcher
prints `Cpus_allowed_list` and its CPU count on every node; for a full `rt_HF`
node it must report all 192 requested CPUs before training starts. Load HPC-X
through `/etc/profile.d/modules.sh` in a non-interactive PBS shell when the
`module` function is not already initialized.

ABCI can expose an allocated GPU list as UUID values in
`CUDA_VISIBLE_DEVICES`, which Ray then returns from `ray.get_gpu_ids()`. Some
Miles SGLang launch paths require numeric local GPU ordinals. For a recipe that
requests exclusive nodes and consumes every `GPUS_PER_NODE` GPU, export a
numeric full-node view inside each Singularity task before starting Ray:

```bash
[[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || exit 1
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$(( GPUS_PER_NODE - 1 ))")"
```

Do not apply this ABCI workaround to partial or nonexclusive GPU allocations;
`0..N-1` may then identify GPUs outside the allocation. Set
`CUDA_VISIBLE_DEVICES`, not `NVIDIA_VISIBLE_DEVICES`, and verify the export
precedes the Ray head and worker startup on every node.

Shrink only workload controls so failures surface quickly. Preserve the
production algorithm, placement, model, and identity inputs. Keep this invariant:

```text
rollout_batch_size * n_samples_per_prompt
  = global_batch_size * num_steps_per_rollout
```

Check these signals in order:

1. The log reports that every PBS node joined the Ray cluster.
2. The placement line accounts for the complete allocation and intended actor
   and rollout split.
3. The checkpoint path contains the intended task, dataset, model, and unique
   configuration identity.
4. The first samples have a reward compatible with the model's response format.
5. Each configured evaluation benchmark reports its own result.
6. One checkpoint is saved, and rerunning the same command reloads it.

Only then restore production sequence lengths and batch sizes and confirm that
one complete step fits without OOM.

## Stage 2: Systems tuning

Give every tuning point a unique `CONFIG_TAG`. Change exactly one class of
setting per comparison and record measured step time.

Training-side settings include tensor/context/expert parallelism,
`MAX_TOKENS_PER_GPU`, actor node count, and actor GPUs per node. Always preserve:

```text
max_tokens_per_gpu * context_parallel_size >= rollout_max_context_len
```

Rollout-side settings include GPUs per engine, SGLang memory fraction, maximum
running requests, CUDA graph batch size, total rollout GPUs, and async sample
concurrency. Use observed utilization, KV-cache pressure, and request
concurrency instead of theoretical maxima.

Do not tune learning rate, batch semantics, dynamic sampling, or staleness bounds
in this stage. Those alter learning rather than systems throughput. For a grid,
use `experiments/sweep.py` in dry-run mode first and retain its manifest joining
configuration identities to full PBS job IDs.

## Stage 3: Production

Use the intended allocation and a workload-appropriate walltime; there is no
four-hour cluster ceiling. Freeze the systems settings selected in stage 2.

Before submission:

- Set a positive save cadence that limits work lost at walltime.
- Confirm HF export and retention for every downstream evaluation dependency.
- Scale the rollout batch deliberately; adding nodes alone changes data
  parallelism but does not define the desired global batch.
- Leave high-volume training dumps off unless a measured short run establishes
  their expected storage cost.
- Use a `CONFIG_TAG` that cannot collide with another run.
- Verify the exact command once with reduced workload while leaving all derived
  names and paths unchanged.

Do not create a long `afterany` dependency chain until the first link is known
to start, save, and resume correctly. `afterany` releases successors after a
failure as well as after success. Prefer success dependencies for asset and
evaluation pipelines.

## Operational rules

- Use `qstat -f <job-id>` and `qdel <job-id>` with the full PBS job ID.
- Submit maintained recipes from the repository root so `PBS_O_WORKDIR` is the
  correct checkout.
- Run payloads through `experiments/common/singularity.sh`; do not reintroduce
  Pyxis or Enroot flags.
- Persistent writes belong below the configured checkpoint, dataset, container,
  or cache roots. Image-internal writable tmpfs state disappears after the job.
- Report measured step times, GPU utilization, checkpoint paths, and result
  counts. Do not describe queue acceptance or job completion alone as a
  successful training or evaluation result.
