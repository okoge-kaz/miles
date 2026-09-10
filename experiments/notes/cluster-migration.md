# cw-dfw experiment integration on OCI (2026-09-09)

Target: `experiments/2026-09-09`, based on main `f1eac9ba`.
Incoming: `origin/experiments/cw-dfw-math-rl`, tip `24eeb68b`.
This is a port across changed core APIs, not a replacement of main with an old tree.

Integration result: cw-dfw changes are ported onto the current main-based branch;
no push was performed. The user's checkpoint/data assets are placed and HF→MCore conversion
is complete. Two-node training/save/eval and a separate checkpoint-resume run
passed under `nemotron_sw_post / batch / interactive`.
See [Step4000 bring-up](oci-step4000-bringup.md) for paths, commands, job IDs and
the distinction between functional validation and production qualification.

## Current architecture and experiment opt-in

- Preserve main's `RolloutExecutor`, `InferenceController`, `TrainerController`,
  `TrainerCell`, shared weight updater, fault-tolerant `TrainStepOutput`, and
  typed per-call `WeightVersionSpan` representation. The retired
  `RolloutManager`, `RayTrainGroup` and old backend-specific weight updaters are
  not restored.
- Standard `--fully-async` retains main's DataBuffer/sample scheduling behavior.
  An explicit `--fully-async-queue-type` or `--use-replay-buffer` selects
  `experimental_fully_async_rollout.FullyAsyncRolloutFn`. Its queue-recycle,
  queue-drop, queue-max, replay persistence, staleness and throughput telemetry
  are ported to the current executor and driver. Shared evaluation pauses
  producer submissions; dedicated evaluation keeps its own fleet.
- Port policy-lag/TIS/ESS diagnostics, truncation treatments, fused one-step
  actor-logprob options, batching metrics and switch/weight-transfer timing.
  Timing is opt-in and preserves the default trainer return contract.
- Preserve current Python model definitions and derive their CLI arguments with
  `python -m miles.utils.external_utils.model_args_utils`; do not resurrect
  deleted `scripts/models/*.sh` files.
- The incoming branch itself removed `experiments/src` and queue analysis in
  `3fd68195`, but retained orphan tests. Those tests and tests of core modules
  already deleted on main are retired, rather than importing nonexistent code.
- The imported experimental queue remains a large lifecycle module to preserve
  its behavior during this API port. A separate mechanical split is deliberately
  outside this integration; this is an exception to the general file-size preference.

## OCI launch contract

See [cluster.md](cluster.md) for the commands that distinguish partition and QoS,
allocation/release commands, filesystem observations and live-limit checks.

- Account: `nemotron_sw_post`, including direct `#SBATCH` headers, submission
  wrappers, sweep defaults and `.env.example`.
- GPU validation: `batch / interactive`; production: `batch / normal`.
  CPU interactive: `cpu / cpu-interactive`. `interactive` is not a partition.
- aarch64 CPU, 4 GB200 GPUs/node. Shapes derive from `GPUS_PER_NODE=4`.
- Workspace/assets are configurable; defaults use the current user's OCI
  workspace. Image import selects aarch64 and uses unique node-local `/tmp`
  scratch. `/tmp` was ext4, not RAM-backed; cw-dfw `/raid` is not assumed.
- Historical eight-H100 matched studies are not GB200 baselines. Matched-mode
  submissions explicitly reject incompatible topology/architecture.

Disposable validation (submit from repository root):

```bash
mkdir -p experiments/outputs
SQSH_IMAGE=/absolute/path/to/qualified-aarch64-image.sqsh \
  sbatch tests/manual/run_oci_validation.sbatch
```

This recipe uses two nodes, exercises all eight GPUs with NCCL, then runs
selected regressions. It does not train or save checkpoints. GPU-free Ray mock
tests advertise logical GPU resources; their local `CUDA_VISIBLE_DEVICES` unset
is not used for the physical GPU communication step or training.

## Evidence collected during integration

- CPU import job `7035384` produced
  `experiments/outputs/miles-oci-aarch64-7035384.sqsh` from `radixark/miles:latest`.
  This local artifact is ignored by Git, not a portable asset guarantee.
- GPU allocation `7035620`: `batch / interactive`, nodes
  `nvl72042-T01` and `nvl72042-T16`, four GPUs each. This allocation preceded
  the account-change request and used `coreai_horizon_dilations`; it is not
  evidence of a GPU runtime job under the new account.
- CPU interactive job `7036198` actually ran under `nemotron_sw_post`.
  `sbatch --test-only -A nemotron_sw_post -p batch --qos=interactive -N 2
  --ntasks-per-node=1 --gpus-per-node=4 --time=00:20:00 --wrap=true`
  was accepted. Test-only admission is not execution.
- CUDA matrix operation passed. Two-node `tests/manual/oci_nccl_smoke.py`
  passed on all eight ranks (all-reduce sum 36); compile checks passed.
- Main `f1eac9ba` and merged code, evaluated in separate processes on the same
  ARM runtime with identical saved inputs: **486 common output/gradient tensors
  across 12 loss configurations matched exactly** (`rtol=0`, `atol=0`). Ten
  additional diagnostic fields were accounted for separately. Evidence:
  `experiments/outputs/loss-parity-{main,merged}.pt` and the comparison scripts
  in that ignored output directory.
- Final focused regression run: **1859 passed, 6 expected failures**, 201.88 s
  (`experiments/outputs/oci-merge-final-focused.log`). It covers both async paths,
  replay/queue contracts, session codecs, loss/gradient and weight-update unit
  tests, SGLang API/argument contracts, launchers and train drivers. A separate
  run passed all 41 executor tests and 3 switch-timing tests; with the four
  passing legacy loss configurations, its log reports 48 passes and the eight
  baseline numerical failures described below.
- Final launcher/resource-contract checks: **104 passed** (1.14 s), with shell
  syntax checks also passing for the manual scripts. The reusable validation
  sbatch recipe was admitted by `sbatch --test-only` (forecast `7036900`);
  this did not submit another job. The updated run-ladder skill passed the
  skill-creator validator inside the interactive container.
- Legacy frozen loss snapshots already fail 8/12 configurations on the clean
  main baseline with this aarch64/PyTorch runtime (for example, TIS exponential
  differences of about 3.8e-6). Their tolerance has not been relaxed. The
  comparison projects out only the ten explicitly named new diagnostic fields;
  all existing values and gradients remain strictly compared.
- The broad trainer/Ray run exceeded its 600-second validation time budget;
  it is not a full-suite pass. The isolated no-live-cell failure-path test passed.
  Unchanged AMD/Kimi launcher snapshot tests also failed in this image; this
  integration does not change those scripts or claim that all repository tests pass.
- Fast-test collection found 8236 tests and six collection errors, all involving
  Hugging Face tokenizer assets absent from the offline validation environment.
  It did not report a remaining import of a retired core/experiment module.

## Workload validation and remaining gates

Checkpoint and dataset transfer is complete; see the
[current asset paths and execution evidence](oci-step4000-bringup.md).
Do not substitute an unrelated checkpoint or unfiltered dataset for the
Step4000/p10–90 cohort. Conversion job `7058753` and initial sync training job
`7059547` succeeded. The latter verified 2-node/8-GPU generation, nonzero reward,
finite loss and nonzero gradient, weight transfer, MCore/HF saving and a named
two-row functional eval. The full HF export is finite and differs from the
initial weights. Resume job `7059888` also completed (`0:0`): it loaded the saved
state, ran step 1 with a finite nonzero gradient, wrote MCore/HF iteration 1 and
evaluated the updated policy. Saved scheduler/data counters advanced correctly.
The optional full `hf/1` tensor comparison on CPU was stopped during prolonged
Lustre I/O wait, after at least 200/398 tensors. It is not a full-scan pass;
the initial `hf/0` export did pass its independent all-tensor finite/delta check.

Bring-up exposed and fixed an OCI ephemeral-port collision, an HTTP-202 router
registration race, and a misplaced parallel-config publication in the merge.
The saved-scheduler option is now explicit for a smoke run whose stop point is
extended from one rollout to two. The corrected resume wrapper passed 290
preflight regression checks on the interactive allocation.

The validated workload is a **small-batch sync/colocated smoke** (4 prompts × 4
responses, max response length 8192), not the production default batch of 3072.
Production-shape memory/throughput and long-run learning still require the rest
of the run ladder. DAPO smoke evaluation overlaps training data; no AIME assets
were transferred and no held-out benchmark score is claimed.

The imported image contains Python 3.12.3, PyTorch 2.13.0+cu130, Ray 2.58.0 and
SGLang `0.5.20.dev54+ga8e5c63` (`a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`).
It supports the basic weight-update request but lacks the experiment's
`first_prefill_weight_version` provenance. Prefill-referenced staleness is
**not qualified**. Use an explicitly compatible ARM fork/image and pass the
prefill smoke before enabling that treatment; do not blindly install the old
0.5.17 fork over this image. Optional evaluator images and fully-async training
are also not yet qualified by the sync bring-up.
