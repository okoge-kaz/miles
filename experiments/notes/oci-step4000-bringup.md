# OCI Step4000 assets and bring-up (2026-09-10)

## Asset placement

The user supplied `qwen3-4b-rl-assets` under the user workspace. Its two asset
directories were moved, without overwriting existing destinations, to:

- `datasets/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/`
- `checkpoints/hf/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000/iter_0004000/`

Paths are relative to `/lustre/fsw/portfolios/coreai/users/kfujii`.
The source/destination device and directory inode numbers match: these were
same-filesystem renames, not re-encoded copies. Empty transfer parent directories
are retained. The source HF checkpoint remains available after MCore conversion.

The HF index names two safetensors shards and 399 tensors (8,822,848,512 payload
bytes). Config and tokenizer files, including `chat_template.jinja`, are present.
The dataset has 10,778 rows with `prompt` message arrays, string `label`, and
`difficulty`. These structural checks do not by themselves validate training.

This checkout's ignored `.env` pins `DATASET_DIR`, `QWEN3_4B_BASE_HF_ROOT` (the
parent of `iter_0004000`) and the previously imported ARM `SQSH_IMAGE`. The
recipes therefore see `/ckpt/hf/iter_0004000` without flattening the original
checkpoint's model/learning-condition hierarchy on the host.

## Validation plan and limits

Rechecked `sinfo`, `scontrol show partition batch`, `sacctmgr show qos`, and the
user associations before allocation. Job `7058753` uses `nemotron_sw_post`,
`batch / interactive`, and nodes `nvl72101-T06` and `nvl72101-T15` (4 GPUs each).

HF→MCore conversion uses `tools/convert_hf_to_torch_dist.py` and the current
`qwen3-4B` Python model arguments, with four local torchrun ranks. Its target is
`checkpoints/megatron/Qwen3-4B-Base-LR2e-5-Step4000_torch_dist`.
The `release` tracker must be written only after conversion succeeds.
Log: `experiments/outputs/oci-step4000-convert-7058753.log`.

Conversion completed successfully: Slurm step `7058753.0`, exit `0:0`, elapsed
5 min 12 s. The output contains four `.distcp` shards and metadata under
`release/`, with `latest_checkpointed_iteration.txt` equal to `release`.
Step `7058753.1` loaded the local tokenizer/chat template and passed 104 launcher
checks. The allocation was released after both steps completed.

Training bring-up starts with the sync/colocated recipe on two nodes. The image
does not qualify the experiment's prefill-provenance API; do not treat a sync
success as validation of prefill-referenced fully-async staleness.

Only the training dataset was transferred, not AIME. The math recipes now expose
`EVAL_DATASET_NAME` and `EVAL_PROMPT_DATA` (defaults unchanged: aime25 and its
usual JSONL). A small, explicitly named `dapo-smoke` slice can exercise evaluation
plumbing without inventing an AIME result. Such an overlapping train/eval slice
is a functional check, not a held-out quality measurement.

Conversion and the initial small-batch training/save/eval run are complete;
checkpoint resume and production-shape qualification are tracked separately.

The first training job, `7058972`, loaded the converted checkpoint successfully
at TP2/PP1 (resharding the converter's TP1/PP4). It formed Ray 2/2 and initialized
all eight training ranks, but one SGLang HTTP server failed to bind port 20001.
`ss -ltnp` found that port already owned by another scheduler in the same job;
the node's ephemeral range was 9000–65000. Cgroup `oom_kill` counters were zero.
The job was stopped before rollout/training. The fix confines experiment-managed
ports to 2000–8999; no reward, optimizer, or parallelism settings were changed.
Allocator and launcher regression checks passed: 132 tests on the interactive
allocation. See `cluster.md` for port discovery and the configuration variables.
Failure evidence: `experiments/outputs/oci-step4000-driver-7058972.log` and
`experiments/outputs/oci-step4000-engine-port-collision-7058972.log`.

Job `7059325` then initialized all four TP2 engines with the bounded port range
and distributed training weights successfully. It exposed a second bring-up
issue: router 0.3.2 queues `POST /workers` with HTTP 202, while Miles immediately
started a 33–36 s engine checksum. Router metadata discovery timed out after
10 s per attempt, before the checksum completed. Direct metadata requests and
manual re-registration after checksum completion succeeded. The API client now
waits, with a 120 s overall deadline, until an accepted worker is listed healthy
before allowing dependent operations to continue. Legacy synchronous responses
retain their old behavior. DP-aware `@rank` URLs are normalized for the check.

The same diagnostic job generated real DAPO responses (first-sample reward 1),
then exposed an integration error: rank-zero parallel-config publication had
been displaced below an unrelated method's `return`. It has been restored to
`TrainRayActor.set_rollout_executor`, alongside the weight-updater wiring.
Existing publication tests and a new combined regression cover both operations.
An AST scan of all staged Python files found no other statements directly after
an unconditional return/raise/break/continue. This job did not reach training or
save and is not a successful smoke qualification.

Job `7059547` was the fresh, no-manual-registration retry. Its wrapper first runs
CPU-only regression checks within the GPU interactive allocation, then launches
the unchanged small-batch learning configuration. Its outcome is recorded below.

The disposable wrapper is `tests/manual/run_oci_step4000_smoke.sbatch`.
It uses a unique, required `SMOKE_TAG`, 4 prompts × 4 responses = global batch
16, and saves MCore + HF after every rollout. First submit with
`SMOKE_NUM_ROLLOUT=1`; resubmit the same tag with `SMOKE_NUM_ROLLOUT=2` to
exercise one resumed optimizer step. Tracking is offline and the two-row
`dapo-smoke` evaluation is explicitly not an AIME benchmark.

Submit from the repository root, with the local asset/image overrides above:

```bash
sbatch --export=ALL,SMOKE_TAG=<unique-tag>,SMOKE_NUM_ROLLOUT=1 \
  tests/manual/run_oci_step4000_smoke.sbatch
# Only after confirming completion and both checkpoint formats:
sbatch --export=ALL,SMOKE_TAG=<same-tag>,SMOKE_NUM_ROLLOUT=2 \
  tests/manual/run_oci_step4000_smoke.sbatch
```

The checked shape is 8 colocated training GPUs, TP2/PP1/CP1/DP4, and four
TP2 rollout engines. Response length is capped at 8192, context at 32768.
Both resumable MCore saving and HF export run every rollout, retaining every
snapshot. The resolved checkpoint directory for the current attempt is:

```text
/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/training/math/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/Qwen3-4B-Base-LR2e-5-Step4000/grpo-clip0.2-0.28-tis2.0/colocated/on-policy/max-weight-staleness-0/oci-step4000-ready-20260910-zero-trunc
```

## Initial training result

Job `7059547` completed with Slurm exit `0:0` in 13 min 4 s on
`nvl72088-T[15,18]`, using `nemotron_sw_post / batch / interactive`.
It passed all 282 preflight checks before launching GPU workers; a separate
focused recheck passed 69 tests. All four router workers registered without
manual intervention.

- Ray formed 2/2 nodes; the converted TP1/PP4 checkpoint loaded at TP2/PP1.
- Rollout 0: 16 samples, mean raw reward 0.5625, mean response length 4350.5,
  truncation fraction 0.1875.
- Optimizer step 0: loss 0.0261717852, gradient norm 0.6445917487, LR 1e-6.
  Both loss and gradient norm are finite and the gradient is nonzero.
- MCore `iter_0000000` saved successfully: eight shards plus metadata, with
  tracker `0`. This recipe uses zero-based rollout IDs for checkpoint iteration
  names; `0` here contains the state after the first optimizer step.
- HF `hf/0` contains `.complete`, config/tokenizer/chat template, an index and
  all 16 referenced safetensors shards (398 tensors, 8,044,936,192 payload bytes).
- Weights were transferred to SGLang after training, then the two-row
  `dapo-smoke` eval completed with reward 1.0 and no truncation. This is a
  functional train/eval-overlap check, not a benchmark result.

Evidence: `experiments/outputs/oci-step4000-driver-7059547.log`,
`experiments/outputs/miles-oci-step4000-smoke-7059547.log`, and the checkpoint's
`dump/` tree. The initial actor-train phase took 145.9 s and includes first-run
kernel compilation; it is not a steady-state throughput measurement.

An independent CPU check in interactive job `7059756` read every HF export
tensor: all 398 are finite, 318 tensors / 62,898,531 elements differ from the
original HF checkpoint, and maximum absolute delta is 1.9073486328125e-6.
The sole omitted source key is the tied `lm_head.weight`; no unexpected keys
were added. The MCore metadata includes optimizer and RNG entries. The saved
dataset state has `sample_offset=4`, `sample_index=16`, and the rollout dump
contains 16 samples with total raw reward 9.
Log: `experiments/outputs/oci-step4000-artifact-checks-7059756.log`.

The first resume attempt, `7059756`, failed before generation (exit `1:0`,
6 min 11 s). Increasing `NUM_ROLLOUT` from 1 to 2 also changed the scheduler's
derived horizon from 16 to 32 samples. Megatron correctly rejected that mismatch
against the saved scheduler instead of silently resetting it. The sync/async
recipes now expose `USE_CHECKPOINT_OPT_PARAM_SCHEDULER` (default 0), which maps
to the existing `--use-checkpoint-opt-param-scheduler` option. This disposable
wrapper sets it to 1, retaining the checkpoint's scheduler state and constant LR
when extending the stop point. Neither optimizer nor RNG loading is disabled.
Do not use that override as an implicit schedule extension for decaying-LR runs.

## Resume result

Corrected resume job `7059888` completed with exit `0:0` in 8 min 9 s on
`nvl72139-T[15-16]` (`nemotron_sw_post / batch / interactive`). Its preflight
passed all 290 checks. It loaded `iter_0000000` and
`global_dataset_state_dict_0.pt` without resetting optimizer or RNG, then ran
rollout/optimizer step 1 (not step 0 again):

- Mean raw reward 0.125, mean response length 5616, truncation fraction 0.5.
- Loss 0.0200671389, gradient norm 0.2368002981, LR 1e-6; all finite and the
  gradient nonzero. Actor-train elapsed 23.1 s; this single smoke sample is not
  a controlled throughput comparison against the first run.
- `iter_0000001` and `hf/1/.complete` were saved, preserving `iter_0000000` and
  `hf/0`. The tracker is now `1`. Both HF exports have 398 tensors / 16 shards.
- The saved scheduler's `num_steps` advanced 16→32 with constant LR retained;
  dataset `sample_offset` advanced 4→8 and `sample_index` 16→32.
- Updated weights were transferred to SGLang, followed by the two-row functional
  eval (reward 0.5, no truncation). The tiny overlapping eval is not evidence of
  model quality improving or regressing.

Evidence: `experiments/outputs/oci-step4000-driver-7059888.log`,
`experiments/outputs/miles-oci-step4000-smoke-7059888.log`, and
`experiments/outputs/oci-resume-artifacts-7059888.log`. The last log confirms
scheduler/data state but its all-tensor HF comparison was interrupted when the
successful GPU allocation ended; that comparison is rerun separately on CPU
interactive resources and must not be counted as passed from this partial log.

The first separate CPU check (`7060110`) stopped while importing Megatron:
Transformer Engine needs `libcuda.so.1`, which CPU nodes do not provide. The
MCore scheduler/data assertions had already passed on the GPU allocation.
The final CPU check (`7060180`) therefore uses only PyTorch CPU tensors and
safetensors, without importing Megatron. It checked at least 200/398 tensors,
then made negligible progress while waiting in Lustre `cl_sync_io_wait`
(only about 167 MB of additional I/O over more than three minutes). It was
stopped as an incomplete optional check, not counted as a pass. Neither CPU
diagnostic changed any checkpoint. The full initial `hf/0` finite/delta check
passed; a full `hf/1` tensor scan remains unverified. Both training jobs and the
MCore scheduler/data resume assertions passed independently of this extra scan.
Reproducer: `experiments/outputs/check_step4000_resume_artifacts.py` (ignored
local diagnostic); partial log: `experiments/outputs/miles-resume-hf-check-7060180.log`.

## Qualification summary

HF→MCore conversion and small-batch sync training, checkpoint save/resume and
functional evaluation are verified. The original HF weights, converted release,
both training checkpoints and both HF exports are retained. No production-scale
or fully-async qualification is claimed. In particular, the prefill-provenance
API, production batch/response memory envelope and held-out AIME evaluation
remain separate gates. See `cluster-migration.md` before scaling this smoke.
