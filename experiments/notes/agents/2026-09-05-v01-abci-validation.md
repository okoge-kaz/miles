# v0.1 ABCI integration and validation, 2026-09-05

## Integrated revisions

Created `v0.1-abci` from `v0.0-abci` at `61def269`. Fetched
`origin/merge/v0.1.0-b300-aws-pdx` at `36cc6a12`. The merge base was
`487a9dfd`, with seven ABCI-only and 232 incoming commits. Merge `4602ba6f`
contains both histories. Python model profiles, v0.1 training/rollout APIs,
configurable completed-group buffers, partial rollout, truncation loss options,
and analysis tools are included. PBS/Singularity, numeric full-node GPU views,
HPC-X's unbound node launch, system NCCL/IB, and ABCI asset paths are retained.

The runtime is the existing `miles-20260902-134430.sif`: Torch 2.13.0+cu130,
Ray 2.58.0, SGLang f994b9aed, Transformers 5.12.1, MCore 0.19.0+c160f0717.
Validation uses the staged public Qwen3-4B HF/MCore checkpoints and the
17,398-row DAPO-Math file. The historical step-4000 SFT model is not staged
here; its configurable recipe remains available.

## CPU verification

The first focused run passed 339 tests and skipped one. Its single failure
identified incoming Slurm job headers, which were ported to PBS. The wider
experiment/launcher run passed 644 tests, with six failures: two missing the
optional `tau2` package in this image, three stale model/index/fixture references,
and the container-build test missing its host Bubblewrap bind. Fixing the stale
references and supplying the real host bind passed the 32-test follow-up. The
code/STEM/multi reward suite passed 11 tests.

Initial real code-reward preflight failed: Singularity marks its root overlay
unbindable, so Bubblewrap could not bind `/usr`. The verifier now enters a
private mount namespace and changes propagation only there before building
the read-only sandbox. The fixed preflight scored Math, Python, MCQA, and
Reasoning Gym correct/wrong pairs as `[1,1,1,1,0,0,0,0]`. The Python probe also
checked that `/root/miles` was absent and only the loopback interface existed.
No process-only reward fallback is used.

## Data preparation

CPU job `2226121.pbs1` finished with exit 0 in 7m30s. It converted complete
training components, rejected evaluation-only rows during merging, and built
32,673 balanced rows: 10,891 Math, 10,891 Code, and 10,891 STEM
(10,578 MCQA plus 313 Reasoning Gym). Reasoning Gym 0.1.25 supplied all 104
task scorers. Job `2226144.pbs1` regenerated the blend with the correct
`dapo-math-17k` source label and finished with exit 0 in 2m43s.

Data and audits are under `datasets/rl/nemotron-performance-transfer/`; the
reproduction entrypoint is `experiments/prepare_abci_multi_env.sbatch`.

## Two-node bring-up and restart

`2226122.pbs1` trained rollout 0, saved model/replay state, and completed with
exit 0. It used 8 actor + 8 rollout GPUs, 8 groups x 8 responses, global batch
64, 8K response / 32K context, and concurrency 64. Inflight replay,
queue-recycle, prefill reference, maximum staleness 8, and exact token-version
segments were enabled. Replay publication preceded the model save.

`2226139.pbs1` resumed this checkpoint and trained rollouts 1--3, completing
with PBS exit 0 in 6m36s. Prepared rollout 1's 363 staleness values matched exactly
before/after restart. It restored 16 pending groups, one prepared batch,
eight inflight groups, and 180,503 generated prefix tokens; it regenerated no
active group. `warm_prepared_batch_hit=1`. All four optimizer losses/gradients
were finite with positive gradient norms. Staleness means were 0, 1, 1.875,
2.125; maxima were 0, 1, 2, 3. Exact token coverage was 1 and invalid counts 0.

The current-vs-old log comparison uses retained ABCI job `2211415.pbs1` in
the `miles-v0.0` checkout. Incoming defaults expanded histogram bins from 16
to 32: 299 staleness keys become 363; four `count_ge_17` fields are replaced
by bins 17--32 and `count_ge_33`. Definitions of group/sample/token lag remain
unchanged. Incoming math recipes also select capacity 6000 at maximum
staleness >= 8 instead of the previous capacity 1000. Distribution comparisons
must account for these configuration changes.

Local evidence is under `experiments/outputs/abci-validation/`, including
`math-2node-resume-report.json`, PBS logs, live logs, checkpoints, replay
manifests, HF exports, and rollout/dashboard dumps. Reproduce the audit with
`python -m experiments.tools.analyze_abci_resume --fresh-log ... --resume-log ...
--baseline-log ... --output ...`.

## Four-node multi-environment training and restart

`2226150.pbs1` completed with PBS exit 0 in 8m12s, training updates 0--2.
It used 8 actor + 24 rollout GPUs, 32 groups x 8 responses, 8K responses,
maximum staleness 4, and sandboxed Python rewards. With save interval 2, the
last committed model was update 1, with prepared batch 2. The validation
wrapper now defaults to four updates to finish on a checkpoint boundary.

`2226156.pbs1` resumed model 1 and completed with PBS exit 0 in 9m29s,
training updates 2--7. Replaying unsaved update 2 is expected. All 365
staleness values for prepared batch 2 matched, as did its optimizer loss.
The restart restored 69 pending groups, one ready group, one prepared batch,
36 inflight groups, and 998,453 generated prefix tokens, with no active-group
regeneration. Exact token coverage stayed 1; invalid token segments/samples
and loss masks stayed 0. Observed maximum sample staleness was 4.

The eight unique trained batches contain 2,048 samples:

| Verifier | Samples | Positive rewards | Mean reward | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Math | 656 | 330 | 0.5030 | 315 |
| Python code | 528 | 140 | 0.2652 | 318 |
| MCQA | 840 | 574 | 0.6833 | 36 |
| Reasoning Gym | 24 | 9 | 0.3333 | 4 |

Both competitive-coding and Skywork code sources were trained. Gym covered
`rotten_oranges`, `decimal_chain_sum`, and `bitwise_arithmetic`; all eight
decimal-chain answers earned 1, and the official scorer assigned partial
credit to one rotten-oranges answer. The 8K validation cap accounts for the
substantial Math/Code truncation. The balanced input mixture does not force
equal domain counts in each asynchronous, completion-ordered batch.

Replay snapshots 1, 3, 5, and 7 passed the core loader's main and external
tensor checksum validation. Each corresponding HF export contains all 16
nonempty indexed shards. Evidence is in `multi-env-resume-report.json`,
`multi-env-samples-report.json`, and `multi-env-artifacts.json` under the
validation output directory. The auditors are
`experiments.tools.analyze_abci_resume` (use `--max-staleness 4
--checkpoint-rollout-id 1` for these logs),
`experiments.tools.analyze_abci_multi_env`, and
`experiments.tools.analyze_abci_artifacts`.

The artifact auditor also compares every retained prepared batch against its
rollout dump: 18 fields, including sample/group IDs, prompts, labels, tokens,
responses, rewards, log probabilities, and weight-version provenance, match
exactly. Loss masks match after expanding the dump's omitted masks to the
all-one masks created by train-data conversion. This passed for the two-node
prepared batches 2--4 and multi-environment batches 2, 4, 6, and 8; batch 2
includes the actual multi-environment restart.

## Reproduction

Submit from this checkout. The validators source `experiments/env.sh` and
use the existing public Qwen3-4B, isolated validation checkpoint directories,
offline W&B, and inflight replay. Keep the namespace and all batch/placement
settings identical for a restart; use a new namespace for a fresh experiment.
`DEBUG_EXIT_AFTER_ROLLOUT` counts updates in the current invocation.
Use a committed, detached checkout for long-running Math validation and do
not edit that checkout until its jobs finish. The submission directory owns
the checkpoint and audit outputs; `MILES_REPO` selects the source mounted
inside the container.

```bash
validation_revision=$(git rev-parse HEAD)
validation_source="$PWD/experiments/outputs/abci-validation/source/$validation_revision"
git worktree add --detach "$validation_source" "$validation_revision"

qsub -P gai51740 -q R9920261300 -N v01-math8-fresh \
  -l select=8:ncpus=192:ngpus=8:mpiprocs=1 -l walltime=02:00:00 \
  -v "RTYPE=rt_HF,VALIDATION_NAMESPACE=my-math8-validation,DEBUG_EXIT_AFTER_ROLLOUT=12,MILES_REPO=$validation_source" \
  -o "$PWD/experiments/outputs/abci-validation/" experiments/validate_abci_math.sbatch

# After successful completion, resume the same namespace for four more updates.
qsub -P gai51740 -q R9920261300 -N v01-math8-resume \
  -l select=8:ncpus=192:ngpus=8:mpiprocs=1 -l walltime=02:00:00 \
  -v "RTYPE=rt_HF,VALIDATION_NAMESPACE=my-math8-validation,DEBUG_EXIT_AFTER_ROLLOUT=4,MILES_REPO=$validation_source" \
  -o "$PWD/experiments/outputs/abci-validation/" experiments/validate_abci_math.sbatch

# Run preparation once; wait for its successful completion before training.
qsub -P gai51740 -q R9920261300 -v RTYPE=rt_HC \
  experiments/prepare_abci_multi_env.sbatch
qsub -P gai51740 -q R9920261300 -N v01-multi-fresh \
  -l select=4:ncpus=192:ngpus=8:mpiprocs=1 -l walltime=02:00:00 \
  -v RTYPE=rt_HF,VALIDATION_NAMESPACE=my-multi-validation \
  -o "$PWD/experiments/outputs/abci-validation/" experiments/validate_abci_multi_env.sbatch
```

For the two-node bring-up use `select=2` and explicitly set
`ROLLOUT_BATCH_SIZE=8,N_SAMPLES_PER_PROMPT=8,ASYNC_MAX_CONCURRENT_SAMPLES=64,MAX_RESPONSE_LEN=8192,SAVE_INTERVAL=1`.
The first invocation used `DEBUG_EXIT_AFTER_ROLLOUT=1`, then the restart used 3.

## Archived replay retention and checkpoint zero

An additional archival audit found a pre-existing retention mismatch, also
present in the starting ABCI branch: Megatron archives checkpoint iteration
`N` when `N % save_retain_interval == 0`, but replay used `(N + 1) % interval`.
The initial bring-up consequently retained model 0 after pruning replay 0.
Its latest-checkpoint restart passed, but that historical archive is unusable.
Commit `911438a3` aligns replay archival IDs with Megatron; 65 replay/resume
tests passed, including preservation and loading of archive 0 after newer
updates. The artifact auditor now rejects any retained distributed model
without a verified replay pair, and detected the known orphan in the original
bring-up. It still passes for the multi-environment run.

Real archival validation `2226244.pbs1` finished with exit 0 in 8m26s (two
nodes, four updates, save interval 1, retain interval 2). Model and replay
iterations both correctly retained 0, 2, and 3. Update 2 had zero GRPO loss and
gradient: its actual eight groups all had constant rewards (two all-zero,
six all-one groups). The original dump is preserved as
`retention-fresh-step2.pt`; `--allow-zero-gradients` permits this documented
small-batch case in the resume auditor while retaining finite-value checks,
requiring some nonzero gradients, and reporting every zero-gradient step.

The first archive restart, `2226246.pbs1`, exposed another existing runtime
issue: Megatron's two checkpoint path resolvers use a truthiness check for
`ckpt_step`, so explicit `CKPT_STEP=0` incorrectly loaded latest iteration 3.
That job was cancelled. `cc29ff40` adds a scoped integer adapter that keeps
zero explicit only during Megatron loading, then restores ordinary integer
arguments and loaded-iteration state, including on exceptions. This avoids
modifying the SIF or checkpoint trackers. The retry `2226274.pbs1` passed all
24 checkpoint-selection/LoRA/in-memory tests on a GPU compute node and then
completed the actual archive-0 restart with PBS exit 0 in 7m45s. The same tests require the CUDA
driver library for this image's Transformer Engine import and cannot collect
on the login node without it.

The Math train entrypoint exposes Megatron's `CKPT_STEP` option. These jobs
use namespace `v01-abci-retention-20260905` and the original 8-group x
8-response bring-up settings. The successful retry loaded model 0, restored
its actor counter to 1, and trained updates 1--3. Prepared batch 1 retained all
363 staleness values and the same optimizer loss as the original batch.
It restored 16 pending groups, eight inflight groups, and 187,897 prefix tokens,
with no active-group regeneration. Every resumed gradient norm was positive.
`math-retention-resume-report.json` and `math-retention-artifacts.json` record
the log and payload audits. The final tracker is 3; retained models 0, 2, 3,
and 4 all have verified replay pairs. Extra model 4 and replay 4/5 are evidence
from the cancelled, incorrectly selected restart, which completed updates 4/5
before cancellation. The successful retry's retained prepared batches match
all 18 audited fields and normalized loss masks.

## Eight-node Math training and restart

`2226149.pbs1` ran from `28a3303e`, with four actor nodes and four rollout
nodes (32 + 32 H200 GPUs), batch 192 x 16 = 3072, concurrency 3072,
16K responses / 32K context, and a two-hour walltime. It completed updates
0--11, saved model/replay/HF checkpoint 11, and reported Ray job success.
All 12 losses and gradient norms were finite with positive gradients;
maximum observed sample staleness was 6, exact token coverage was 1, and
invalid segment/sample/loss-mask counts were 0.

PBS nevertheless ended with exit 127 after 1h36m46s. During this job I edited
the running shared `train.sh` to add `CKPT_STEP`. Bash resumed reading at the
old byte offset after `ray job submit` returned and interpreted a trailing
`--rollout-num-gpus-per-engine` flag as a command. This was an operational
error after successful learning and checkpoint publication. The dependent
restart `2226167.pbs1` and audit `2226187.pbs1` consequently never ran.

Commit `8661acbb` lets the Math/artifact validators use a fixed `MILES_REPO`.
The replacement eight-node resume `2226346.pbs1` uses the detached checkout
`experiments/outputs/abci-validation/source/8661acbb` and the same checkpoint
namespace, batch shape, placement, and two-hour walltime. It resumes saved
update 11 for four further updates. Final acceptance is pending its completion
and CPU artifact audit `2226347.pbs1`, which depends on its PBS success.

The first eight-node snapshot, at update 1, contains 318 inflight groups and
40,277,569 generated prefix tokens, totaling 1,682,249,411 bytes. Capture took
382.348s and publication 1.159s; this capture overhead is material at batch
3072. CPU audit `2226185.pbs1` passed with exit 0 in 43s, using about 11GB
peak memory. It verified the complete replay payload and 16 HF shards;
`math-8node-first-artifacts.json` records the result. Large deserialization can
be run with `experiments/validate_abci_artifacts.sbatch`, setting
`VALIDATION_CHECKPOINT_RELATIVE` beneath the validation checkpoint root and
`ARTIFACT_REPORT_NAME` beneath the validation output root. Final CPU audit
`2226347.pbs1` uses the same detached source and will produce
`math-8node-artifacts.json` in the original submission checkout.
