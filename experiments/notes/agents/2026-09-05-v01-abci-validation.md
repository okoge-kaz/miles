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

## Accepted results

| Check | Evidence | Result |
| --- | --- | --- |
| Eight-node Math and checkpoint resume | `2226432.pbs1`, fixed revision `aa9290b1` | PBS exit 0, 22m04s; updates 16/17, finite positive gradients, complete response/loss-token coverage |
| Full replay payload and provenance | `2226434.pbs1` | PBS exit 0, 1m07s; model 17 paired, replay 15/17 verified, all 8704 samples / 80,091,002 tokens in snapshot 17 covered |
| Archived checkpoint zero restart | `2226244.pbs1`, `2226274.pbs1` | PBS exit 0; model/replay retention aligned and archive 0 actually resumed |
| Math/Code/STEM simultaneous RL and resume | `2226150.pbs1`, `2226156.pbs1` | PBS exit 0; all four verifiers represented in 2048 unique trained samples |
| Focused regressions | GPU preflights and replay suite | 111 SGLang/staleness/Sample, 24 checkpoint, and 65 replay/resume tests passed |

The initial eight-node run completed 12 updates in 1h36m46s but had a
post-training shell failure caused by editing an active launcher. Its first
restart then exposed a synthetic-abort provenance gap in update 14. Both
findings, fixes, and the successful fixed-checkout reruns are detailed below;
the failed historical checks are not counted as successful acceptance.
All PBS terminal results are preserved in `pbs-job-results.json` under
`experiments/outputs/abci-validation/`.

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

# After the resumed job succeeds, audit its finalized checkpoints on a CPU node.
validation_checkpoint_relative="math/dapo-math-17k/Qwen3-4B/grpo-clip0.2-0.28-tis2.0/async/off-policy/max-weight-staleness-8-from-prefill/my-math8-validation-zero-trunc-rb-inflight-concurrency-3072-tbq6000"
qsub -P gai51740 -q R9920261300 \
  -v "RTYPE=rt_HC,MILES_REPO=$validation_source,REQUIRE_COMPLETE_POLICY_PROVENANCE=1,VALIDATION_CHECKPOINT_RELATIVE=$validation_checkpoint_relative,ARTIFACT_REPORT_NAME=my-math8-artifacts.json" \
  -o "$PWD/experiments/outputs/abci-validation/" experiments/validate_abci_artifacts.sbatch

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
update 11 for four further updates. It completed updates 12--15 and exited
with PBS status 0 in 34m44s. All four losses/gradient norms were finite with
positive gradients. CPU artifact audit `2226347.pbs1` exited 0 in 54s;
model 15 had its verified replay pair, replay 13/15 passed checksums, prepared
batches 14/16 matched all 18 fields and normalized masks, and HF exports
contained all 16 indexed shards. The ignored W&B `atexit` BrokenPipe messages
occurred after offline run finalization and did not change Ray/PBS success.

However, the strict final log audit rejected update 14: exact response-token
coverage was 0.9984355506704807 and exact loss-token coverage was
0.9971485200289981. Five of 3072 samples lacked metadata for 40,400 prefix
tokens generated before the restart. Their IDs are 42078, 42079, 42853,
42857, and 42858. The original samples and descriptors are preserved in
`incomplete-provenance-step14.pt/json`; CPU inspection was `2226406.pbs1`.
Earlier progress commentary incorrectly generalized complete coverage to all
resumed updates; the final auditor caught this and that statement was corrected.

The pinned SGLang `TokenizerManager._handle_abort_req` returns accumulated
tokens for a synthetic abort echo with only the coarse `weight_version`,
omitting scheduler first-prefill/forward versions and token-version segments.
The regular batch-output path has those fields. The new scoped server hook
caches the scheduler-authored metadata per live request and preserves it on
synthetic abort replies, even after an intermediate output has been consumed.
It does not infer missing versions or modify the SIF. The hook is installed
inside the spawned HTTP server when exact segments are enabled. Regression
tests cover consumed intermediate outputs, clipped prefixes, missing scheduler
output, duplicate installation/finished requests, already authoritative abort
metadata, and the actual installed SGLang abort handler.

CPU inspection `2226415.pbs1` found complete provenance in every materialized
sample of checkpoint 15, so it is a valid source for the corrected restart.
The corrected run will audit source update 15 and prepared batch 16; earlier
source updates 12--14 remain explicitly excluded from that boundary audit.
The original update-14 failure remains evidence, not a passing result. The
corrected real run and stronger artifact audit below also check complete
ready/inflight/prepared token provenance.

Corrected validation `2226432.pbs1` uses detached revision `aa9290b1`, two
updates from checkpoint 15, and the unchanged eight-node placement/batch
settings. Its GPU preflight passed all 111 SGLang/abort/fully-async/queue/
staleness/Sample tests, including the real installed abort handler. CPU audit
`2226434.pbs1` depends on its success and enables
`REQUIRE_COMPLETE_POLICY_PROVENANCE=1`; its report is
`math-8node-provenance-fixed-artifacts.json`. The source-window resume audit
uses `--fresh-start-rollout-id 15 --checkpoint-rollout-id 15` and reports all
excluded earlier updates explicitly. It retains the strict 100% response- and
loss-token coverage requirements for every accepted update. The corrected
training run finished with PBS exit 0 in 22m04s. Updates 16/17 had finite
losses and positive gradient norms; maximum staleness was 4/5. Response-token,
loss-token, and loss-sample coverage were all 1, with invalid counts 0.
Prepared batch 16's 365 staleness values matched exactly across this restart.
Replay restored 509 pending groups, 92 ready groups, one prepared batch,
225 inflight groups, and 28,951,398 prefix tokens, with no regeneration.
Read plus restore took 86.063s. `math-8node-resume-report.json` records the
accepted boundary audit; `math-8node-original-provenance-failure.json` retains
the earlier failing update separately.

Actual KV-cache retractions were logged during the corrected run. The final
snapshot 17 contains 222 inflight groups and 31,823,008 prefix tokens, totaling
1,386,438,599 bytes. Capture took 381.681s and publication 3.771s. Capture time
remains a material overhead at this concurrency, even after fixing metadata
preservation. The original incomplete samples also passed a negative-control
check: the stronger artifact auditor rejects sample 42078's missing prefix.

Final CPU audit `2226434.pbs1` completed with PBS exit 0 in 1m07s. Retained
model 17 has its verified replay pair. Replay 15 contains 8144 materialized
samples and 74,144,894 response tokens; replay 17 contains 8704 samples and
80,091,002 tokens. Every ready/inflight/prepared response has complete,
contiguous, valid version segments. Prepared batches 16 and 18 each contain
3072 samples and match all 18 audited fields and normalized loss masks.
All nine HF exports (odd iterations 1--17) contain their 16 nonempty indexed
shards. No acceptance work remains pending.

Reproduce the accepted restart log audit from this checkout:

```bash
python -m experiments.tools.analyze_abci_resume \
  --fresh-log experiments/outputs/abci-validation/2226346.pbs1.live.log \
  --resume-log experiments/outputs/abci-validation/2226432.pbs1.live.log \
  --fresh-start-rollout-id 15 --checkpoint-rollout-id 15 \
  --baseline-log /groups/gai51740/kazuki_fujii/src/miles-v0.0/experiments/outputs/training/math/dapo-math-p10-90/qwen3-4b/2211415.pbs1.OU \
  --output experiments/outputs/abci-validation/math-8node-resume-report.json
```

CPU source audit `2226361.pbs1` completed with exit 0 in 2m43s, using about
20GB peak memory, before the restart's checkpoint pruning. It verified all
six replay snapshots (1, 3, 5, 7, 9, 11), the retained model 11's replay pair,
and all corresponding 16-shard HF exports. Prepared batch 12 contains 3072
samples and 25,120,908 response tokens; all 18 audited sample fields and
normalized loss masks match the original rollout dump. This audit is preserved
as `math-8node-resume-source-artifacts.json`.

The restart loaded model/replay 11 and restored the actor's weight-version
counter to 12. Replay restored 526 pending groups, 124 ready groups, one
prepared batch, 210 inflight groups, and 17,174,786 prefix tokens, regenerating
no active group. Read plus restore took 75.275s. Prepared batch 12 hit the
warm path and all 365 staleness values matched before/after restart. Besides
the 363 fields seen in the initial two-node uniform-lag batch, this batch
also has the existing reward/length correlation fields, which are only
reported when the inputs have sufficient variance.

The first eight-node snapshot, at update 1, contains 318 inflight groups and
40,277,569 generated prefix tokens, totaling 1,682,249,411 bytes. Capture took
382.348s and publication 1.159s; this capture overhead is material at batch
3072. CPU audit `2226185.pbs1` passed with exit 0 in 43s, using about 11GB
peak memory. It verified the complete replay payload and 16 HF shards;
`math-8node-first-artifacts.json` records the result. Large deserialization can
be run with `experiments/validate_abci_artifacts.sbatch`, setting
`VALIDATION_CHECKPOINT_RELATIVE` beneath the validation checkpoint root and
`ARTIFACT_REPORT_NAME` beneath the validation output root. Final CPU audit
`2226347.pbs1` used the same detached source and produced
`math-8node-artifacts.json` in the original submission checkout.
