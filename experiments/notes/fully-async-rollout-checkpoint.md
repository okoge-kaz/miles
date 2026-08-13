# Fully-async rollout checkpoint/replay

Enable with `--fully-async-rollout-checkpoint`, or set
`FULLY_ASYNC_ROLLOUT_CHECKPOINT=1` for the Qwen3-4B math-async recipe. The CLI
default is off so a checkpoint made by an older Miles build keeps its legacy
cursor-only resume behavior.

Treat the flag as part of the run identity: enable it from the start with a
fresh `CONFIG_TAG`/checkpoint directory, and do not toggle it while resuming a
run. Enabling it on a legacy checkpoint fails because the committed model has no
matching replay sidecar. Disabling it on a full-replay checkpoint would discard
the pending-lease ledger, so Miles detects the sidecar and rejects that downgrade.

## Failure semantics

The distributed model checkpoint tracker is the commit record. For rollout `N`
the order is:

1. the trainer successfully applies batch `N` and ACKs its batch token;
2. Miles atomically writes `rollout/fully_async_state_N.pt` and its checksum;
3. Megatron writes model/optimizer/RNG checkpoint `N` and publishes its tracker;
4. Miles prunes old rollout sidecars.

If step 2 fails, model saving does not start. If model saving fails, sidecar `N`
is an uncommitted orphan and resume loads the sidecar named by the older model
tracker. A model tracker that names a missing, corrupt, wrong-schema, or
wrong-dataset sidecar fails resume instead of silently resetting the dataset.
Distributed model saves are forced to synchronous completion in this opt-in
mode, even when `--async-save` is configured, so step 4 cannot run before the
tracker in step 3 is durable. This adds checkpoint-boundary latency but does not
change non-checkpoint training steps or the default cursor-only path.

The sidecar owns the allocation cursor plus a prompt-lease ledger. This avoids
treating “submitted to generation” as “trained”: prompt groups leave the ledger
only after a successful trainer ACK, or after a terminal dynamic-filter drop.
At a scheduled distributed checkpoint, Miles first finishes the next rollout
future that was already prefetched by `train_async.py`. The failure-free path
would wait for that same future immediately before its weight push; moving the
wait before the snapshot makes the next batch a complete prepared batch and
preserves its pre-update admission/version boundary across resume.

| Snapshot state | Resume behavior |
| --- | --- |
| Prepared next trainer batch | Reuse the complete trajectories immediately; do not re-run the admission/staleness check |
| Groups already admitted into a partial drain | Keep the admitted trajectories and continue filling that batch |
| Completed ready queue | Restore full trajectories in queue order, then re-run drain-time staleness/filter checks |
| Completed but blocked on the queue capacity | Restore as ready trajectories |
| Active generation or retry-buffer lease | Restore the original prompt identity/retry count and regenerate |
| Trainer-ACKed batch | Do not restore; the matching model checkpoint already contains its optimizer update |

Consequently, resume can start training from a prepared batch without waiting
for a cold rollout fill. It is not bitwise continuation: work after the last
published checkpoint is lost, active requests regenerate with a new inference
RNG stream, and external reward/tool services may return different results.

## Weight-version continuity

The sidecar stores the rollout engines' last globally applied weight version.
Before the first resume-time weight push, Miles restores that value into every
trainer failover cell's updater. The startup push therefore advances from `v`
to `v+1`, matching the failure-free `--update-weights-interval 1` path. Ready
groups compare their saved provenance against that resumed current version;
prepared/admitted groups retain the decision already made before the snapshot.

The first implementation intentionally requires dense, critic-free Megatron
GRPO, the standard FIFO global data source and built-in reward/data conversion,
`--update-weights-interval 1`, and a non-`disk-delta` weight transport. It also
rejects debug rollout injection and any postprocessing that trims part of a
prepared batch. These are fail-fast guards, not silent fallbacks.

## Storage and observability

Full `Sample` trajectories are saved for the prepared batch, partial drain, and
the entire ready queue. This can make a sidecar large when the queue is deep or
responses are long. Checkpoint write time and bytes are logged. The newest two
sidecars are retained by default, in addition to IDs selected by
`--save-retain-interval`.

Resume metrics are emitted on the first restored rollout:

- `resume/fully_async/warm_prepared_batch_hit`
- `resume/fully_async/current_applied_weight_version`
- `resume/fully_async/pending_groups_restored`
- `resume/fully_async/ready_groups_restored`
- `resume/fully_async/regenerated_active_groups`
- `resume/fully_async/partial_drains_restored`
- `resume/fully_async/prepared_batches_restored`
- `resume/fully_async/applied_weight_version_restored`

For the math-async recipe:

```bash
CONFIG_TAG=my-run-full-replay FULLY_ASYNC_ROLLOUT_CHECKPOINT=1 sbatch \
  experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
```

`experiments/verify_resume.sh` enables the feature in its two-job kill/resume
test and reports the retained replay sidecars and warm-resume log records.
