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
The sidecar records the queue policy and effective group capacity. Resume rejects
a different policy or capacity instead of interpreting one algorithm's queue as
another. Sidecars written before this field existed are treated as
`queue-recycle`, preserving backward compatibility without allowing an ambiguous
cross-policy restore.

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
only after a successful trainer ACK or a terminal queue/filter disposition.

For `queue-recycle`, a scheduled distributed checkpoint first finishes the next
rollout future that was already prefetched by `train_async.py`. The failure-free
path would wait for that same future immediately before its weight push; moving
the wait before the snapshot makes the next batch a complete prepared batch and
preserves its pre-update admission/version boundary across resume. `queue-max`
and `queue-drop` intentionally do not reserve that next batch before the
preceding update, so their scheduled sidecars normally preserve the ready queue
rather than manufacturing a warm prepared batch.

| Snapshot state | Resume behavior |
| --- | --- |
| Prepared next trainer batch | Reuse the complete trajectories immediately; do not re-run the admission/staleness check |
| Groups already admitted into a partial drain | Keep the admitted trajectories and continue filling that batch |
| Completed ready queue | Restore full trajectories in queue order, then re-run drain-time staleness/filter checks |
| Completed but blocked on `queue-recycle`/`queue-max` safety capacity | Restore as ready trajectories; restored over-capacity state keeps producer backpressure until consumption falls below the cap |
| `queue-drop` completion pending worker admission | Apply the same oldest-first overflow eviction at the snapshot boundary, then restore only the bounded queue |
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

The implementation supports `queue-recycle`, `queue-max`, and `queue-drop`. It
intentionally requires dense, critic-free Megatron GRPO, the standard FIFO
global data source and built-in reward/data conversion,
`--update-weights-interval 1`, and a non-`disk-delta` weight transport. It also
rejects debug rollout injection and any postprocessing that trims part of a
prepared batch. These are fail-fast guards, not silent fallbacks.

## Storage and observability

Full `Sample` trajectories are saved for the prepared batch, partial drain, and
the entire ready queue. This can make a sidecar large when the queue is deep or
responses are long. Schema 2 stores per-token lists in contiguous CPU tensors
and records repeated references to the same live `Sample` only once. Schema 3
pre-packs each completed prompt group while generation is running and writes
the immutable per-token tensors and UTF-8 response bytes as checksum-verified,
256-MiB-bounded binary parts in parallel. The small lifecycle/queue manifest
remains atomically published as the main `.pt` file. This moves list conversion
and hashing away from the checkpoint boundary and avoids serializing the entire
queue through one tensor/file stream. It costs additional rollout-manager RAM
approximately equal to the packed payload (tokens, masks, rollout log-probs,
and response bytes); the cache entries are released with their `Sample` objects.

Loading still creates an independent `Sample` at every occurrence, matching
schema 1's mutation semantics, and schema 1/2 sidecars remain readable. Every
binary slice, the main manifest, byte size, dtype, offset, and length are
validated before restore. Capture time and write time are logged separately.
The newest two sidecars are retained by default, in addition to IDs selected by
`--save-retain-interval`.

`queue-drop` never serializes more than its effective completed-group capacity.
If generation tasks finish at the snapshot boundary before the worker records
their admission, capture applies the policy's overflow decisions in memory. The
dropped trajectories themselves are omitted; only compact eviction counts,
response-length populations, and optional lifecycle/reward records remain.
`queue-recycle` and `queue-max` preserve capacity-blocked completions because
their failure-free behavior is backpressure rather than eviction.

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
