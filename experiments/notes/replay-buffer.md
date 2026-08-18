# Replay buffer for fully-async rollout

Replay-buffer persistence is opt-in:

```text
--use-replay-buffer
--replay-buffer-type rollout   # default when enabled
--replay-buffer-type inflight
```

Without `--use-replay-buffer`, resume keeps the existing cursor-only behavior
and does not persist rollout data. The Qwen3-4B math-async recipe exposes the
same settings as `USE_REPLAY_BUFFER=1` and `REPLAY_BUFFER_TYPE=rollout|inflight`.

Treat both settings as part of the run identity. Enable the replay buffer from
the start with a fresh `CONFIG_TAG`/checkpoint directory, and do not change its
type while resuming a run. Miles rejects a missing buffer, a disabled resume of
a model that has one, and a stored type that differs from the requested type.

## Buffer types

Both types preserve completed rollout data: prepared trainer batches, partial
drains, the ready queue, queue-policy state, prompt leases, data-source state,
and the rollout engines' applied weight version.

| Type | Active generation at save | Resume behavior |
| --- | --- | --- |
| `rollout` | Save the original prompt lease only | Regenerate the active group from its original prompt |
| `inflight` | Interrupt the request and save its returned partial `Sample`, including token IDs, decoded text, logprobs, status, and policy provenance | Submit the complete saved token prefix as `input_ids`, perform one prefill, and generate only the remaining token budget |

`inflight` never writes an SGLang KV cache. At capture time Miles pauses the
fully-async producer, asks every rollout worker to abort active requests, waits
for their partial replies, encodes the resulting token prefixes, resets the
generation abort state, and immediately restarts the producer. A sample that
had not emitted a token is still recorded as unfinished and resumes from its
prompt.

The token IDs are authoritative during continuation. Decoded text can be empty
when a partial prefix consists only of special tokens, so the single-turn
generator uses `response_length` and validates that the stored prompt/response
token boundary is consistent before prefill.

This is semantic continuation, not bitwise continuation. The resumed request
has a fresh KV cache and inference RNG state, and can run after a policy-weight
update. Its existing token provenance is retained and new provenance is
appended. `inflight` is therefore currently restricted to the built-in
single-turn generate function; custom generate functions fail argument
validation. Use `rollout` for Search-R1 and other custom generators.

## Failure and commit semantics

The distributed model checkpoint tracker is the commit record. For rollout
`N`, the order is:

1. the trainer successfully applies batch `N` and ACKs its replay-buffer token;
2. Miles atomically writes `rollout/replay_buffer_N.pt`, tensor parts, and the checksum manifest;
3. Megatron writes model/optimizer/RNG checkpoint `N` and publishes its tracker;
4. Miles marks the matching replay buffer committed and prunes old buffers.

If step 2 fails, model saving does not start. If model saving fails, replay
buffer `N` is an uncommitted orphan; resume loads the buffer named by the older
model tracker. A tracker that names a missing, corrupt, wrong-schema, or
wrong-dataset buffer fails resume instead of silently resetting rollout state.
Distributed model saves are forced to synchronous completion in this mode so
pruning cannot run before the model tracker is durable.

The replay buffer owns the allocation cursor plus a prompt-lease ledger. Prompt
groups leave that ledger only after a successful trainer ACK or a terminal
queue/filter disposition; submission to generation alone is not consumption.

For `queue-recycle`, a scheduled distributed checkpoint first finishes the
next rollout future already prefetched by `train_async.py`. The failure-free
path waits for the same future before its weight push, so the buffer contains a
complete prepared batch with the same pre-update admission boundary.
`queue-max` and `queue-drop` do not reserve the next trainer batch before that
update and usually preserve the ready queue instead.

| Saved state | Resume behavior |
| --- | --- |
| Prepared next trainer batch | Reuse complete trajectories immediately without rerunning admission/staleness checks |
| Groups admitted into a partial drain | Keep admitted trajectories and continue filling the batch |
| Completed ready queue | Restore full trajectories in queue order and rerun drain-time staleness/filter checks |
| Completion blocked by `queue-recycle`/`queue-max` capacity | Restore it as ready; retain producer backpressure until consumption falls below the cap |
| `queue-drop` completion awaiting admission | Apply the same oldest-first overflow eviction at capture, then restore only the bounded queue |
| Active generation in a `rollout` buffer | Regenerate the original prompt/retry lease |
| Active generation in an `inflight` buffer | Prefill the saved token prefix and continue |
| Trainer-ACKed batch | Do not restore; the model checkpoint already contains its optimizer update |

## Compatibility

New writes use replay-buffer schema 4 and the `replay_buffer_N.pt` name. The
reader remains compatible with schema 1-3 files named
`fully_async_state_N.pt`; those legacy files are interpreted as type `rollout`.
The legacy CLI option and Python module names are intentionally not aliases:
current configurations must use replay-buffer terminology, while compatibility
code for the old disk format stays isolated in `miles.rollout.replay_buffer`.

The buffer records queue policy and effective group capacity. Resume rejects a
different policy or capacity instead of interpreting one algorithm's queue as
another. Older schemas without this field are treated as `queue-recycle`.

## Storage and observability

Completed and inflight `Sample` payloads are encoded through one deduplicated
sample table. Per-token arrays and UTF-8 response bytes are written as
checksum-verified, 256-MiB-bounded binary parts in parallel; the compact
lifecycle/queue manifest is atomically published as the main `.pt` file. The
completed-sample packing cache is never used for a mutable inflight sample.

Every binary slice, main manifest, byte size, dtype, offset, and length is
validated before restore. Capture and durable-write time are logged separately.
The newest two buffers are retained by default; change this with
`--replay-buffer-keep-last`. IDs selected by `--save-retain-interval` are kept
in addition to the recent set.

`queue-drop` never serializes more than its configured completed-group
capacity. Capture applies pending oldest-first overflow decisions in memory;
the dropped trajectories are omitted while compact eviction metrics remain.

Resume metrics are emitted on the first restored rollout:

- `resume/replay_buffer/warm_prepared_batch_hit`
- `resume/replay_buffer/current_applied_weight_version`
- `resume/replay_buffer/pending_groups_restored`
- `resume/replay_buffer/ready_groups_restored`
- `resume/replay_buffer/regenerated_active_groups`
- `resume/replay_buffer/inflight_groups_restored`
- `resume/replay_buffer/inflight_tokens_restored`
- `resume/replay_buffer/partial_drains_restored`
- `resume/replay_buffer/prepared_batches_restored`
- `resume/replay_buffer/applied_weight_version_restored`

For the math-async recipe:

```bash
CONFIG_TAG=my-run-replay USE_REPLAY_BUFFER=1 REPLAY_BUFFER_TYPE=rollout sbatch \
  experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
```

Use `REPLAY_BUFFER_TYPE=inflight` for token-prefix continuation with the
built-in single-turn generator. `experiments/verify_resume.sh` exercises the
two-job save/resume transaction and reports retained replay buffers and warm
resume log records.
