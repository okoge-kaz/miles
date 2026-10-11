# ReplayDataBuffer

ReplayDataBuffer saves and restores data waiting for training in DataBuffer.
The `inflight` mode also saves and restores inflight rollout batches, including unfinished groups' token prefixes.
Restored groups are consumed once, subject to the usual staleness filter.

## Configuration

| Environment variable | CLI | Default |
| --- | --- | --- |
| `USE_REPLAY_DATA_BUFFER` | `--use-replay-data-buffer` | `0` |
| `REPLAY_DATA_BUFFER_MODE` | `--replay-data-buffer-mode` | `rollout` |
| `REPLAY_DATA_BUFFER_ASYNC_RESTORE` | `--replay-data-buffer-async-restore` | `0` |

`--use-replay-buffer` and `--replay-buffer-async-restore` remain aliases.
Legacy environment variables are fallbacks; the new variables take precedence, including explicit `0` values.

## Implementation and checkpoint format

| Module | Responsibility |
| --- | --- |
| `miles/rollout/replay_data_buffer.py` | Lifecycle, snapshot, restore |
| `miles/rollout/replay_data_buffer_checkpoint.py` | Parts, index, hashes, commit marker |
| `miles/rollout/replay_data_buffer_inflight.py` | Abort, capture prefixes, resume generation |

Checkpoints live in `<save>/rollout/replay_<iteration:07d>/`.
New checkpoints use schema 2; schema 1 is read as `rollout` mode.

| File | Contents |
| --- | --- |
| `header.json` | Iteration, served version, cursor, counts, mode, fingerprint, index hash |
| `index.jsonl` | Part order, group counts, byte sizes, SHA256 hashes |
| `part_*.pt` | Completed groups in batch-sized parts, including tokens, logprobs, masks, rewards, versions, routing data |
| `source_metadata.pt` | Source metadata |
| `auxiliary.pt` | Retry prompts and inflight groups' original prompts/prefixes |
| `COMMITTED.json` | Model iteration and header hash |

`train_async.py` waits for model/optimizer checkpoint completion before saving the buffer,
even when model checkpoint writes normally run asynchronously. It defers the next batch drain until after saving.
After the model/optimizer save, the producer's event loop captures the source cursor, queued groups,
completed groups waiting for capacity, and unfinished work.
The checkpoint directory is synced and renamed before writing the commit marker. Iteration 0 is valid.
Initial weights without rollout history start fresh; resuming training requires a buffer checkpoint at the same iteration.
Corruption and mismatches in mode, source, or fingerprinted sampling/filter/ordering settings stop restoration.

## Restore modes

### rollout

Completed groups retain their saved outputs and rewards. Unfinished groups are regenerated from their original prompts.
DataBuffer selects restored groups according to `--async-data-buffer-order` as described below.

### inflight

In addition to completed groups, this mode saves inflight rollout groups.
Saving pauses the producer and aborts SGLang requests to collect partial responses.
Prefixes, logprobs, loss masks, versions, and metadata are retained; generation resumes with the remaining token budget.
Completed siblings are retained, along with the original submission and first-prefill provenance.

This mode requires the standard single-turn text generator. Custom generators, sessions, multimodal inputs,
and rollout routing/indexer replay are rejected. KV caches, engine RNG, and external environment state are not saved.

## Scheduling after restore

`--async-data-buffer-order` (`ASYNC_DATA_BUFFER_ORDER`) controls selection in both restore modes.

| Order | Restored-group selection |
| --- | --- |
| `fifo` | Preserve the saved queue order. |
| `oldest-policy-first` | Rebuild the priority queue and select the group with the smallest observed generation version. Ties keep insertion order; unknown versions come last. |

Priority snapshots are also written in priority order, including completed groups waiting for capacity.
The restored buffer continues to apply this algorithm as new groups arrive.
The ordering setting must match the checkpoint and is independent of `--async-staleness-metric`.
Staleness is checked against the current version when a group is consumed; stale groups follow the configured retry/drop policy.
See [staleness.md](staleness.md) for the version definitions and filters.

## Asynchronous restore

With asynchronous restore disabled, all parts and auxiliary data are loaded and verified before training starts.
When enabled, the header, source metadata, and first part are loaded first; the remaining parts are restored in the background.
Training can proceed once an eligible batch is available. New rollout production waits until the entire restore is verified.
The index is streamed. If the first part cannot supply an eligible batch, training waits for later parts or regenerated groups.

A save requested during restore removes capacity backpressure and loads the entire unread tail before saving.
Memory use and waiting time for this save therefore depend on the remaining buffer size.

## Rollout routing replay (R3)

R3 (`--use-rollout-routing-replay`) reuses MoE expert-routing decisions during training.
Its compatibility with ReplayDataBuffer depends on the restore mode:

| Mode | Current implementation |
| --- | --- |
| `rollout` | No explicit R3 prohibition. Completed samples' `rollout_routed_experts` arrays are serialized and restored; unfinished groups regenerate their routing data with their outputs. End-to-end R3 checkpoint/resume has not been validated on GPUs. |
| `inflight` | Rejected by `validate_buffer_arguments()` in `miles/utils/async_buffer_arguments.py`. This also applies to `--use-rollout-indexer-replay`. |

Keep routing/indexer replay settings unchanged across resume; these flags are not currently included in the configuration fingerprint.

## Supported scope

Requires fully-async rollout, the global prompt dataset, and a single Megatron actor policy.
Critic training, independent-DP recovery, multi-LoRA, debug injection, and custom buffer snapshots are unsupported.
Automatic migration of legacy CW/ABCI pickle files, repeated sampling of consumed data, and saving unread parts without loading them are not implemented.
