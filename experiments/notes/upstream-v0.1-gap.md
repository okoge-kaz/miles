# Upstream Miles v0.1 gap analysis

This note compares the committed tree of `experiments/b300-aws-pdx` at
`5efc197158d77c6a83ec63610c9439c14b5bdcd0` with the official
`radixark/miles` `v0.1.0` release commit
`78527b9102b3a5f15891128cbec3ac6c2dc4089e`. The local annotated tag
`upstream-v0.1.0` was verified against the peeled `v0.1.0` tag published by
the official GitHub repository on 2026-08-23.

This is a tree and implementation comparison, not a claim that the current
branch is simply older. The histories diverge at
`5400905d59d4c03ed7a8fd559b287e4337ff3897`: there are 151 current-only and
193 v0.1-only commits. The final trees differ in 1,343 files (80,620
insertions and 51,810 deletions), so a blind merge is not an appropriate
upgrade strategy.

## High-confidence capabilities absent from the current branch

| Capability in v0.1 | Implementation evidence absent from snapshot `5efc1971` | Operational value | Upstream introduction |
|---|---|---|---|
| Fully-async evaluation | `miles/ray/rollout/eval_dispatch.py`, `eval_fleet.py`, and `miles/rollout/checkpoint_eval.py` | Adds shared-engine pause evaluation, a dedicated `--eval-num-gpus` fleet, and an external `CheckpointEvalFn` backend with checkpoint snapshots and bounded in-flight evaluation | `1a661812` |
| Replaceable fully-async data buffer and sample-level submission scheduling | `miles/rollout/fully_async_data_buffer.py` and `submission_scheduler.py`; flags `--async-data-buffer-capacity-factor`, `--async-unused-samples-handler`, `--custom-async-data-buffer-path`, and `--rollout-submission-granularity` | Bounds producer lead, supports drop/retry policy for unused or stale groups, exposes buffer staleness, and backfills concurrency as individual samples complete | `eff558e9`, `96c11ccc` |
| Partial final optimizer step and FLOPs-aware balancing | `miles/utils/dp_schedule.py`; flags `--allow-partial-train-step` and `--balance-by-flops` | Trains a trailing batch smaller than the configured GBS and balances variable-length microbatches using attention-aware FLOPs estimates | `11cc2326` |
| Actor forward-only elimination | `--skip-actor-forward-only` plus detached training-log-prob reuse in the loss path | Removes a redundant Megatron forward-only pass for supported one-step policy-loss configurations | `23ec9e53` |
| Session server v2 trajectory trees | `miles/rollout/session/v2/` and the `--use-session-server v2` path | Supports append-only branching trajectories, multiple lineages, retry-aware leaf selection, post-processing hooks, and additional R3 rows under in-place weight updates | `abd748bd` through `fd2b6eaf` |
| Configurable TITO replay matching | `miles/utils/chat_template_utils/message_matcher_hub/`; `--session-message-matcher` | Lets harnesses choose strict, normalized tool-call, role/content-only, or custom replay equivalence instead of treating serialization differences as divergence | `bc1233ee` |
| Stable FSDP2 backend with hybrid sharding and FSDP R3 | Stable `miles/backends/fsdp_utils/parallel.py`, `adaptations/routing_replay.py`, and `models/replay_routers.py`; `--dp-replicate-size` | Promotes the experimental backend, adds replicate-by-shard device meshes, and makes MoE rollout routing replay available to FSDP | `424c59e6`, `ea1c9186`, `800274af` |
| Megatron colocate weight rematerialization | `miles/backends/megatron_utils/rematerialize_utils.py`; `--rematerialize-param-from-master-weight` | Rebuilds low-precision actor weights from optimizer master weights and removes the bf16 CPU backup, saving about two bytes per parameter per rank in eligible colocated runs | `319716c0` |
| Expanded agent/environment connectors | `examples/experimental/verifiers/`, HUD, AgentENV/E2B and Modal OpenEnv backends, and `swe-agent-harbor-daytona/` | Adds full Verifiers rollout integration, computer-use RL, two per-episode sandbox backends beyond Daytona, and a Harbor-on-Daytona example | `d2010d29`, `095984c5`, `6bc45ad3`, `60aef7f5`, `41b9ae23` |
| Dashboard MFU and run-health advisory v2 | `miles/utils/device_flops.py`, MFU metrics, and the expanded `miles/dashboard/advisory.py` | Reports actor-train MFU, scrapes engines directly with DP-aware views, and distinguishes stalls/abort storms/degenerate reward signals from configuration tuning advice | `d0e8e7c6`, `1b266286`, `35c701ba`, `6afacc18` |
| Tested Python launcher migration | v0.1 replaces the remaining top-level model/launch shell scripts with Python command builders and snapshot tests | Makes generated commands testable and removes a large set of quoting/path errors in shell recipes | `2b2ef6f8` and its prerequisite launcher-test series |
| New official model support and recipes | `miles_plugins/models/inkling/`, `scripts/models/qwen3.8-27B.py`, and v0.1 recipes/docs for newer large models | Adds Inkling and native Inkling LoRA integration, Qwen3.8-27B, and maintained launch guidance for newer model families | `5c517599`, `92ccb87d`, `bb6f7dbd` |

Other useful v0.1 additions include a shared actor/critic Qwen3-4B PPO example,
the ability to use `--stream-optimizer-state-to-disk` without trainer offload,
an in-tree AMD Triton attention bridge for FSDP, and stronger rollout reward
group invariants. These are narrower recipes, relaxations, or correctness changes
rather than entirely new top-level subsystems.

## Capabilities that are not missing

The following already exist in the `5efc1971` snapshot and should not be counted as
v0.1 gaps:

- Fully-async RL itself and `--max-weight-staleness`. This branch has both, and
  the 8B and 30B-A3B validation runs used staleness 4 successfully.
- `--use-dynamic-global-batch-size`. v0.1 adds partial-step scheduling and
  FLOPs balancing around it, not the basic flag.
- Base TITO/session support. The missing part is the v2 branching tree and its
  replay matcher/picker/postprocessor stack.
- Base rollout routing replay for Megatron. The missing part is FSDP R3 and the
  v0.1 session-v2 extensions.
- An experimental FSDP2 backend. v0.1 promotes it to the stable namespace and
  adds hybrid sharding, routing replay, and additional validation.
- The dashboard and its first-generation tuning advisory. The gaps are MFU,
  direct/DP-aware engine telemetry, run-health advisory v2, and later dump/UI
  fixes.
- Python launchers and many of the named model configurations. The gap is the
  completed shell-to-Python migration, snapshot coverage, and specific new
  adapters/recipes.

## Recommended adoption order

1. Port the fully-async evaluation cluster and the data-buffer/submission
   scheduler cluster together. They are the closest match to the current async
   RL and staleness experiments, but must be reconciled with this branch's
   custom persisted replay buffer rather than replacing it mechanically.
2. Port session v2 plus the message-matcher commits as one unit if branching
   agent trajectories are needed. Their data model, server API, and sample
   post-processing are coupled.
3. Port `--skip-actor-forward-only`, partial-step/FLOPs scheduling, and weight
   rematerialization independently after focused numerical and resume tests.
4. Adopt the stable FSDP namespace, hybrid sharding, and FSDP R3 only as a
   coordinated backend migration; imports and model adaptations moved at the
   same time.
5. Cherry-pick only the environment connectors that match the intended
   workload. Verifiers is directly useful for RLVR; E2B, Modal, HUD, and Harbor
   add external service and dependency requirements.
6. Treat the dashboard and launcher work as operational improvements after the
   training-path ports, unless observability is the immediate bottleneck.

For every cluster, compare against commits made after the snapshot first, port onto a
temporary integration branch, and run the same 4B/8B/30B-A3B training and async
staleness smokes. The size and bidirectional divergence of the trees make a
whole-tag merge substantially riskier than feature-cluster cherry-picks or a
manual forward port.
