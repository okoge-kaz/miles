# AReaL Tau2 user-simulator RL

This recipe trains only on the pinned 1,982-row
`inclusionAI/AReaL-tau2-data/tau2_rl_train.jsonl` split. Each trajectory owns
an isolated source DB snapshot and runs the official Tau2 user simulator. Reward
is assigned at terminal state from the task's declared DB, environment
assertion, expert-action, and communication components. LLM-judged natural-
language assertions are rejected.

This is conversational multi-turn RL. It is separate from both the static
`tool_call_pivot` next-action task and the Workplace single-turn multi-step
environment.

The default policy is the locally trained `Qwen3-4B-Base` agentic-tool-call
SFT checkpoint at iteration 953, not `Qwen3-4B-Instruct-2507`. Its recipe name
is `qwen3-4b-agentic-sft-953`, and RL checkpoints are rooted below
`/ckpt/training/tau_bench/areal-tau2-user-simulator/Qwen3-4B-Agentic-SFT-953/`.

Multi-turn inflight replay is enabled. In addition to the prepared trainer
batch, completed queue, prompt leases/cursor, queue state, and policy-version
provenance, an interrupted Tau episode stores its policy token prefix,
logprobs/loss mask, official message history, DB hashes, and orchestrator
step/error counts. The message history includes user-simulator messages and
tool calls/results, so it is also the DB apply-command log.

Checkpoint capture finishes any tool or user-simulator transition already in
progress and records the next agent boundary. Resume asks the pinned official
Tau2 environment to replay mutating tool calls, verifies their recorded tool
outputs and both DB hashes, reconstructs agent/user histories without calling
the user simulator for old turns, then prefills the complete saved policy token
prefix. No SGLang KV cache or full DB copy is stored. This is semantic rather
than bitwise continuation: future policy sampling and future external user-
simulator calls occur in the resumed process.

The four-node fresh/resume gate completed in jobs 333450 and 333451 through
rollout 9 and committed model checkpoint plus inflight replay buffer 9. A
post-fix resume at checkpoint 9 restored 207,990 response tokens across three
inflight groups; rollout 11 contained 15 genuine version-10-to-11 mixed
continuations with zero DB/message hash failures, zero invalid provenance, and
zero staleness-bound violations. CPU job 334117 passed 332 tests and exercised
all 1,982 prepared tasks plus a real mutating trajectory through save, fresh-DB
replay, history/DB/counter verification, and policy-prefix continuation.

## Prepare data

From the repository root:

```bash
bash experiments/setup/download/stage_areal_tau2.sh
```

The staging path downloads only the RL JSONL and nine DB snapshots, verifies
the pinned revision and SHA-256 values, validates all Task schemas against
`tau2==1.0.1`, and precomputes each gold terminal DB hash. It does not download
the 33,531-row SFT file.

## Update count

With one optimizer step per rollout batch:

`updates = ceil(1,982 * epochs / rollout_batch_size)`

| Epochs | RBS 16 | RBS 32 | RBS 63 | RBS 64 | RBS 192 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 620 | 310 | 158 | 155 | 52 |
| 6 | 744 | 372 | 189 | 186 | 62 |

The production default is 6 epochs, RBS 63, and 16 sampled trajectories per
prompt group: GBS is therefore 1,008 samples. It schedules 189 weight updates,
11,907 prompt groups (15 final-batch repeats), and 190,512 trajectories. Five
epochs schedules 158 updates, 9,954 prompt groups (44 repeats), and 159,264
trajectories. The group size increases trajectory count for GRPO but does not
increase the number of weight updates.

The epoch cap is enforced at six. Because only 1,982 published task rows exist,
more updates obtained by making RBS smaller do not create more task diversity.
The pinned source has 1,981 unique Task+DB payloads (one exact duplicate row);
its task ID field is not an identity key, so preparation uses the source row
index and preserves every published row.

## Batch topology

The batch choice is compatible with all four possible eight-node
trainer:rollout splits:

| Train:rollout nodes | Train GPUs | Rollout GPUs | Trainer DP | GBS / DP rank |
| ---: | ---: | ---: | ---: | ---: |
| 1:7 | 8 | 56 | 4 | 252 |
| 2:6 | 16 | 48 | 8 | 126 |
| 3:5 | 24 | 40 | 12 | 84 |
| 4:4 | 32 | 32 | 16 | 63 |

Qwen3-4B uses TP=2 and CP=1, so the relevant divisibility constraint is the
trainer data-parallel size `4, 8, 12, 16`, not merely the trainer node count.
Their least common multiple is 48. With `n=16`, `GBS=RBS*16`; the closest valid
GBS to 1,024 is therefore 1,008, giving RBS 63. RBS does not have to divide the
number of rollout nodes because the fully asynchronous worker schedules whole
trajectory groups through a shared queue.

The current staleness study intentionally uses only the 1:7 and 2:6 rows. Its
dedicated launcher is described below. A direct recipe launch uses the default
1:7 split:

```bash
source experiments/env.sh
source experiments/common/pbs.sh
pbs_submit --profile=gpu --nodes=8 --time="${PBS_DEFAULT_WALLTIME}" \
  --export=MILES_WORKSPACE_ROOT \
  experiments/scripts/tau_bench/async/areal-tau2/qwen3-4b-agentic-sft-953/run.sbatch
```

Set `TRAIN_EPOCHS=5` to use the shorter schedule. Detailed tool/user-simulator
wait timing is off by default and is enabled only with `TAU_LOG_OVERHEAD=1`.
The official Tau runtime log level defaults to `TAU_LOG_LEVEL=ERROR` so its
per-step DEBUG/INFO messages do not dominate a full run.

## 40K staleness and truncation study

The dedicated launcher below is dry-run by default and submits 12 independent
arms with `--submit`:

```bash
bash experiments/scripts/tau_bench/async/areal-tau2/submit_staleness_truncation_sweep.sh
bash experiments/scripts/tau_bench/async/areal-tau2/submit_staleness_truncation_sweep.sh --submit
```

The grid is max weight staleness `{8,16,20}` x trainer:rollout nodes
`{1:7,2:6}` x truncation treatment `{zero-reward,zero-loss}`. Every arm fixes
max context and response length to 40,960, RBS=63, n=16, GBS=1,008, TIS,
LR=1e-6, inflight replay, and DB-restore/prefill overlap. It runs 180 updates:
11,340 prompt groups, 181,440 trajectories, and 5.721 effective passes over the
1,982 rows. This lies between the exact five-epoch (158 update) and six-epoch
(189 update) schedules and ends on a ten-update checkpoint boundary.

`zero-reward` assigns raw scalar reward zero to a truncated trajectory and
retains its response loss mask. `zero-loss` retains the environment reward for
logging and GRPO group normalization but sets that trajectory's complete
response loss mask to zero. The treatments are separate run/checkpoint
identities and are never enabled together.

Both Megatron and HF artifacts are written every ten updates. HF snapshots are
all retained. `SAVE_RETAIN_INTERVAL=181` is beyond the run horizon, so Megatron
retention keeps only the latest resumable checkpoint; replay retention is also
one committed buffer. Fifteen minutes before each 24-hour wall limit the
segment creates an external-save sentinel and submits exactly one `afterany`
successor. Reusing the same `RUN_NAMESPACE` resumes that identity. Once update
180 is committed, training exits normally and no new successor is submitted.
`CHAIN_MAX_SEGMENTS` defaults to 64 only as a runaway-chain guard.

Submission is idempotent with respect to active PBS jobs. Re-running the
launcher with the same `RUN_NAMESPACE` skips matching pending or running arms
and submits only missing arms, which also makes recovery from a partial
controller outage safe.

Detailed reset, tool, user-simulator, terminal, cleanup, and resume-overlap
timings are enabled for this study. They are retained per trajectory and the
non-generation total also feeds the shared rollout timing metrics.

Configured staleness must be interpreted together with realized staleness. The
queue holds 1,000 completed groups and this study has six groups in flight, so
its structural lag ceiling is approximately `(1000 + 6) / 63 = 15.97` updates.
Consequently M=20 is the effectively unbounded control and M=16 is near the
natural ceiling; M=8 is the arm expected to exercise recycling. The launcher
prints this warning and expands the trainer-side staleness histogram through
32. Compare Tau v3 scores from the retained HF snapshots alongside realized
lag, recycled groups, wasted tokens, throughput, truncation rate, and API/user-
simulator timing rather than treating configured M as the observed exposure.

Tau v3 v1.0.1 test evaluation is a separate held-out job. The training recipe
forces in-run evaluation off and never materializes official Tau v3 train/base
rows:

```bash
export RESULT_ROOT=/path/to/results
export TRAINING_HF_ROOT=/path/to/training/hf
pbs_submit --profile=gpu \
  --export=MILES_WORKSPACE_ROOT,RESULT_ROOT,TRAINING_HF_ROOT \
  experiments/scripts/tau_bench/evaluate.sbatch
```
