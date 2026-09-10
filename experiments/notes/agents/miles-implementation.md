# Work log — miles implementation

## 2026-08-03 — code reading for the experiments/ scripts

Repo state: `main` at `5400905d`.

### Launch path

Two launcher styles coexist:

- `scripts/run-*.sh` — bash arrays (`MODEL_ARGS`, `CKPT_ARGS`, …) then
  `ray start --head` + `ray job submit -- python3 train.py`. `experiments/*/train.sh`
  follows this shape; only the paths differ.
- `scripts/run_*.py` — typer + dataclass (`miles/utils/typer_utils.py:25`,
  `dataclass_cli`, env prefix `MILES_SCRIPT_`). The shared implementation is
  `miles/utils/external_utils/command_utils.py:109` (`execute_train`): it kills
  stale processes, starts ray unless `MILES_SCRIPT_EXTERNAL_RAY=1`, assembles a
  runtime-env JSON (PYTHONPATH, `CUDA_DEVICE_MAX_CONNECTIONS`, NVLS detection,
  NCCL socket vars) and submits the job. `ExecuteTrainConfig.num_nodes` defaults
  to `$SLURM_JOB_NUM_NODES` (`:104`), so the `.py` launchers are already
  Slurm-aware.

`train.py` itself is 158 lines: placement groups → rollout manager (router +
SGLang engines) → training models → weight sync → loop.

### Plug points

`docs/user-guide/environments.md` defines three nested layers; verified against
the flags in `miles/utils/arguments.py`:

| Flag | Default implementation |
|---|---|
| `--rollout-function-path` | `miles/rollout/sglang_rollout.py` |
| `--custom-generate-function-path` | `generate_hub/single_turn.py` |
| `--custom-agent-function-path` | consumed by `generate_hub/agentic_tool_call.py` |
| `--rm-type` / `--custom-rm-path` | `rm_hub/` (`deepscaler`, `math`, `dapo`, `gpqa`, `f1`, `ifbench`, `remote_rm`, …) |

`generate_hub/multi_turn.py` is the framework-side multi-turn loop the ReTool v2
example rides on; the example supplies only tool specs, a tool executor and a
reward function. `generate_hub/agentic_tool_call.py` documents the agent-function
contract (`base_url, prompt, request_kwargs, metadata` → optional metadata dict).

### Batch-size invariant

`miles/utils/arguments.py:3057`:

```
global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout
```

and it asserts consistency if `--global-batch-size` is also given. Dynamic
sampling adds `over_sampling_batch_size` (`:3072`), defaulted to
`rollout_batch_size` and asserted `>=` it. In fully-async mode
`--async-max-concurrent-samples` (`:630`) decouples generation concurrency from
the training batch, with a floor of `n_samples_per_prompt` (`:3048`).

Relevance to the throughput study: **concurrent sandboxes** =
`max(async_max_concurrent_samples, over_sampling_batch_size × n_samples_per_prompt)`,
not simply `rollout_batch_size × n_samples_per_prompt`.

### Routing / session

- `miles/ray/rollout/router_manager.py:37` — `assert not has_pd_disaggregation`
  for the miles router: **PD disaggregation requires the SGLang router**.
- `miles/rollout/session/core.py:307` — the session server injects
  `X-SMG-Routing-Key: <session_id>`; with router policy `manual` or
  `consistent_hashing` (`generate_utils/generate_endpoint_utils.py:37`) this pins
  a session to one engine. `--use-session-server` auto-selects policy `manual`
  (`backends/sglang_utils/arguments.py:159`).
- `--session-server-port <start> <end>` launches one server per port; the
  session URL carries the affinity (`router_manager.py:96`).

### PD disaggregation

`--prefill-num-servers N` → `SglangConfig.from_prefill_num_servers`
(`backends/sglang_utils/sglang_config.py:170`), mutually exclusive with the YAML
`sglang_config` and with `--rollout-external` (`arguments.py:3129`). KV transfer
is handled by the SGLang Model Gateway.

### Not verified

- No run has been executed yet; every claim above is from reading the code, not
  from observing behaviour.
- FSDP backend paths (`scripts/run_*_fsdp.py`) were not read.

## 2026-08-09 — the staleness bound measures queue residency, and `in_place` hides the rest

Repo state: `experiments/cw-dfw-math-rl` at `653ff029`, plus the change described
below. Measured on job 15429055 (`interactive`, 2 nodes, `pool0-[00005,00056]`).

### The claim

`staleness = current - oldest_weight_version` (`fully_async_rollout.py:409`) is a
difference against the version a group **finished** under. `Sample.weight_versions`
gets one entry per generate *call* (`types.py:285-286`), stamped by SGLang when it
builds the reply (`tokenizer_manager.py:1982`, sglang 0.5.17.dev32+g3fe50ed — read
inside the image that runs, not from a GitHub tag). `Req` has no weight-version
field in `schedule_batch.py` or `scheduler.py`, so no arrival-time value exists to
report, and `/model_info` (`http_server.py:716`) returns the same
`server_args.weight_version` that miles polls for `current`. Single-turn generation is one call, so the list has one element.

`PAUSE_GENERATION_MODE=in_place` — the recipe default (`run.sbatch:57`) and what
every sweep passes — returns from `pause_generation` before touching scheduler
state (`sglang/srt/managers/scheduler.py:4465-4475`), so a request frozen across a
weight update and resumed on the old KV cache produces one reply and **zero**
retractions (`retraction_count` is only bumped by `Req.reset_for_retract`,
`schedule_batch.py:1603-1606`). Nothing in the reply records the span.

### The measurement

Shrunk smoke: `NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=64 MAX_RESPONSE_LEN=2048 CONFIG_TAG=smoke-genspan-20260809
SAVE_HF=0`, everything else at recipe defaults (`MAX_WEIGHT_STALENESS=2`,
`in_place`, 1 train node + 8 rollout GPUs, `tp2 cp1 -> dp4`).

| rollout | `weight_version/min..max` | `mixed_version_ratio` | `staleness/total/mean` | `staleness/pre_queue/mean` | `staleness/total/max` |
|---|---|---|---|---|---|
| 0 | 1 .. 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 1 .. 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | **2 .. 2** | **0.0** | **0.0** | **1.0** | **1.0** |

At rollout 2 every one of the 8 groups started under v1 and finished under v2.
Every pre-existing instrument reports on-policy: `weight_version` min and max are
both 2, `mixed_version_ratio` is 0, `staleness/train` is 0. `queue_size` was 0
throughout, so the residency really was zero — the lag was entirely inside
generation.

**`mixed_version_ratio` and `dump/mixed_version_frac` are structurally 0 for this
study.** They are `len(set(weight_versions)) > 1`, and the set has one element
unless a sample took more than one generate call — multi-turn or partial-rollout
resume, and fully-async rejects partial rollout (`arguments.py:54`). A flat zero
there is not evidence.

### What was added

`_generate_group` reads the version before generating and stamps it into
`Sample.metadata["submission_weight_version"]` (`fully_async_rollout.py:64-90`),
the same mechanism multi-LoRA already uses for `slot_version`
(`multi_lora/async_rollout.py:141-146`). The accepted-batch decomposition is
emitted whether or not `MAX_WEIGHT_STALENESS` is configured; `staleness/rollout`
also records the offered population before admission. With S the selected
reference, Q the group-ready version, D the dequeue version, and T the train
version:

| key | quantity |
|---|---|
| `staleness/pre_queue/*` | `Q - S` — updates crossed before generation/reward/finalization made the group ready |
| `staleness/in_queue/*` | `T - Q` — updates crossed from ready to training |
| `staleness/total/*` | `T - S` = `pre_queue + in_queue` |
| `staleness/rollout/*` | `T* - S` — would-be train staleness before admission |

Names taken from Applied Compute's PQS/IQS decomposition. **Q is the version
when the whole group becomes queue-ready.** It is stamped after generation,
reward, and finalization, and can exceed the group's last-forward version when
an update lands during post-forward work. Keying on an earlier sample boundary
would charge pre-ready work to in-queue staleness and invert the split.

For `queue-recycle`, the bound admits `D-S < max`; with one scheduled update
before training this is equivalent to `T-S <= max`.
The minimum bound of 1 follows from the strict rule and a nonnegative dequeue
gap. It is independent of startup: startup uses `T=D`, while normal prefetched
batches use `T=D+1`.

Missing reference provenance stays absent rather than being reported as zero.

Two things the change had to fix to be safe:

- `_CachedWeightVersion.get` is now called from every in-flight group, so the
  first fill would have put one `/model_info` request per group on the router at
  once. It is single-flighted behind an `asyncio.Lock` with a re-check.
- An unreadable version on the submission path would reach `_worker_loop`'s
  `task.result()` and kill the rollout. `_current_weight_version` swallows it and
  returns None — instrumentation must not be able to stop generation.

### Caveats on this run

- `truncated_ratio` was 0.88–0.98 because the smoke ran at `MAX_RESPONSE_LEN=2048`;
  `raw_reward` was 0.031/0.016/0.109, so the verifier matches the checkpoint.
- Three rollouts is one weight update's worth of evidence. It shows the metric
  fires and is consistent (`total = pre_queue + in_queue`, `num_groups` =
  `rollout_batch_size`); it says nothing about the magnitude at production
  response length, which is the number the study actually needs.
- **The table above was measured with an earlier revision that keyed the split on
  the group's oldest sample.** In that run all 8 samples of the batch landed under
  v2, so oldest == newest and the numbers are what the shipped `newest`-keyed code
  would report. Re-verified on the shipped code below.

### The straggler case, observed (job 15433024, 4 rollouts, same shape)

Same recipe, `NUM_ROLLOUT=4`, `CONFIG_TAG=smoke-pqs-iqs-20260809`. All four
rollouts satisfy `total = pre_queue + in_queue`, no old key survives, and the
router was readable throughout. Rollout 2 is the interesting one:

| family | mean | max | histogram |
|---|---|---|---|
| `pre_queue` | 1.0 | 1 | `count_1: 8` |
| `in_queue` | 0.0 | 0 | `count_0: 8` |
| `total` | 1.0 | 1 | `count_1: 8` |
| oldest-reference lag (historical diagnostic) | 0.125 | 1 | `count_0: 7`, `count_1: 1` |

`in_queue` is 0 for all eight groups, so the single group reading 1 under the
bound has `Q - oldest = 1`: **its samples split across v1 and v2**. That is the
straggler, and it showed up in 1 of 8 groups at 2048 response length on the first
run — an earlier revision of this note predicted it would not be observable at
this scale, which was wrong. Under the superseded `oldest`-keyed code that group
would have been logged `pre_queue = 0, in_queue = 1`; the shipped code logs
`pre_queue = 1, in_queue = 0`. The inversion is real and the fix corrects it.

Still not measured: the magnitude at production response length, and any run in
which `in_queue` is non-zero (`queue_size` was 0 in every rollout of both smokes,
so nothing ever waited).

- Checkpoint save/resume was not exercised (`NUM_ROLLOUT=3` < `SAVE_INTERVAL=10`).
  The change does not touch that path.

## 2026-08-11 — deferred actor startup warmup

Status: design note only; do not implement yet.

The first trainer call has a large cold-start transient that is not explained by
batch size. Across the staleness/node-ratio jobs, rollout step 0 was 3.1–10.1x
slower than the median of steps 1–9. For example, `s4-t2r6` spent 383.2 s in
`train_wait`, 617.9 s in actor log-prob forward, 286.9 s in actor training and
14.9 s updating weights. The next step's corresponding log-prob and actor-train
times were about 31.0 s and 120.9 s. The first batch contained fewer tokens than
steady state, and its log-prob completion times ranged from about 48 s to 618 s
across ranks before converging to 30–31 s on the next call. The likely cause is
distributed CUDA first use (module/kernel loading, allocator state, pipeline/TP
communication, autograd/gradient-reduction setup), with every rank waiting for
the coldest one; the logs do not isolate one compiler or kernel as the sole
cause.

If this becomes worth addressing, the deliberately small first change is to
overlap a **forward-only actor warmup** with the first rollout generation:

1. Perform the initial actor-to-rollout-engine weight sync as today.
2. Start the first `rollout_manager.generate` future.
3. Concurrently run a synthetic packed batch through actor `forward_only` on
   every training rank, using the real dtype and TP/PP/CP layout and a
   representative sequence length; discard the log-prob result.
4. Wait for both futures, then train the real rollout batch normally.

This warmup must not call `optimizer.step`, advance the scheduler, change model
weights or increment the policy version. Its objective is not to make startup
staleness exactly zero; it is to move the anomalous first log-prob setup work
onto time in which the independent rollout nodes are already generating. A
producer startup gate is therefore intentionally out of scope for the first
change. The fully-async producer may still accumulate an initial version-0
backlog during the overlap, but the interval before the first weight update
should be shorter than the current serial `initial generation -> cold trainer`
path.

If forward-only warmup leaves a material first-backward penalty, a second change
can add a forward/backward dry run with the optimizer disabled, followed by
gradient clearing and RNG restoration. Do not use a zero learning rate as a
substitute: Adam moments and step counters would still change. Warming the actual
optimizer path by snapshotting and restoring the full model/optimizer/scheduler
state is not justified unless measurements show that optimizer first use is a
large remaining term.

Before adding either code path, first test `CUDA_MODULE_LOADING=EAGER` in one
otherwise identical run. For any implementation, verify per-rank warmup time,
real step-0 log-prob/actor-train time versus steady state, unchanged model and
optimizer checksums/state, and the resulting startup
`staleness/{pre_queue,in_queue,total}` trace.
