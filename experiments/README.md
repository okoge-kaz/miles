# experiments/

Slurm (pyxis + enroot) recipes for running miles on **cw-dfw**. Non-agentic RL
first: these establish that the container, the checkpoints, the rollout loop and
the reward path all work, before anything needs a sandbox provider.

Everything here is deliberately explicit — paths, argument groups, and mounts
are spelled out rather than inherited from `scripts/`, so a diff between two
experiments shows exactly what changed.

## Layout

```
experiments/
  env.sh                        shared paths / account / container mounts
  status.sh                     queue + logs at a glance
  container/import_image.sbatch docker image -> dated .sqsh
  setup/download_assets.sbatch  HF model + datasets
  setup/convert_checkpoint.sbatch  HF -> torch_dist (Megatron)
  setup/stage_model.sh          one model: download + convert, chained
  sweep.py                      submit a grid of runs; sweeps/ holds the grids
  configs/eval_math.yaml        multi-benchmark eval (--eval-config)
  common/run_identity.sh        run name, config tag, checkpoint path
  common/placement.sh           derive + validate the GPU split, before srun
  common/ray_cluster.sh         multi-node ray bring-up
  math_sync/<dataset>/          README.md + one directory per model
  math_sync/<dataset>/<model>/  single-turn math RL (GRPO, deepscaler reward)
  math_async/<dataset>/<model>/ the same task on the fully-async rollout
  tool_multiturn/<model>/       multi-turn tool-calling RL (ReTool v2 style)
  notes/                        reference notes + notes/agents/ work log
  outputs/                      job logs (git-ignored)
```

A recipe is `<task>/<dataset>/<model>/`, e.g. `math_sync/dapo-math-p10-90/qwen3-4b-instruct-2507/`.
The task directory fixes the RL setup (rollout mode, reward, generate function),
the dataset directory fixes the prompt file and the eval benchmarks, and the
model directory fixes the weights, the parallelism and the batch shape.

Every level is a directory rather than a conditional inside a script: there is
no `if` picking a dataset or an eval set at runtime, so the file that runs is
the file you read.

### Dataset directories

**Every `<task>/<dataset>/` directory carries a `README.md`, and it has the same
three parts in the same order.** A new dataset is not staged until its README
exists — the point of the directory is that the numbers it is aiming at and the
ranges it may be moved through are written down before any GPU time is spent.

1. **Prior work.** A table of published reference points for this dataset or
   task: system, base model, reported result, and an **arXiv link**. Then a short
   paragraph on which of those papers actually determine the algorithm the
   recipes run, and a provenance note separating what was read from the abstract
   from what was taken from a paper's body and not re-verified.
2. **Reference hyperparameters vs. this recipe.** A side-by-side of the leading
   paper's settings and the recipe's defaults, so the gaps are explicit rather
   than discovered later.
3. **What to search, and over what range.** A hyperparameter here is any value
   that can move **throughput** or **downstream performance** — global batch
   size, samples per prompt, rollout batch size, off-policy steps, learning rate,
   max response and context length, the GPU split, the engine geometry. Per
   knob: the current default, the range worth sweeping, **which of the two it
   affects** (`throughput` / `quality` / `both`), and *why* — naming the metric
   that says whether the move worked. Grouped by class and ordered by expected
   effect, because the run ladder moves one class at a time; the `throughput`
   rows are exactly the ones that can be swept without invalidating a
   comparison.

A local baseline section — what this cluster actually measured — belongs between
2 and 3 once it exists. Settings that are correctness rather than search space
(`RM_TYPE` against a non-thinking checkpoint, for instance) go in a closing
"not a hyperparameter" note so they are not swept by accident.

`math_sync/dapo-math-p10-90/README.md` and `math_async/dapo-math-p10-90/README.md` are the
worked examples. The two differ only where the task differs: the async one
searches staleness and the actor/rollout split, the colocated one does not have
those knobs at all.

Each recipe is a pair: `run.sbatch` allocates the node and starts the
container, `train.sh` runs inside it and holds the actual argument groups.
`train.sh` is a full copy per model rather than a shared file with overrides —
one file is the whole configuration, and a diff between two models shows every
difference at once.

### Models

Both math tasks carry **one** checkpoint. The 1.7b / 4b / 8b / 30b-a3b recipes
were deleted on 2026-08-05: none of them had ever been run, so their parallelism
and token budgets were unverified, and a stale recipe is worse than no recipe.
Re-add one by copying the surviving pair and measuring it.

| model | sync (16 GPU colocated) | async (8 train + 8 rollout) | verifier |
|---|---|---|---|
| `qwen3-4b-instruct-2507` | TP2 CP4, 9216 tok/GPU | TP2 CP1, 32768 | **`math`** |

Two rules the row encodes:

* `max_tokens_per_gpu × cp_size ≥ rollout_max_context_len`, because a single
  sample has to fit in that budget
  (`miles/backends/training_utils/data.py:473`). With the shared 32768 context
  the row clears it.
* **`qwen3-4b-instruct-2507` defaults to `--rm-type math`, not `deepscaler`.**
  deepscaler returns 0 unless the response contains a `</think>` delimiter
  (`rm_hub/deepscaler.py:36-44`), and this checkpoint is non-thinking, so under
  deepscaler the entire run would score reward 0 without erroring anywhere.

### Off-policy control

Both tasks pass `--num-steps-per-rollout` explicitly (default 1). That makes
the number of optimizer steps taken per rollout batch a named knob instead of
an implicit consequence of `--global-batch-size`, and it is what turns the
four-knob invariant into a startup assert (`arguments.py:3056` only checks it
when the flag is present).

Dynamic sampling is **off**: no `--dynamic-sampling-filter-path`, and difficulty
is handled once, offline, by `experiments/src/difficulty_filter` — which is what
the `p10-90` in the dataset name is. The online filter drops a variable number of
groups per step, so the generation cost of a step depends on the policy rather
than on the configuration; with wall-clock as the study's primary axis, that is
uncontrolled variance in the metric being reported. Measured, it also tripled
rollout time (253 s → 761 s). `--over-sampling-batch-size` is likewise not
passed, so the rollout loop submits exactly what it needs.

The offline filter is keyed to a *policy*: `dapo-math-p10-90` is the 0.1–0.9 pass
rate window measured with Qwen3-4B-Instruct-2507. Another model needs its own
measurement and its own dataset directory — and with the online filter gone there
is no safety net when the window is stale.

`--partial-rollout` is on in `math_sync`, to carry over generation still in flight
when a rollout batch closes. It does not exist in `math_async` —
`--fully-async` rejects the flag (`arguments.py:54`); the equivalent question
there is `PAUSE_GENERATION_MODE`, which decides what happens to in-flight
generation at a weight update.

`--mask-offpolicy-in-partial-rollout` is **never** passed. It zeroes the loss
over everything a resumed sample produced under the previous weights, which
discards exactly the tokens partial rollout exists to keep.

`math_async` additionally bounds staleness with `--max-weight-staleness`
(default 2) — miles' own default is *no bound*, which lets an arbitrarily old
group reach the optimizer — and corrects the remaining lag with `--use-tis`.

### Multi-node

Node counts and parallelism are set at the top of `run.sbatch`, in one block of
`: "${VAR:=value}"` defaults. Edit that block for a lasting change, or override
any of it on the command line — the `:=` form means an exported value wins:

```bash
sbatch -A $ACC -N 4 \
  --export=ALL,ACTOR_NUM_NODES=2,ACTOR_GPUS_PER_NODE=8,ROLLOUT_NUM_GPUS=16 \
  experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
```

Nothing is inferred from the allocation. `-N 4` allocates four nodes;
`ACTOR_NUM_NODES` and `ROLLOUT_NUM_GPUS` say what they are used for, and
`common/placement.sh` rejects the job **before `srun`** — in seconds, without
starting a single container — if:

* the split does not add up to the allocation,
* `tp × cp` does not divide the training GPUs,
* the global batch is not divisible by the resulting `dp`,
* the engine pool is not divisible by `--rollout-num-gpus-per-engine`.

It prints the resolved shape on success:

```
placement 2x8 train (16 GPU) + 16 rollout, tp2 cp2 -> dp4
```

`train.sh` reads these as plain variables and defines no defaults of its own,
so the value in `run.sbatch` is the value that runs. More nodes raise `dp` and
shrink the per-rank batch; raise `ROLLOUT_BATCH_SIZE` with the allocation.

### Checkpoint layout

Checkpoints are keyed by configuration, not by job id:

```
/ckpt/training/<task>/<dataset>/<model>/<rl-algorithm>/<placement>/<policy-regime>/max-weight-staleness-<s>/<config>

/ckpt/training/math/dapo-math-p10-90/Qwen3-4B/grpo-clip0.2-0.28-tis2.0/
    colocated/on-policy/max-weight-staleness-0/
    rollout-length-32k-lr1e-6-rbs256-gbs2048-tseed1234-rseed42
```

`RL_ALGORITHM` is *derived* from the loss knobs (`ADVANTAGE_ESTIMATOR`,
`EPS_CLIP`, `EPS_CLIP_HIGH`, `USE_TIS`, `TIS_CLIP`, `TIS_CLIP_LOW`,
`KL_LOSS_COEF`), never set by hand, so the directory name cannot drift from what
the run passes. `POLICY_REGIME` is `on-policy` only when
`MAX_WEIGHT_STALENESS == 0` *and* `NUM_STEPS_PER_ROLLOUT == 1` -- the two ways a
sample goes off-policy. `PLACEMENT=colocated` refuses a non-zero staleness or a
pause mode, because neither flag exists on that path. The pause mode is not in
the tag -- it is checked off-grid, so that experiment sets `CONFIG_TAG` itself.

The run name carries the same information —
`math-dapo-math-p10-90-Qwen3-4B-dapo-async-off-1step-rollout-length-32k-lr1e-6-tseed1234-rseed42` — so the
wandb group, the log and the checkpoint directory all name the same thing.
`common/run_identity.sh` computes all of it once and is sourced by both
`run.sbatch` and `train.sh`, so the two cannot disagree.

A resumed run therefore finds its own history and two settings never share a `--load`.
`CONFIG_TAG` defaults to the rollout mode, the steps per rollout, the response
length, the learning rate and both seeds; `RL_ALGORITHM` is a directory level
above it. `TRAIN_SEED` (`--seed`) and `ROLLOUT_SEED` (`--rollout-seed`) move
independently so replicates can separate training-seed variance from
rollout-sampling variance. Set `CONFIG_TAG` explicitly for anything that default
does not encode — the staleness bound, the pause mode, the batch shape, a
filtered prompt file, a tuning job.
The dump directory follows the same path.

wandb is always on: math uses project `off-policy-<dataset>` (e.g.
`off-policy-dapo-math-p10-90`), while both Search-R1 placements use
`async-search-r1`; the group is `RUN_NAME`. `run.sbatch` fails at submit time if
`WANDB_API_KEY` is unset rather than quietly running without logging; `env.sh`
resolves it from `~/.netrc`.

### Eval

`--eval-prompt-data` takes name/path pairs and accepts as many as you give it, so
several benchmarks each report their own series. The `dapo-math-p10-90` recipes
evaluate **AIME-2025 only**, at `n=8`. In-training eval sits on the critical path
of a wall-clock measurement, so it is kept small deliberately: it exists to show
that a run is learning, not to produce a reported number. The pairs are written
out in `train.sh` — a different eval set is a different dataset directory, not a
branch.

Reported numbers come from `experiments/src/offline_eval/run_eval.sbatch` instead:
every AIME year at `n=16` and a 32768 generation budget, run against the HF
snapshots `--save-hf` leaves behind, in a separate job that costs the training run
nothing. See `math_async/dapo-math-p10-90/README.md` §2.

Search-R1 uses the analogous
`experiments/src/offline_eval/run_search_r1_eval.sbatch`.  Its dedicated client
reuses the training multi-turn generation/reward functions, runs the local
wiki-18 retriever, and reports both exact match and interaction cost (search
calls, LLM turns, and masked observation tokens).  The Search-R1 recipe exports
HF snapshots every 20 rollouts and leaves in-run evaluation off by default.

Everything listed must already be in `prompt`/`label` shape, since the global
`--input-key`/`--label-key` apply to eval too. `setup/build_math_jsonl.py`
converts a raw `problem`/`answer` file into that shape; AIME-2025 and MATH-500
were converted with it.

Per-dataset settings (a cheaper `n` for a 500-problem set, a different verifier)
are not expressible as pairs and need `--eval-config`, which **overrides**
`--eval-prompt-data` entirely. `configs/eval_math.yaml` is that config, adding
MATH-500 at `n=4`; a recipe that wants it replaces its `--eval-prompt-data`
lines with `--eval-config /root/miles/experiments/configs/eval_math.yaml`.

Eval shares the engines with training rollout, so its cost is real: that config
is 2960 generations per eval interval.

### Telemetry

The rule is: anything without overhead is always on, and the one expensive
artifact is opt-in.

Always on — wandb, `--dump-details` and `--use-miles-dashboard`. The collector is
fire-and-forget (a few ms per step, nothing on the training path waits on it) and
the rollout dumps are one file per step from a single writer. View a run, live or
finished, from anywhere that can see the directory:

```bash
python -m miles.dashboard.serve --dump-details <ckpt-path>/dump --follow
```

Off by default — `DUMP_TRAIN_DATA=1` turns it on. The per-rank train dumps are
the expensive half of `--dump-details`: every rank writes its shard each rollout,
`torch.save` runs inline on the training path, and TP/CP ranks write duplicate
copies deduplicated only at read time, so the volume scales with the training
world size rather than with `dp`. There is no retention or interval knob — every
rollout and every eval, for the whole run.

`--use-rollout-entropy` is tied to the same switch, because entropy is computed
on the train side and has no consumer without the train dumps.

What each level gives you:

| | collector + rollout dumps (default) | `DUMP_TRAIN_DATA=1` |
|---|---|---|
| timeline, GPU util, engine metrics | yes | yes |
| `dump/zero_std_group_frac`, `truncated_frac`, reward stats | yes | yes |
| `weight_version`, `mixed_version` (staleness) | yes | yes |
| conversation view | yes | yes |
| `lp_diff`, `imp_ratio` per token and per sample | — | yes |
| `advantages`, `returns`, `loss_mask`, entropy | — | yes |

Turn it on for a diagnostic run — is the train/rollout logprob gap spread out or
one pathological sample, is the resumed partial-rollout prefix entering the loss
where you think — and leave it off for the long ones.

### Sweeping

`sweep.py` submits a grid over any of the knobs above, for one model or several:

```bash
experiments/sweep.py --sweep experiments/sweeps/offpolicy.txt \
  --recipe math_async/dapo-math-p10-90/qwen3-4b-instruct-2507 \
  -- -N 2 -p batch_short --time=02:00:00
```

It prints the grid and stops. `--submit` launches it; `--max-jobs` (32) is a
deliberate ceiling. A grid file is one knob per line — several values enter the
Cartesian product, a single value is applied to every point without appearing in
the tag:

```
NUM_STEPS_PER_ROLLOUT   1 2 4
MAX_WEIGHT_STALENESS    1 4
NUM_ROLLOUT             30
```

Two things it takes care of that are easy to get wrong by hand:

* **Every point gets its own `CONFIG_TAG`** (`sweep-<name>-step2-stale4`), so no
  two points share a checkpoint directory or a wandb group. Submitting a grid
  without this silently has each run resume from the previous one's optimizer
  state.
* **The four-knob invariant is closed per point.** miles asserts
  `rollout_batch × n_samples = global_batch × num_steps` at startup, so a grid
  over the batch shape has to recompute the other side. Whichever of
  `GLOBAL_BATCH_SIZE` / `NUM_STEPS_PER_ROLLOUT` the file does not pin is derived;
  pin both and they are checked. Pinning the global batch keeps the optimizer
  step a fixed size while the rollout grows (DAPO's scaling); pinning the step
  count grows it.

Submitted points are recorded in `outputs/sweeps/<name>-<timestamp>.jsonl` with
their job ids and full environment, so results can be joined back to their
configuration.

Sweep on `batch_short`, not `batch` — a grid is measurement, and the ranges
worth covering are in each dataset's README.

### Running one

`.claude/skills/miles-run-ladder` is the intended progression: `interactive` for
bring-up on 2 nodes, `batch_short` for parallelism and rollout tuning, `batch`
for the real 4 h job.

## Logs

Every job writes one combined stdout+stderr file to
`experiments/outputs/<job-name>-<jobid>.log`. The `#SBATCH --output` path is
relative, so **submit from the repo root**.

```bash
experiments/status.sh            # queue + the 10 newest logs
experiments/status.sh -f         # follow the newest log
experiments/status.sh 15004856   # job detail + follow that job's log
```

`experiments/outputs/` is git-ignored (`outputs/.gitignore`), so logs never end
up in a commit.

## Paths on lustre

| Purpose | Host path | In container |
|---|---|---|
| Datasets | `/lustre/fsw/portfolios/coreai/users/kfujii/datasets` | `/data` |
| HF weights | `…/checkpoints/hf` | `/ckpt/hf` |
| Megatron (`torch_dist`) weights | `…/checkpoints/megatron` | `/ckpt/megatron` |
| Training checkpoints | `…/checkpoints/training` | `/ckpt/training` |
| Container images | `…/container` | — |
| Caches (HF, enroot, JIT) | `…/cache` | `/root/.cache` |
| miles checkout | `…/src/miles` | `/root/miles` |

The image ships miles at `/root/miles`; the mount shadows it with this checkout,
so edits here take effect without rebuilding.

`$HOME` is not mounted, so everything that writes to `~/.cache` already lands on
that mount and survives the job — that is how `huggingface/`, `tvm-ffi/`
(sgl_kernel) and `deep_gemm/` got there. `env.sh` additionally redirects the
caches that default *outside* `~/.cache` and would otherwise be recompiled every
job: torch inductor (`/tmp/torchinductor_$USER`, and `/tmp` is RAM-backed here),
triton (`~/.triton/cache`), the CUDA PTX JIT cache (`~/.nv/ComputeCache`), and
SGLang's DeepGEMM cache, which miles otherwise pins under `/tmp`
(`ray/rollout/server_group.py:107`).

CUDA graphs are not part of this: they are captured in-process at engine start
and never written to disk. What a warm cache saves is the torch.compile and
kernel-JIT work before the capture, not the capture itself.

A corrupt entry after a killed job surfaces as a JIT or JSON decode error
(`docs/faq.md:112`); delete that directory under `$CACHE_DIR` and rerun.

## Order of operations

```bash
export ACC=coreai_horizon_dilations

# 1. Image (CPU node, ~30-60 min). Writes miles-latest-YYYYMMDD.sqsh and
#    repoints the miles-latest.sqsh symlink.
sbatch -A $ACC experiments/container/import_image.sbatch

# 2. Model + datasets (CPU node).
sbatch -A $ACC experiments/setup/download_assets.sbatch

# 3. HF -> torch_dist (GPU node, ~10 min).
sbatch -A $ACC experiments/setup/convert_checkpoint.sbatch
```

Adding one more model later does not need the whole list re-staged.
`stage_model.sh` submits that model's download and chains its conversion behind
it; both halves are skipped if already done.

```bash
experiments/setup/stage_model.sh Qwen3-4B-Instruct-2507
```

Then the training recipes:

```bash
# 4a. Math RL, colocated.
sbatch -A $ACC experiments/math_sync/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch

# 4b. The same task on the fully-async rollout.
sbatch -A $ACC experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
```

A fast smoke variant (a few rollouts, short responses), for checking a change
before spending a slot on a full run:

```bash
sbatch -A $ACC -p interactive --time=01:00:00 \
  --export=ALL,NUM_ROLLOUT=3,ROLLOUT_BATCH_SIZE=8,N_SAMPLES_PER_PROMPT=8,GLOBAL_BATCH_SIZE=64,MAX_RESPONSE_LEN=1024 \
  experiments/math_sync/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
```

Keep the four-knob invariant when changing batch sizes:

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

The recipes pass `--num-steps-per-rollout` (default 1), so miles checks this at
startup and aborts if it does not hold. Change `GLOBAL_BATCH_SIZE` and
`NUM_STEPS_PER_ROLLOUT` together — raising the step count is how a run is made
deliberately off-policy, and the assert is what stops it happening by accident.

## Cluster facts these scripts assume

Measured on cw-dfw, 2026-08-03:

- GPU partitions all serve the same `pool0-*` nodes: **H100 x8, 128 CPUs, 2 TB RAM**.
  The partition also selects the QoS (`batch_short` -> `p_batch_short`), so
  `--qos` is never passed explicitly — **choosing the partition is the only
  scheduling lever**:

  | partition | MaxTime | node cap |
  |---|---|---|
  | `interactive` | 4 h | **2** (GPU 16) |
  | `batch_short` | 2 h | 4 |
  | `batch` | 4 h | 768 |
  | `batch_long` | 8 h | 768 |
  | `batch_large_long` | 14 d | 768 |

  **Default to `interactive`.** Every recipe here is single-node and 4 h, which
  is exactly what it covers, and it schedules far ahead of `batch`. Reach for
  `batch_long` / `batch_large_long` only when a run genuinely needs more than
  4 h, and for `batch` only above 2 nodes.
- CPU partitions (`cpu` 1d, `cpu_long` 7d): 96 CPUs, no GPU. `cpu_interactive`
  (1 d) is the fast lane for downloads and image imports.
- **No docker and no apptainer**; `enroot` + pyxis (`srun --container-image`) only.
  `/etc/subuid` is empty, so nested rootless docker / `--fakeroot` is not an option.
- Compute nodes have egress (huggingface.co, ghcr.io, app.daytona.io all reachable).
- `/tmp` is RAM-backed — never point enroot scratch at it (see `import_image.sbatch`).

## Optional

- **wandb**: set `WANDB_API_KEY` (and optionally `WANDB_PROJECT`) in the
  submitting shell; `--export=ALL` carries it in. Without it, wandb stays off.
- **Docker Hub rate limits**: if `enroot import` returns HTTP 429, put
  credentials in `$ENROOT_CONFIG_PATH/.credentials`
  (`machine auth.docker.io login <user> password <token>`).

## Next step (agentic RL)

These two recipes need no sandbox. The agentic recipes in `examples/`
(Harbor, OpenEnv, NeMo-Gym) each need a container runtime or an external sandbox
service; on this cluster that means the internal OpenSandbox service, which is
reachable from cw-dfw compute nodes (verified: HTTP 401 without a key). See
`docs/user-guide/environments.md` for how the connectors plug in.
