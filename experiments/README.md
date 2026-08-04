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
  configs/eval_math.yaml        multi-benchmark eval (--eval-config)
  common/ray_cluster.sh          multi-node ray bring-up, sourced by every train.sh
  math_sync/<model>/            single-turn math RL (GRPO, deepscaler reward)
  math_async/<model>/           the same task on the fully-async rollout
  tool_multiturn/<model>/       multi-turn tool-calling RL (ReTool v2 style)
  notes/                        reference notes + notes/agents/ work log
  outputs/                      job logs (git-ignored)
```

A recipe is `<task>/<model>/`, e.g. `math_sync/qwen3-4b/`. The task directory
fixes the RL setup (rollout mode, reward, generate function); the model
directory fixes the weights, the parallelism and the batch shape. Adding a
model to an existing task means one new subdirectory, not a new task.

Each recipe is a pair: `run.sbatch` allocates the node and starts the
container, `train.sh` runs inside it and holds the actual argument groups.
`train.sh` is a full copy per model rather than a shared file with overrides —
one file is the whole configuration, and a diff between two models shows every
difference at once.

### Models

Both math tasks carry the same five checkpoints. Only the parallelism, the
per-GPU token budget and the engine size differ; the RL settings are identical.

| model | sync (8 GPU colocated) | async (4 train + 4 rollout) | verifier |
|---|---|---|---|
| `qwen3-1.7b` | TP1 CP4, 12288 tok/GPU | TP1 CP4, 12288 | `deepscaler` |
| `qwen3-4b` | TP2 CP4, 9216 | TP2 CP2, 16384 | `deepscaler` |
| `qwen3-4b-instruct-2507` | TP2 CP4, 9216 | TP2 CP2, 16384 | **`math`** |
| `qwen3-8b` | TP4 CP2, 16384 | TP4 CP1, 32768 | `deepscaler` |
| `qwen3-30b-a3b` | TP4 EP8, 32768, R3 | TP4 EP4, 32768, R3 | `deepscaler` |

Two rules the table encodes:

* `max_tokens_per_gpu × cp_size ≥ rollout_max_context_len`, because a single
  sample has to fit in that budget
  (`miles/backends/training_utils/data.py:473`). With the shared 32768 context
  every row clears it.
* **`qwen3-4b-instruct-2507` defaults to `--rm-type math`, not `deepscaler`.**
  deepscaler returns 0 unless the response contains a `</think>` delimiter
  (`rm_hub/deepscaler.py:36-44`), and this checkpoint is non-thinking, so under
  deepscaler the entire run would score reward 0 without erroring anywhere.

`qwen3-30b-a3b` is the only MoE and the only recipe with R3
(`--use-rollout-routing-replay`), which replays the routing SGLang used during
generation in the training forward pass. It also moves the fp32 Adam state to
the host (`--optimizer-cpu-offload`), since 30B of optimizer state does not fit
beside the weights on one node.

### Off-policy control

Both tasks pass `--num-steps-per-rollout` explicitly (default 1). That makes
the number of optimizer steps taken per rollout batch a named knob instead of
an implicit consequence of `--global-batch-size`, and it is what turns the
four-knob invariant into a startup assert (`arguments.py:3056` only checks it
when the flag is present).

Dynamic sampling and partial rollout are **on by default** in both tasks
(`DYNAMIC_SAMPLING=0`, `PARTIAL_ROLLOUT=0` opt out). They belong together:
over-sampling aborts whatever is still generating once enough groups have passed
the filter, and without `--partial-rollout` that work is discarded rather than
resumed.

`--mask-offpolicy-in-partial-rollout` is **off** (`MASK_OFFPOLICY=1` turns it
on). It zeroes the loss over everything a resumed sample produced under the
previous weights, which discards exactly the tokens partial rollout was enabled
to keep.

`math_async` additionally bounds staleness with `--max-weight-staleness`
(default 2) — miles' own default is *no bound*, which lets an arbitrarily old
group reach the optimizer — and corrects the remaining lag with `--use-tis`.

### Multi-node

Every recipe takes its node count from the allocation, so `-N` is the only thing
that changes:

```bash
sbatch -A $ACC -N 4 experiments/math_sync/qwen3-8b/run.sbatch
```

`run.sbatch` runs one task per node and each `train.sh` sources
`common/ray_cluster.sh`, the one copy of the bring-up: node 0 is the Ray head and
the driver, the rest join it and idle until the driver signals completion
through a flag file under `experiments/outputs/.ray/`. Workers `exit` from
inside that file, so nothing after the `source` line runs on them. The driver
waits for every worker to *register* — reachable is not registered — because
miles sizes its placement groups from the node count.

`train.sh` itself carries no comments: it is the configuration, and the
reasoning behind each setting is here instead.

Placement is derived and then checked before anything expensive starts. The
per-model parallelism is fixed, so **data parallelism is whatever is left over
and grows with the allocation**; the recipe prints it and refuses to start if
`tp × cp × pp` does not divide the training GPUs or if the global batch is not
divisible by the resulting `dp`:

```
placement: 4 node(s) x 8 GPU = 32 training GPUs, tp4 cp2 -> dp4
```

`ACTOR_NUM_NODES`, `ACTOR_GPUS_PER_NODE` and (async) `ROLLOUT_NUM_GPUS` override
the derived split. `math_async` defaults to 4+4 GPUs within one node, and to a
half-and-half split by node from two nodes up.

More nodes raise `dp` and shrink the per-rank batch; they do not by themselves
make a step more informative. Raise `ROLLOUT_BATCH_SIZE` with the allocation.

### Checkpoint layout

Checkpoints are keyed by configuration, not by job id:

```
/ckpt/training/<task>/<dataset>/<model>/<config>
/ckpt/training/math/dapo-math-17k/Qwen3-4B/async-off-1step-rollout-length-24k-lr1e-6
```

so a resumed run finds its own history and two settings never share a `--load`.
`CONFIG_TAG` defaults to the rollout mode, the steps per rollout, the response
length and the learning rate; set it explicitly for anything that default does
not encode — a filtered prompt file, dynamic sampling turned off, a tuning job.
The dump directory follows the same path.

wandb is keyed the same way: the project defaults to `off-policy-<dataset>`
(e.g. `off-policy-dapo-math-17k`) so every run varying the off-policy knobs on
one dataset is directly comparable, with `CONFIG_TAG` separating the settings.

### Eval

`--eval-prompt-data` takes name/path pairs and accepts as many as you give it, so
several benchmarks each report their own series. The default is **AIME-2024 plus
AIME-2025**: same size and difficulty, but 2025 is outside the 2024 contamination
window, so a gap between the two is the cheapest memorisation signal available.

Everything listed must already be in `prompt`/`label` shape, since the global
`--input-key`/`--label-key` apply to eval too. `setup/build_math_jsonl.py`
converts a raw `problem`/`answer` file into that shape; AIME-2025 and MATH-500
were converted with it.

Per-dataset settings (a cheaper `n` for a 500-problem set, a different verifier)
are not expressible as pairs and need `--eval-config`, which **overrides**
`--eval-prompt-data` entirely. `configs/eval_math.yaml` is that config, adding
MATH-500 at `n=4`:

```bash
--export=ALL,EVAL_CONFIG=/root/miles/experiments/configs/eval_math.yaml
```

Eval shares the engines with training rollout, so its cost is real: the config
above is 2960 generations per eval interval.

### Telemetry

`--dump-details` and `--use-miles-dashboard` are **on by default**
(`DUMP_DETAILS=0`, `USE_DASHBOARD=0`, `ROLLOUT_ENTROPY=0` opt out). View a run,
live or finished, from anywhere that can see the directory:

```bash
python -m miles.dashboard.serve --dump-details <ckpt-path>/dump --follow
```

The collector is fire-and-forget and costs a few ms per step. **The dumps are the
part that costs**: one rollout dump plus one train dump *per rank* every rollout
and every eval, `torch.save` inline on the training path, and no retention or
interval knob anywhere. Per-step volume scales with the training world size, not
with `dp`, because TP/CP ranks write duplicate copies that are deduplicated only
at read time. Measure it on a tuning run before committing a production
allocation to it.

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
| Caches (HF, enroot) | `…/cache` | `/root/.cache` |
| miles checkout | `…/src/miles` | `/root/miles` |

The image ships miles at `/root/miles`; the mount shadows it with this checkout,
so edits here take effect without rebuilding.

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
sbatch -A $ACC experiments/math_sync/qwen3-4b/run.sbatch

# 4b. The same task on the fully-async rollout.
sbatch -A $ACC experiments/math_async/qwen3-4b/run.sbatch
```

A fast smoke variant (a few rollouts, short responses). `qwen3-1.7b` is the
cheapest place to check a change before spending a slot on a bigger model:

```bash
sbatch -A $ACC -p interactive --time=01:00:00 \
  --export=ALL,NUM_ROLLOUT=3,ROLLOUT_BATCH_SIZE=8,N_SAMPLES_PER_PROMPT=8,GLOBAL_BATCH_SIZE=64,MAX_RESPONSE_LEN=1024 \
  experiments/math_sync/qwen3-1.7b/run.sbatch
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
