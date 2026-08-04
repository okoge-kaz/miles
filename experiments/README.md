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
  common/run_identity.sh        run name, config tag, checkpoint path
  common/placement.sh           derive + validate the GPU split, before srun
  common/ray_cluster.sh         multi-node ray bring-up
  math_sync/<dataset>/<model>/  single-turn math RL (GRPO, deepscaler reward)
  math_async/<dataset>/<model>/ the same task on the fully-async rollout
  tool_multiturn/<model>/       multi-turn tool-calling RL (ReTool v2 style)
  notes/                        reference notes + notes/agents/ work log
  outputs/                      job logs (git-ignored)
```

A recipe is `<task>/<dataset>/<model>/`, e.g. `math_sync/dapo-math/qwen3-4b/`.
The task directory fixes the RL setup (rollout mode, reward, generate function),
the dataset directory fixes the prompt file and the eval benchmarks, and the
model directory fixes the weights, the parallelism and the batch shape.

Every level is a directory rather than a conditional inside a script: there is
no `if` picking a dataset or an eval set at runtime, so the file that runs is
the file you read.

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

Dynamic sampling and partial rollout are **always on** — not toggles. They
belong together: over-sampling aborts whatever is still generating once enough
groups have passed the filter, and without `--partial-rollout` that work is
discarded rather than resumed.

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
  experiments/math_async/dapo-math/qwen3-8b/run.sbatch
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
/ckpt/training/<task>/<dataset>/<model>/<config>
/ckpt/training/math/dapo-math-17k/Qwen3-4B/async-off-1step-rollout-length-24k-lr1e-6
```

The run name carries the same information —
`math-dapo-math-17k-Qwen3-4B-async-off-1step-rollout-length-24k-lr1e-6` — so the
wandb group, the log and the checkpoint directory all name the same thing.
`common/run_identity.sh` computes all of it once and is sourced by both
`run.sbatch` and `train.sh`, so the two cannot disagree.

A resumed run therefore finds its own history and two settings never share a `--load`.
`CONFIG_TAG` defaults to the rollout mode, the steps per rollout, the response
length and the learning rate; set it explicitly for anything that default does
not encode — a filtered prompt file, dynamic sampling turned off, a tuning job.
The dump directory follows the same path.

wandb is always on and keyed the same way: project `off-policy-<dataset>` (e.g.
`off-policy-dapo-math`), group `RUN_NAME`. `run.sbatch` fails at submit time if
`WANDB_API_KEY` is unset rather than quietly running without logging; `env.sh`
resolves it from `~/.netrc`.

### Eval

`--eval-prompt-data` takes name/path pairs and accepts as many as you give it, so
several benchmarks each report their own series. The `dapo-math` recipes evaluate
**AIME-2024 and AIME-2025**: same size and difficulty, but 2025 is outside the
2024 contamination window, so a gap between the two is the cheapest memorisation
signal available. The pairs are written out in `train.sh` — a different eval set
is a different dataset directory, not a branch.

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
sbatch -A $ACC experiments/math_sync/dapo-math/qwen3-4b/run.sbatch

# 4b. The same task on the fully-async rollout.
sbatch -A $ACC experiments/math_async/dapo-math/qwen3-4b/run.sbatch
```

A fast smoke variant (a few rollouts, short responses). `qwen3-1.7b` is the
cheapest place to check a change before spending a slot on a bigger model:

```bash
sbatch -A $ACC -p interactive --time=01:00:00 \
  --export=ALL,NUM_ROLLOUT=3,ROLLOUT_BATCH_SIZE=8,N_SAMPLES_PER_PROMPT=8,GLOBAL_BATCH_SIZE=64,MAX_RESPONSE_LEN=1024 \
  experiments/math_sync/dapo-math/qwen3-1.7b/run.sbatch
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
