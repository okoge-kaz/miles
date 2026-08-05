# SWE-RL: train on SWE-Gym, measure on SWE-bench Verified

A task shaped so that "did RL make the model better?" has an answer you can
defend: train and eval are disjoint by repo, both run through the same
NeMo-Gym grading harness, and the eval set is SWE-bench Verified.

Sandboxes run as Apptainer SIFs on a CPU-only Slurm allocation, so SWE-bench
test execution never competes for CPU with the SGLang rollout workers.

```
GPU allocation                        CPU allocation
┌────────────────────────┐            ┌──────────────────────────────┐
│ miles trainer          │  /run      │ NeMo-Gym mini_swe_agent_2    │
│  Megatron + SGLang     │ ─────────► │   └ Apptainer sandbox / task │
│  session server ◄──────┼────────────┤     (prebuilt .sif)          │
│                        │ policy     │   └ swebench harness → reward│
└────────────────────────┘ requests   └──────────────────────────────┘
```

## Why this split

| | Choice | Reason |
|---|---|---|
| Eval | SWE-bench Verified, `subset=verified` | Official images and official eval specs. The grading gap that breaks SWE-Gym (`KeyError: 'getmoto/moto'` at `make_test_spec`) does not exist on this path. |
| Train | SWE-Gym, `subset=gym`, filtered | Prebuilt SIFs exist (`xingyaoww_*`), and `preflight.py` drops every repo that also appears in Verified, so a gain on Verified is not memorization. |
| Harness | Same for both | Baseline and post-training numbers come from an identical scoring path — no cross-harness comparison. |

Two filters are applied to the training pool and both matter:

1. **Contamination** — any repo present in Verified is removed from training.
   Verified draws from 12 repos; overlap would make the eval meaningless.
2. **Gradability** — instances the `swebench` package cannot build a test spec
   for are removed. They do not crash the run; they score a constant 0, which
   drags every GRPO group toward zero advantage and quietly wastes the whole
   experiment.

## Prerequisites

- Prebuilt SIFs. **Already available — nothing to build** (checked 2026-08-03):

  ```
  /lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench/
  ```

  57,823 images, ~1 GiB each. All **500 SWE-bench Verified** instances are
  present (also as a dedicated `swe-bench-verified-v1/` subdirectory), plus
  SWE-Gym (`xingyaoww_*`), R2E-Gym (`namanjain12_*`) and SWE-rebench.

  The `__` in an instance id is encoded three different ways depending on the
  publisher — `_1776_`, `_s_`, or kept verbatim — so filenames must be probed
  rather than formatted. `preflight.py` handles all three.

  Task JSONL for the training sets: `/lustre/fsw/portfolios/llmservice/users/nliudvig/swe-bench/data/`
  (e.g. `swe-gym-local-2401-localrepos.jsonl`). Their `container_formatter`
  fields are container-relative against a `/swe-bench-images` mount of the
  directory above.
- A policy with a **non-zero** SWE-bench pass rate. This is not optional: the
  miles recipe notes a 4B policy solves none of these tasks, rewards come back
  uniformly 0, and `rollout/zero_std` fires — GRPO then has no gradient signal
  at all. Start from a ~30B-class coding model (GLM-4.7-Flash, which the
  maintained Harbor recipe uses, or Qwen3-Coder-30B-A3B).
- NeMo-Gym from the `mini-swe-agent-per-request-policy-url` branch until
  [NVIDIA-NeMo/Gym#2166](https://github.com/NVIDIA-NeMo/Gym/pull/2166) merges.
- x86_64 CPU nodes. SWE sandbox images are amd64-only; on an arm cluster most
  of the pool is simply not built.

## Steps

### 1. Preflight (no GPU, no server, minutes)

```bash
cd experiments/swe_rl_verified
SIF_POOL=/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench
SWE_DATA=/lustre/fsw/portfolios/llmservice/users/nliudvig/swe-bench/data

python preflight.py \
    --sif-dir     $SIF_POOL \
    --train-input $SWE_DATA/swe-gym-local-2401-localrepos.jsonl \
    --eval-input  princeton-nlp/SWE-bench_Verified \
    --out-dir     ./data
```

Produces `data/train_swegym.jsonl`, `data/eval_verified_{dev,full}.jsonl`
(dev = a deterministic 100-instance slice for per-checkpoint evals),
`data/preflight_report.json`, plus a coverage report per filter stage.

Then normalize the SIF names so one template covers both datasets — the
`container_formatter` this prints is the value for
`configs/apptainer_overrides.yaml`:

```bash
python link_sifs.py --sif-dir $SIF_POOL \
    --link-dir /lustre/fsw/portfolios/coreai/users/$USER/sif_by_id \
    --input data/train_swegym.jsonl --input data/eval_verified_full.jsonl
```

Symlinks, so this costs no disk and leaves the shared pool untouched.

**Stop here if the report says BLOCKED** — a collapsed train pool is cheaper to
fix now than after a GPU allocation.

### 2. Golden scan (CPU only)

Proves the sandbox → image → SWE-bench harness chain end to end with no model
involved. Every gold patch must score 1.0.

```bash
mkdir -p logs
GOLDEN=1 EXP_DIR=$PWD CONTAINER_IMAGE=<your nemo-gym enroot image> \
    sbatch sbatch/gym_server.sbatch

# once the server reports ready:
python ../../examples/experimental/nemo-gym/eval_nemogym_via_api.py \
    --input data/eval_verified_dev.jsonl --golden --limit 5 \
    --nemo-gym-url $(cat .nemo_gym_url)
```

The sbatch deliberately fails fast on the two known Apptainer-in-enroot
breakages before starting the server: a missing `/dev/fuse` (`squashfuse_ll
exited: fuse: device not found`) and SIF mounting, probed with a real image.
`--userns` is used rather than `--fakeroot`, which fails inside enroot with
`no mapping entry found in /etc/subuid`.

Repeat with `data/train_swegym.jsonl` — the training pool needs the same proof.

### 3. Baseline on Verified

Serve the untrained checkpoint with SGLang, then:

```bash
./run_eval.sh baseline http://<sglang-host>:<port>/v1
```

This number is the thing the whole experiment is measured against. Record the
full 500-instance run too (`EVAL_SET=full`) — the 100-instance dev slice has a
±9-point 95% interval, wide enough to hide any realistic single-run gain.

Optionally add a reference point from `inference-api.nvidia.com/v1` (other
teams use it for mini-swe-agent baselines) to sanity-check that the harness
produces sane pass rates before trusting it on your own checkpoint.

### 4. Train

Start the server without `GOLDEN=1`, then launch the trainer with
`NEMO_GYM_URL` pointing at the CPU allocation:

```bash
export NEMO_GYM_URL=$(cat .nemo_gym_url)
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
python ../../examples/experimental/nemo-gym/run.py \
    --prompt-data $PWD/data/train_swegym.jsonl \
    ...
```

Wiring (from the recipe README): `--custom-generate-function-path
miles.rollout.generate_hub.agentic_tool_call.generate`,
`--custom-agent-function-path nemogym_agent_function.run`, `--custom-rm-path
nemogym_generate.reward_func`, `--dynamic-sampling-filter-path
miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted`,
`--use-session-server`, `--session-server-ip 0.0.0.0`.

Watch `rollout/zero_std` from the first step. If it is high, the policy is
solving nothing and no amount of training time will help — change the policy
or the task pool rather than waiting.

### 5. Measure

Evaluate each saved checkpoint on the dev slice, and the baseline plus the
final checkpoint on the full 500:

```bash
./run_eval.sh step0040 http://<sglang-host>:<port>/v1
EVAL_SET=full ./run_eval.sh final http://<sglang-host>:<port>/v1
```

`run_eval.sh` prints pass@1 with a 95% interval and counts harness errors
separately from model failures — an episode that died in the sandbox is not
evidence about the model, and lumping the two together is the easiest way to
report a gain that is not there.

## Sizing and the real bottleneck

Per-sandbox figures NVIDIA aligned on internally for SWE-bench Verified:
**4 vCPU / 8 GB RAM / 50 GiB scratch**. Keep the server's `concurrency` below
`allocated_cpus / 4`.

Moving sandboxes to CPU nodes removes CPU contention but does not make the
run contention-free. Two things remain:

- **Trainer-side CPU** — session server, router, tokenization and the agent
  function's HTTP handling all stay on the GPU nodes.
- **Lustre** — SIF reads are random reads against a filesystem tuned for large
  parallel block reads, on the same mount the trainer writes checkpoints to.
  At high concurrency this is the likely next bottleneck. If sandbox startup
  dominates, stage the SIFs to node-local storage and repoint
  `container_formatter` there.

## Files

| File | Purpose |
| --- | --- |
| `preflight.py` | Contamination / gradability / SIF-coverage gate; emits the datasets. |
| `link_sifs.py` | Symlink farm normalizing SIF names to `{instance_id}.sif`. |
| `configs/apptainer_overrides.yaml` | Apptainer provider settings (`--userns`, `--no-mount cwd`). |
| `sbatch/gym_server.sbatch` | NeMo-Gym server on a CPU allocation; `GOLDEN=1` for the golden scan. |
| `run_eval.sh` | Verified eval of one endpoint, with pass@1 and a confidence interval. |

## Unverified pieces

Flagging these rather than letting them look settled:

- The exact NeMo-Gym config key paths in `configs/apptainer_overrides.yaml`
  follow a working internal config but were not run against the installed
  provider here. A mistyped Hydra key is ignored, not rejected — confirm the
  provider actually picked up `container_formatter` on the first golden scan.
- Repo disjointness between SWE-Gym and Verified is asserted by the dataset's
  construction but is *computed from the data* by `preflight.py` rather than
  assumed.
