# Search-R1 experiments

For a destination-cluster checklist, including the B300/CUDA 13 and Enroot
`.sqsh` qualification procedure, see
[`CLUSTER_MIGRATION.md`](CLUSTER_MIGRATION.md).

The maintained 4B recipe starts from the user's step-4000 SFT checkpoint:

- `search_r1/async/nq-hotpotqa-p10-90/qwen3-4b`: one 8-GPU actor
  node plus one 8-GPU rollout node, with TIS and replay-buffer resume enabled.

It logs to the W&B project `async-search-r1`. Submit it with:

```bash
experiments/submit_training.sh \
  search_r1/async/nq-hotpotqa-p10-90/qwen3-4b search-r1-async \
  --qos=interactive
```

All Search-R1 experiment entrypoints live in this directory; no compatibility
copies are maintained elsewhere.

For an end-to-end training validation on interactive nodes with no W&B network
traffic, keep the production environment/reward path but limit the optimizer
work to one accepted rollout batch:

```bash
experiments/submit_training.sh \
  search_r1/async/nq-hotpotqa-p10-90/qwen3-4b search-r1-offline-smoke \
  --qos=interactive --time=04:00:00 \
  --export=ALL,WANDB_MODE=offline,NUM_ROLLOUT=1,ROLLOUT_BATCH_SIZE=4,N_SAMPLES_PER_PROMPT=2,GLOBAL_BATCH_SIZE=8,SAVE_INTERVAL=1
```

This still requires the fixed p10-90 prompt file; a smoke must not silently
fall back to the raw, policy-dependent training population.

## Held-out evaluation

The evaluator under `evaluation/` reuses the training multi-turn generator,
the local E5/FAISS retriever, and outcome-normalized exact match. It records
exact match together with search calls, turns, truncation, generated tokens,
and loss-masked observation tokens. Evaluation runs against immutable HF
checkpoints outside the optimizer loop and forces W&B offline mode.

Validate NQ and HotpotQA on two prompts each on an interactive node:

```bash
sbatch -A coreai_horizon_dilations -p batch --qos=interactive \
  --export=ALL,WANDB_MODE=offline,EVAL_MODE=smoke \
  experiments/search_r1/evaluation/run.sbatch
```

Set `EVAL_MODE=full` to run NQ, HotpotQA, TriviaQA, PopQA,
2WikiMultiHopQA, MuSiQue, and Bamboogle. For a trained policy, point
`MODEL_PATH` at its exported HF snapshot under `/ckpt/training`. Results and
sampling provenance are written under `experiments/outputs/evaluation/search_r1`
by default;
the top-level `summary.json` contains per-benchmark and macro exact match plus
the interaction-cost metrics. Reusing `RESULT_ROOT` resumes completed prompts;
the provenance sidecar rejects a checkpoint or sampling mismatch.

## Current decision record

| Axis | Maintained default |
|---|---|
| Tracking | W&B project `async-search-r1`; run name is the group |
| Prompt population | One immutable step-4000 SFT-policy p10-90 offline-filtered set |
| Online filtering | None; no dynamic sampling, reward top-up, or abort-only top-up |
| Optimizer | Adam, LR `5e-7`; first sweep `3e-7` / `5e-7` / `1e-6` |
| Batch | 32 prompts, 8 samples per prompt, 256 trajectories per update |
| Budget | 200-update gate, 1,600-update primary run, conditional extension to 3,000 |
| Reward | Outcome normalized EM; format score 0; retrieved tokens loss-masked |
| Async placement | One 8-GPU trainer plus one 8-GPU rollout node, TIS enabled |
| Async log-probs | Fused actor denominator; no reference forward at KL 0; behavior log-probs retained for TIS/alignment |
| Async resume | Replay buffer (type `rollout`) is mandatory |
| Protocol | Existing think/search/information/answer text tags; no tokenizer resize |
| B300 | CUDA 13 image is a compatibility candidate; target-cluster `sm_103` smoke still required |

## Fixed offline difficulty set

Search-R1 does **not** use dynamic sampling or reward-dependent online
filtering. The recipe reads the immutable
`/data/searchr1-nq-hotpotqa/searchr1-nq-hotpotqa-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl`
and does not pass
`--dynamic-sampling-filter-path`. This keeps prompt distribution,
generated trajectories per update, and rollout cost independent of the policy
being compared.

The filter matches the math convention. Run
`Qwen3-4B-Base-LR2e-5-Step4000` eight times per raw prompt using the training
protocol (temperature/top-p 1, 512 tokens per action, three turns, E5 top-k 3,
outcome-only EM), then keep inclusive pass rate 0.1--0.9. With eight samples
this is exactly 1/8--7/8: only all-wrong and all-correct prompts are dropped.
Difficulty is a property of this complete policy/sampling/retrieval tuple, so
changing the model, max turns, retriever, or group size requires a new measured
dataset tag.

Prepare the assets, validate the measurement path on 64 prompts, then run the
resumable full measurement:

```bash
sbatch -A coreai_horizon_dilations experiments/setup/environments/prepare_search_r1.sbatch

sbatch -A coreai_horizon_dilations --export=ALL,LIMIT=64 \
  experiments/tools/difficulty_filter/run_measure_search_r1.sbatch

sbatch -A coreai_horizon_dilations \
  experiments/tools/difficulty_filter/run_measure_search_r1.sbatch
```

The full job writes the cached per-prompt measurement under `/data/difficulty/`
and materializes the fixed JSONL automatically. It is append/resume safe. The
measurement turns off SGLang log-prob collection because filtering consumes
only responses and rewards; training keeps rollout log-probs for TIS. A recipe
fails before allocating Ray if the fixed file is absent.

## Default training budget

The raw local parquet has exactly 169,615 rows. The exact filtered size `N` is
known only after measuring the initial policy; inferring it from the raw size
would silently invent a retention rate. `NUM_ROLLOUT` is the number of
rollout/update batches. At 32 prompts and eight trajectories per prompt:

| `NUM_ROLLOUT` | prompt exposures | trajectories |
|---:|---:|---:|
| 200 | 6,400 | 51,200 |
| 1,600 | 51,200 | 409,600 |
| 3,000 | 96,000 | 768,000 |
| 8,000 | 256,000 | 2,048,000 |

After filtering, one pass through unique rows is `ceil(N / 32)` updates and the
repeat factor at step `S` is `32*S/N`. Obtain the exact reference rather than
using the raw 5,300-step value:

```bash
source experiments/env.sh
N=$(wc -l < "${DATASET_DIR}/searchr1-nq-hotpotqa/searchr1-nq-hotpotqa-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl")
echo "filtered rows=${N}, one-pass updates=$(( (N + 31) / 32 ))"
```

Use 200 updates as an early learning/format gate and 1,600 as the primary run.
Evaluate HF exports throughout, then extend the same checkpoint to 3,000 only
while OOD exact match, entropy, valid-action rate, search-call rate, and abort
rate remain healthy. Treat 8,000 as a published-exposure reference, not an
automatic target: GRPO collapse late in training is reported in the original
and follow-up studies. A dataset pass is bookkeeping, not an early-stop rule;
the studies reuse prompts and are better compared by prompt/trajectory exposure.

This conversion matters because the published batch sizes are prompt batches,
whereas Miles' `GLOBAL_BATCH_SIZE` counts trajectories and enforces
`ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT == GLOBAL_BATCH_SIZE *
NUM_STEPS_PER_ROLLOUT`. The original Search-R1 configuration uses 512 prompts,
five responses per prompt, LR `1e-6`, and 500 steps. A later empirical study
uses 512 prompts, five responses, LR `5e-7`, and up to 600 steps with early
stopping. A Qwen3-4B reproduction uses 256 prompts, eight responses, LR `1e-6`,
KL 0, and 200 steps. Its 51,200 prompt exposures motivate the local 1,600-step
starting budget; the optimizer dynamics are not identical because Miles uses
smaller, more frequent updates.

Recommended first sweep:

- LR: `3e-7`, `5e-7` (default), `1e-6`.
- Batch: keep the validated bring-up shape at 32 prompts / 8 samples / 256
  trajectories, then test 64 / 8 / 512 if retriever and rollout throughput are
  healthy.
- Group size: keep `n=8` for the Qwen3 reproduction arm and add `n=5` for the
  official Search-R1 arm. Do not use `n=1` with Miles GRPO: without a separate
  baseline, within-prompt advantages degenerate.
- Retrieval: keep exact E5 and top-k 3. The Search-R1 ablation found top-k 3
  stronger than 1 or 5.

Sources: [Search-R1 paper](https://arxiv.org/pdf/2503.09516), [official GRPO
recipe](https://github.com/PeterGriffinJin/Search-R1/blob/main/train_grpo.sh),
[follow-up empirical study](https://arxiv.org/pdf/2505.15117), and [Qwen3-4B
reproduction](https://huggingface.co/orbit-ai/searchr1-repro-4b).

## Reward, loss mask, and async log-prob work

The default reward is outcome-only normalized exact match. Format shaping is
implemented but disabled (`SEARCH_FORMAT_SCORE=0`), because Search-R1 reports
that outcome-only reward is sufficient and the follow-up study shows that
format shaping often hurts 3B/7B instruct GRPO models. Intermediate retrieval
reward is intentionally absent; the same follow-up found that it did not help
and could reduce final performance.

Retrieved `<information>...</information>` tokens have loss mask zero. This is
the most strongly validated Search-R1 feature: the paper reports a large drop
without retrieved-token masking. The fixed offline filter is an analysis-control
choice inherited from math, not a claimed Search-R1 accuracy gain. The
asymmetric upper PPO clip likewise remains an ablation rather than a
literature-established Search-R1 gain.

The async recipe enables `--fuse-one-step-actor-logprobs`. It reuses the
gradient-bearing training forward as the actor denominator, eliminating the
standalone actor scoring forward. KL defaults to zero and `--use-kl-loss` is
omitted, so no reference log-prob forward is created. SGLang rollout log-probs
remain enabled: multi-turn search needs exact generated token/log-prob alignment,
and async TIS compares the current policy with the behavior policy.

The following math features are wired but should be enabled only as controlled
Search-R1 experiments:

- M2PO (`IS_CORRECTION=m2po`) as an alternative stale-policy correction.
- OPSM (`USE_OPSM=1`) for token filtering.
- Dual clip (`EPS_CLIP_C`) and staleness-gradient diagnostics.
- Format reward (`SEARCH_FORMAT_SCORE`), preferably only if an observed invalid
  action-rate problem justifies it.

## Tags and tokenizer

The complete protocol already uses eight textual delimiters:
`<think>...</think>`, `<search>...</search>`,
`<information>...</information>`, and `<answer>...</answer>`. That is sufficient
for the environment and matches Search-R1. In the current Qwen3 tokenizer, only
the think pair is an added token; the search, information, and answer strings
tokenize into multiple ordinary tokens. This is expected: SGLang stops on the full strings
`</search>` and `</answer>`, and `no_stop_trim` retains the generated closing
tag in text, token IDs, and log-probs.

The initial 32-trajectory GPU smoke confirms the operational tags but not strict
format adherence: 19 trajectories searched and 20 emitted a closing answer tag,
while none wrapped all free-form reasoning in `<think>...</think>`. The reward
mean was 0.40625, so outcome learning and search execution work; strict format
reward remains zero by default and the missing think wrappers do not block the
environment. Track this separately from answer EM. If full grammar adherence is
a product requirement, run a small `SEARCH_FORMAT_SCORE=0.2` arm or a short
format SFT rather than changing the vocabulary. The follow-up literature makes
this an ablation, not an automatic default, because format shaping sometimes
reduced instruct-model GRPO accuracy.

Do not add new tokenizer special tokens to an existing run. Doing so changes
the vocabulary and embedding shape, requires a fresh HF/Megatron conversion,
and makes trainer/SGLang checkpoints incompatible. Revisit this only as a
cold-start SFT/RL ablation if invalid actions remain high; native Qwen tool-call
tokens would likewise be a different protocol rather than a resume-compatible
fix.

## Cluster portability

The recipes are ready for another Slurm cluster that provides Pyxis/Enroot,
eight GPUs per requested node, a shared filesystem, and routable TCP between
nodes. Override `SHARED_WS`, `WS`, `SQSH_IMAGE`, `SLURM_ACCOUNT_NAME`, and, when
needed, `GPUS_PER_NODE` before submission. Stage assets with
`experiments/setup/download/stage_all.sh` and
`experiments/setup/environments/prepare_search_r1.sbatch`, then materialize the policy-specific
fixed dataset with `experiments/tools/difficulty_filter/run_measure_search_r1.sbatch`.
Copying the resulting JSONL and its pass-rate/meta artifacts is sufficient when
the other cluster uses the same policy, tokenizer, E5 index/corpus, and sampling
settings; otherwise re-measure there.

The minimum Search-R1 assets currently occupy about 77 GB: a 61 GB mmap FAISS
index, 14 GB corpus, 2.1 GB E5 encoder, and the training/eval data. Budget at
least 256 GB host RAM for the index page cache, Python corpus strings, encoder,
and job processes. Bake `faiss-cpu`, `fastapi`, and `uvicorn` into the cluster
image when compute nodes have no package-index egress; the worker installs them
only when imports are missing.

An arbitrary non-Slurm or non-Pyxis cluster is not turnkey yet. It needs a
launcher adapter that preserves `/root/miles`, `/data`, and `/ckpt` mounts, Ray
head/worker discovery, the retriever port, two-node async GPU placement, and
durable shared storage for the replay buffer. Before a long run,
perform a one-update smoke and verify the retriever's real `/retrieve` probe,
valid tagged actions, non-empty search observations, W&B project, checkpoint,
and replay-buffer restore.
