# Dataset inventory — what is staged, by genre

Everything under `/lustre/fsw/portfolios/coreai/users/kfujii/datasets`, mounted at
`/data` in the container. Paths below are the in-container ones, which is what a
recipe's `--prompt-data` and an eval config's `path:` take.

Companion documents:

- [datasets.md](datasets.md) — how miles reads a JSONL, how to inspect one
- [domains.md](domains.md) — measured sequence lengths, reward density, turn structure
- `src/difficulty_filter/sweeps.sh` — the (prompt file, verifier) table, executable

**Status column.** The distinction that matters is between "converted" and
"verified". A wrong verifier does not crash: it returns 0.0 on every row, and a
dataset that looks uniformly impossible is indistinguishable from one that is.
So nothing counts as ready until a smoke run has produced a pass rate that is
neither 0.000 nor 1.000.

| | meaning |
|---|---|
| ✅ | smoke run produced a sane pass rate |
| ⏳ | staged and converted; smoke queued, not yet run |
| ⛔ | blocked, see the note |

Counts are `wc -l` as of 2026-08-05.

---

## 1. Math

The reference workload, and the only genre with a difficulty-filtered training
set in use.

| Role | Path | Rows | Verifier | Status |
|---|---|---|---|---|
| train | `/data/dapo-math-p10-80/dapo-math-p10-80.jsonl` | 3,962 | `math` | ✅ |
| train (source) | `/data/dapo-math-17k/dapo-math-17k.jsonl` | 17,398 | `math` | ✅ |
| train | `/data/skywork-or1-rl/skywork-or1-math-miles-20k.jsonl` | 20,060 | `math` | ⏳ |
| train | `/data/nemotron-rl-math-v2/nemotron-rl-math-v2-miles.jsonl` | 7,732 | `math` | ⏳ |
| eval | `/data/aime-2024/aime-2024.jsonl` | 30 | `math` | ✅ |
| eval | `/data/aime-2025/aime-2025.jsonl` | 30 | `math` | ✅ |
| eval | `/data/aime-2026/aime-2026.jsonl` | 30 | `math` | ✅ |
| eval | `/data/math-500/math-500.jsonl` | 500 | `math` | ✅ |
| eval (spare) | `/data/aime-2023/aime-2023.jsonl`, `/data/amc-2023/` | 30, parquet | `math` | not in a config |

Eval config: `configs/eval_math.yaml` (aime24/25/26 + math500).
Recipes: `math_sync/dapo-math-p10-90/qwen3-4b-instruct-2507/`, `math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/` (the only surviving pair; the other four models were deleted unrun on 2026-08-05)
for qwen3-1.7b / 4b / 4b-instruct-2507 / 8b / 30b-a3b.

**`--rm-type math`, not `deepscaler`.** `deepscaler.py:36-44` gates on the
response containing `</think>`, so it returns 0 for every non-thinking model —
Qwen3-4B-Instruct-2507 answering `\boxed{37}` to the label `37` scored 0.

**DAPO-Math is saturated for this policy**: mean pass rate 0.770, 56.4% of prompts
solved 8/8. GRPO's advantage is proportional to the group std `sqrt(p(1-p))`, so
those rows contribute no gradient. `dapo-math-p10-80` is the 3,962-row window that
survives the cut, and is what both recipe families train on.

## 2. Knowledge / MCQA

| Role | Path | Rows | Verifier | Status |
|---|---|---|---|---|
| train | `/data/nemotron-rl-mcqa/knowledge-mcqa-miles-20k.jsonl` | 19,789 | `gpqa` | ✅ |
| eval | `/data/mmlu-pro/mmlu-pro-miles-2k.jsonl` | 2,012 | `gpqa` | ⏳ |
| eval | `/data/gpqa/gpqa-diamond-miles.jsonl` | 198 | `gpqa` | ⏳ |
| eval (spare) | `/data/gpqa/gpqa-{main,extended}-miles.jsonl` | 448, 546 | `gpqa` | not in a config |

Eval config: `configs/eval_knowledge.yaml`. GPQA-Diamond runs at `n=8` — 198
questions are noisy at the default 4.

`--rm-type gpqa` reads `metadata.valid_letters` per row, because the option count
varies: MMLU-Pro has up to ten (chance floor 0.1), GPQA four (0.25).

**GPQA is gated.** It needs `HF_TOKEN` in `.env` *and* the terms accepted by that
same account, then `setup/prepare_gpqa.sbatch`. The gate fails silently — `hf
download` exits 0 having fetched only the public README — which is why both that
script and `setup/download_dataset.sbatch` check for data files rather than the
exit status.

The GPQA converter shuffles the four options with a seed derived from the question
digest, so the ordering is deterministic but the answer letter is uniform
(chi-square 1.8 on diamond, df=3). Ordering by anything stable puts the correct
answer on A far too often and hands a model free accuracy for always answering A.
`prepare_gpqa.sbatch` fails rather than warns if that regresses.

Knowledge-MCQA is the **best-conditioned training set staged**: mean pass rate
0.498, and 43.3% of rows land in the 0.2–0.8 window against 19.3% for DAPO-Math.

## 3. Abstract reasoning

| Role | Path | Rows | Verifier | Status |
|---|---|---|---|---|
| train | `/data/nemotron-rl-arc-agi/arc-agi-train-miles.jsonl` | 10,000 | `grid_and_ast.arc_agi_reward` | ⏳ |
| eval | `/data/nemotron-rl-arc-agi/arc-agi-val-miles.jsonl` | 514 | same | ⏳ |
| train | `/data/nemotron-rl-reasoninggym/reasoning-gym-miles.jsonl` | 14,259 | `rewards.reasoning_gym_reward` | ✅ |

Eval config: `configs/eval_arc_agi.yaml`.

ARC answers are exact — one wrong cell is a wrong answer — so `arc_agi_reward` has
no tolerance anywhere. It accepts the `\boxed{...}` grid the prompt asks for and
falls back to the last balanced `[[...]]` span for a model that drops the wrapper.

**ARC-AGI ships its own difficulty** (a continuous score plus easy/hard buckets in
`metadata`), so it is the one training set where a difficulty window can be cut
without measuring one. Its sweep is for verifier confidence, not for the filter.

reasoning-gym covers 104 procedurally generated environments; its sweep is partial
(4,749 of 14,259, mean 0.495).

## 4. Code

| Role | Path | Rows | Verifier | Status |
|---|---|---|---|---|
| train | `/data/nemotron-rl-comp-coding/competitive-coding-miles.jsonl` | 16,083 | `code_exec.code_exec_reward` | ⏳ |

No eval config yet — LiveCodeBench is staged raw (`/data/livecodebench-lite`) but
not converted.

`code_exec.py` runs the generated program in a subprocess with a wall-clock
timeout and an `RLIMIT_AS` cap, the same weak-isolation pattern
`examples/retool_v2/tool_sandbox.py` already uses. **That is not a security
boundary and is not claimed to be one** — it is enough to grade code the policy
wrote against tests we supply, on a node we already trust. The child gets a
stripped environment rather than an inherited one, so a generated program cannot
read cluster credentials.

Two problem shapes, both handled: `fn_name` present means the input is a literal
argument list and the solution defines a class method or a bare function;
otherwise input goes to stdin and stdout is compared. Scoring is all-or-nothing
across the test list, since partial credit rewards special-casing the samples.

## 5. Tool use — single step

One assistant turn compared against an expert action. No environment, no state.

| Role | Path | Rows | Verifier | Status |
|---|---|---|---|---|
| train | `/data/nemotron-rl-fncall-pivot/fncall-pivot-miles.jsonl` | 9,620 | `rewards.tool_call_match_reward` | ✅ |
| train | `/data/nemotron-rl-conv-tooluse/conv-tooluse-miles-20k.jsonl` | 20,065 | same | ✅ |
| train | `/data/nemotron-rl-swe-pivot/swe-pivot-miles-20k.jsonl` | 19,667 | same | ⏳ |
| eval | `/data/bfcl/bfcl-ast-miles.jsonl` | 3,641 | `grid_and_ast.bfcl_ast_reward` | ⏳ |

Eval config: `configs/eval_tool.yaml`, with `tool_key: tools`.

**Tools are per row.** BFCL ships a different function set per question, so a
global tool list cannot describe it. miles reads them via `--tool-key`
(`data.py:211-217`); BFCL keeps them at the top level, the Nemotron converters put
them under `metadata`, and both the eval path and the measurement driver accept
either. Getting this wrong does not error — the prompt renders with no functions
to call and every row scores 0.

BFCL is restricted to the 13 **AST** categories. `exec_*` calls live APIs and
`multi_turn_*` needs a stateful environment; including ungradable rows would push
the score down by the fraction the harness cannot judge rather than by anything
the policy did. The `irrelevance` categories are kept — an empty ground truth
means "call nothing", and spurious tool use is a real failure mode.

**SWE-Pivot needed no new verifier.** NeMo-RL's `stage2_swe1.yaml` grades it with
`swe_pivot_single_step_tool_use_with_argument_comparison`, which is what
`tool_call_match_reward` already does — a single step against an expert call, no
container, no repo checkout.

conv-tooluse is the **hardest staged set**: mean 0.303, 63.0% of rows scoring 0/8,
only 7.4% inside the 0.2–0.8 window.

## 6. Multi-turn — a non-LLM response arrives, then decoding resumes

The genre with genuinely different rollout shape: an observation is appended with
`loss_mask=0` (`tool_call_utils.py:58-64`) and generation continues.

| Role | Path | Size | Verifier | Status |
|---|---|---|---|---|
| source train | `/data/searchr1-nq-hotpotqa/train.parquet` | 169,615 rows / 340 MB | `--rm-type search_r1` | ✅ staged; unfiltered bring-up smoke only |
| fixed train | `/data/searchr1-nq-hotpotqa/searchr1-nq-hotpotqa-p10-90.jsonl` | determined by offline Qwen3-4B-Instruct-2507 n=8 pass-rate measurement | same | ⏳ measurement required before controlled sync/async runs |
| eval | `/data/flashrag-datasets/{nq,triviaqa,popqa}/test.jsonl` | 3,610 / 11,313 / 14,267 | same | ⛔ |
| eval | `/data/flashrag-datasets/{hotpotqa,2wikimultihopqa,musique}/dev.jsonl` | 7,405 / 12,576 / 2,417 | same | ⛔ |
| eval | `/data/flashrag-datasets/bamboogle/test.jsonl` | 125 | same | ⛔ |
| train | `/data/tau-bench/tau1_train.jsonl` | 500 | environment-internal | ⛔ |
| eval | `/data/tau-bench/tau1_retail_test.jsonl` | 115 | environment-internal | ⛔ |
| eval | `/data/tau-bench/tau1_airline_test.jsonl` | 50 | environment-internal | ⛔ |

Eval config: `configs/eval_search_r1.yaml` (the seven FlashRAG sets Search-R1
reports, sliced to 500 with the `path@[0:500]` syntax so the numbers stay
comparable to published ones).
Recipes: `search_r1_sync/nq-hotpotqa-p10-90/qwen3-4b-instruct-2507/`,
`search_r1_async/nq-hotpotqa-p10-90/qwen3-4b-instruct-2507/`, and
`tau_bench/tau1/qwen3-4b-instruct-2507/`.

The colocated Search-R1 path completed an unfiltered one-rollout/one-update GPU
bring-up smoke (job 15729407) with a checkpoint; it is not evidence for the
final fixed-dataset design. The old dynamic-filter async smoke (job 15789560)
was cancelled before allocation. The fully-async recipe, fused actor-logprob
path, and replay sidecar pass unit/static validation; its final GPU smoke waits
for the fixed p10-90 artifact. Difficulty-pipeline smoke outputs are isolated
under `/data/difficulty/smoke/` and are never training inputs. The assets are
staged:

- `/data/search-r1/e5_Flat.index` — 64 GB, reassembled from `part_aa`+`part_ab`
- `/data/search-r1/wiki-18.jsonl` — 14 GB corpus
- `/ckpt/hf/e5-base-v2` — the query encoder

`wiki-18.jsonl.gz` is a gzipped **tar** despite the name, so `gunzip -c` yields tar
framing and the retriever dies on a stray `0x80`. `prepare_search_r1.sbatch` asks
`tar -tzf` directly rather than sniffing the first 512 bytes — the ustar magic
sits at offset 257 and a short fragment does not see it reliably.

The retriever (`src/search_r1/retrieval_server.py`) mmaps the index and refuses to
start if the corpus and index counts disagree. The worker probes the real
`/retrieve` endpoint before rollout; later request failures abort and filter the
affected trajectory rather than training on an empty observation.

**tau-bench needs an external LLM** for its user simulator: `TAU_USER_MODEL` and
`NVIDIA_INFERENCE_API_KEY` in `.env`, no default (see `.env.example`). Its recipe
takes `--input-key index` and no `--custom-rm-path` — `generate_with_tau.py:140`
does `int(sample.prompt)` and the environment scores the trajectory itself.

## 7. Instruction following / structured output

| Role | Path | Rows | Verifier | Status |
|---|---|---|---|---|
| train | `/data/nemotron-rl-ifollow/instruction-following-miles-20k.jsonl` | 20,096 | `ifeval_g.ifeval_reward` | ⏳ |
| train | `/data/nemotron-rl-ifollow-struct/structured-outputs-miles.jsonl` | 9,437 | `rewards.structured_output_reward` | ⏳ |

No eval config yet.

**Not `--rm-type ifbench`.** IFBench's registry has 58 instruction ids and this
dataset uses 48, with *zero* overlap — the built-in verifier graded 0/20,096.
`ifeval_g.py` loads `open_instruct.IFEvalG.instructions_registry` instead, which
covers all 48. Scoring is strict AND across a row's constraints; an empty response
is 0.0.

Requires `nltk langdetect immutabledict absl-py`, installed per job
(`EXTRA_PIP` in `sweeps.sh`). That knob is `+`-separated, not comma —
`sbatch --export` splits on commas, so a comma list silently delivers only the
first package.

## 8. SWE — full agentic

Staged raw, **no converter and no recipe**. Listed so the gap is explicit.

| Path | Notes |
|---|---|
| `/data/swe-bench-verified`, `-lite`, `-full` | eval splits |
| `/data/swe-gym`, `/data/swe-rebench`, `/data/swe-smith`, `/data/r2e-gym-v1`, `/data/swe-bench-extra` | training instances |

These need a container per instance and a repo checkout, which is the whole reason
SWE-Pivot (§5) is attractive: it captures the tool-choice signal with none of the
sandbox machinery. See `swe-bench-sif-image-pool` and
`swe-gym-not-gradable-by-swebench-package` for what has already been built here.

---

## Measured pass rates

`Qwen3-4B-Instruct-2507`, 8 samples per prompt, temperature 1.0.
Files: `/data/difficulty/<name>.<policy>.passrate.jsonl` (+ a `.meta.json` sidecar
and a `.audit.jsonl` of full responses). Keyed by policy, so several models
accumulate side by side.

| Dataset | n | mean | 0/8 | 8/8 | in 0.2–0.8 |
|---|---|---|---|---|---|
| conv-tooluse | 20,065 | 0.303 | 63.0% | 23.8% | 7.4% |
| knowledge-mcqa | 19,789 | 0.498 | 19.1% | 17.1% | **43.3%** |
| reasoning-gym | 4,749 *(partial)* | 0.495 | 35.9% | 31.5% | 20.6% |
| fncall-pivot | 7,374 *(partial)* | 0.590 | 37.0% | 54.9% | 4.5% |
| dapo-math-17k | 17,398 | 0.770 | 9.9% | 56.4% | 19.3% |

The last column is the one to read: it is the fraction of prompts that produce a
non-zero GRPO advantage. fncall-pivot at 4.5% means ~95% of a rollout batch is
wasted work unless it is filtered first.

## What is not done

- **8 of 14 sweeps are unrun or partial.** Queued on `interactive`; that partition
  allows 2 nodes per user, so they serialise at up to 4 h each.
- **Recipes exist for 4 of the genres.** math (×10), search_r1, tau_bench,
  tool_multiturn. Everything in §2–§5 and §7 has data and a verified-or-queued
  verifier but no `run.sbatch`/`train.sh`.
- **Search-R1 and tau-bench have never completed a rollout.**
- **No eval config** for code (§4) or instruction following (§7).
- **SWE (§8) has no converter.**
