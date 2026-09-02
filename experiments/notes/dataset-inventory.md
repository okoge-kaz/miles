# Dataset inventory and validation state

Training and evaluation data are stored below the host `DATASET_DIR` and are
mounted at `/data` in the training container. Paths in this note are therefore
container paths, which is the form accepted by `--prompt-data` and the offline
evaluation runners.

This inventory deliberately keeps four different claims separate:

- **converted/audited**: row counts and schema checks completed;
- **reward checked**: the current verifier accepted known-good inputs and
  rejected known-bad inputs;
- **RL forward/backward**: the current checked-in recipe completed at least one
  real optimizer update;
- **resume/eval**: a separate current-code resume or held-out evaluation
  completed.

A nonzero pass rate is useful difficulty evidence, but it is not by itself an
RL, resume, or downstream-evaluation proof. Likewise, an old job that imports a
module which has since been removed is historical evidence only.

## Audited artifacts

The following table records the latest full conversion evidence available in
the repository logs as of 2026-08-26.

| Artifact | Container path | Rows | Evidence |
|---|---|---:|---|
| DAPO-Math source used by the SFT filters | `/data/dapo-math-17k/dapo-math-17k.jsonl` | 17,398 | all three policy measurements completed |
| Nemotron Knowledge-MCQA train / validation | `/data/nemotron-rl-mcqa/miles-{train,validation}.jsonl` | 617,020 / 68,553 | conversion job 306571 |
| Nemotron Reasoning Gym train | `/data/nemotron-rl-reasoninggym/miles-train.jsonl` | 15,000 | conversion job 306571 |
| Nemotron structured-output train / validation | `/data/nemotron-rl-ifollow-struct/miles-{train,validation}.jsonl` | 9,437 / 512 | conversion job 306571 |
| Nemotron IFEvalG train | `/data/nemotron-rl-ifollow/miles-train.jsonl` | 46,391 | conversion job 306571; current-input job 306648 |
| Nemotron conversational tool use | `/data/nemotron-rl-conv-tooluse/miles-train.jsonl` | 96,968 | conversion job 306571 |
| Nemotron function-calling pivot | `/data/nemotron-rl-fncall-pivot/miles-train.jsonl` | 9,620 | conversion job 306571 |
| Nemotron SWE pivot | `/data/nemotron-rl-swe-pivot/miles-pivot-train.jsonl` | 50,661 | conversion job 306571; single-action only |
| Nemotron competitive code train / validation | `/data/nemotron-rl-comp-coding/miles-{train,validation}.jsonl` | 23,971 / 322 | jobs 306571 and 306615 |
| Skywork math / code train | `/data/skywork-or1-rl/miles-{math,code}-train.jsonl` | 105,045 / 14,057 | conversion job 306571; ten published math rows had empty labels and were skipped |
| GPQA diamond / main / extended | `/data/gpqa/gpqa-{diamond,main,extended}-miles.jsonl` | 198 / 448 / 546 | conversion job 306588; job 306854 passed five actual-artifact tests, split balance/source-preservation audits, and a scorer correct/wrong probe at revision `565d50c1` |
| LiveCodeBench release v6 | `/data/livecodebench-lite/livecodebench-release-v6-miles-eval.jsonl` | 1,055 | job 306604; evaluation only |
| IFBench test | `/data/ifbench/IFBench_test_miles.jsonl` | 300 | held-out evaluation only |
| MATH-500 | `/data/math-500/math-500.jsonl` | 500 | job 306822: canonical eval-only math rows and source-provenance audit; data/config revision `2c800d2e` |
| AIME24 / AIME25 / AIME26 | `/data/aime-{2024,2025,2026}/aime-{2024,2025,2026}.jsonl` | 30 / 30 / 30 | job 306823: canonical eval-only math rows with source/output SHA-256 provenance; data/config revision `5135c7aa` |
| Calendar train / validation | `/data/nemotron-rl-ifollow-calendar-v2/miles-local-env-{train,validation}.jsonl` | 9,659 / 256 | local conversion/solver job 305108 |

`Idavidrein/gpqa` is gated upstream. The current GPQA job used the complete,
user-owned NeMo Skills copy derived from that repository. A future refresh must
receive `HF_TOKEN` through the submitted job environment after the account has
accepted the terms. No setup script reads `.env`, and a public README-only
download is not considered success.

The static conversion suite passed 199 tests in job 306655. Job 307584 then
audited the actual 83,013-row static Nano file, confirmed the five verifier and
six source counts, resolved all 48 IFEvalG ids, exercised known-good/known-bad
IFEvalG, math, and MCQA probes, and loaded a stratified sample through Miles'
real Dataset/chat-template path. Neither job is a substitute for a GPU optimizer
update, and job 307584 did not execute generated code or JSON-schema probes.

## DAPO-Math difficulty filters

All three requested SFT-model measurements used 17,398 prompts, 16 samples per
prompt, a 16,384-token response cap, a 32,768-token context cap, DeepScaler, and
zero reward on truncation. The filter keeps the inclusive pass-rate interval
0.1--0.9.

| Policy checkpoint | Mean pass rate | Kept rows | Filtered path |
|---|---:|---:|---|
| Qwen3-4B Base step 4000 | 0.5560 | 10,891 | `/data/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl` |
| Qwen3-8B Base step 4000 | 0.6298 | 9,816 | `/data/dapo-math-p10-90-qwen3-8b-base-lr1.5e-5-step4000/dapo-math-p10-90-qwen3-8b-base-lr1.5e-5-step4000.jsonl` |
| Qwen3-30B-A3B Base step 4000 | 0.7470 | 7,425 | `/data/dapo-math-p10-90-qwen3-30b-a3b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-30b-a3b-base-lr2e-5-step4000.jsonl` |

The original 8B pass-rate job left one failed prompt group. Resume job 294347
filled it and completed all 17,398 measurements before the 9,816-row output was
finalized. Filtering is therefore complete for 4B, 8B, and 30B-A3B.

Only the 4B math recipe currently exists under
`experiments/scripts/math/{sync,async}/dapo-math-p10-90/qwen3-4b/`. The completed
8B and 30B-A3B filters do **not** establish that those model sizes have completed
current-code RL forward/backward or resume validation.

## Maintained training mixtures

The benchmark-transfer names describe the training source rather than the held-
out benchmark. Conversion jobs reject rows marked `metadata.eval_only=true`.

| Training file | Composition | Current reward entry point | Current GPU evidence |
|---|---|---|---|
| `/data/nemotron-performance-transfer/nemotron3-nano-competitive-code-train.jsonl` | 38,028 code rows: 23,971 Nemotron + 14,057 Skywork | `experiments.src.reward_sets.code.reward` | jobs 306787/306788 completed the current same-identity fresh+resume gate, restoring iteration 0 and `replay_buffer_0`, advancing optimizer step 1, and publishing iteration 1 plus `replay_buffer_1` |
| `/data/nemotron-performance-transfer/nemotron3-nano-knowledge-mcqa-reasoning-gym-train.jsonl` | 632,020 rows: 617,020 MCQA + 15,000 Reasoning Gym | `experiments.src.reward_sets.stem.reward` | jobs 306790/306792 completed the current same-identity fresh+resume gate, restoring iteration 0 and `replay_buffer_0`, advancing optimizer step 1, and publishing iteration 1 plus `replay_buffer_1` |
| `/data/nemotron-performance-transfer/math-code-stem-balanced-train.jsonl` | 32,673 rows, exactly 10,891 per domain; STEM slice is 10,578 MCQA + 313 Reasoning Gym | `experiments.src.reward_sets.math_code_stem.reward` | jobs 306793/306796 completed the current 4-node, 16K, n=16 fresh/resume gate. The second restored iteration 0 plus 15 pending, 3 ready, and 6 inflight groups and one prepared batch, then published iteration 1 plus `replay_buffer_1` |
| `/data/nemotron-rl-ifollow/miles-train.jsonl` | 46,391 IFEvalG rows | `experiments.src.reward_sets.instruction_following.reward` | jobs 306686/306687 completed a current-code fresh+resume pair with inflight replay; iteration 0 was restored and iteration 1 was trained/saved |
| `/data/nemotron-agentic-conv-tooluse-pivot/nemotron-agentic-conv-tooluse-pivot-train.jsonl` | Pinned NVIDIA conversational tool-use Pivot: 63,559 exact function-call rows after reserving 2,000 held-out calls; 31,409 message actions are excluded | `experiments.src.reward_sets.tool_call_pivot.reward` | Step4000 Qwen3-4B, 16K, n=16, four-node recipe is implemented; current-SFT fresh/resume and held-out GPU evidence remain pending |
| `/data/areal-tau2-data/miles-tau2-rl-train.jsonl` | Pinned AReaL Tau2 RL-only split: 1,982 rows (1,148 airline, 563 retail, 271 telecom) plus nine source DB snapshots | `experiments.src.environments.areal_tau2.generator.generate` | CPU job 331861 passed 328 tests, all-row Task/DB checks, and real mutating event-log/DB-hash restore; six-epoch/RBS-63/n=16 recipe schedules 189 updates and 190,512 trajectories with `inflight` replay, but GPU fresh/resume evidence remains pending |

The pinned Pivot source contains 96,968 rows: 65,559 exact function calls and
31,409 free-form message actions. Preparation writes a deterministic,
zero-overlap 2,000-row held-out file at
`/data/nemotron-agentic-conv-tooluse-pivot/nemotron-agentic-conv-tooluse-pivot-heldout.jsonl`.
Message-action rows are intentionally excluded because the exact-action
verifier cannot establish their semantics.

The maintained production-shaped recipes use four `R9920261300` PBS nodes with
a 24-hour wall clock, 16,384 maximum response tokens, 192 prompts per rollout,
16 samples per prompt, and no in-run evaluation. Resume jobs use the same
deterministic checkpoint identity; see
[checkpoints.md](checkpoints.md). A recipe default or static test is not marked
as resume-verified until a second GPU job actually restores and advances it.

## Evaluation-only data and validated offline runner

LiveCodeBench, GPQA, and IFBench remain outside all training mixtures. The
current offline suite runner is
`experiments/scripts/reasoning_eval/run-suite.sbatch`, implemented by
`experiments/tools/reasoning_eval/suite.py`. Jobs 305175 and 305176 completed
generation and sandboxed/offline scoring:

| Benchmark | Full shape | Job 305176 sample accuracy |
|---|---:|---:|
| LiveCodeBench release v6 | 1,055 prompts x 1 | 0.409478672985782 |
| GPQA Diamond | 198 prompts x 8 | 0.4185606060606061 |
| IFBench | 300 prompts x 8 | 0.20375 |

The artifacts are under
`experiments/outputs/domain_eval/env-{smoke,full}-20260825/`; this is the retained
legacy artifact location from before the runner was consolidated under
`reasoning_eval`.

Job 306776 additionally completed a YAML-entry smoke for AIME24, MATH500, and
GPQA Diamond: it opened each staged dataset, generated and scored two responses,
and wrote summaries plus `_SUCCESS`. This validates only those selected
dataset/reward contracts, not all AIME/GPQA splits or every
`experiments/configs/eval_*.yaml` file. Current reasoning-runner job 306691 also
completed all 30 AIME24 prompts with one repeat and a checksummed artifact. Job
307365 completed the unified current-SFT smoke for all nine tasks: two prompts
each for AIME24/25/26, MATH500, LiveCodeBench, GPQA Diamond/Main/Extended, and
IFBench, at each task's smoke repeat count. These are runner contracts, not full
benchmark estimates; neither job establishes the current 64-repeat AIME default.
See [offline-eval.md](offline-eval.md).

Tau three v1.0.1 has its own held-out evaluator at
`experiments/scripts/tau_bench/evaluate.sbatch`. Preparation validates the
official train/test/base split contract but materializes only 100 test tasks
(retail 40, airline 20, telecom 40). The runner accepts only these `eval_only`
test rows. Official Tau v3 train/base artifacts remain absent. The separate
training recipe uses only the external AReaL Tau2 RL split and its source DB
snapshots; it does not consume the held-out Tau v3 evaluation file.

## Implemented but not admitted end to end

| Task | Current implementation | Missing evidence or work |
|---|---|---|
| Structured output | JSON Schema converter and `reward_sets.structured_output` CPU checks | dedicated GPU forward/backward and resume |
| Calendar | local converter, deterministic solver, and constraint verifier | official-grader parity is not established; GPU RL/replay/resume not run |
| Workplace assistant (single-turn multi-step) | local policy/tool generator and runtime/verifier using pinned resource modules | add a bounded environment service/pool and obtain GPU fresh/resume evidence; it must not be reported as conversational multi-turn RL |
| Search-R1 | consolidated dataset/retrieval environment code; job 307366 generated a two-prompt NQ/HotpotQA smoke and 307427 revalidated its data/services/protocol-matched artifacts | fixed-dataset current GPU training and a policy baseline that actually issues search calls; the smoke scored zero with no searches |
| Full SWE | pinned SWE-ReBench V2 and SWE-Gym payloads, candidate normalization, Harbor/E2B admission and evaluator code | 32,033 ReBench and 2,438 SWE-Gym tasks normalized, but no rows have passed live E2B admission; no 4-node RL or official-comparable downstream result |
| Lean proofs | 1,376,663 rows staged for proof data | pinned Lean/Mathlib execution sandbox, compile reward, GPU/replay validation |
| Identity/adversarial IF/MultiTurnChat | source data staged | pinned judge/GenRM semantics and multi-turn loop where applicable |
| Safety/GenRM/RLHF | source data staged | separate RM/DPO/GenRM pipeline; must not be relabeled as exact-match RLVR |

For the complete repository-level coverage analysis, see
[nemotron-rl-training-data.md](nemotron-rl-training-data.md) and
[nemotron3-nano-super-task-coverage.md](nemotron3-nano-super-task-coverage.md).
