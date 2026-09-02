# Nemotron RL datasets, benchmark guards, and verifier readiness

This inventory covers the original 21 Hugging Face repositories requested for
the Nemotron 3 Nano / RLVR review. Public payloads live under `/data` in the
Miles container. The current
`experiments/setup/manifests/nemotron_rl_datasets.tsv` has 26 data rows: those
21 plus five later repository-SWE sources (R2E-Gym, two SWE-ReBench-V2 sources,
SWE-Gym, and SWE-bench Verified). Their execution status is covered in
`nemotron3-nano-super-task-coverage.md` rather than being folded into the table
below.

The important distinction is:

- **policy RLVR**: a generated response has a deterministic, local reward;
- **environment RL**: an action changes episode state and requires a generation
  loop plus stateful execution. That implementation may be local or service-
  backed; it does not inherently require the NeMo Gym Python package;
- **judge / preference / SFT**: valid training data, but not a deterministic
  policy-RL reward;
- **eval only**: must not be mixed into training without contaminating the
  reported benchmark.

## Download and conversion result

All 21 repositories are staged. `Idavidrein/gpqa` remains gated for a fresh
Hugging Face download, but the same user already owned the complete NeMo Skills
26.03 copy prepared directly from that repository. `prepare_gpqa.sbatch` stages
that existing user-owned exact-source copy under `/data/gpqa/source/` and converts every row;
it does not fetch from an alternate distributor or treat a public README as a
successful gated download.

| Repository / local directory | Published rows used here | Role and status |
|---|---:|---|
| `nvidia/Nemotron-3-Nano-RL-Training-Blend` / `nemotron-3-nano-rl-training-blend` | 93,244 | Split and restored; details below |
| `livecodebench/code_generation_lite` / `livecodebench-lite` | 1,055 | Converted, **eval only** |
| `Idavidrein/gpqa` / `gpqa` | 198 diamond, 448 main, 546 extended | Staged and converted from the existing user-owned NeMo Skills source; **eval only** |
| `BytedTsinghua-SIA/DAPO-Math-17k` / `dapo-math-17k-byted` | 1,791,700 physical Parquet rows | Policy RLVR, math verifier |
| `Skywork/Skywork-OR1-RL-Data` / `skywork-or1-rl` | 105,055 math + 14,057 code | Math and sandboxed-code RLVR; 10 math rows have empty published labels |
| `nvidia/Nemotron-Math-Proofs-v1` / `nemotron-math-proofs-v1` | 1,376,663 | Lean SFT ready; Lean RL needs a compiler sandbox |
| `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` / `nemotron-rl-conv-tooluse-pivot` | 96,968 | Single expert-action RLVR: 65,559 function calls and 31,409 message actions |
| `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1` / `nemotron-rl-swe-pivot` | 50,661 | Single expert-action RLVR; not full SWE execution |
| `nvidia/Nemotron-RL-Identity-Following-v1` / `nemotron-rl-identity-following` | 21,660 | Principle-only rows; GenRM/judge required |
| `nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1` / `nemotron-rl-ifollow-adversarial` | 1,000 | Rubric judge required |
| `nvidia/Nemotron-RL-Instruction-Following-Calendar-v2` / `nemotron-rl-ifollow-calendar-v2` | 9,659 train + 256 validation | Converted for a local deterministic constraint verifier; GPU RL unverified |
| `nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1` / `nemotron-rl-ifollow-multiturnchat` | 2,011 | Rubric judge required |
| `nvidia/Nemotron-RL-ReasoningGym-v1` / `nemotron-rl-reasoninggym` | 15,000 | Official task-specific verifier for all 104 tasks |
| `nvidia/Nemotron-RL-Safety-v1` / `nemotron-rl-safety` | 89,068 | Preference/RM or DPO data, not policy RLVR |
| `nvidia/Nemotron-RL-agent-workplace_assistant` / `nemotron-rl-workplace-assistant` | 1,255 train + 545 validation | Local runtime and final-state verifier exist; a Miles custom-generate loop and production-scale GPU/replay path do not |
| `nvidia/Nemotron-RL-coding-competitive_coding` / `nemotron-rl-comp-coding` | 23,971 selected train + 322 validation | Sandboxed-code RLVR |
| `nvidia/Nemotron-RL-instruction_following-structured_outputs` / `nemotron-rl-ifollow-struct` | 9,437 train + 512 validation | JSON Schema RLVR |
| `nvidia/Nemotron-RL-instruction_following` / `nemotron-rl-ifollow` | 46,391 | Official Open-Instruct IFEvalG RLVR |
| `nvidia/Nemotron-RL-knowledge-mcqa` / `nemotron-rl-mcqa` | 617,020 train + 68,553 validation | Per-row template-regex MCQA RLVR |
| `nvidia/Nemotron-RLHF-GenRM-v1` / `nemotron-rlhf-genrm-v1` | 299,517 | Generative reward-model training, not policy RLVR |
| `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` / `nemotron-rl-fncall-pivot` | 9,620 | Single expert-action RLVR |

Counts are physical JSONL line counts or Parquet metadata from the staged
artifacts, not estimates from directory size.

GPQA outputs are `/data/gpqa/gpqa-{diamond,main,extended}-miles.jsonl`.
Conversion asserts zero skipped rows, exact split sizes, `eval_only=true`,
source-to-output label and option equality, prompt retention, and answer-letter
balance (chi-square 2.04, 0.45, and 1.21 respectively). A new token and accepted
terms are needed only to refresh the upstream HF CSV payload.

CPU container job 306854 subsequently passed five tests, opened all three actual
artifacts (198/448/546 rows), checked balance and source preservation, and ran a
canonical GPQA scorer probe (`correct=1`, `wrong=0`). This validates the prepared
data/scorer contract at revision `565d50c1`; it is not a model evaluation.

The Byted/Tsinghua Parquet above is not the same artifact as the 17,398-prompt
`zhuzilin/dapo-math-17k` used by the policy-specific 4B/8B/30B-A3B difficulty
filters. The shared name is misleading; their paths and row-count assertions are
kept separate.

## Nemotron 3 Nano blend

The published blend masks its two math components behind row references. The
offline restoration command joins those references against the already staged
DAPO and Skywork Parquet files; it does not re-download data at conversion time.

| Output | Rows | Meaning |
|---|---:|---|
| `/data/nemotron-3-nano-rl-training-blend/train-restored.jsonl` | 93,244 | Published ordering with DAPO/Skywork rows restored |
| `/data/nemotron-3-nano-rl-training-blend/miles-train-static.jsonl` | 83,013 | Deterministic local rewards |
| `/data/nemotron-3-nano-rl-training-blend/miles-train-workbench.jsonl` | 10,229 | Environment rows separated from the static blend; runtime/verifier helpers exist, but no checked-in Miles generator or production GPU admission exists |
| `/data/nemotron-3-nano-rl-training-blend/miles-unverifiable.jsonl` | 2 | Referenced Skywork source rows have empty ground truth |

The static file contains:

| Verifier | Rows |
|---|---:|
| `math` | 22,054 |
| `mcqa_regex` | 19,670 |
| `python_code` | 19,169 |
| `ifeval_g` | 16,575 |
| `json_schema` | 5,545 |

The two invalid Skywork labels remain isolated and visible. They are not silently
dropped into a different category and must not be treated as zero-reward prompts.

Job 307584 audited the actual static file and reproduced the 83,013-row total,
all five verifier counts, and all six source counts. It resolved all 48 IFEvalG
ids, passed known-good/known-bad IFEvalG, math, and MCQA probes, and loaded ten
stratified rows through Miles' Dataset/chat-template path. It did not execute a
generated-code or JSON-schema reward probe, so those retain their separate test
evidence.

## Verifier implementations

Recipe-specific modules under `experiments/src/reward_sets` dispatch by
`metadata.verifier` and reject verifier ids outside their audited domain.
Converters preserve the verifier inputs inside each row so the reward is
reproducible after shuffling or mixing datasets. Training recipes select a
restricted `reward_sets.<recipe>` entry point instead of the broad
`reward_sets.all_domains` diagnostic entry point.

| Verifier | Implementation and guard |
|---|---|
| math | Miles' symbolic boxed-answer checker |
| MCQA | The row's own `template_metadata.output_regex`; a single global GPQA parser is incorrect for the 126+ output templates |
| structured output | `jsonschema.validate`, including legacy `definitions` handling |
| instruction following | Pinned Open-Instruct IFEvalG registry; all 48 IDs used by the data must resolve |
| Reasoning Gym | Pinned `reasoning-gym==0.1.25`; task-specific scoring for all 104 task names, including non-unique Rubik/graph/planning answers |
| expert action | Exactly one expected message/tool call with exact argument keys and normalized scalar values |
| competitive code | Bubblewrap filesystem/process isolation plus an outer network namespace; all selected tests must pass |
| LiveCodeBench | Pinned official LiveCodeBench evaluator; explicit `LCB_ALLOW_LOCAL_EXECUTION=1` and `eval_only` guard |
| GPQA | Deterministically shuffled A-D options and Miles' GPQA answer parser; `eval_only` guard in converted metadata |

Skywork's 14,057 code rows contain three published test formats: 9,256
stdin/stdout rows, 2,040 named-function rows, and 2,761 LeetCode-style
`entry_point + import_prefix + test_code` harnesses. The converter and sandbox
support all three; no harness row is relabeled as an exact-text task.

The IFEvalG and Reasoning Gym dependency closures are staged under
`/data/open-instruct-deps` and `/data/reasoning-gym-deps`. They are appended to
`sys.path`, so the CUDA-tested NumPy/SymPy packages in the immutable image remain
authoritative.

Generated code is untrusted. The default code verifier refuses to run when its
sandbox is unavailable. On this cluster the base image uses the read-only host
binary mount:

```bash
--bind="${CONTAINER_MOUNTS},/usr/bin/bwrap:/usr/local/bin/bwrap"
```

`CODE_EXEC_SANDBOX=process` is only an explicit escape hatch for an already
isolated disposable worker; no preparation or training script enables it.

## What is not a static policy-RL verifier

- **Workplace:** `experiments.src.environments.workplace.runtime` and `.verifier`
  load pinned standalone resource modules and use the published final-state
  comparator without importing or serving NeMo Gym. No checked-in custom-
  generate loop currently connects those helpers to Miles, so multi-turn GPU RL
  has not run. Implement the loop behind a bounded worker/service pool, then run
  fresh/resume/replay validation. Grading only the last call would be wrong.
- **Calendar:** `experiments.src.environments.calendar.verifier` locally checks
  a complete JSON schedule against the published expected state. It is a static
  response verifier, not a live tool loop, and exact-text grading is not used.
  Job 305108 solved all 9,915 converted rows locally. This does not establish
  parity with an official external grader or a GPU RL path.
- **Identity, adversarial IF, MultiTurnChat:** their supervision is a principle or
  rubric, so a pinned judge/GenRM service and judge-version telemetry are needed.
- **Safety:** pairwise preference rows belong in RM/DPO training. Turning the
  preference into a policy-RL exact-match label changes the task.
- **GenRM:** trains a generative reward model. It can later supply judge rewards,
  but is not itself a prompt/answer RLVR set.
- **Lean proofs:** the `messages` field is directly usable for SFT. RL requires a
  Lean 4 + Mathlib image, per-sample project state, compilation timeout, and
  compiler-result reward.
- **Full SWE:** SWE-Pivot verifies the next expert action. Full patch-level SWE
  still needs a repository container, tool loop, and test harness such as
  OpenHands/SWE-bench.

## Benchmark contamination policy

LiveCodeBench release v6 and every GPQA split are evaluation-only. To improve
LiveCodeBench, train on the staged competitive-code/Skywork-code sets and evaluate
on LCB. To improve GPQA, train on Knowledge-MCQA/ReasoningGym and evaluate on
GPQA. Do not train on either benchmark's hidden answers or tests.

`prepare_performance_transfer_blends.sbatch` materializes those two safe training
mixtures as the single JSONL path accepted by Miles:

| Training file | Rows and verifiers | Held-out evaluation |
|---|---|---|
| `/data/nemotron-performance-transfer/nemotron3-nano-competitive-code-train.jsonl` | 38,028 `python_code` rows (23,971 Nemotron 3 Nano competitive-code + 14,057 Skywork code) | LiveCodeBench v6 |
| `/data/nemotron-performance-transfer/nemotron3-nano-knowledge-mcqa-reasoning-gym-train.jsonl` | 632,020 rows (617,020 Knowledge-MCQA `mcqa_regex` + 15,000 `reasoning_gym`) | GPQA diamond/main/extended |

The streaming merger rejects any row with `metadata.eval_only=true`, validates
the expected row and verifier counts, and atomically replaces its output only
after a complete successful pass. The benchmark files are not merger inputs.

## Miles launch contract for a converted training file

At minimum, a recipe-specific file uses:

```bash
--prompt-data /data/nemotron-3-nano-rl-training-blend/miles-train-static.jsonl \
--input-key prompt \
--label-key label \
--tool-key tools \
--apply-chat-template \
--custom-rm-path experiments.src.reward_sets.<recipe>.reward
```

The current recipe modules are `code`, `stem`, `math_code_stem`,
`instruction_following`, and `tool_call_pivot`. Each rejects verifier ids that are
outside its audited dataset contract. The broad
`experiments.src.reward_sets.all_domains.blend_reward` remains for converter
smoke tests and diagnostics, not as a new training-job entry point. Code-bearing
mixtures also need the Bubblewrap mount shown above. Workplace rows remain in a
separate file: runtime/verifier helpers exist, but a custom loop must be
implemented and bounded before fresh/resume GPU checks or admission to a
production blend.

### Current GPU admission, not inferred from conversion

As of 2026-08-26, Code jobs 306787/306788 completed the current same-identity
fresh+resume gate: the second job restored iteration 0 plus `replay_buffer_0`,
trained step 1, and published iteration 1 plus `replay_buffer_1`. IFEvalG jobs
306686/306687 completed a current 4-node, 16K, n=16 fresh+resume pair with
inflight replay: the first job saved iteration 0 and `replay_buffer_0`, and the
second restored iteration 0, trained step 1, and saved iteration 1 plus
`replay_buffer_1`.

STEM jobs 306790/306792 completed the current same-identity gate: the second
restored iteration 0 plus `replay_buffer_0`, trained step 1, and published
iteration 1 plus `replay_buffer_1`. Math+Code+STEM jobs 306793/306796 also passed:
the resume restored iteration 0 and 15 pending, 3 ready, and 6 inflight groups
plus one prepared batch, then published iteration 1 plus `replay_buffer_1`.
Exact-tool jobs 306920/306921 passed replay/resume only on the retired
Qwen3-4B-Instruct-2507 checkpoint. A Step4000 Qwen3-4B replacement recipe now
uses the pinned Pivot source, but its fresh/resume and held-out GPU checks are
still pending. Tau v3 remains evaluation-only. The maintained stateful training
path uses the external AReaL Tau2 RL split with multi-turn `inflight` event-log
replay; it restores active episodes at an agent boundary and still needs a
current fresh/resume GPU proof. Older Tau jobs remain historical checkpoint-
mechanism evidence outside that recipe.
Structured output, Calendar, and Workplace have CPU verifier evidence but no
dedicated current GPU forward/backward+resume pair. Conversion success must not
be reported as this missing evidence. `BrokenPipeError` lines at the end of the
IFEvalG jobs occurred during Ray shutdown after successful checkpoint/replay
publication and job completion; they are teardown noise, not a training failure.
The same interpretation applies to Code 306787/306788: both jobs exited 0 after
their durable artifacts were published.

## Reproduction and queues

| Work | PBS queue | Typical walltime |
|---|---|---:|
| Hugging Face / Git transfer | `R9920261300` (CPU-only) | `24:00:00` |
| Full checkpoint/data conversion | `R9920261300` | `08:00:00` |
| CPU schema/unit/integration validation | `R9920261300` (CPU-only) | minutes to one hour |
| GPU bring-up validation | `R9920261300` | task-specific short run |
| Maintained GPU training and resume jobs | `R9920261300` | `24:00:00` |

The maintained recipes request 24 hours and preserve a stable run identity
across an intentional resume. A forward/backward validation may exit after a few
real optimizer steps; it does not need to consume the allocation.

Entry points:

- `experiments/setup/download/stage_nemotron_rl_datasets.sh`
- `experiments/setup/datasets/prepare_nemotron_nano.sbatch`
- `experiments/setup/datasets/prepare_nemotron_static_components.sbatch`
- `experiments/setup/datasets/prepare_livecodebench.sbatch`
- `experiments/setup/datasets/prepare_gpqa.sbatch`
- `experiments/setup/datasets/prepare_performance_transfer_blends.sbatch`
- `experiments/setup/environments/prepare_ifeval_dependencies.sbatch`
- `experiments/setup/environments/prepare_reasoning_gym_dependencies.sbatch`
- `tests/slurm/test_nemotron_training_input.sbatch`
