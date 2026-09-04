# Offline evaluation

Maintained RL recipes set `EVAL_INTERVAL=0`; held-out evaluation runs after a
Hugging Face checkpoint has been exported. The runners are consolidated under
`reasoning_eval`, but their benchmark-specific scoring contracts remain
distinct. Evidence for one contract must not be used to mark another as
validated.

## 1. General reasoning suite: MATH, LiveCodeBench, GPQA, and IFBench

The current validated entry point is
`experiments/scripts/reasoning_eval/run-suite.sbatch`. Its implementation is
`experiments/tools/reasoning_eval/suite.py`.

The job performs two phases:

1. validate the source Hugging Face checkpoint, create a job-local unpadded
   vocabulary view if needed, and generate candidates through an eight-GPU vLLM
   server;
2. stop the server, then score candidates offline. AIME/MATH use the local math
   answer verifier, LiveCodeBench scoring runs in an unroutable network namespace
   with Bubblewrap and the pinned evaluator, GPQA uses the converted row label,
   and IFBench uses the pinned constraint scorer.

Candidate writes are resumable through `.partial` files, final outputs are
atomically renamed, and each completed task receives `_SUCCESS`. `EVAL_MODE=smoke`
uses two prompts per task; `EVAL_MODE=full` uses the complete requested split.

Jobs 305175 and 305176 completed the current generate-and-score path:

| Task | Smoke shape | Full shape | Full sample accuracy |
|---|---:|---:|---:|
| LiveCodeBench release v6 | 2 x 1 | 1,055 x 1 | 0.409478672985782 |
| GPQA Diamond | 2 x 8 | 198 x 8 | 0.4185606060606061 |
| IFBench | 2 x 8 | 300 x 8 | 0.20375 |

The retained pre-consolidation summaries are under
`experiments/outputs/domain_eval/env-{smoke,full}-20260825/<task>/summary.json`.
These are checkpoint baselines, not proof that a particular RL run improved the
benchmark. An effectiveness claim requires a pinned pre/post checkpoint pair
evaluated with the same contract.

Job 307365 completed the current unified runner against the user's Qwen3-4B Base
step-4000 SFT checkpoint. It ran all nine smoke tasks, published `_SUCCESS` and
artifact manifests, and produced:

| Task | Smoke shape | Sample accuracy |
|---|---:|---:|
| AIME24 / AIME25 / AIME26 | 2 prompts x 16 | 0.40625 / 0.84375 / 0.84375 |
| MATH500 | 2 x 4 | 1.0 |
| LiveCodeBench v6 | 2 x 1 | 0.0 |
| GPQA Diamond / Main / Extended | 2 x 8 | 0.4375 / 0.5 / 0.375 |
| IFBench | 2 x 8 | 0.0 |

The IFBench smoke had four length-finished empty responses. All values above are
two-prompt plumbing checks, not benchmark estimates.

`experiments/scripts/reasoning_eval/run-suite-after-training.sbatch` selects a complete
post-training export before invoking the same evaluator.
`experiments/scripts/reasoning_eval/score-suite.sbatch` scores already
generated candidates. Neither script runs inside the optimizer loop.

## 2. Reasoning evaluation: AIME24, AIME25, and AIME26

The reportable reasoning path is under
`experiments/scripts/reasoning_eval/`, with helpers in
`experiments/tools/reasoning_eval/`. It is separate from
`experiments/configs/eval_aime.yaml`.

The intended pinned protocol uses:

- a vLLM 0.20.2 SquashFS image to serve a Qwen3-4B Hugging Face checkpoint;
- the NeMo Evaluator/NeMo Skills 26.03 image to prepare and grade AIME24/25/26;
- Qwen3 thinking plus `--reasoning-parser qwen3`;
- a 32,768-token context and at most 28,672 generated tokens;
- temperature 0.6, top-p 0.95, top-k 20;
- one repeat and two prompts in smoke mode, or 64 repeats over all 30 prompts
  per year in the current full default.

The setup entry points are:

```bash
sbatch experiments/scripts/reasoning_eval/import-evaluator-images.sbatch
sbatch experiments/scripts/reasoning_eval/prepare-aime-data.sbatch
```

Image validation job 306707 confirmed the two pinned SquashFS artifacts. The
evaluator validates every indexed checkpoint shard, records a checkpoint
manifest, refuses to reuse a result directory with a changed protocol, and
finalizes each task atomically with an artifact checksum. The unpadding helper
is `experiments/tools/reasoning_eval/unpad_vocab.py`; it builds a job-local view
and never modifies the training checkpoint.

Prepared-data job 306823 completed ten contract tests and then opened the actual
AIME24/25/26 artifacts, confirming 30 canonical eval-only math rows per year and
matching source/output SHA-256 provenance. This admits the data/config revision
`5135c7aa`; it is setup evidence, not a generation or scoring run.

Earlier pre/post jobs 301121 and 301122 completed AIME24/25/26 with eight repeats
per prompt. They are useful historical end-to-end evidence and their logs are
under `experiments/outputs/reasoning_eval/`.

Current-refactor job 306691 subsequently completed AIME24 over all 30 prompts
with one repeat and exited successfully. It exported 30 request/response records,
wrote `_SUCCESS`, and published a checksummed artifact manifest under
`experiments/outputs/reasoning_eval/revalidate-aime1-20260826/aime24/`. This
validates the current runner for that AIME24 single-repeat contract. Job 307365
then exercised AIME24/25/26 at two prompts x 16 repeats. Neither run validates
the 30-prompt x 64-repeat full default. Do not infer that a setup-image job is an
evaluation job.

For a post-training run, use
`experiments/scripts/reasoning_eval/run-after-training.sbatch`. It selects the
newest structurally complete numeric HF export and records the selected path.
For a sweep, use
`experiments/scripts/reasoning_eval/submit-staleness-sweep.sh`; rerunning the
launcher skips completed task suites and resumes unfinished ones.

## 3. Miles-native `eval_*.yaml` configs

Files under `experiments/configs/` describe datasets for Miles' native eval
arguments. They are not the same reportable protocol as the pinned reasoning
runner. `tests/fast/experiments/test_eval_configs.py` proves that
`eval_aime.yaml` and `eval_gpqa.yaml` parse and expose the expected sampling
fields; job 304741 ran those two static tests.

Three later container jobs exercised the prepared data rather than just parsing
YAML. Job 306822 passed four MATH-500 contract tests and audited all 500 actual
eval-only rows plus source provenance (revision `2c800d2e`). Job 306823 passed
ten AIME data/marker tests and audited all three 30-row files with source/output
SHA-256 provenance (revision `5135c7aa`). Job 306854 passed five GPQA tests,
opened the 198/448/546-row actual splits, audited balance/source preservation,
and ran a scorer correct/wrong probe (revision `565d50c1`). None generated model
responses.

Job 306776 added a real YAML-entry smoke for the selected AIME24, MATH500, and
GPQA Diamond contracts. For each task it opened the configured staged data,
generated two responses, invoked the corresponding math or GPQA scorer, wrote
two score records and a summary, and published `_SUCCESS`. The retained results
are under `experiments/outputs/domain_eval/yaml-entry-e2e-20260826/`. This is
end-to-end evidence for those three selected dataset/reward contracts, not for
every dataset listed in their YAML files and not for the unrelated configs.

Current audit status:

| Config family | Static state | End-to-end state |
|---|---|---|
| `eval_gpqa.yaml` | job 306854 passed five tests, audited all 198/448/546-row actual splits, and ran a scorer correct/wrong probe | job 306776 generated and scored two GPQA Diamond prompts; main and extended have not completed this current YAML-entry smoke |
| `eval_aime.yaml` | job 306823 passed ten tests and audited all three canonical 30-row, SHA-256-provenanced eval-only files | job 306776 generated and scored two AIME24 prompts; AIME25/26 have not completed this YAML-entry smoke, and this path is not the pinned NeMo Skills protocol |
| `eval_math500.yaml` | job 306822 passed four tests and audited all 500 canonical eval-only rows plus source provenance | job 306776 generated and scored two prompts successfully; the complete 500-prompt config evaluation has not run |
| `eval_ifbench.yaml`, `eval_livecodebench.yaml` | YAML present; their datasets and dedicated scorers are validated separately | the YAML-native entry itself has not completed a current smoke; use the unified reasoning runner for the recorded evidence |
| Legacy aggregate `eval_math.yaml`, `eval_knowledge.yaml`, `eval_search_r1.yaml`, and `eval_tool.yaml` | files remain in the committed tree | stale paths/settings and removed integrations make them unvalidated; do not advertise or submit them |
| `eval_arc_agi.yaml` | YAML present | its old converter/verifier path was removed; unvalidated and not safe to advertise |

Before committing a config as validated, require a job that opens every
configured path, generates at least one real response, invokes the intended
verifier/generator, and writes a complete result. A YAML parse test alone is
insufficient.

## Search-R1 evaluation

Search-R1 uses `experiments/search_r1/evaluation/run.sbatch`, because each
trajectory may call the pinned retrieval service between policy turns.
Job 307366 generated and scored the two-prompt NQ and HotpotQA smoke against the
user's step-4000 SFT checkpoint. Job 307427 completed a current-code revalidation:
it audited all seven staged eval files, passed retriever and SGLang health/content
probes, recognized both protocol-matched task artifacts as complete, and
republished the aggregate result. Both tasks scored exact match 0;
`search_calls_mean=0` and `searched_frac=0`, so this validates the pipeline but
not effective retrieval use. Full seven-benchmark evaluation and current GPU
RL/resume remain separate gates.

## Tau and static `tool_call_pivot` evaluators

Tau and exact tool action diagnostics have dedicated entry points:

- `experiments/scripts/tau_bench/evaluate.sbatch`
- `experiments/scripts/tool_call_pivot/evaluate.sbatch`

Tau three is the primary downstream benchmark for the conversational tool-use
Pivot training recipe. It accepts only the held-out v1.0.1 test split and uses
either the verified NVIDIA-hosted Gemini user simulator or direct Gemini.
Credentials are allowlisted at the Slurm job boundary; the Python evaluator does
not parse dotenv files. The exact tool-action evaluator remains a training-domain
diagnostic and is not the reported downstream benchmark. Both evaluation paths
still require successful current-checkpoint GPU execution evidence.

## Validation checklist

For a quick evaluation validation, a small smoke is sufficient; it need not
consume the four-hour allocation. Check all of the following:

1. the exact checkpoint and tokenizer pass structural validation;
2. the job opens the real held-out data path;
3. generation returns the expected number of unique `(prompt, repeat)` records;
4. the intended scorer runs, including the sandbox for generated code;
5. summaries and success markers are present and nonempty;
6. W&B is offline or disabled;
7. the full evaluation uses a new, protocol-locked result root rather than
   silently mixing sampling settings.

Slurm `COMPLETED` by itself is not sufficient: inspect result counts and success
markers. Conversely, a smoke score of exactly zero on two prompts can still
validate plumbing; it is not enough to judge model quality.
