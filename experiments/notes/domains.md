# Domain contracts and current validation state

This note describes how the maintained experiments differ operationally. It is
not a leaderboard table. Reward rates and sequence lengths depend on the exact
policy, prompt file, sampling parameters, and verifier revision; old
Qwen3-4B-Instruct measurements must not be presented as properties of a dataset
or as evidence for the current Qwen3-4B Base recipes.

## Three execution shapes

| Shape | Current tasks | What runs after generation | Replay implication |
|---|---|---|---|
| Static local verifier | Math, MCQA/GPQA, Reasoning Gym, IFEvalG, JSON Schema, exact next tool action, Calendar | deterministic parser/scorer; code additionally launches a local sandbox | inflight replay can be considered after a real fresh/resume proof |
| Stateful local environment | Tau | a policy action changes episode state and produces a loss-masked observation before generation resumes | begin with completed-rollout replay; inflight state requires explicit environment snapshot semantics |
| External execution sandbox/service | full SWE; future Lean/browser/desktop work | repository/compiler/VM state plus a clean terminal grader | replay must pin the image, task, grader, and any restorable environment artifact |

Search-R1 is multi-turn but read-only: an action issues a retrieval query and the
retrieved passages are appended as loss-masked observations. Its infrastructure
and failure modes are still closer to a service-backed environment than to a
static reward.

Conversation history alone does not make a task interactive. Nemotron
conversational tool-use, function-calling pivot, and SWE pivot ask for one expert
next action from a fixed prefix. They do not execute that action and therefore
remain static single-step RLVR.

## Maintained domain implementations

| Domain | Training input and routing | Verification semantics | Current evidence |
|---|---|---|---|
| Math | 4B policy-filtered DAPO; built-in `deepscaler` reward | normalize and compare the final/boxed answer; truncation is forced to reward zero by the recipe | jobs 307062/307063 completed a current 16K-response fresh/resume smoke, restoring iteration 1 plus inflight replay and advancing through iteration 3; the smoke reduced `n` and batch size, while the checked-in production defaults remain n=16. 8B/30B filters are complete, but current RL/resume for those sizes is unverified |
| Code | competitive-code blend; `experiments.src.reward_sets.code.reward` permits only `python_code` | extract Python and require every published stdin/function/harness test to pass in a Bubblewrap filesystem/process sandbox inside an unroutable network namespace | jobs 306787/306788 completed the current same-identity fresh+resume gate: iteration 0 plus `replay_buffer_0` was restored, optimizer step 1 ran, and iteration 1 plus `replay_buffer_1` was published |
| STEM | Knowledge-MCQA + Reasoning Gym; `experiments.src.reward_sets.stem.reward` permits `gpqa`, `mcqa_regex`, and `reasoning_gym` | per-row MCQA regex/letter scoring or the pinned Reasoning Gym task scorer | jobs 306790/306792 completed the current same-identity fresh+resume gate: iteration 0 plus `replay_buffer_0` was restored, optimizer step 1 ran, and iteration 1 plus `replay_buffer_1` was published |
| Math+Code+STEM | balanced JSONL; `experiments.src.reward_sets.math_code_stem.reward` permits only `math`, `python_code`, `mcqa_regex`, and `reasoning_gym` | route each row by `metadata.verifier`, group a reward batch by verifier, score groups concurrently, and restore original order | jobs 306793/306796 completed the current 4-node, 16K, n=16 fresh/resume gate. The resume restored iteration 0 and replay state (15 pending, 3 ready, 6 inflight groups, and one prepared batch), trained step 1, and published iteration 1 plus `replay_buffer_1` |
| IFEvalG | Nemotron instruction following; `experiments.src.reward_sets.instruction_following.reward` | pinned Open-Instruct IFEvalG registry, hidden-thinking removal, mean constraint satisfaction | current jobs 306686/306687 completed a 4-node, 16K, n=16 fresh+resume pair with inflight replay; iteration 0 restored and iteration 1 advanced |
| Exact tool action | balanced function-call-only split; `experiments.src.reward_sets.tool_call.reward` | exactly one expected tool call, exact tool and argument keys, normalized scalar values | jobs 306920/306921 proved 16K, n=16 inflight replay/resume for the verifier, but used the retired Qwen3-4B-Instruct-2507 checkpoint. That recipe is now fail-closed, so the current SFT model still needs a replacement recipe, fresh/resume validation, and held-out evaluation |
| Tau Bench | pinned Tau v1 task identities; `experiments.src.environments.tau_bench.generator.generate` plus `experiments.src.reward_sets.tau.reward` | execute official state transitions, append user/tool observations with loss mask zero, use the terminal environment reward | current-SFT local-policy jobs 307433/307434 completed 16K, n=16 rollout replay/resume: the second restored iteration 1 plus 6 pending, 2 ready, 2 regenerated active groups, and one prepared batch, then saved iteration 2. The replacement SFT recipe is not yet committed, and downstream evaluation still fails, so Tau is not an effectiveness result |
| Calendar | converted expected calendar state; `experiments.src.environments.calendar.verifier.score_calendar_response` | require the complete event set, exact durations, allowed windows/constraints, and global non-overlap | job 305108 solved and locally verified all 9,915 converted rows; official-grader parity and GPU RL are not proved |
| Workplace | `experiments.src.environments.workplace.runtime` and `.verifier` | isolated fixture state, multiple tool calls, terminal state comparison | runtime/verifier correctness tests exist, but no Workplace custom-generate entry point is checked in; GPU training, resume, and production lifecycle are therefore unverified |
| Full SWE | Harbor/E2B candidates and source-specific graders | agent edits in one sandbox; apply the captured patch and run task tests in a separate clean grader sandbox | implementation/contracts exist, but zero rows have passed live E2B admission and no 4-node RL/downstream result exists |

Recipe-specific reward modules reject unexpected verifier ids. The broad
`experiments.src.reward_sets.all_domains` module is for conversion diagnostics,
not a maintained training entry point.

STEM CPU job 306819 completed with exit code 0 after 121 passing tests plus an
official Reasoning Gym correct/wrong probe (`correct=1`, `wrong=0`). It is
supporting verifier evidence; the separate 306790/306792 GPU pair supplies the
current forward/backward and replay-resume evidence.

GPQA CPU job 306854 completed five actual-artifact tests, audited all three
198/448/546-row splits, and ran a scorer correct/wrong probe. This validates the
prepared held-out data/scorer contract, not downstream model effectiveness.

## How the static multi-environment recipe dispatches

`/data/nemotron-performance-transfer/math-code-stem-balanced-train.jsonl`
contains 32,673 rows: 10,891 math, 10,891 code, and 10,891 STEM. The STEM slice
contains 10,578 MCQA and 313 Reasoning Gym rows. Every row carries a
`metadata.verifier` value.

`experiments.src.reward_sets.math_code_stem.reward` calls
`dispatch_restricted_reward`. The dispatcher:

1. reads the verifier id from every generated sample;
2. fails the whole call if any id is missing or outside the four-value allowlist;
3. groups the batch by verifier;
4. evaluates the independent groups concurrently; and
5. returns rewards in the original sample order.

This is a static heterogeneous reward mixture, not a scheduler that launches
three remote environments. Only `python_code` starts subprocesses. Its verifier
owns one semaphore per asyncio event loop (`CODE_EXEC_CONCURRENCY`, default 4),
so scalar reward calls cannot bypass the process cap.

Miles also places an `asyncio.Semaphore` at
`GenerateState.generate_fn_semaphore` in
`miles/rollout/inference_rollout/inference_rollout_common.py`. That semaphore
bounds concurrent custom-generate calls according to SGLang request concurrency;
it is not an independent CPU environment worker pool.

## Tau and the user simulator

The checked-in Tau generator supports two user-simulator backends:

- `TAU_USER_BACKEND=local-policy` uses the same local SGLang checkpoint and
  needs no external API key;
- `TAU_USER_BACKEND=gemini` defaults to `gemini-2.5-flash-lite` and requires
  `GEMINI_API_KEY` to be exported into the submitted job environment.

Secrets must be present at the Slurm job boundary; the Python environment code
does not discover or parse dotenv files. The local-policy backend is sufficient
for cluster bring-up and controlled pre/post comparisons, but it is not directly
comparable with a Tau leaderboard run that uses a different user simulator. The
Gemini path exists in the checked-in generator and evaluator; it must not be
called validated until both an evaluation and an RL optimizer update actually
complete with that backend.

Jobs 307433/307434 used the local-policy backend. Downstream job 307463 attempted
eight episodes and rejected the result because all eight failed before reaching
a terminal state (`mean_reward=0`). This is a fail-closed evaluator result, not a
zero-quality benchmark score. It proves neither Tau effectiveness nor that a
specialized SFT is unnecessary.

IFBench does not require Gemini. Its released 300-prompt test set is generated
with the policy checkpoint and scored offline by the pinned IFBench constraints.
Current full offline evidence is job 305176: 2,400 samples and sample accuracy
0.20375. Current-SFT smoke job 307365 also completed two prompts x eight samples,
with accuracy 0 and four length-finished empty responses. These validate the
held-out evaluator contracts at their respective scopes, not reward improvement
during a training run.

## Offline evaluation versus analysis tools

`experiments/scripts/reasoning_eval/run-suite.sbatch` and
`experiments/tools/reasoning_eval/suite.py` form the generate-then-score runner
for the AIME/MATH diagnostics, LiveCodeBench, GPQA, and IFBench. Job 307365
completed all nine requested current-SFT smoke tasks and published checksummed
artifacts. Generation
completes before code scoring starts; the scoring step uses an unroutable
network namespace and Bubblewrap. It lives beside the stricter NeMo-backed AIME
runner rather than in a second `domain_eval` hierarchy.

`experiments/tools/training_analysis/` summarizes already-written training logs
and rollout dump files. It does not train a model or implement a benchmark; its
output is diagnostic evidence about a run, not an independent effectiveness
measurement. The directory name deliberately describes that responsibility and
replaces the ambiguous former `domain_rl` name.

Search-R1 has its own interactive held-out runner under
`experiments/search_r1/evaluation/`. Job 307366 generated the two-prompt NQ and
HotpotQA smoke; job 307427 audited all seven staged eval sets, revalidated the
retriever/SGLang services, and accepted the protocol-matched completed artifacts.
Both tasks scored exact match 0 and the policy made zero search calls, so this is
environment/evaluator plumbing evidence rather than retrieval effectiveness.

Miles-native files under `experiments/configs/eval_*.yaml` are a separate path.
Job 306776 completed a two-prompt generate-and-score YAML-entry smoke for
AIME24, MATH500, and GPQA Diamond, including `_SUCCESS` markers. This validates
only those selected dataset/reward contracts: the other AIME and GPQA splits and
the unrelated configs remain unverified. Legacy aggregate configs in the tree
must not be treated as validated merely because their YAML parses. See
[offline-eval.md](offline-eval.md) before using one in a report.

## Admission rule

For a domain to be called end-to-end validated on the current revision, require:

1. audited input and current reward/generator correct-vs-wrong probes;
2. a real GPU job with generation, forward, backward, and at least one applied
   optimizer update;
3. a second same-identity job that restores the checkpoint actually saved by
   the first job and advances it (iteration 0 is valid after optimizer step 0);
4. if replay is enabled, explicit replay-artifact restoration and reuse evidence;
5. a held-out offline-evaluation smoke, followed by the intended full benchmark
   for any effectiveness claim.

The four-hour `batch`/`interactive` allocation is a maximum and can be chained.
Validation does not require consuming all four hours; a few real optimizer steps
plus a real resume are sufficient. Do not promote a pending, failed, or old-path
job to “verified.”
