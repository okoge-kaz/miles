# Nemotron 3 Nano / Super task coverage

This note distinguishes staged data, a local reward implementation, and an
end-to-end RL run. A task is not called supported merely because its JSONL can
be parsed. Stateful environments and judge-based rewards must preserve their
published semantics.

Primary dataset cards:

- [Nemotron 3 Nano RL Training Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend)
- [Nemotron RL Super Training Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends)
- [Nemotron RL Ultra Training Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends)
- [Nemotron RL Lightning Training Blend](https://huggingface.co/datasets/nvidia/Nemotron-RL-Lightning-Training-Blend)
- [IFBench](https://github.com/allenai/IFBench)
- [Tau Bench](https://github.com/sierra-research/tau-bench)
- [Tau three](https://github.com/sierra-research/tau2-bench)

The Nano card publishes 93,244 rows from seven components. The Super card
publishes six curriculum files: `rlvr1` (138,712), `rlvr2` (156,278), `rlvr3`
(107,037), `swe1` (50,661), `swe2` (1,444), and `rlhf` (25,171), for 479,303
rows total. Super also states that additional unreleased data was used, so the
public blend cannot reproduce the complete NVIDIA recipe by itself.

## Nano coverage

| Component | Local state | Remaining work |
|---|---|---|
| DAPO math | The 4B/8B/30B-A3B difficulty filters completed with 10,891/9,816/7,425 rows; the maintained RL recipe is 4B only | Add current 8B and 30B-A3B forward/backward+resume recipes/jobs before claiming all model sizes train |
| Skywork math/code | Converted with all three published code harness formats; Code jobs 306787/306788 completed a current same-identity fresh+resume replay gate through iteration 1 | Keep code in Bubblewrap and network namespaces; downstream LiveCodeBench improvement remains unverified |
| Knowledge MCQA | Converted with each row's own answer regex; STEM jobs 306790/306792 completed a current same-identity fresh+resume replay gate through iteration 1 | Current replay/resume gate passed; downstream GPQA improvement remains unverified |
| Competitive code | Sandboxed reward; Code jobs 306787/306788 restored iteration 0 plus `replay_buffer_0`, trained step 1, and published iteration 1 plus `replay_buffer_1` | Current replay/resume gate passed; downstream effectiveness remains unverified |
| IFollow / IFEvalG | All 48 registry IDs and correct/wrong probes pass; jobs 306686/306687 completed a current 4-node, 16K, n=16 inflight replay fresh+resume pair and advanced iteration 0 to 1 | Current replay gate passed; held-out IFBench pre/post effectiveness remains a separate requirement |
| Structured outputs | JSON Schema reward verified on CPU | Add a dedicated GPU smoke before describing this component as end-to-end verified |
| Workplace assistant | Standalone runtime and final-state verifier implemented without importing or serving NeMo Gym; current source release converted to 1,255 train + 545 validation rows; resource/tool correctness tests pass | Add a checked-in Miles custom-generate loop behind a bounded environment worker/service pool, then run fresh and replay GPU validation |

Nano's static components now have implementation paths, but this is not the same
as end-to-end admission. Structured output still lacks a dedicated GPU run, and
Workplace has no checked-in custom-generate loop and is not admitted for
production-scale or replay training. The standalone Workplace runtime imports
only pinned resource modules and fixtures from the NeMo
Gym checkout at commit `48d5b9c01e3fc59a49f19674d0034a6f06396074`; it has no
NeMo Gym package or server dependency. Two Skywork source rows have empty
published ground truth and remain quarantined rather than being assigned reward
zero.

## Super coverage by task semantics

| Task family | Status in Miles | Work required for full support |
|---|---|---|
| DAPO/Skywork math, competitive code, knowledge MCQA, Reasoning Gym | Deterministic rewards implemented; corresponding Math/Code/STEM paths exercised | Preserve verifier pins and sandbox policy |
| IFollow and structured outputs | Deterministic rewards implemented; current IFollow jobs 306686/306687 completed fresh+resume inflight replay validation | Dedicated structured-output GPU fresh/resume smoke |
| Conversational tool use and function-calling pivot | Exact single function-call verifier, deterministic 9,400/400 train/eval split, and offline action evaluator implemented; jobs 306920/306921 completed replay/resume on Qwen3-4B-Instruct-2507 | The old recipe is now fail-closed. Add a current-SFT recipe and repeat fresh/resume plus held-out evaluation; free-form message actions remain excluded until a semantic judge is pinned |
| SWE Pivot | Exact single expert-action verifier implemented | This is next-action imitation/RLVR, not repository-level SWE execution |
| Lean proof | 1,376,663 Lean rows staged; no Gym dependency is necessary in principle | Build a pinned NeMo-Skills Lean/Mathlib compiler sandbox, expose a small execute service, add timeout/cache and compiler-result reward, then GPU/replay validation |
| Identity following | Principle-only rows staged | Pinned judge/GenRM service, calibration set, judge version telemetry, and stored judge outputs |
| Adversarial IF and MultiTurnChat | Rubric rows staged | Multi-turn generator plus a pinned rubric judge; validate judge drift and deterministic replay artifacts |
| Calendar | Standalone deterministic verifier implemented; local job 305108 found a feasible schedule for all 9,915 converted rows | Official-grader parity is not established; add dedicated GPU fresh/resume validation, and use a service only if future Calendar data becomes genuinely interactive |
| Workplace assistant | Standalone runtime and the pinned upstream `is_correct` final-state comparator are wired through local resource modules; no checked-in Miles generator exists | Implement the policy/tool loop, bounded worker/service pool, health/reset/cleanup telemetry, and 4-node fresh/rollout-replay validation; no independent parity audit has been recorded |
| Safety | Preference/RM data staged | Separate RM or DPO pipeline; do not convert pairwise labels into an exact-match policy reward |
| GenRM | Generative reward-model corpus staged | GenRM training/serving recipe, calibration, versioned inference, and failure policy |
| SWE 1/2 (R2E-Gym and SWE-Gym) | Native Harbor-to-E2B execution, dataset normalization, live semantic admission, task materialization, pinned graders, and a fail-closed 4-node recipe are implemented. The public Super split normalizes to 1,172 R2E-Gym and 272 SWE-Gym candidates. Offline contracts have passed, but zero rows have completed live E2B admission. | Finish the E2B/template pin and full-scale cost gate, admit empty=0 and oracle=1 tasks with a live `E2B_API_KEY`, then run 4-node fresh RL and an external downstream evaluation. Replay remains disabled. |
| RLHF curriculum | Not a GRPO/RLVR dataset | Implement the intended RLHF/GenRM pipeline and its model/service dependencies |

The completely unsupported Super families are Identity, adversarial IF,
MultiTurnChat, Safety, GenRM, and the RLHF stage. Lean, Calendar, Workplace, and
repository-level SWE now have implementation paths but have not passed their
production admission gates. Conversational tool use, function calling, and SWE
Pivot are supported only at the exactly verifiable single-function-call
granularity, not as full stateful environments.

## Ultra and Nemotron 3.5 Lightning delta

Ultra publishes 337,721 rows across `rlvr1`, `rlvr2`, `ifbench`, `rlhf`,
`reasoning`, `swe`, and `mopd`. It retains the Super task families and adds
ARC-AGI, QA abstention, Litmus Bench, citation formatting, free-form formatting,
InverseIFEval, science, indirect prompt injection, and a dedicated multi-subject
reasoning blend. Its `swe` subset is 97.36% SWE-rebench-v2 and 2.64% SWE-Gym.

| Ultra addition | Current Miles coverage | Required work |
|---|---|---|
| ARC-AGI | Not implemented | Dataset adapter plus grid parser and exact output verifier; isolated Python only if transformations execute generated code |
| QA abstention | Not implemented | Preserve answerability labels, implement exact abstain/answer contract, and add held-out calibration |
| Citation formatting | Not implemented | Deterministic required-marker/string-match verifier; separate factual citation correctness from formatting |
| Free-form formatting | Not implemented | Regex verifier registry with adversarial correct/wrong probes and timeout limits |
| Litmus Bench | Not implemented | Inspect each released verifier type, port deterministic cases, quarantine judge-only cases |
| InverseIFEval / MultiTurnChat | Not implemented | Multi-turn generator plus pinned rubric/GenRM judge and replay artifacts |
| Indirect prompt injection | Not implemented | Stateful tool environment, immutable policy/secret fixtures, attack-aware terminal grader, safety evaluation |
| Multi-subject reasoning | Partial through Math/STEM/ReasoningGym | Route each source to its published verifier and audit unsupported subject scorers |
| Repository SWE | 7,610 SWE-ReBench-V2 and 206 SWE-Gym candidates are normalized from the public Ultra split; the native Harbor/E2B path exists but no candidate is live-admitted | Finish immutable template/publisher-pin validation and cost controls, perform live semantic admission, then run fresh 4-node RL and separate downstream evaluation |

The 92,684-row Lightning blend is a useful current coverage target. Static or
partially implemented paths cover Math-v4, competitive code, MCQA/science,
instruction following, structured outputs, Calendar, and exact function-call
actions. Repository SWE has a local execution path but remains outside the
end-to-end coverage count until live admission and training/evaluation complete.
Other gaps are full conversational tool use, GenRM, Safety, MultiTurnChat, QA
abstention, free-form formatting, citation formatting, and message-action
grading. NVIDIA's current public Lightning recipe describes
six NeMo Gym environments (math-with-judge, code generation, MCQA, instruction
following, Workplace, and structured JSON), while its current dataset card has
more components and does not list Workplace. Treat this as recipe/dataset
version drift, not proof that unlisted components are unused.

## IFBench and Tau Bench scope

IFBench's released test set has 300 prompts and is held out for offline
evaluation. It must not become training data. The RL recipe instead uses 46,391
Nemotron IFollow rows with the pinned IFEvalG verifiers; IFBench is used to
measure out-of-distribution instruction-following generalization. The official
IFBench repository describes 58 OOD constraints and 29 separate RLVR training
constraints. Current full offline job 305176 evaluated 300 prompts with eight
samples each and obtained sample accuracy 0.20375. This validates the held-out
pipeline on that checkpoint; it is not a post-RL improvement claim. Current-SFT
smoke job 307365 also completed two prompts x eight samples with accuracy 0 and
four length-finished empty responses. That is runner evidence only.

Tau Bench is not itself a Nemotron dataset. The raw 500-task retail train split
is retained for audit. Before mixing, every task is checked with the pinned
official reward: its ground-truth trajectory must score one and an empty
trajectory must score zero. Five published tasks (indices 161, 214, 229, 288,
and 349) fail the latter condition because their published gift-card action is
rejected for insufficient balance, leaving the database unchanged. They are not
safe binary-RL examples and are excluded. The local balanced training file thus
combines 495 reward-verified Tau retail tasks, 495 Nemotron conversational
tool-use rows, and 495 Nemotron function-calling rows. Tau rows run in the
official stateful retail environment; Nemotron rows retain their single
expert-action verifier. The user simulator is a loss-masked local-policy turn
generator, so no external model key is required for cluster smoke tests.

Jobs 299793--299795 and summary job 299801 are historical replay-mechanism
evidence: they imported the since-removed `experiments.src.nemo_blends` and
`experiments.src.tau_bench` layouts. Their recorded staleness and replay metrics
must not be attributed to the current generator. Current canonical job 305093
used `experiments.src.environments.tau_bench.generator.generate` and
`experiments.src.reward_sets.tau.reward`, with the local-policy user simulator
and replay disabled, and completed one optimizer update. Current-SFT local-policy
jobs 307433/307434 subsequently completed a 16K, n=16 rollout-replay fresh/resume
sequence through iteration 2. The second restored 6 pending, 2 ready, and 2
active groups plus one prepared batch in 0.349 seconds. The replacement SFT
recipe is not yet checked in, so this is runtime evidence rather than a complete
maintained recipe admission. CPU audit jobs 306786/306797 each completed 20
replay/guard tests. The initial n=2 job 306809 was canceled; jobs 306813/306814
later completed but used the now-prohibited Qwen3-4B-Instruct-2507 checkpoint and
are historical evidence only.

For STEM, CPU job 306819 completed with exit code 0 after 121 passing tests and
official Reasoning Gym correct/wrong probes (`correct=1`, `wrong=0`). Fresh GPU
job 306790 completed optimizer step 0, `replay_buffer_0`, and the iteration-0
checkpoint. Resume 306792 restored that state, trained step 1, and published
iteration 1 plus `replay_buffer_1`; both jobs exited 0.

The checked-in code also supports a Gemini user simulator, defaulting to
`gemini-2.5-flash-lite`, when `TAU_USER_BACKEND=gemini` and `GEMINI_API_KEY` are
available at the Slurm job boundary. Python environment/evaluator modules do not
parse dotenv files. This backend still needs both RL and held-out evaluation
execution evidence before it can support a comparability claim. Local-policy
downstream job 307463 rejected its result because all eight episodes failed
before a terminal state; `mean_reward=0` there is not a model-quality score.

The pinned Tau v1 environment is retained for the first Miles compatibility
run. Its own repository now labels the airline/retail tasks outdated and points
to Tau three. A production migration requires replacing the compatibility layer
and task converter with the current Gym interface, pinning a Tau-three release,
adding its `retail`, `airline`, `telecom`, and optional `banking_knowledge`
domains, and re-running reward-equivalence and replay tests. Voice mode is an
evaluation modality and is out of scope for text RL.

## Replay eligibility

| Reward/generator class | Eligible replay mode | Admission rule |
|---|---|---|
| Static deterministic reward, including IFEvalG | `inflight` or `rollout` | Admit only after fresh and resumed GPU jobs show optimizer updates and restore telemetry |
| Tau stateful custom generator | `rollout` only | Runtime fresh/resume passed in 307433/307434; keep the mode limited to completed trajectories, commit the replacement recipe/tests, and do not reconstruct an inflight environment halfway through a turn |
| Judge/GenRM reward | `rollout` only initially | Store judge output, model/version, rubric hash, and failure status with the sample |
| Workplace environment | `rollout` only initially | Store environment version and deterministic seed/reset identity; first move execution behind a bounded lifecycle |
| Calendar static constraint reward | `inflight` or `rollout` in principle | First pass dedicated GPU fresh/resume validation and record verifier version; current CPU evidence is not replay admission |
| E2B SWE environment | Disabled | First pass live fresh and resumed E2B validation. If admitted later, start with completed `rollout` artifacts containing the immutable template/image, repository/commit, patch, test command/version, result artifacts, and environment seed. |

Custom rewards or generators stay outside the replay allowlist until their
fresh and resume jobs both pass. This avoids interpreting a serializable buffer
as proof that an environment can be safely reconstructed.

IFEvalG has completed the current fresh+resume inflight replay gate in jobs
306686/306687: iteration 0 and `replay_buffer_0` were saved, restored, advanced
to step/iteration 1, and republished with `replay_buffer_1`. End-of-job
`BrokenPipeError` messages occurred during shutdown after successful publication
and exit, so they are teardown noise. Code jobs 306787/306788 passed the same
current-code gate: the resume restored iteration 0 and `replay_buffer_0`, applied
optimizer step 1, and published iteration 1 plus `replay_buffer_1`. Both exited
0; their teardown `BrokenPipeError` messages likewise followed durable
publication. STEM jobs 306790/306792 also passed the current gate by restoring
iteration 0 and `replay_buffer_0`, training step 1, and publishing iteration 1
plus `replay_buffer_1`. Tau current-SFT jobs 307433/307434 restored completed-
rollout replay and advanced through iteration 2, but the replacement recipe is
not yet committed and downstream evaluation has no successful episode. The older IFEvalG
299318/299319 and Tau 299794/299795 pairs used removed import layouts and are
historical checkpoint-mechanism evidence only. Tau `inflight` remains prohibited
because its environment cannot be reconstructed from a partial token prefix.

## E2B SWE implementation and admission boundary

Miles now has a native Harbor/E2B path rather than a NeMo-Gym dependency. It
normalizes R2E-Gym V1, SWE-ReBench-V2, SWE-Gym, and the corresponding Nemotron
Super/Ultra subsets; resolves allowed OCI publishers to immutable image
identities; creates one-use E2B sandboxes; runs a bounded shell/edit/test agent;
uploads the hidden verifier late into a fresh verifier sandbox; and emits a
binary terminal reward. Candidate and task-tree digests bind the dataset row,
materialized task, image, agent, and verifier evidence. Provider credentials
remain on the Harbor service; Miles rollout workers receive only a scoped
`/run` bearer token. No `.env` file is loaded.

The implementation has passed offline contract tests, production-container CPU
tests, a GPU metadata/reward plumbing test, and one-task inspection
materialization for SWE-ReBench-V2 and SWE-Gym. R2E-Gym cannot be inspection-
materialized from its raw row: its oracle patch and exact test mapping are
derived in a live gold-parent sandbox by design. These checks establish the
software contract, not E2B availability or task correctness at scale.

As of 2026-08-26, `E2B_API_KEY` is not exported to the job environment and
`/data/miles-swe/admitted` contains no promoted training dataset. Consequently:

- live empty-patch=0/oracle-patch=1 semantic admission has admitted zero rows;
- template build cost, concurrency/latency, and all publisher/image pins still
  need the final full-scale live gate;
- the 4-node, 16K-response, 16-samples-per-prompt RL recipe has not run; and
- neither hardened-local nor official-comparable downstream SWE-bench results
  exist.

Replay is deliberately fail-closed (`USE_REPLAY_BUFFER=0`). Completed-rollout
replay may be considered only after fresh and resumed live E2B jobs pass;
inflight replay remains prohibited without exact sandbox snapshot restoration.

There are two explicit trust boundaries. First, owner-only files and a pruned
agent environment do not isolate mutually hostile processes running as the same
Unix UID on the Harbor task host; do not co-locate untrusted same-UID workloads.
Second, Harbor's bearer-authenticated HTTP control path is intended for a
private cluster fabric. It is not an Internet-facing zero-trust perimeter; use
an authenticated TLS/mTLS proxy before crossing an untrusted network.

The provided SWE-bench Verified evaluator is intentionally named
`hardened-local`: it uses pinned official parser/grading code but adds local
security/path policy and requires pre-bound admitted task rows. Its score is not
leaderboard-comparable. An exact official score must be produced separately
with the unmodified pinned official evaluation harness and its expected result
artifacts.
