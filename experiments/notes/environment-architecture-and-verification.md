# Miles environment architecture and verification

Status date: 2026-08-26. This note separates implemented code, completed
validation, and recommended architecture. Dataset coverage is tracked in
`nemotron3-nano-super-task-coverage.md`.

## Recommendation

Use a hybrid boundary. Do not make Miles depend on the NeMo Gym Python package,
but also do not keep every stateful environment inside the rollout manager.

1. Keep CPU-cheap, deterministic, side-effect-free verifiers in process:
   Math, MCQA/GPQA, IFEvalG, JSON schema, Calendar constraints, exact next tool
   action, and ReasoningGym scorers.
2. Put mutable state, expensive execution, judges, and sandboxes behind a
   framework-neutral async episode service. The minimum contract is
   `create/reset`, `step`, `verify`, and `close`, with an episode id, task id,
   seed, environment version, deadline, and structured failure class.
3. Write one Miles agent-function adapter to that contract. Provide adapters
   for NeMo Gym's seed/verify and Gymnasium protocols instead of importing Gym
   into the training image.
4. Reuse NeMo Gym environments out of process when their published semantics
   or NVIDIA dataset compatibility is valuable. Keep direct implementations
   only where the verifier is small enough to audit and own.
5. Run full SWE, desktop, browser, and Office-GUI environments in a sandbox/VM
   pool. Never execute their mutable state in the trainer or rollout manager.

This keeps high-throughput static rewards cheap while obtaining bounded
concurrency, fault isolation, independent upgrades, and horizontal CPU scaling
for agentic workloads.

## What Miles and slime currently provide

Miles is a trainer/rollout framework, not an environment catalog. Its documented
extension levels are:

| Extension | Owner of agent loop | Owner of token trajectory | Typical use |
|---|---|---|---|
| custom reward | Miles | Miles | static RLVR verifier |
| custom agent function | environment adapter | Miles session server | external HTTP environment/harness |
| custom generate function | experiment code | experiment code | Tau, local multi-turn tools |
| custom rollout function | experiment code | experiment code | custom batching/data source/multi-agent |

`miles/rollout/generate_hub/agentic_tool_call.py` is the current in-tree
boundary for an external episode service: it sends policy calls through the
Miles session server, which records token ids, log probabilities, and loss masks.
The current user guides for NeMo Gym, Harbor, and rollout endpoints all use
adapters around this boundary. Treating Miles as the connector and trajectory
owner is therefore the repository-supported direction; a future roadmap item is
not counted as implemented until its code is present.

slime follows the same plugin policy. Its documented default for tool,
sandbox, browser, and multi-turn work is a custom generate function plus a
custom reward. The current coding-agent example creates a fresh execution
sandbox, obtains a diff, and grades that diff in a second clean sandbox. slime
therefore owns hook contracts and trajectory plumbing, not a universal
environment protocol or environment lifecycle service.

## Why current `experiments/src` code does and does not scale

Static verifier dispatch is the lower-cost path. The restricted reward
dispatcher groups a mixed batch by `metadata.verifier` and evaluates those
groups concurrently. The competitive-programming verifier, not the dispatcher,
owns one `asyncio.Semaphore` per event loop (`CODE_EXEC_CONCURRENCY`, default 4),
so scalar reward calls cannot bypass the subprocess cap. Calendar is an in-
memory constraint check. MCQA, IF, schema, and math do not maintain episode
state.

The checked-in Workplace generator/runtime/verifier implements a local Miles
rollout environment for one user request and multiple model/tool steps:

- `experiments.src.environments.workplace.runtime._load_resource_functions` is
  cached per Python process, but every
  future rollout worker would still load pinned modules and fixtures;
- runtime helpers create pandas/CSV-backed databases and expose 27 tool
  functions in process;
- `experiments.src.environments.workplace.generator` wires the bounded step loop
  to Miles, but it has no conversational user simulator;
- there is no cross-worker environment backpressure, health endpoint, lease,
  cleanup retry, or isolation from a crashing resource module.

At the current production shape, `rollout_batch_size=192` and `n=16` can produce
3,072 simultaneous episode trajectories. Even if policy generation is bounded,
allowing that many database/tool environments in rollout workers creates CPU,
memory, GIL, and long-tail contention. A production service should expose:

- a bounded worker/actor pool and queue depth;
- idempotent episode creation and close;
- deterministic seed and fixture version;
- per-step and whole-episode deadlines;
- health/readiness, retry classification, and lease expiry;
- structured reward artifacts and environment version in replay;
- independent CPU autoscaling without restarting policy workers.

Calendar should remain local until it acquires real tools or mutable state.
Workplace should move to the episode service. Tau can use the same service
contract around its pinned official environment. Lean needs a compiler execution
service. SWE needs a sandbox provider plus a separate clean grader.

### Current validation boundary

Implementation shape and GPU admission are separate:

- Code jobs 306787/306788 completed the current same-identity fresh+resume gate.
  The second job restored iteration 0 and `replay_buffer_0`, trained step 1, and
  published iteration 1 plus `replay_buffer_1`. STEM jobs 306790/306792 passed
  the same gate through iteration 1 and `replay_buffer_1`; balanced Math+Code+STEM
  jobs 306793/306796 completed the same gate. The resume restored iteration 0,
  15 pending, 3 ready, and 6 inflight groups plus one prepared batch, then
  published iteration 1 and `replay_buffer_1`. Older jobs 305094/305095/305096
  each established one optimizer update only.
- Exact tool action jobs 306920/306921 completed an inflight replay fresh/resume
  pair, but used the now-prohibited Qwen3-4B-Instruct-2507 recipe. They validate
  the verifier/replay mechanism historically. The replacement Step4000 recipe
  trains on conversational tool-use Pivot data. AReaL Tau2 user-simulator RL is
  a separate recipe with multi-turn `inflight` event-log replay, and Tau v3 is
  held out for downstream evaluation.
- IFEvalG jobs 306686/306687 completed the current 4-node, 16K, n=16 inflight-
  replay fresh/resume gate. The second job restored iteration 0 and its replay
  state, trained step 1, and published iteration 1 plus `replay_buffer_1`.
  `BrokenPipeError` messages during final Ray shutdown occurred after successful
  publication and completion.
- Calendar and Workplace have local CPU correctness evidence only. They do not
  have dedicated GPU forward/backward+resume evidence.
- STEM CPU job 306819 completed with exit code 0 after 121 passing tests and
  official Reasoning Gym correct/wrong probes (`correct=1`, `wrong=0`). This
  complements the completed GPU pair; it does not establish downstream GPQA
  improvement.
- Legacy domain-eval jobs 305175/305176 completed LiveCodeBench, GPQA Diamond,
  and IFBench generation+scoring. Job 306776 completed two-prompt YAML-entry
  generation+scoring for AIME24, MATH500, and GPQA Diamond; reasoning job 306691
  completed the current-refactor 30-prompt, one-repeat AIME24 contract. Current
  job 307365 then completed all nine current-SFT smoke tasks: AIME24/25/26,
  MATH500, LiveCodeBench, all three GPQA splits, and IFBench. Its two-prompt
  samples validate the runner, not full benchmark accuracy. Separate CPU jobs 306822/306823/306854
  audited the full 500-row MATH-500, three 30-row AIME, and all three GPQA
  prepared-data/scorer contracts; those are data/config evidence, not additional
  model evaluations.
  The Tau three held-out evaluator and exact tool-call diagnostic still lack a
  current-SFT execution result.
- Full SWE has no live-admitted E2B rows and no 4-node RL/downstream result.

## NeMo Gym trade-off

NeMo Gym already separates Agent, Model, and Resources servers. Its resources
server owns tools, state, and verification; its agent server owns the episode
loop. Current deployment documentation supports a shared Ray cluster, a Gym-owned
Ray cluster, or fully separate clusters connected over HTTP. The last topology
avoids Python/Ray dependency coupling with Miles and is the preferred Gym mode
here.

Advantages:

- published Nemotron environment semantics and data adapters;
- per-session state isolation and async FastAPI server conventions;
- existing stateful, judge, tool, and sandbox integrations;
- independent CPU/GPU resource servers and a growing upstream catalog;
- less local code for complex NVIDIA tasks.

Costs and risks:

- evolving config, manifests, APIs, and environment layout;
- pin/container upgrade work and occasional recipe/card version drift;
- another distributed service to deploy, observe, and secure;
- same-cluster mode imposes Ray/Python compatibility;
- upstream environment bugs can affect rewards even when Miles is healthy.

The mitigation is not to fork all of Gym. Pin Gym and each resource image by
commit/digest, deploy it separately, validate gold/no-op/adversarial fixtures,
and connect through a small versioned HTTP adapter. Miles remains usable when
Gym is absent.

## Cost estimates

These are engineering estimates for this repository, not vendor commitments.
They assume dataset preparation, correct/wrong fixtures, metrics, one GPU smoke,
and replay/resume admission tests.

| Environment class | Own implementation | Reuse a working NeMo Gym environment | Typical ongoing cost |
|---|---:|---:|---:|
| Static exact verifier | 0.5–2 engineer-days | 1–3 days | low; pin parser/dependency |
| Deterministic stateful tools | 1–3 engineer-weeks | 3–7 days | own: 0.25–0.5 engineer-month/quarter; Gym: 2–5 days per major upgrade |
| Fixed compiler/code sandbox | 2–4 weeks | 3–10 days plus image build | own: 0.5–1 engineer-month/quarter |
| SWE/browser/desktop | 1–3 engineer-months or more | 2–4 weeks integration | high in either case; images, benchmark drift, security, cost |

For Math/Code/STEM/IF/schema/Calendar, local ownership is cheaper. For full SWE,
browser/desktop, multi-turn judged chat, and a broad Nemotron-compatible catalog,
reuse is cheaper. Workplace is near the boundary: its verifier is auditable, but
the lifecycle/service layer should be shared rather than task-specific.

## Existing verifier semantics

### Math

Math rows are prompt JSONL, not an interactive environment. `metadata.verifier`
routes to `grade_answer_verl`. The model's final/boxed answer is normalized, then
compared with the reference using the Miles math utilities and symbolic
equivalence where applicable. DAPO and Skywork adapters retain published ground
truth and append the required final-answer format. Difficulty filtering is a
separate offline model-pass-rate selection step; it is not the reward.

### Code

Competitive-code rows include published tests in metadata. The verifier extracts
the final Python block and runs every selected stdin/function/published-harness
test in a fresh temporary directory. The default security boundary is a new user
and network namespace plus Bubblewrap filesystem isolation and `prlimit` caps for
address space, CPU, files, descriptors, output, and wall time. Reward is one only
when all selected tests pass. This requires a local sandbox, but no external
sandbox API.

### STEM and GPQA

Knowledge MCQA uses each row's published regex and expected option. GPQA uses a
seeded option ordering and the Miles GPQA parser. ReasoningGym loads a pinned
task-specific scorer. These are static prompt/reward environments. The current
Math+Code+STEM “multi env” is one balanced JSONL whose rows carry different
verifier ids; the fail-closed `reward_sets.math_code_stem` entry point dispatches
each row and rejects every other verifier id. It is not an HTTP environment
scheduler.

### Instruction following and IFBench

Training uses 46,391 Nemotron IFEvalG rows. The pinned Open-Instruct registry
constructs every constraint checker, strips hidden thinking, and returns the
mean of satisfied constraints. IFBench's 300 released prompts are held out. Its
official strict offline scorer gives success only when all constraints for a
prompt pass. Both are deterministic; neither needs Gemini.

### Tau Bench

The training path uses the external AReaL Tau2 RL split: 1,982 serialized tasks
and nine DB snapshots. A thin `AgentGymEnv` extension injects each task and a
deep-copied DB into the pinned official user-simulator/orchestrator lifecycle.
Terminal reward follows the task-declared DB, environment assertion, action,
and communication basis. Natural-language-judge reward is rejected. Stateful
inflight replay stores the policy prefix and official message history, replays
mutating tool calls, and validates the restored DB hashes before generation
continues.

Tau v3 v1.0.1 remains the downstream evaluator. It runs all 100 held-out retail,
airline, and telecom test tasks through the official DB-backed environment.
Official Tau v3 train/base tasks are used transiently to validate the split
contract and are not materialized or used for training.

### Calendar and Workplace

Calendar expects a complete JSON event list. The verifier checks exact event ids,
durations, time windows, before/after/at/between constraints, and global
non-overlap. Local preparation job 305108 produced a feasible schedule accepted
by this verifier for all 9,915 converted rows. No independent run of an official
grader was recorded, so this establishes local constraint consistency only.

Workplace's checked-in runtime exposes email, calendar, analytics, project-
management, and CRM tools, and its verifier uses the pinned upstream
`is_correct` final-state comparison. It does not import or run NeMo Gym, but
reuses pinned Workplace resource modules and data from a Gym checkout. Its
custom generator performs multiple model/tool steps for one fixed user request;
without a user simulator, it is explicitly single-turn multi-step rather than
conversational multi-turn.

### Static conversational/function tool actions

Only rows whose expected next action is a function call are admitted to the new
tool-call split. Reward requires exactly one call, the exact tool name, identical
argument keys, and matching values. `expected_action=message` rows are excluded:
the existing “non-empty message and no tool” check cannot establish semantic
correctness. The held-out evaluator reports exact action, name, arguments,
single-call, no-call, error, and per-source metrics.

## SWE-RL verification

It is test-driven, but “run unittest” is incomplete. One episode and its grader
should be separate:

1. Start a writable agent sandbox from a pinned instance image/repository state.
2. Let the agent inspect, edit, and run its own tests. Record all model tokens and
   loss-mask tool observations.
3. Capture a normalized `git diff`; reject empty, oversized, binary, disallowed
   path, submodule, symlink, and secret-bearing patches.
4. Start a second clean grader sandbox from the same pinned base. Never reuse the
   agent's workspace.
5. Apply only the captured patch. A successful `git apply` is necessary but has
   no reward by itself.
6. Apply the benchmark's hidden test patch or mount hidden tests read-only, then
   run the instance-specific `eval.sh`/test command with a timeout.
7. Parse results with the benchmark's repository/version-specific parser.
8. For SWE-bench, full resolution means every `FAIL_TO_PASS` test now passes and
   every `PASS_TO_PASS` regression test still passes. Partial counts are useful
   diagnostics but should not silently become the terminal binary reward.
9. Persist base image digest, commit, patch hash, test-spec hash, stdout/stderr,
   exit code, timeout/failure class, F2P/P2P sets, and final reward.

The official harness uses Docker and tries `git apply`, a reject variant, then
`patch`; E2B can replace Docker while retaining the canonical grader. A separate
clean grader also addresses a known risk in the local harness: unrestricted
patches can edit tests or configuration and poison evaluation. Enforce path
policy and keep hidden tests unavailable to the agent.

For RL, use completed-rollout replay first. Store the diff and grader artifacts;
do not rerun an old trajectory against a changed image or test spec. Inflight
replay is unsafe unless the exact sandbox snapshot and agent state can be
restored.

## Word, Excel, PowerPoint, and Mail RL

Choose capability scope before choosing a sandbox.

### File-native agent (recommended first)

This trains document operation, not mouse/GUI skill. Give the agent a disposable
directory and typed tools:

- Word: paragraph/run/style/table/comment/relationship operations through
  `python-docx` or direct OOXML;
- Excel: cell/formula/style/table/chart/named-range operations through
  `openpyxl`, with LibreOffice headless recalculation for formula caches;
- PowerPoint: slide/layout/shape/text/table/image/theme operations through
  `python-pptx` or OOXML;
- render/convert/inspect through a pinned headless LibreOffice container;
- Mail: a fake inbox/CRM/calendar/knowledge-base service with read/search,
  draft/send/forward/escalate/schedule/update tools and a durable action ledger.

Each episode starts from immutable input artifacts and a fresh work directory or
database. Terminal verification should combine:

- package validity and security checks (ZIP/OOXML paths, macros, external links,
  decompression limits);
- native structural assertions: text, styles, tables, formulas, names, charts,
  slide hierarchy, relationships, comments, and untouched-state invariants;
- formula recalculation and semantic value checks for spreadsheets;
- deterministic render checks for visual requirements, using tolerance or image
  regions rather than raw whole-image equality;
- for Mail, exact final database/ledger state, policy constraints, recipient and
  attachment correctness, evidence provenance, and explicit penalties for
  unauthorized sends or secret leakage.

DocOps and OfficeVal are useful starting verifier/task sources: they inspect
native documents and include executable per-task checks. WorkBench and the email
response benchmark demonstrate state/ledger grading for workplace and mail
tasks. Use train/eval artifact hashes to prevent leakage.

### GUI/desktop agent

Use this only when the capability target is actual Word/Excel/PowerPoint/Outlook
UI operation. Each episode needs a snapshot-restored VM, isolated user identity,
screen plus accessibility/UIA observations, keyboard/mouse actions, application
health checks, and deterministic teardown. Grade final files and application
state outside the UI whenever possible. OSWorld provides real desktop task and
evaluator patterns; Windows Agent Arena supplies a Windows 11 VM/snapshot and
UIA/Win32-oriented scaling path.

GUI RL is much more expensive and noisy: boot/reset latency, dialogs, font and
render drift, focus races, Office licensing, Windows images, account isolation,
and evaluator brittleness. Start with file-native Linux containers, then add a
small Windows/GUI curriculum only for tasks where UI navigation itself matters.

### Suggested implementation order

1. Import a small leakage-free DocOps/OfficeVal subset and run gold/no-op/
   adversarial verifier audits.
2. Build one generic artifact episode service and Miles agent adapter.
3. Add Word, Excel, and PowerPoint typed tools plus a clean terminal grader.
4. Add the fake Mail world and final-state ledger verifier.
5. Run offline baselines, one-node GPU smoke, then 4-node fresh training.
6. Admit completed-rollout replay only after fresh and resumed runs preserve
   artifact hashes and verifier versions.
7. Add OSWorld/Windows VM environments only after the file-native path is stable.
