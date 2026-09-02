# notes/

Reference notes for running Miles with PBS Pro and SingularityCE. Historical
measurements in `notes/agents/` retain their original scheduler, image format,
paths, and job IDs; they are evidence, not current operating instructions. Two
kinds of documents:

| Directory | Contents | Update policy |
|---|---|---|
| `notes/*.md` | Curated reference — how things work, how to inspect them | Edit in place; keep current |
| `notes/agents/*.md` | Work log — what was investigated/measured on a given day, with evidence | Append; never rewrite history |

Index:

- [cluster.md](cluster.md) — cluster resources and asset paths
- [containers.md](containers.md) — OCI image to SIF, non-root permissions, and mount layout
- [dataset-inventory.md](dataset-inventory.md) — what is staged, by genre: path, row count, verifier, whether it is verified
- [nemotron-rl-training-data.md](nemotron-rl-training-data.md) — the original 21 requested Nemotron/LCB/GPQA repositories, later SWE manifest delta, conversion outputs, and verifier/environment readiness
- [nemotron3-nano-super-task-coverage.md](nemotron3-nano-super-task-coverage.md) — Nano/Super task coverage, IFBench/Tau scope, replay admission, and E2B SWE work
- [environment-architecture-and-verification.md](environment-architecture-and-verification.md) — Nano/Super/Ultra/Lightning coverage strategy, Miles/slime/NeMo-Gym boundary, verifier semantics, SWE grading, and Office/Mail RL design
- [upstream-v0.1-gap.md](upstream-v0.1-gap.md) — capabilities present in official Miles v0.1 but absent from this branch, plus porting priorities
- [datasets.md](datasets.md) — how miles reads a JSONL, how to inspect one
- [domains.md](domains.md) — current domain routing, verifier/generator shape, and validation boundary
- [checkpoints.md](checkpoints.md) — `huggingface/` `megatron/` `training/`, conversion, resume
- [miles-architecture.md](miles-architecture.md) — the four objects, the directory map, the plug points
- [parallelism.md](parallelism.md) — why every recipe is CP=1, the measured cost of context parallelism, per-model memory headroom
- [node-ratio-procedure.md](node-ratio-procedure.md) — how the train:rollout split is chosen, and why the staleness measurement has to come first
- [algorithm-ablation.md](algorithm-ablation.md) — the frame the algorithm arms run in, and what miles can express today
- [rollout-scaling.md](rollout-scaling.md) — the two floors under rollout time, and when adding rollout GPUs stops helping
- [offline-eval.md](offline-eval.md) — the separately validated domain, pinned reasoning, Miles-native config, Tau, and tool-call evaluation paths
- [replay-buffer.md](replay-buffer.md) — persisted fully-async queue state, buffer types, commit semantics, and resume observability
- [replay-buffer-validation.md](replay-buffer-validation.md) — correctness, restart-distribution, latency, and save-cost measurements for `rollout` and `inflight`
- [telemetry.md](telemetry.md) — what the runs record to W&B, what the analysis needs, and known gaps
- [off-policy-variables.md](off-policy-variables.md) — the off-policy / async variable space, fixed controls, and evidence needed for sample-efficiency claims

Maintained recipes live under `experiments/scripts/<domain>/<placement>/...`;
the production-shaped Math reference is
`experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/`. Settings and their
rationale should stay aligned with the notes above. See
`.claude/rules/experiment-recipes.md`.

> **Caution.** This directory lives inside a checkout of an open-source repo. Keep
> internal endpoints, API keys, hostnames and verbatim internal discussion out of
> it — reference the channel or ticket instead.
