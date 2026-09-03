# notes/

Reference notes for running miles on cw-dfw. Two kinds of documents:

| Directory | Contents | Update policy |
|---|---|---|
| `notes/*.md` | Curated reference — how things work, how to inspect them | Edit in place; keep current |
| `notes/agents/*.md` | Work log — what was investigated/measured on a given day, with evidence | Append; never rewrite history |

Index:

- [cluster.md](cluster.md) — cw-dfw partitions, node specs, what is and is not installed
- [containers.md](containers.md) — docker image → `.sqsh`, pyxis flags, mount layout
- [dataset-inventory.md](dataset-inventory.md) — what is staged, by genre: path, row count, verifier, whether it is verified
- [datasets.md](datasets.md) — how miles reads a JSONL, how to inspect one
- [checkpoints.md](checkpoints.md) — `hf/` `megatron/` `training/`, conversion, resume
- [miles-architecture.md](miles-architecture.md) — the four objects, the directory map, the plug points
- [parallelism.md](parallelism.md) — why every recipe is CP=1, the measured cost of context parallelism, per-model memory headroom
- [node-ratio-procedure.md](node-ratio-procedure.md) — how the train:rollout split is chosen, and why the staleness measurement has to come first
- [pipeline-balance-model.md](pipeline-balance-model.md) — producer/consumer equations, queue-policy staleness predictions, and node-scaling fits
- [algorithm-ablation.md](algorithm-ablation.md) — the frame the algorithm arms run in, and what miles can express today
- [rollout-scaling.md](rollout-scaling.md) — the two floors under rollout time, and when adding rollout GPUs stops helping
- [offline-eval.md](offline-eval.md) — how a checkpoint is scored after the fact: the three-step procedure, why AIME-2023 is excluded, and the failure taxonomy
- [replay-buffer.md](replay-buffer.md) — persisted fully-async queue state, buffer types, commit semantics, and resume observability
- [replay-buffer-validation.md](replay-buffer-validation.md) — correctness, restart-distribution, latency, and save-cost measurements for `rollout` and `inflight`
- [telemetry.md](telemetry.md) — what the runs record to W&B, what the analysis needs, and known gaps

Recipes under `experiments/math_*/` carry settings only; the reasoning for any
setting lives in the notes above. See `.claude/rules/experiment-recipes.md`.

> **Caution.** This directory lives inside a checkout of an open-source repo. Keep
> internal endpoints, API keys, hostnames and verbatim internal discussion out of
> it — reference the channel or ticket instead.
- `off-policy-variables.md` — the variable space for the off-policy / async study:
  what miles rejects at startup, the algorithm and importance-sampling surface,
  what is swept, what is fixed, and what must be recorded to support a
  sample-efficiency claim after the fact.
