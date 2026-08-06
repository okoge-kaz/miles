# Experiment recipes: keep rationale out of the scripts

Applies to everything under `experiments/` — `run.sbatch`, `train.sh`, and the
submission helpers.

## The rule

**A recipe holds settings. It does not hold the argument for those settings.**

Measurements, comparisons, job IDs, benchmark numbers, cluster-policy
explanations and "why not the obvious alternative" belong in `experiments/notes/`
and are referenced from there — never inlined as a comment block above a
variable.

Not this:

```bash
# CP=1. Context parallelism was only here to satisfy mtpg * cp >= 32768, but it
# splits the sequence and pays an all-to-all per layer. Measured at 1.9x:
#   cp2, mtpg 16384   actor_train 42.7-44.5s   164 TFLOP/s
#   cp1, mtpg 32768   actor_train 23.4-24.1s   327 TFLOP/s
# (jobs 15150858/60/62 vs 15150863; logged TFLOP/s undercounts by 4/3 ...)
: "${CONTEXT_PARALLEL_SIZE:=1}"
```

This:

```bash
: "${CONTEXT_PARALLEL_SIZE:=1}"
```

with the table, the job IDs and the reasoning in `experiments/notes/parallelism.md`.

## Why

- A recipe is read to answer "what is this run configured as". Ten lines of
  justification per knob buries that.
- The rationale ages differently from the setting. A note gets revised as new
  measurements land; a comment block gets copied into nine sibling recipes and
  then silently contradicts them.
- Recipes are duplicated per model and per mode. Anything written in one has to
  be maintained in ten. Notes are written once.
- `experiments/notes/` is indexed by `notes/README.md` and is where someone
  actually looks for a finding.

## Where things go

| Kind of content | Home |
|---|---|
| Parallelism, recompute, memory/speed trade-offs | `notes/parallelism.md` |
| Partitions, QOS, queue behaviour, submission hazards | `notes/cluster.md` |
| Rollout throughput, scaling, concurrency | `notes/rollout-scaling.md` |
| What is logged and what the analysis needs | `notes/telemetry.md` |
| Checkpoint layout, retention, resume | `notes/checkpoints.md` |
| The variable space of the study | `notes/off-policy-variables.md` |
| Dated measurements with raw evidence | `notes/agents/*.md` (append-only) |

## What a comment in a recipe may still say

Short, local, and about mechanics rather than findings — a unit, a non-obvious
coupling between two variables, or a guard that would otherwise look like a typo.
One line. If it needs a second line, it needs a note.
