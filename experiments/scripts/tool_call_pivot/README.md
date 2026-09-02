# Conversational tool-use RL

The canonical recipe uses NVIDIA's pinned
`Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` release and the local
Qwen3-4B Step4000 SFT checkpoint:

```text
async/nemotron-agentic-conv-tooluse-pivot/qwen3-4b/
```

The source represents each expert assistant step as a separate next-action RL
problem. The multi-turn prefix is fixed expert history: this recipe generates
one next action, does not execute it against a database, and is not end-to-end
agent-environment RL. Dataset preparation admits only `function_call` actions
with a declared tool and exactly verifiable name and arguments. The pinned
payload has 96,968 rows: 65,559 exact function calls and 31,409 free-form
message actions.
It reserves 2,000 deterministically selected eligible rows for offline
evaluation and uses the remaining 63,559 unique function calls for training.
Free-form message actions are retained in the raw and converted source audit
but excluded from this recipe because exact-match reward cannot verify their
semantics.

For throughput and staleness studies this domain is always identified as
`TASK_FAMILY=tool_call_pivot` with
`ROLLOUT_SEMANTICS=static_single_turn_pivot`. Prepared rows carry the same
`interaction_mode` plus `stateful_environment=false`, and both the recipe and
reward reject stateful or custom multi-turn generation. Report this arm
separately from `workplace`, whose environment executes tools and returns new
observations across turns.

The training recipe uses 16K response length, 16 samples per prompt, four
nodes, W&B offline mode, no inline evaluation, the validated tool-call reward,
and inflight replay. The primary downstream evaluation is the stateful Tau
three v1.0.1 held-out test split. `tool_call_pivot/evaluate.sbatch` remains a
training-domain diagnostic for exact action, tool-name, and argument accuracy;
it is not the reported downstream benchmark.

Run these stages in order:

```bash
bash experiments/setup/download/stage_swe_tool_rl_datasets.sh
source experiments/env.sh
source experiments/common/pbs.sh
pbs_submit --profile=cpu --time="${PBS_PREP_WALLTIME}" \
  experiments/setup/datasets/prepare_agentic_tool_use_pivot.sbatch
pbs_submit --profile=cpu --time="${PBS_PREP_WALLTIME}" \
  tests/slurm/test_tool_call_pivot_environment.sbatch
pbs_submit --profile=cpu --time="${PBS_PREP_WALLTIME}" \
  experiments/setup/environments/prepare_tau_bench.sbatch
```

`submit_effectiveness_when_idle.sh` runs Tau three pre-evaluation, Pivot
training, and Tau three post-evaluation when this user's PBS queue is empty.
There is no Tau training recipe under `experiments/scripts/tau_bench/`; that
directory is evaluation-only.
