# Training environments

Each directory owns the runtime, generator, or verifier for one environment
family:

- `common`: loss-masked observation insertion shared by stateful environments;
- `calendar`: local schedule-constraint verification;
- `competitive_programming`: sandboxed execution of published Python tests;
- `instruction_following`: pinned Open-Instruct IFEvalG constraint checking;
- `reasoning_gym`: pinned task-specific Reasoning Gym scoring;
- `search_r1`: Search-R1 retrieval service runtime;
- `areal_tau2`: stateful conversational multi-turn Tau2 training with an
  official user simulator, per-row DB snapshots, and terminal reward;
- `tau_bench`: shared Tau policy loop and held-out Tau v3 rollout lifecycle;
- `tool_call_pivot`: deterministic static single-turn Pivot verification; it
  never executes a tool or advances environment state;
- `workplace`: pinned resource runtime for one user request followed by multiple
  model/tool steps, and official final-state verification. It is explicitly a
  single-turn multi-step environment, not conversational multi-turn RL;

Dataset conversion and mixture assembly are deliberately outside this tree.
Static recipes access verifiers through a restricted
`experiments.src.reward_sets.<recipe>.reward` entry point. Stateful recipes use
their explicit custom-generate entry point. Both paths validate
`metadata.verifier` before execution.

Dataset preparation lives under `experiments/src/datasets`; callers import the
dataset and environment implementations directly from their canonical paths.
