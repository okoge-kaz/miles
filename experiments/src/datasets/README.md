# Dataset preparation

Dataset modules convert source records into the stable Miles JSONL contract.
They may depend on protocol adapters and environment verifiers for conversion
audits, while environment modules must not import dataset conversion modules
except for immutable task-identity validation needed during rollout.
Training reads the converted JSONL directly; it does not carry source-specific
Nemotron schema handling in the generic data loader.

- `common`: streaming I/O, structural audit, and deterministic merge utilities;
- `nemotron`: Nemotron schema adapters, conversion, and Nano restoration;
- `calendar`, `workplace`, `tool_call_pivot`, `tau_bench`: environment-specific data
  preparation;
- `areal_tau2`: pinned 1,982-row RL-only Task/DB preparation for the Tau2
  user-simulator training environment;
- `search_r1`: Search-R1 evaluation-data preparation.
