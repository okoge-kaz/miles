# Dataset preparation

Dataset modules convert source records into the stable Miles JSONL contract.
They may depend on protocol adapters and environment verifiers for conversion
audits, while environment modules must not import dataset conversion modules
except for immutable task-identity validation needed during rollout.
Training reads the converted JSONL directly; it does not carry source-specific
Nemotron schema handling in the generic data loader.

- `common`: streaming I/O, structural audit, and deterministic merge utilities;
- `nemotron`: Nemotron schema adapters, conversion, and Nano restoration;
- `calendar`, `workplace`, `tool_call`, `tau_bench`: environment-specific data
  preparation;
- `search_r1`: Search-R1 evaluation-data preparation.
