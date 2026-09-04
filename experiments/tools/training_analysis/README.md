# Training analysis

These tools inspect artifacts that an RL run has already written. They do not
launch training and they are not held-out evaluators.

- `summarize_log.py` reads the textual training log and reports rollout reward,
  truncation, staleness, optimizer-step, and weight-update evidence.
- `summarize_dump.py` reads Miles dashboard dumps and reports the same evidence
  by verifier and environment, including response weight-version coverage.
- `summarize_dump.sbatch` runs the dump reader against a host-side dump
  directory in the pinned Miles container.

Use `experiments/tools/reasoning_eval/` for checkpoint evaluation and
`experiments/tools/replay_buffer_validation/` for replay-buffer-specific
experiments.
