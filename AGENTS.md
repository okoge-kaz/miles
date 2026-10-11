# Codex Instructions

When creating or substantially modifying `miles/**/*.py`, `scripts/**/*.py`, `tools/**/*.py`, `train.py`, or `train_async.py`, read and follow `.claude/rules/general-code-style.md`.

When creating or substantially modifying a launcher under `scripts/` or `examples/`, or a model definition under `scripts/models/`, also read and follow `.claude/rules/launch-and-model-scripts.md`.

Per user preference, do not add opening module docstrings to new Python files. Put design explanations in the relevant notes and retain useful function/class documentation.

For ABCI experiment implementation, debugging, or notes, use
`.claude/skills/abci-experiments/SKILL.md` and consult the relevant files in
`experiments/notes/agents/`. Keep analysis and measurements in `experiments/notes/logs/`.
