# Codex Instructions

When creating or substantially modifying `miles/**/*.py`, `scripts/**/*.py`, `tools/**/*.py`, `train.py`, or `train_async.py`, read and follow `.claude/rules/general-code-style.md`.

When creating or substantially modifying a launcher under `scripts/` or `examples/`, or a model definition under `scripts/models/`, also read and follow `.claude/rules/launch-and-model-scripts.md`.

For Slurm experiments on this cluster, read `experiments/notes/cluster.md` and
`.claude/skills/miles-run-ladder/SKILL.md` before allocating resources. The default
account is `nemotron_sw_post`. GPU validation uses partition `batch` with QoS
`interactive` (not partition `interactive`); nodes have 4 GB200 GPUs and aarch64
CPUs. Recheck live limits using the discovery commands in the cluster note.
Use `tests/manual/run_oci_validation.sbatch` for disposable multi-node NCCL and
regression checks. For the staged Step4000 assets, use the training/save/resume
wrapper `tests/manual/run_oci_step4000_smoke.sbatch`; paths, commands, and results
are in `experiments/notes/oci-step4000-bringup.md`.
See `experiments/notes/cluster-migration.md` for the exact
qualification limits; infrastructure tests do not establish training readiness.
