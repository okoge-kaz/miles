# Experiment tests

| Directory | Scope |
| --- | --- |
| `common/` | PBS submission, job configuration, training launch and service lifecycle |
| `container/` | Container validation and publication |
| `checkpoint_convert/` | HF / Megatron conversion, assets, comparison and shared checkpoint state |
| `datasets/` | Source conversion, parallel preparation and training mixture contracts |
| `difficulty_filtering/` | Offline difficulty selection, resume/provenance, PBS recipes, merge dependencies, RL sampling parameters, independent inference engines, and CPU verifier / GPU measurement supervision |
| `nemo_gym/` | Reward, environment and HTTP bridge contracts |
| `training/` | Truncation, preflight, checkpoint controls, replay and training metrics |

These are CPU checks, including Miles core extensions used by the experiments.
They do not establish GPU training performance or convergence. Run them in the
prepared Miles image:

```bash
python3 -m pytest --confcutdir=tests/fast/experiments tests/fast/experiments
```

The `--confcutdir` option excludes the repository-wide GPU/Ray fixtures for this
standalone CPU run. Tests are collected recursively; no package initializer is needed.

Difficulty filtering checks include fractional IF rewards, unscored verifier failures, HTTP retries,
live journal reads, source snapshots, and the sixteen shared PBS dependency chains across 4B/8B/30B.
Native NeMo Gym probes are recorded separately from CPU/mock tests in the experiment logs.
