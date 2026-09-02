# PBS/Singularity integration tests

These launchers are persistent cluster integration tests. They run on the
the `R9920261300` reservation with W&B offline by construction. The historical
`tests/slurm` path is retained so existing test references remain stable.

| Test | Coverage | Prerequisite |
| --- | --- | --- |
| `test_nemotron_components.sbatch` | Fast tests plus real staged Reasoning Gym, GPQA, IFEvalG, LiveCodeBench, and Bubblewrap verifier probes. | Canonical prepared datasets and `/usr/bin/bwrap`. |
| `test_nemotron_training_input.sbatch` | Exact Nano input through `Dataset`, tokenizer, chat template, and every admitted static reward. | Nano conversion and `allenai/open-instruct`. |
| `test_performance_transfer_input.sbatch` | Exact competitive-code and STEM training inputs through `Dataset`, tokenizer, and rewards. | Performance-transfer blend preparation. |
| `test_tau_bench_environment.sbatch` | All pinned Tau v1 retail tasks, positive/negative reward checks, and Qwen tool/chat serialization. | Pinned Tau checkout and prepared Tau data. |
| `test_areal_tau2_environment.sbatch` | All 1,982 pinned AReaL Task/DB identities, gold terminal state, mutating event-log/DB-hash continuation, and an offline user-simulator Gym lifecycle. | Prepared AReaL Tau2 RL data and the Tau v3 runtime image. |
| `test_harbor_e2b_preflight.sbatch` | Native Harbor E2B lifecycle/exec/file/artifact mocks, fresh separate-verifier transfer, Miles provider selection, and graceful worker cancellation; no E2B API call. | Harbor `harbor-miles-v0.20.0` at the pinned commit, `uv sync --extra e2b`, and the Miles overlay applied. |
| `test_harbor_e2b_live.sbatch` | One real native E2B sandbox create/exec/upload/download/kill lifecycle; W&B offline and no `.env` loading. | The pinned Harbor checkout with its E2B extra and `E2B_API_KEY` exported in the submitting process. |

Submit from the repository root, for example:

```bash
source experiments/env.sh
pbs_submit --profile=cpu tests/slurm/test_nemotron_components.sbatch
```

Small deterministic checks remain under `tests/fast`; these jobs exist only for
checks that require cluster mounts, staged data, or the production container.
