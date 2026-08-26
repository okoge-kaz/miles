# Experiment setup

`experiments/setup` contains reproducible asset staging and preparation jobs.
Runtime environments and rewards do not live here: their implementations are
under `experiments/src/environments`, dataset adapters under
`experiments/src/datasets`, and persistent cluster tests under `tests/slurm`.

## Layout

```text
experiments/setup/
  download/       reusable transfer jobs and staging orchestration
  models/         checkpoint conversion and model-only orchestration
  manifests/      declarative model and dataset inventories
  datasets/       deterministic dataset conversion and blend construction
  environments/   pinned environment assets and external dependencies
tests/slurm/       persistent cluster/container integration tests
```

### Download and staging

| File | Purpose |
| --- | --- |
| `download/download_dataset.sbatch` | Download one Hugging Face dataset and reject README-only gated downloads. |
| `download/download_model.sbatch` | Download one Hugging Face model and write a completion marker. |
| `download/download_git_dependency.sbatch` | Clone and optionally pin an external source checkout. |
| `download/stage_all.sh` | Stage the generic model and dataset manifests. |
| `download/stage_model.sh` | Download and convert one model from the model manifest. |
| `download/stage_nemotron_rl_datasets.sh` | Stage the role-annotated Nemotron dataset manifest. |
| `download/download_swe_datasets.sbatch` | Selectively fetch pinned Super SWE2, Ultra SWE, R2E-Gym, SWE-rebench V2, SWE-Gym, or SWE-bench Verified without downloading complete Nemotron blends. |
| `download/download_swe_verifiers.sbatch` | Fetch and checksum-pin the official R2E-Gym and SWE-rebench V2 parser modules. |

The old `download_assets.sbatch` was removed. It hard-coded one model and two
datasets and was fully superseded by these manifest-driven jobs.

### Models and manifests

| File | Purpose |
| --- | --- |
| `models/convert_checkpoint.sbatch` | Convert Hugging Face weights to a Megatron `torch_dist` checkpoint. |
| `models/stage_sft_checkpoints.sh` | Validate or submit serialized conversion of existing SFT checkpoints. |
| `manifests/models.txt` | Model source, Megatron type, conversion overrides, and node count. |
| `manifests/datasets.txt` | General Hugging Face dataset inventory. |
| `manifests/nemotron_rl_datasets.tsv` | Nemotron inputs annotated by training/evaluation/environment role. |
| `manifests/sft_checkpoints.txt` | Existing Hugging Face-format SFT checkpoints to convert. |

### Dataset preparation

| File | Output or purpose |
| --- | --- |
| `datasets/prepare_nemotron_rl_math.sbatch` | Restored Nemotron Math v2 Miles JSONL. |
| `datasets/prepare_nemotron_static_components.sbatch` | Deterministic-reward Nano components as per-dataset Miles JSONL. |
| `datasets/prepare_nemotron_nano.sbatch` | Restored Nano blend partitioned by verifier availability. |
| `datasets/prepare_gpqa.sbatch` | GPQA Diamond/Main/Extended evaluation JSONL. |
| `datasets/prepare_livecodebench.sbatch` | LiveCodeBench evaluation prompts with public/private tests. |
| `datasets/prepare_competitive_code_validation.sbatch` | Sampled competitive-programming validation split. |
| `datasets/prepare_performance_transfer_blends.sbatch` | Benchmark-safe competitive-code and STEM training blends. |
| `datasets/prepare_math_code_stem_blend.sbatch` | Balanced Math/Code/STEM multi-environment blend. |
| `datasets/prepare_tool_call_env.sbatch` | Disjoint function-call-only train/evaluation splits. |
| `datasets/build_math_jsonl.py` | Legacy raw-math JSONL adapter used by the math preparation job. |
| `datasets/prepare_swe_rl.sbatch` | Split pinned SWE sources by executable schema into gold-free rows plus owner-only Harbor manifests; unsupported R2E/SWE-Gym rows remain quarantined. |

These jobs only perform deterministic conversion, validation, sampling, or
merging. New conversion logic belongs in `experiments/src/datasets`; the
explicitly marked legacy math adapter is the remaining migration target.

### Environment assets and dependencies

| File | Purpose |
| --- | --- |
| `environments/prepare_calendar_env.sbatch` | Convert Calendar v2 rows and preflight the local constraint verifier. |
| `environments/prepare_workplace_local_env.sbatch` | Convert Workplace rows against pinned resource fixtures. |
| `environments/prepare_tau_bench.sbatch` | Pin and convert Tau v1 tasks and build the Tau/Nemotron blend. |
| `environments/prepare_ifbench.sbatch` | Pin IFBench dependencies and build held-out evaluation data. |
| `environments/prepare_search_r1.sbatch` | Assemble Search-R1 index, corpus, encoder, and benchmark data. |
| `environments/prepare_ifeval_dependencies.sbatch` | Pin the IFEvalG dependency closure. |
| `environments/prepare_reasoning_gym_dependencies.sbatch` | Pin the official Reasoning Gym scorer. |
| `environments/materialize_harbor_swe_tasks.sbatch` | Materialize fresh-verifier Harbor task trees; immutable images and live R2E admission are mandatory outside explicit dry-run mode. |

The one-time `inspect_nemotron_schemas.sbatch` probe was removed after its
schema findings were encoded in the canonical adapters and regression tests.
Durable data/container checks are documented in `tests/slurm/README.md`; no
ad-hoc smoke or test launchers remain in setup.
