# Experiment implementation layout

ABCI 実験用のデータ変換・NeMo Gym 接続・学習拡張を置く。
独立評価の設定と対応状況は [experiments/eval](../eval/README.md) を参照する。
CPU 検証と GPU 学習の実測は [検証記録](../notes/logs/README.md) で区別する。

- `nemo_gym/bridge/`: Miles reward, generation, HTTP and policy-session connections;
- `nemo_gym/runtime/`: pinned CPU environment setup and service startup;
- `nemo_gym/sandbox/`: rootless code execution boundary for upstream graders;
- `nemo_gym/patches/`: upstream error propagation and transport/execution patches;
- `datasets/`: training source adapters, parallel conversion, and preparation CLIs;
- `training/`: training treatments implemented through official extension hooks.
- `checkpoint_convert/`: offline adapter using official Megatron Bridge weight APIs.
- `cluster.py`: container release/package/GPU validation.

Only these canonical namespaces are supported. Dataset preparation and
environment verification are imported directly from their owning packages;
there are no forwarding aliases between them.

NeMo Gym owns migrated environments and rewards. Their source configs point directly to
`experiments.src.nemo_gym.bridge.reward.reward`; there are no domain wrappers for Code,
IF, MCQA, QA, structured outputs or LLM judges. SGLang uses its official launcher through PBS shell recipes.

Search-R1 and SWE experiment implementations, preparation utilities and evaluation
entry points have been removed. Dataset converters are limited to the active Code, MCQA, IF and Tau2 sources.
DAPO uses an already compatible JSONL file.

The NeMo Gym reward entry exposes `async reward(args, sample_or_samples, **kwargs)`.
Dataset-only Responses/Chat helpers live in `datasets/common/openai_responses.py`.
Mixture source paths, ratios, and reward hooks are defined in each job's `train.sh`.
Evaluation-only benchmark adapters are not kept here. Independent evaluation is planned through
NeMo Evaluator / Launcher; environment preparation and evaluation jobs are not implemented yet.
External E2B / AgentENV execution must also be connected behind NeMo Gym, not directly to Miles.
See [reward contracts](../notes/agents/data-contracts.md).
