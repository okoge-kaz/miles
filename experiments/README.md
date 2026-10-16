# ABCI Experiments

## 参照先

- [実装・デバッグ](notes/agents/README.md)
- [分析・検証記録](notes/logs/README.md)
- [v0.1.2 への ABCI 統合と検証](notes/logs/2026-10-16-v012-abci-merge.md)

## 配置

```text
experiments/
  env.sh                         workspace、mount、cache、image
  common/                        PBS、container、Ray、設定検証
  container/                     miles.def、import_image.sh、SGLang patch
  scripts/
    models/                      学習 preset
    datasets/train/              学習 dataset preset
    checkpoint_convert/         共通 model preset、両方向の run/convert
    staleness_math/              truncation 実験
    staleness_multi_env/          Math/Code/MCQA/IF の staleness・配置比較
    multi_env/                   Math/Code/MCQA/IF/Tau2 混合、IF データ準備
    nemo_gym/                    CPU 採点・環境、固定 user / judge model の起動
  src/                          データ・環境・採点・学習の拡張
  eval/                         評価設定と NeMo Evaluator 接続状況
  notes/{agents,logs}/           仕様・手順 / 分析・記録
```

## 環境の準備

repository root で実行する。model/data はあらかじめ配置する。

```bash
export PBS_PROJECT=gai51740
export MILES_WORKSPACE_ROOT=/groups/gai51740/kazuki_fujii
bash experiments/setup.sh init
bash experiments/setup.sh status
APPLY_SGLANG_PROVENANCE_PATCH=1 bash experiments/setup.sh container  # preview
MODEL_VARIANT=sft bash experiments/setup.sh hf_mcore qwen3_4b       # preview
```

`setup.sh` の job は `--submit` を付けると投入する。
手順は [container](notes/agents/containers.md)、[checkpoint](notes/agents/checkpoints.md)。

## 学習

```bash
MILES_LAUNCH_DRY_RUN=1 \
  bash experiments/scripts/staleness_math/zero_loss_truncation/run.sh
RUN_NAMESPACE=truncation-study CHAIN_JOBS=1 \
  bash experiments/scripts/staleness_math/zero_loss_truncation/sweep.sh
# 投入する場合は sweep.sh に --submit を追加する。
```

model/dataset は `MODEL_PRESET` / `DATASET_PRESET` で選び、詳細は各 `train.sh` で設定する。
配置は `disaggregate` / `colocated`。W&B は既定 online、dump と学習中 eval は OFF。
認証情報は compute 側で解決する。

[Multi-env](notes/agents/multi-env.md)、[学習 script](notes/agents/training-scripts.md)、
[保存・resume](notes/agents/checkpoints.md)、[外部採点](notes/agents/external-environments-and-judge.md) を参照する。
