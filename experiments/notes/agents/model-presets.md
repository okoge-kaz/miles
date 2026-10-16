# モデル設定

## Preset の場所

| Model | 学習 `scripts/models/` | 変換 `scripts/checkpoint_convert/models/` | 公式 `MODEL_PROFILE` |
| --- | --- | --- | --- |
| Qwen3-4B | `qwen3-4b.sh` | `qwen3_4b.sh` | `qwen3-4B` |
| Qwen3-8B | `qwen3-8b.sh` | `qwen3_8b.sh` | `qwen3-8B` |
| Qwen3-30B-A3B | `qwen3-30b-a3b.sh` | `qwen3_30b_a3b.sh` | `qwen3-30B-A3B` |
| Qwen3-4B SFT | `qwen3-4b-sft.sh` | `qwen3_4b.sh` | `qwen3-4B` |
| Qwen3-8B SFT | `qwen3-8b-sft.sh` | `qwen3_8b.sh` | `qwen3-8B` |
| Qwen3-30B-A3B SFT | `qwen3-30b-a3b-sft.sh` | `qwen3_30b_a3b.sh` | `qwen3-30B-A3B` |
| Nemotron3.5 Lightning | `nemotron3.5-lightning.sh` | `nemotron3_5_lightning.sh` | `nemotron-3-nano-30b-a3b` |
| Qwen3.5 35B-A3B | `qwen3.5-35b-a3b.sh` | `qwen3_5_35b_a3b.sh` | `qwen3.5-35B-A3B` |

構造引数は公式 `scripts/models/*.py`、学習 default は実験 preset に置く。
モデル固有 SFT path は preset に置き、`env.sh` は checkpoint root のみ扱う。

## 学習モデルの選択

`MODEL_PRESET` は拡張子なしのファイル名。個別の環境変数は preset の default より優先する。
Qwen3 は Base SFT iter4000、Nemotron3.5 / Qwen3.5 は公開 post-trained model が既定。
`-sft` preset は配置済みの SFT iter4000 を参照する。旧名の preset は従来の checkpoint default を保持する。
`HF_MODEL_NAME`（`/ckpt/hf` からの相対 path）と `MODEL_NAME`（Megatron checkpoint 名）は個別に上書きできる。
既存 run の再開は、同じ `HF_MODEL_NAME`・`MODEL_NAME`・`CKPT_PATH` を使う。

### Math の SFT preset

`staleness_math/{zero_reward_truncation,zero_loss_truncation}/train.sh` は
下表の `MODEL_PRESET` を明示指定する。
未指定・空文字・対応外の名前は `exit 2` で終了し、4B への fallback は行わない。
両 recipe とも、`MODEL_PRESET` に応じて次の dataset を選ぶ。明示した `DATASET_PRESET` は優先する。

| `MODEL_PRESET` | `MODEL_NAME` | 既定 `DATASET_PRESET` |
| --- | --- | --- |
| `qwen3-4b-sft` | `Qwen3-4B-SFT-iter4000` | `dapo-math-nonzero-std-qwen3-4b-sft-iter4000` |
| `qwen3-8b-sft` | `Qwen3-8B-SFT-iter4000` | `dapo-math-nonzero-std-qwen3-8b-sft-iter4000` |
| `qwen3-30b-a3b-sft` | `Qwen3-30B-A3B-SFT-iter4000` | `dapo-math-nonzero-std-qwen3-30b-a3b-sft-iter4000` |

3モデルとも `scripts/difficulty_filtering/<model>-sft/dapo-math.sh` の統合出力を使う。
16回答中1〜15正解の prompt を採択し、offline filtering では打ち切り回答の報酬を0にする。
出力 path は `/data/<DATASET_TAG>/<DATASET_TAG>.jsonl`。生成・採択条件は [difficulty-filtering.md](difficulty-filtering.md) を参照する。

モデル別の LR・TP/CP/PP/EP・R3・token 上限・SGLang 設定は `scripts/models/*-sft.sh` で変更する。
値は外部指定 → SFT preset → 既存 model preset → recipe fallback の順で優先する。
R3 と ReplayDataBuffer の組合せには [recipe の固定条件](replay-data-buffer.md#staleness_math-の設定) を適用する。
モデル構造は公式 `MODEL_PROFILE` を使い、学習 CLI 配列は各 `train.sh` が所有する。
並列数と R3 の既定値・制約は [parallelism.md](parallelism.md#math-sft-preset-の既定値) を参照する。

```bash
MODEL_PRESET=qwen3-8b-sft WANDB_MODE=disabled MILES_LAUNCH_DRY_RUN=1 \
  bash experiments/scripts/staleness_math/zero_reward_truncation/run.sh
MODEL_PRESET=qwen3-30b-a3b-sft LR=5e-7 \
  CHAIN_JOBS=1 bash experiments/scripts/staleness_math/zero_reward_truncation/sweep.sh
# sweep は preview。投入時は --submit を追加する。
```

`zero_reward_truncation/sweep.sh` の既定は3モデル × TIS / TIS＋score centeringの比較。
この sweep では LR / batch を共通化し、30B の R3 ON / Replay OFF を固定する。
全条件と loss / forward の違いは [zero-reward-study.md](zero-reward-study.md) を参照する。

filtered dataset は各モデルの offline filtering の merge 完了後に利用する。
初回学習前に、同じ SFT の HF weights と `/ckpt/megatron/<MODEL_NAME>_torch_dist` を用意する。
SFT preset の checkpoint を変換する場合は、変換 preset の旧 default を使わず入力と名前を合わせる。

```bash
source experiments/env.sh
source experiments/scripts/models/qwen3-8b-sft.sh
MODEL_VARIANT=sft HF_SFT_REF="${HF_CKPT_DIR}/${HF_MODEL_NAME}" MODEL_SFT_NAME="${MODEL_NAME}" \
  bash experiments/setup.sh hf_mcore qwen3_8b
# 変換も preview。投入時は --submit を追加する。
```

### Response / context length の暫定値

`scripts/models/*.sh` の `MAX_RESPONSE_LEN` と `MAX_CONTEXT_LEN` は、現状の実験用の仮の値。
モデル固有の最大対応長や、品質・速度・GPU memory を検証して決めた推奨値として扱わない。
agent は既定値を確定済みの実験条件として引き継がず、実験の目的・入力長・GPU 構成に合わせて
確認し、必要に応じて環境変数または recipe で明示する。
変更時は `MAX_RESPONSE_LEN <= MAX_CONTEXT_LEN` を保ち、prompt と response が収まる長さを選ぶ。

## 変換モデルの選択

両方向とも同じ変換 preset を source する。

| `MODEL_VARIANT` | 参照 |
| --- | --- |
| `base` | preset 内の HF Base URL と配置済み directory |
| `rl`（既定） | HF RL/post-trained URL と配置済み directory |
| `sft` | `HF_SFT_REF` の絶対 host path |

preset は download しない。Qwen3 の SFT default は `iter_0004000`。
Nemotron3.5 / Qwen3.5 の SFT は `HF_SFT_REF` を指定する。

```bash
MODEL_VARIANT=base bash experiments/setup.sh hf_mcore qwen3_4b
MODEL_VARIANT=sft HF_SFT_REF='/groups/<group>/<user>/checkpoints/hf/lightning-sft/iter_0000042' \
  bash experiments/setup.sh hf_mcore nemotron3_5_lightning
```

変換 backend と重みの保持範囲は [checkpoints.md](checkpoints.md)。
