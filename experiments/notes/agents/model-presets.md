# モデル設定

## Preset の場所

| Model | 学習 `scripts/models/` | 変換 `scripts/checkpoint_convert/models/` | 公式 `MODEL_PROFILE` |
| --- | --- | --- | --- |
| Qwen3-4B | `qwen3-4b.sh` | `qwen3_4b.sh` | `qwen3-4B` |
| Qwen3-8B | `qwen3-8b.sh` | `qwen3_8b.sh` | `qwen3-8B` |
| Qwen3-30B-A3B | `qwen3-30b-a3b.sh` | `qwen3_30b_a3b.sh` | `qwen3-30B-A3B` |
| Nemotron3.5 Lightning | `nemotron3.5-lightning.sh` | `nemotron3_5_lightning.sh` | `nemotron-3-nano-30b-a3b` |
| Qwen3.5 35B-A3B | `qwen3.5-35b-a3b.sh` | `qwen3_5_35b_a3b.sh` | `qwen3.5-35B-A3B` |

構造引数は公式 `scripts/models/*.py`、学習 default は実験 preset に置く。
モデル固有 SFT path は preset に置き、`env.sh` は checkpoint root のみ扱う。

## 学習モデルの選択

`MODEL_PRESET` は拡張子なしのファイル名。個別の環境変数は preset の default より優先する。
Qwen3 は Base SFT iter4000、Nemotron3.5 / Qwen3.5 は公開 post-trained model が既定。
filtered DAPO dataset は Qwen3-4B SFT 用なので、モデル変更時は dataset も選ぶ。

```bash
MODEL_PRESET=qwen3-8b DATASET_PRESET=dapo-math-17k MILES_LAUNCH_DRY_RUN=1 \
  bash experiments/scripts/staleness_math/zero_loss_truncation/run.sh
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
