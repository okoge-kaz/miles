# Dataset

## 入力設定

preset は `scripts/datasets/train/`。path は container から見える値を使う。
独立評価の dataset は NeMo Evaluator 側で管理する。
Math / Code / MCQA / IFのoffline difficulty filteringは [専用手順](difficulty-filtering.md)。
Qwen3-4B / 8B / 30B-A3B SFTで共通の採択条件を使う。
filtering済みデータの学習用mixture設定は [multi-env.md](multi-env.md#sft-checkpointごとの事前filtering) を参照する。

| 入力 | 環境変数 | CLI |
| --- | --- | --- |
| 単一 file | `PROMPT_DATA` | `--prompt-data` |
| 複数 file | `PROMPT_DATA_FILES`：path の JSON 配列 | `--prompt-data-files` |
| source 別設定 | `PROMPT_DATA_CONFIG`：JSON/YAML path または `base64:` JSON | `--prompt-data-config` |
| 比率 | `PROMPT_DATA_RATIOS`：数値の JSON 配列 | `--prompt-data-ratios` |
| epoch の prompt group 数 | `PROMPT_DATA_EPOCH_SIZE` | `--prompt-data-epoch-size` |

複数 source は `DATASET_PRESET=multi-env` を選ぶ。single/files/config は併用しない。
wrapper は mixture 指定時に preset の単一 path を渡さない。
mixture は global rollout dataset が必要。入力の検証は
`miles.utils.dataset_mixture.validate_prompt_arguments` が担当し、async buffer の検証とは分離する。

```bash
PROMPT_DATA_FILES='["/data/math/train.jsonl", "/data/code/train.jsonl"]'
PROMPT_DATA_RATIOS='[0.75, 0.25]'
```

各 source は公式 Dataset で読み、JSONL、Parquet、`@[start:end]` slice に対応する。
comma 区切りや glob は展開しない。

## Source config

```json
{
  "epoch_size": 1000,
  "datasets": [
    {"name": "math", "path": "/data/math/train.jsonl", "ratio": 0.75, "category": "math", "rm_type": "deepscaler"},
    {"name": "qa", "path": "/data/qa/train.jsonl", "ratio": 0.25, "category": "qa", "rm_type": "f1"}
  ]
}
```

- source・比率・hook の標準定義は各 `train.sh` 内に置く。
- file config の相対 path は config directory 基準。inline JSON は絶対 path のみ。source 名は重複不可。
- source ごとに `input_key`、`label_key`、`metadata_key`、`tool_key`、
  `generate_function_path`、`custom_rm_path` を指定できる。
- category は config → 行の `metadata.category` → source 名の順で選ぶ。
- `metadata.dataset_source` に source 名を記録する。

## Ratio・shuffle・resume

ratio 省略時は既存行数の比率で連結する。指定時は全 source に指定し、最大剰余法で
一 epoch の整数 quota を求める。既定 epoch size は filter 後の総行数。
epoch size は学習 step 数ではなく prompt group の巡回単位で、正の整数のみ受け付ける。
実験 recipe の `PROMPT_DATA_EPOCH_SIZE=12000` は明示的な既定値で、学習量から自動算出しない。
不足分は source を巡回する。ratio 0 は除外し、正の ratio で空 source なら停止する。

`--rollout-shuffle` は `rollout_seed + epoch_id` の独立 RNG で source 内と混合後を shuffle する。
cursor、epoch、sample/group ID、source 内容・ratio・schema・template の fingerprint を保存する。

ratio は epoch の投入 group 比率。完了順・filter・retry により batch 比率は変わる。
実際の割合は [logging.md](logging.md) で確認する。

## Reward と評価

通常の sample dispatch は source の `RewardSpec.custom_rm_path` が global RM より優先する。
source custom scorer を使う場合、truncation wrapper は明示的に合成する。
NeMo Gym 採点は source の `custom_rm_path=experiments.src.nemo_gym.bridge.reward.reward` を指定する。
旧 judge の global group wrapper は使わない。

学習 job の eval は固定 OFF。評価の配置と対応状況は [experiments/eval](../../eval/README.md)。
Miles の `--eval-config` と NeMo Evaluator の評価設定は別形式。
評価専用 split を training mixture に混ぜない。
