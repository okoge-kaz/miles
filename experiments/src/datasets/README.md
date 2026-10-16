# Dataset preparation

現在の学習 recipe が使う Code、MCQA、IF、Tau2 を Miles の `prompt` / `label` / `metadata` に変換する。
DAPO-Math-17k は取得済みの Miles 形式 JSONL を直接使い、追加の数学変換は置かない。
データ変換は学習前の処理。採点・環境 loop は NeMo Gym、混合比率は各 job の `train.sh` が所有する。

`scripts/multi_env/prepare/prepare.sh` が直接実行するのは IF と Tau2 の準備。
Code / MCQA は変換済み JSONL を学習時に読み、元データからの再生成には下記の `convert.py` を使う。
学習 job から直接 import されないことは、未使用の変換処理であることを意味しない。

## 構成

| Path | 担当 |
| --- | --- |
| `convert.py` / `source_adapters.py` | Nemotron competitive coding と knowledge MCQA の source schema 変換 |
| `instruction_following/download.py` | AllenAI IF training の固定 revision の download |
| `instruction_following/prepare.py` | 単一 user prompt と複数制約の変換 |
| `tau2/prepare.py` | native NeMo Gym Tau2 rows を公式 train/test ID で分離 |
| `common/io.py` | JSONL / Parquet / Arrow / CSV の入力と JSONL の出力 |
| `common/openai_responses.py` | Responses 形式の prompt を Chat messages に変換 |
| `common/prepare.py` | IF 変換、source/output checksum と schema manifest |
| `common/parallel.py` | 順序を保つ process 並列変換 |

QA、Calendar、Workplace、Nano blend、Skywork、Reasoning Gym、structured output、native judge / C++ の
専用変換は現在の学習で使わないため提供しない。NeMo Gym の追加 service を使う場合は、
その時点で必要な source を確認する。一般の接続 capability と、準備済みの学習 recipe は別である。
評価は NeMo Evaluator / Launcher を使い、評価専用 dataset を学習形式へ変換しない。

## Code / MCQA

```bash
python3 -m experiments.src.datasets.convert \
  --dataset competitive-coding \
  --input /data/nemotron-rl-comp-coding/opencodereasoning_filtered_25k_train.jsonl \
  --output /path/new/code.jsonl --summary /path/new/code.conversion.json \
  --num-proc 8

python3 -m experiments.src.datasets.convert \
  --dataset knowledge-mcqa --input '/path/source/train-*.parquet' \
  --output /path/new/mcqa.jsonl --summary /path/new/mcqa.conversion.json \
  --num-proc 8
```

Code は問題文と native inputs/outputs / fn_name を保持し、独自 test_code / entry_point harness は拒否する。
MCQA は問題文、正解選択肢、選択肢 ID、output_regex を保持する。
2個以上の単一英字の選択肢と、その中の正解ラベルを要求する。公開 source にある
全選択肢が A の本文に潰れた行、選択肢外の正解、空の prompt/label は除外し、summary の skipped に数える。
本文から選択肢や正解を推定して補修しない。
元データの train split を明示し、validation を input glob に含めない。

## IF / Tau2

IF / Tau2 は `scripts/multi_env/prepare/prepare.sh` で準備する。IF は download と変換を行う。
出力の既定は `/data/multi_env/if-training.jsonl`。
既存の schema 2 manifest は checksum を照合して再利用し、異なる既存出力は上書きしない。

Tau2 の native prepare は [NeMo Gym 接続](../nemo_gym/README.md) の image build 中に行い、
`python -m experiments.src.datasets.tau2.prepare --input ... --train-output ... --validation-output ...`
で変換する。参照した公式 task ID、train/test の重複、不明 ID を検査する。
出力ごとの manifest に source commit、入力・出力 checksum、split と件数を記録する。
現行 recipe は標準3領域の train を使い、Telecom full を自動的には含めない。

## 並列変換

各変換 CLI は `--num-proc`（既定1）/ `--chunk-size`（既定128行）を受け取る。
worker は spawn で起動し、行変換を担当する。親 process は入力と JSONL serialization・出力を担当する。
入力順を保持し、先読み量を制限する。`--limit` は変換済み行数までの入力 prefix に適用する。
変換失敗は伝播し、単一の出力 file は全行成功後に公開する。複数出力の transaction は実装しない。
Tau2 の split 検査、checksum、download はこの並列化の対象外。
Tau2 の変換済み行は split ごとにメモリに保持する。

IF の CPU PBS `multi_env/prepare/run.sh` は既定8 workers、shell 直接実行は既定1。
`DATASET_NUM_PROC` / `DATASET_CHUNK_SIZE` で変更する。IF download の並列数は別設定で2 workers。
I/O が支配的な場合は process 数を増やしても高速化するとは限らない。
