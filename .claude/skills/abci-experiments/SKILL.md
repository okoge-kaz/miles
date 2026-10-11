---
name: abci-experiments
description: ABCI 用 Miles の実験 script、container、checkpoint、データ・採点・buffer・logging 拡張を実装またはデバッグするとき、experiments/notes/agents の該当仕様を参照する。実験 notes の整理にも使う。
---

# ABCI 実験の実装・デバッグ

## 参照先

作業前に [notes/agents の索引](../../../experiments/notes/agents/README.md) を開き、
変更する機能の文書を読む。作業中に対象が広がったら、その機能の文書を追加で参照する。
全 notes を一括で読む必要はない。

| 作業 | 参照する文書（`experiments/notes/agents/`） |
| --- | --- |
| recipe・引数・配置 | `training-scripts.md`、`parallelism.md` |
| PBS・Ray・container | `cluster.md`、`containers.md`、`common-layout.md` |
| 変換・保存・resume | `checkpoints.md`、`replay-data-buffer.md` |
| データ・multi-env・reward | `datasets.md`、`multi-env.md`、`data-contracts.md` |
| staleness・metric | `staleness.md`、`logging.md` |
| 外部環境・SGLang 採点 | `external-environments-and-judge.md` |

## 更新と検証

- 設定や契約を変更したら該当する agent note を更新する。実装と記載が食い違う場合は確認して直す。
- `##` / `###` で設定・手順・制約を分け、背景や注意書きを重複させない。
- branch 比較、実測、障害原因は [notes/logs](../../../experiments/notes/logs/README.md) に置く。
  参照は過去の測定や分析が必要なときに限る。編集履歴だけの log は追加しない。
- 変更した script の構文・preview と、変更した処理の既存テストを確認する。
  CPU/mock と GPU 実測の結果を区別して報告する。
