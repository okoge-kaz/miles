# Experiment Instructions

## 参照

実装・デバッグ・文書整理では [abci-experiments skill](../.claude/skills/abci-experiments/SKILL.md) を使い、
[notes/agents](notes/agents/README.md) から該当する仕様を読む。root の `AGENTS.md` も適用する。

## Script と設定

- `scripts/<project>/` または `scripts/<project>/<experiment>/` に配置する。`sweep.sh` は条件展開、`run.sh` は scheduler、
  `train.sh` は学習設定。変換処理は `checkpoint_convert/{hf_mcore,mcore_hf}/convert.sh`。
- preset/common を読んでも、学習設定と実際の CLI 配列を `train.sh` に列挙する。
  モデル構造は公式 `scripts/models/*.py`、学習 default は実験側 `scripts/models/` を使う。
- container payload はファイルに置く。Python の先頭に module docstring を追加しない。
- 共通 script の責務は [common-layout](notes/agents/common-layout.md)、既定値は
  [training-scripts](notes/agents/training-scripts.md) に従う。

## Notes

- `notes/agents/`：現在の設定、操作手順、実装上の契約。見出しで整理し、簡潔に書く。
- `notes/logs/`：分析・実測・障害記録。revision、条件、結果、artifact を残す。
- 古い手順や編集履歴を蓄積しない。`notes/README.md` と `notes/agents/README.md` は英語、
  その他の新しい文書は日本語で書く。

## 実装・検証

- Python の追加・大幅変更は `.claude/rules/general-code-style.md` に従う。
- credential を qsub 引数、Ray runtime metadata、log に含めない。
- shell 構文、submission preview、引数生成、変更した処理を検査する。
  実 GPU 学習・resume の検証結果は CPU/mock と分けて記録する。
