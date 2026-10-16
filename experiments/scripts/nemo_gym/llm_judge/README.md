# 固定 judge モデルの提供

NeMo Gym の LLM judge が呼ぶ SGLang 推論 API を ABCI 上に立てる、任意の起動 script。
独自の reward 計算・judge prompt・回答 parser は持たない。

- `submit.sh`: PBS job の preview / 投入。
- `run.sh`: GPU allocation と Singularity の起動。
- `serve.sh`: 公式 `python3 -m sglang.launch_server` の引数。

NeMo Gym の `math_with_judge` / `equivalence_llm_judge` / `multichallenge` を使い、
固定モデルの API をローカルに用意する場合に使う。既存 API があればこの script は不要。
multi-env recipe の標準採点には LLM judge を含まず、自動では起動しない。

現在の接続は NeMo Gym resource → `judge_model` API adapter → SGLang `/v1/chat/completions`。
NeMo Gym の CPU runtime は固定モデルの GPU engine を起動しない。
起動後の接続先は `--judge-model-config` の JSON で明示する。
設定・投入例・log の場所は [固定モデルの起動](../../../notes/agents/external-environments-and-judge.md)。
