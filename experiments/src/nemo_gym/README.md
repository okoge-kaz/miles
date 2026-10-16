# NeMo Gym 接続

独自の採点器や tool/user loop を実装せず、固定版の NeMo Gym を呼ぶ。
NeMo Gym は SIF 内の `/opt/nemo-gym/.venv` に導入し、学習 Python と依存を分離する。
導入には `INSTALL_NEMO_GYM=1` を指定する。[container build](../../notes/agents/containers.md) を参照する。
ABCI の同じ allocation 内の CPU で動作し、Docker、sudo、外部 CPU server は不要。
Tau2 の user simulator と LLM judge は固定モデルの推論 API を別途必要とする。
CPU runtime はモデルの GPU engine を起動しない。
追加可能な task と実行要件は [task 一覧](../../notes/agents/nemo-gym-tasks.md)。

## ファイルの責務

| Directory | 担当 |
| --- | --- |
| `bridge/` | Miles の verifier から service への対応付け、HTTP payload、reward/generation hook、policy session 接続、retry・未採点・metrics |
| `runtime/` | service から upstream 実装への対応付け、依存・資産の導入、CPU process の起動・ready/終了監視、導入済み環境の検査 |
| `sandbox/` | `execution.py` の rootless Bubblewrap 境界、`worker.py` の native Python grader 呼び出し |
| `patches/` | upstream の例外伝搬、model transport、実行境界の変更 |

`bridge/payloads.py` が `metadata.verifier` を service 名と API request に変換し、
`runtime/serve.py` の `SERVICES` が service 名を NeMo Gym の実装 directory に対応付ける。
`runtime/setup.py` は専用 venv と学習時に必要な資産を準備し、`sources.lock.json` と
`requirements.lock.txt` で source と Python 依存を固定する。
`runtime/verify.py` は asset checksum と実 service の採点・接続を検査する。
`sandbox/` は native grader の実行隔離と timeout の分類を担当し、採点ルール自体は NeMo Gym に置く。

各 job・preflight・container probe はこの配置を直接参照し、旧 module path の alias は置かない。
旧 module path を含む patch を適用済みの runtime は、新しい `NEMO_GYM_ROOT` に setup する。

## 委譲先

| Task | NeMo Gym の実装 | metadata.verifier |
| --- | --- | --- |
| IF | resources_servers/instruction_following / ifbench | ifeval_g / ifbench |
| Reasoning Gym | resources_servers/reasoning_gym | reasoning_gym |
| MCQA | resources_servers/mcqa | mcqa_regex |
| QA | resources_servers/hotpotqa_qa の answer_em | qa_exact_match |
| Python code | resources_servers/code_gen | python_code |
| C++ | resources_servers/competitive_coding_challenges | nemo_gym_cpp |
| Math judge | resources_servers/math_with_judge | nemo_gym_math_judge |
| Equivalence judge | resources_servers/equivalence_llm_judge | nemo_gym_equivalence_judge |
| Rubric judge | resources_servers/multichallenge | nemo_gym_rubric_judge |
| Structured output | resources_servers/structured_outputs | json_schema |
| Calendar | resources_servers/calendar | calendar_constraints |
| Workplace | resources_servers/workplace_assistant + responses_api_agents/simple_agent | nemo_gym_workplace |
| Tau2 | responses_api_agents/tau2 | nemo_gym_tau2 |

NeMo Gym を使う source の `custom_rm_path` は `experiments.src.nemo_gym.bridge.reward.reward`。
Miles 組み込み採点を使う Math 等は `rm_type` を指定する。
Code / IF / MCQA / QA / LLM judge / Structured output 専用の `custom_rewards` wrapper は持たない。

テキストの `/verify` に渡す前に、共通の `bridge/payloads.py:final_answer` で
最後の `</think>` 以降を切り出す。開始 tag が prompt 側にある場合も同じ条件。
終了していない `<think>` が残る場合は空の回答、tag がない場合は非思考形式の回答として全文を使う。
元の Sample.response / token trace は変更しない。Tau2 等の stateful reward は native episode の結果を使う。

`allenai/IF_multi_constraints_upto5` は IFEvalG 系54制約で、`verifier=ifeval_g` を使う。
旧出力名 `ifbench.jsonl` に反して、公式 IFBench の58制約とは別物。
公式 IFBench のデータは `verifier=ifbench` で resources_servers/ifbench に接続する。
どちらも `grading_mode=fraction`、制約の合格率を reward とする。
Reasoning Gym の task scorer、Calendar の採点、Workplace の状態遷移は全て upstream が所有する。

QA は `</think>` より後にある最後の `<answer>` 内を抽出し、なければ残りの全文を使う。
末尾の EOS token を除き、native の `{"answer": "..."}` に包んで `/verify` へ送る。
label は文字列または非空文字列の list。正解 alias を順に照合し、`answer_em=1` で終了、
すべて0なら reward0。途中で採点が失敗した場合は retry 後に未採点とする。
回答の正規化と完全一致は NeMo Gym に委譲し、標準 `reward` の部分一致は使わない。
`answer_em` が欠落・不正でも標準 `reward` に fallback しない。
HTTP event 件数は alias ごとの request を数え、sample metadata には最終照合の結果を記録する。
空文字への正規化を含む EM の判定条件は upstream に従う。

## Structured output

`--services structured_outputs` を指定する。追加依存と例外伝搬 patch を導入するため、
既存 CPU runtime では `setup.sh` を再実行する。source revision 自体は変更しない。
必要なのは CPU process と Python 依存だけで、Docker や sandbox は不要。

source の `custom_rm_path` は共通の `experiments.src.nemo_gym.bridge.reward.reward` に変更する。
既存 `verifier=json_schema` / `schema_str` / `schema_type` のデータを使える。
対応するのは text response。native tool-call 出力の採点経路はまだ接続していない。

思考部分を除いた最終回答全体を native parser に渡す。Markdown fence や説明から JSON を抽出しない。
NeMo Gym は object の properties を必須化し、追加 property を禁止する strict schema を適用する。
`definitions` を `$defs` に補う旧補正も廃止。これらは学習条件の変更として扱う。
不正な生成形式・schema に反する回答は native reward0。
不正な入力 schema、model API / HTTP エラー、採点器の実行例外は未採点とし、理由を記録する。
`parse_error` / `validation_error` は正当な不正解の `scoring_reason` として保存する。

## CPU runtime の用意

```bash
export NEMO_GYM_ROOT=/path/to/writable/nemo-gym
bash experiments/scripts/nemo_gym/setup.sh
```

`uv` と git が必要。`runtime/sources.lock.json` の NeMo Gym / Tau / verifiable-instructions と
Python 3.13.14 を専用 venv に導入する。IFBench source pin は NeMo Gym 側が所有する。
NLTK、spaCy、Tau の DB も setup 時に配置し、起動時の download を避ける。
Python 依存は `runtime/requirements.lock.txt` を `uv pip sync` で導入する。
source、依存 lock、patch、asset の checksum と package 一覧を `miles-runtime.json` に記録する。
0.7 系は `r0.7.0` branch の固定 commit であり、正式 v0.7.0 tag ではない。

patch は verifier の例外伝搬、判定不能な judge 出力の未採点化、Code の実行隔離、
固定 judge model への HTTP 試行数・timeout を変更する。
通常の正誤判定は upstream に任せる。起動時に patch 対象一覧・適用後 hash・source lock を照合する。

## allocation 内で起動

固定 user simulator の native 接続 JSON を用意し、GPU job の割当 node 上で実行する。
`--host` は Miles worker から到達できる当該 compute node の IP。

```bash
bash experiments/scripts/nemo_gym/serve.sh \
  --host <compute-node-ip> --port-base 30190 \
  --state-dir /shared/path/nemo-run-001 \
  --user-model-config /shared/path/user-model.json
```

必要なものだけ `--services instruction_following reasoning_gym calendar` のように選べる。
既定は instruction_following / tau2 / code_gen / mcqa。
QA 接続を個別に使う場合は `--services hotpotqa_qa` を明示する。通常の multi-env には含めない。
IFBench、Reasoning Gym、Calendar、Workplace、各 judge は明示的に選ぶ。
C++ は `--services competitive_coding_challenges --code-test-file /path/native-test-metadata.jsonl`。
LLM judge は `--services math_with_judge --judge-model-config /shared/judge-model.json`。
model JSON と公式 SGLang 起動は [固定モデルの設定](../../notes/agents/external-environments-and-judge.md)。
Tau2 / Workplace の既定 `--max-steps=200` は仮値。Workplace だけの実行にも変更できる。
全 service の ready 後に `state-dir/endpoints.json` を公開し、終了時に削除する。
各 process の log は `state-dir/<service>.log`。起動 wrapper は子 process の終了を監視する。
学習側の `NEMO_GYM_ENDPOINTS` は、そのファイルを **全 worker の container から参照できる path** にする。
manifest 内は HTTP URL のため host/container のパス変換は不要。
`scripts/multi_env/run.sh` は最初の node で Gym を開始し、ready 後に全 node の学習を起動する。
終了時に Gym も停止し、Gym が先に終了した場合は学習を失敗させる。別 CPU allocation は不要。
`NEMO_GYM_ENDPOINTS` を明示した場合は、その既存 service を使う。固定 user simulator は別途起動する。

## 対話と学習データ

`--use-session-server v2`、`--custom-generate-function-path experiments.src.nemo_gym.bridge.generate.generate`、
`--custom-agent-function-path experiments.src.nemo_gym.bridge.agent.run` を使う。
stateful source の `custom_rm_path` は `experiments.src.nemo_gym.bridge.reward.reward`。

Native row を `metadata.nemo_gym.request` に保持し `/run` へ渡す。
policy 呼び出しだけを共通 `policy_server` から Miles session に転送する。
Responses ↔ Chat 変換は NeMo Gym の `VLLMConverter` を使う。
実際の token ID / logprob / version / loss mask は Miles session server が取得する。
テキストから学習用 token を再構築しない。user LLM の推論は別の固定 model server を通る。

Workplace は単一 user request に対する複数 tool step。
Tau2 は user simulator と policy の multi-turn 対話で、公式 terminal reward を受け取る。
AReaL 固有 DB snapshot の旧 JSONL は非互換。旧 `tool.jsonl` は流用しない。

## Tau2 データ

container build 中に専用 NeMo Gym venv で native データを準備する。データ生成元も固定する。
通常は `scripts/multi_env/prepare/run.sh` で、image 内の原形式 JSONL から Miles train/test を生成する。
以下は image 外で新しい runtime を準備する際の upstream 操作例であり、read-only SIF 内では実行しない。

```bash
cd "$NEMO_GYM_ROOT"
NEMO_GYM_TAU2_BENCH_DATA_REF=ce4013b0afe03c873488878b72851414f92f458b \
  .venv/bin/python benchmarks/tau2/prepare.py
```

生成された原形式の JSONL を `experiments.src.datasets.tau2.prepare` に `--input` で渡す。
`--train-output /path/tau2-train.jsonl --validation-output /path/tau2-test.jsonl` を指定する。
この変換は pinned Tau runtime の公式 train/test task ID で分離し、重複と不明 ID を拒否する。
標準3領域の train は30/74/74の178件、test は20/40/40の100件で、base は合計278件。
Telecom には別に full 2,285件があるが、現在の prepare/学習 recipe には含めない。
train の178件は task 定義数であり、固定された対話 trajectory の件数ではない。
NeMo Gym が除去する NL_ASSERTION 採点は使わないため、外部 judge を含む benchmark score と同一ではない。
旧 Tau v3 evaluator は廃止し、held-out Native rows の評価にも NeMo Gym を使う。

## Code とデータ変換

upstream Python code_gen は Ray worker / multiprocessing で native grader を呼び、
そのままではこの環境で必要な filesystem/network 隔離を提供しない。
Code は NeMo Gym の問題処理・採点条件を使う。実行 boundary だけ `sandbox/execution.py` で
rootless Bubblewrap に接続する。ネットワークを分離し、request ごとの作業 directory のみ書き込み可能。
Python は upstream の LiveCodeBench grader、C++ は upstream CCC の compile/run script と subtask 集計。
C++ の native test metadata が必要で、Python 用 stdin tests を言語名だけ変えて流用するものではない。
`g++` / `ld` / `as` / `timeout` と Bubblewrap / util-linux を実行 host に用意する。
Docker、NeMo Skills server、E2B はこの構成では起動しない。

`--code-concurrency=4`、Python test timeout `--code-timeout=10`、memory4 GiB、出力16 MiB が既定。
C++ のコンパイル・実行時間管理は native CCC の設定と test script に従う。
外側 deadline と test が報告する time limit は未採点、通常の誤答・compile error は native reward0。
データに含む compile/run script は信頼済みの公式 test metadata を使う。

現在の学習用 Code / MCQA は [dataset conversion](../datasets/README.md) の変換を使う。
未使用の native judge / C++ / structured-output 変換 CLI は置かない。
これらの追加 service を使う場合は native request を保持した Miles 入力を別途用意する。

Miles が `prompt` / `label` / `metadata` を読む。NeMo Gym がこの loader を置き換えるわけではない。
正解・unit tests・rubric は prompt に混ぜず、native request を metadata に保持する。
Python の独自 `test_code` / `entry_point` harness は受け付けず、native inputs/outputs / fn_name 形式への明示的な変換が必要。
MCQA は思考を除いた最終回答に native regex と grading_mode を適用し、選択肢と正解 ID が一致すれば1、
それ以外は0。output_regex の最後の一致を優先し、失敗時は lenient_answer_colon に fallback する。
思考部分にだけ存在する Answer や Code は採点に使わない。

## エラーと再開

- `/verify` は最大3試行。HTTP・通信 timeout・mask_sample・上流 failure は未採点 `None`。
- Code の明示的な execution time limit / deadline は再試行せず未採点。失敗理由は区別する。
- `/run` の失敗は Miles session を新しくして最大3 episode。途中の token trace は学習に使わない。
- 上限後も reward 0 に置換しない。未採点がある prompt group は Miles の既存 filter が除外する。
- `metadata.nemo_gym_result` と warning に service・試行数・failure_kind を記録する。
- `nemo_gym.bridge.metrics.log_rollout` は除外分も含む process-local event 件数を記録する。
- 完了 sample の replay は利用可能。外部 session の途中状態を保存する inflight replay は未対応で拒否する。
- GPU RL の学習・収束は CPU 接続検証とは別に確認する。
