# Staleness / ReplayDataBuffer / policy lag の検証

## 対象と方法

staleness logging、staleness filter / queue ordering、ReplayDataBuffer、policy lag logging の
差分だけを HEAD からの作業コピーに適用して検証した。HF 保存間隔、rollout 時間計測、
multi-source dataset の読み込み拡張など、別用途の未commit差分は含めていない。
ReplayDataBuffer が使う source 定義の読み取り・fingerprint・cursor 保存は含める。

既存 `miles-v0.1.1-multi-env.sif` の Python 3.12 / PyTorch を CPU で使用した。
driver テストは既存 fake actor / executor で保存と次の batch 取得の順序を検証する。
GPU のないホストで SGLang dumper の任意 Megatron plugin を読み込まないよう、
テスト起動 process 内だけ `sys.modules["megatron"] = None` とした。

## 結果

- 関連 pytest: **206 passed, 10 skipped**。skip は既存生成テストの未対応 variant。
- 各 commit 対象の Python 構文・ruff、変更差分の whitespace 検査: 成功。
- `staleness_math/{zero_loss_truncation,zero_reward_truncation}` と
  `staleness_multi_env/math_code_mcqa_if` の実際の CLI 出力を確認。
  既定で `--log-policy-lag-metrics` があり、`LOG_POLICY_LAG_METRICS=0` では省略される。

## 検証した契約

- response-token version span と最初の prefill version の保持、不正な span の拒否。
- 各 staleness 定義による採用判定、provenance 欠損の拒否、priority 同値の挿入順保持。
- checkpoint の checksum・commit marker、source 変更の検出、iteration 0、同期／非同期復元。
- inflight prefix の token・logprob・mask と元の provenance の保持、残り token 予算での再生成。
- 非同期モデル保存の完了後に buffer を保存し、保存中に次の batch を取り出さないこと。
- 現在の training forward と rollout の logprob 差を使い、診断値で loss・gradient を変更しないこと。
- token 数の異なる microbatch・category を sum/count で集約し、学習 tracking に渡すこと。

GPU 分散学習、実 engine の abort/resume、W&B への実送信はこの検証には含めていない。

## Multi-env staleness recipe

`staleness_multi_env/math_code_mcqa_if` の run / train / sweep を実験設定として別commitにした。
math・code・MCQA・IF の比率は各0.25。policy lag logging は既定ON。
全3レシピの既定ON／明示OFFを検査する pytest は **6 passed**。
shell 構文と sweep のpreviewを確認した。job は投入していない。

このrecipeは別途commitする共通起動・データ混合基盤に依存する。
具体的には `run_nemo_gym_training.sh`、`sweep_training.sh`、`submit_training.sh`、
`multi_env_preflight.py` と multi-source dataset / CLI の未commit差分が必要。
recipe のcommitだけをcheckoutした状態での学習起動は未検証。
