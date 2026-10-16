# DAPO-Math の TIS / staleness 比較

## 実行範囲

`scripts/staleness_math/zero_reward_truncation/sweep.sh` は、既定で次の直積60設定を展開する。
引数なしでは PBS command の preview、`--submit` を指定した場合だけ投入する。

| 軸 | 値 |
| --- | --- |
| `MODELS` | `qwen3-4b-sft qwen3-8b-sft qwen3-30b-a3b-sft` |
| loss | `tis` のみ |
| `RATIOS` | `1:7 2:6 3:5 4:4`（train:rollout の node 数） |
| `STALENESS_LEVELS` | `12 16 20 24 28` |

全モデルは SFT iter4000 と、モデル別の nonzero-std filtering 済み DAPO-Math を使う。
checkpoint と dataset path は [model-presets](model-presets.md) を参照する。
filter 条件は16回答中1〜15正解で共通だが、採択された prompt 集合はモデル間で異なる。

```bash
WANDB_MODE=disabled PBS_PROJECT=gai51740 \
  bash experiments/scripts/staleness_math/zero_reward_truncation/sweep.sh

# 1モデルに限定した preview
MODEL_PRESET=qwen3-8b-sft WANDB_MODE=disabled \
  bash experiments/scripts/staleness_math/zero_reward_truncation/sweep.sh
```

`MODELS` を明示しない場合、単数の `MODEL_PRESET` でも絞り込める。
score centering は今回のstudyから除外した。`LOSS_VARIANT` / `LOSS_VARIANTS` にTIS以外を指定すると投入前に拒否する。
run 名には model、loss、staleness、node 比を含める。別条件の checkpoint を共有しない。

## 共通設定

| 項目 | 値 |
| --- | --- |
| LR / schedule | `1e-6` / constant |
| global batch | 3072 responses |
| rollout batch | 192 prompts × 16 responses = 3072 responses |
| optimizer step | 1 / rollout、合計300 |
| optimizer | Adam、β1=0.9、β2=0.98、weight decay=0.1 |
| response / context 上限 | 16384 / 32768 tokens |
| sampling | temperature=1、top-p=1、top-k=-1 |
| seed | train=1234、rollout=42 |
| advantage | GRPO、prompt 内平均を減算し標本標準偏差＋1e-6で除算。標準偏差0なら分母1 |
| loss 集約 | 各 response の token 平均を sample 単位で平均 |
| KL / entropy 係数 | 0 / 0 |
| buffer | fully async、FIFO、容量係数30（5760 prompt groups）、concurrency=4096 |
| staleness | first-prefill、超過 group は retry |
| weight update | 1 rollout ごと、broadcast、generation pause=in_place |
| RM | deepscaler、Ray CPU workers=4 |
| 保存 | resume checkpoint=10 step ごと、最新＋50 stepごとの世代を保持。HF=10 stepごと・全世代保持 |
| 計測 | policy lagとTIS pre/post統計は常時ON、in-run eval / rollout dump / repetition 計測 OFF |
| W&B | online、project=`miles-v0.1.2-abci-math` |
| PBS | `R9920261300` / `rt_HF`、8 nodes × 8 GPU、`168:00:00`、1 allocation |
| SIF | `miles-v0.1.2.sif`（provenance patch 適用済み） |

LR、global batch、rollout batch は sweep から共通値を export し、recipe でも model preset より先に
既定値を定義する。変更する場合は全設定に共通の環境変数で指定する。
`SAVE_RETAIN_INTERVAL=50` をmodel presetより先に設定する。保持判定を0始まりの保存番号に
対応させた新SIFが必要（build job 2670420）。例えば120 step時点では
`iter_0000049`、`iter_0000099`、`iter_0000119` が残る。
全世代を残すには `SAVE_RETAIN_INTERVAL=none` を指定する。
HFは全世代を保持する。Replayとdataset cursorは保存完了後に同じ保持間隔で整理する。
モデルの削除がまだ完了していない世代は安全のため残し、次回保存時に再確認する。
修正前のSIFでは50 stepごとの保持にならない。詳細は
[GPU検証記録](../logs/2026-10-17-staleness-smoke.md#checkpoint-の保存と保持) を参照する。
policy lagとTIS統計はrecipeで常時有効にし、外部のOFF指定より優先する。
`train/tis/pre/{mean,std,min,max}`、`post/{mean,std,min,max}`、`train/tis/clip_fraction` の
母集団と集約方法は [logging](logging.md#tis-の-pre--post) を参照する。

## モデルごとの設定

| Model | Train TP / CP / PP / EP / expert TP | Rollout GPUs/engine / EP | R3 | ReplayDataBuffer |
| --- | --- | --- | --- | --- |
| 4B | 2 / 1 / 1 / 1 / 1 | 1 / 1 | OFF | ON、非同期復元 |
| 8B | 2 / 1 / 1 / 1 / 1 | 1 / 1 | OFF | ON、非同期復元 |
| 30B-A3B | 1 / 1 / 1 / 8 / 1 | 2 / 1 | ON | OFF |

共通で BF16、fused attention、max tokens/GPU=32768、full recompute（uniform、1 layer）、
SGLang memory fraction=0.70、chunked prefill=4096。
sweep は上表の R3 / Replay 設定を外部指定より優先する。30B-A3B の resume chain は拒否する。
Replay は学習待ちデータの永続化であり、同一 sample を繰り返し学習する機構ではない。

## TIS loss

`policy_loss` と `--use-tis` を使う。TIS clipは[0, 2]、PPO clipは下0.2 / 上0.28。
`skip-actor-forward-only=1`、`use-rollout-logprobs=0`、rollout候補確率の追加記録は行わない。
学習forwardは実行し、actorの事前log-prob forwardを省略する。

## Zero reward on truncation

`experiments.src.training.truncation.zero_reward` は `Sample.Status.TRUNCATED` を先に判定し、
該当 sample には正答が含まれていても0点を返す。loss mask は0にせず学習に残す。
非打ち切りは deepscaler を呼び、`</think>` 等の reasoning 終了 marker より後の
boxed answer を label と数式照合し、正答1・その他0を返す。
全16回答の報酬が等しい group は、平均減算による advantage が0になる。

## Walltime と検証記録

今回の指定は各設定168時間・1 allocation。30B は Replay を保存しないため、
24時間の複数 allocation を継ぎ足して同一の buffer 状態を再現する方法は使わない。
300 stepの時間は実際の生成長・staleness の影響で変わり、完走保証ではない。
既存計測と今回の全設定は [準備・検証記録](../logs/2026-10-17-zero-reward-study.md) を参照する。
