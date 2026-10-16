# DAPO-Math：TIS / score centering × staleness の実験準備

## 現在の対象

GPU smokeの途中でユーザーからscore centering除外の指示があり、TISのみ60設定へ変更した。
現在の [60設定CSV](../../outputs/validation/20261017-zero-reward-study/settings.csv) と
[CLI付きJSON](../../outputs/validation/20261017-zero-reward-study/settings.json) は変更後のsweepから再生成する。
当初の120設定artifactは同じdirectory内の `initial-120-*` に保存した。以下の準備時の分析は当初の記録。
GPU検証とSC中止の経緯は [smoke記録](2026-10-17-staleness-smoke.md) を参照する。

## 対象と合意事項

基準 revision は v0.1.2 ABCI merge `d278456cd358346505dc65f18fca2623f2990f28`。
`zero_reward_truncation/train.sh` と `sweep.sh` にモデル・loss の比較を追加した。
今回は設定の準備と表示までで、実 GPU job は投入していない。

- SFT iter4000 とモデル別 filtered DAPO-Math を使用する。
- 3モデル × node 比4通り × staleness 5通り × loss 2通りの120設定。
- TIS＋score centering は v0.1.2 の標準 loss を使う。PPO clipped loss への項の追加ではない。
- LR / global batch / rollout batch はすべて共通。30B は R3 ON、ReplayDataBuffer OFF。

現在の全契約と起動方法は [zero-reward-study](../agents/zero-reward-study.md)。

## 固定値と入力

LR=`1e-6`、global batch=3072、rollout batch=192 prompts ×16 responses、1 step/rollout、300 step。
response/context=16384/32768、temperature=1、top-p=1、top-k=-1、train/rollout seed=1234/42。
first-prefill staleness、FIFO、容量係数30、生成concurrency=4096、超過groupはretry。
GRPOの標準偏差正規化とsample平均を両lossで維持し、KL/entropy係数は0。
resume/HF保存は10 step間隔（最終step・epoch境界の保存は既存の制御に従う）。

| Model | Dataset行数 | Train TP/CP/PP/EP | Rollout GPU/engine | R3 | Replay |
| --- | ---: | --- | ---: | --- | --- |
| Qwen3-4B SFT iter4000 | 13,505 | 2/1/1/1 | 1 | OFF | ON、非同期復元 |
| Qwen3-8B SFT iter4000 | 12,530 | 2/1/1/1 | 1 | OFF | ON、非同期復元 |
| Qwen3-30B-A3B SFT iter4000 | 10,911 | 1/1/1/8 | 2 | ON | OFF |

HF config、変換済みMegatron checkpointのtracker、dataset、SIFの存在を全モデルで確認した。
datasetはモデルごとの採択集合であり、同一のprompt集合ではない。

## Loss と forward

- `tis`: `policy_loss`、`--use-tis`、TIS [0,2]、PPO clip下0.2/上0.28、`skip-actor-forward-only=1`。
- `tis_sc`: `score_centering`、`--score-centering-is tis`、clip=2、候補記録128、
  `use-rollout-logprobs=1`、`skip-actor-forward-only=0`。PPO clipは不使用。

両条件でactorの事前log-prob forwardは省略される。score centeringも学習forwardのlogitsを使う。
rollout記録から得る生成時のqと、学習forwardから得る現在のπを組み合わせて、
TIS重み付きscoreからrollout分布での期待値を近似して引く。
`skip-actor-forward-only=0`はscore centeringで余分なforwardを強制する意味ではない。

候補記録は約1 KiB/token増える。完了bufferが満杯で平均応答長5000なら、
候補配列だけで `5760 groups × 16 responses × 5000 tokens × 1024 bytes` ≈472 GBになる。
これは配列の単純容量見積もりで、実測peak memoryではない。SCでは転送・保存負荷も増加する。

## Walltime の根拠

既存v0.1.1の30-step pilotを読み取り参照した。
source: `/groups/gai51740/kazuki_fujii/src/miles-v0.1.1/experiments/outputs/validation/20261016-staleness-tuning/comparison.csv`。
実行中の既存studyやsnapshotは変更していない。

30B-A3B、train:rollout=2:6、TIS、R3 ON、Replay OFF、staleness=16、
最後のcheckpointのみ保存する条件で、完了したrollout設定の定常step時間は286.2〜344.7秒。
今回のpresetと同じrollout TP2/EP1では304.4秒/step、300 stepの単純換算は25.4時間。
この時点で24時間を超える。今回の保存頻度・他のnode比・SCの追加負荷はこの換算に含まれない。

全設定を `WALL=168:00:00`、`CHAIN_JOBS=1` にした。30Bのresume chainはscriptで拒否する。
queueは既存の `R9920261300`、resourceは `rt_HF`。2026-10-17の `qstat -Qf` で
enabled/started、group `gai51740`、node上限302、明示的なwalltime上限がないことを確認した。
[ABCI公式ガイド](https://docs.abci.ai/v3/ja/job-execution/) では通常のSpot `rt_HF` 上限も168時間。
予約サービスには予約終了による打ち切りがある。今回の専用queueの終了日時は `qrstat` では取得できなかった。
168時間は余裕を持たせた指定であり、SCを含む300 stepの実測による完走保証ではない。

## 全設定と artifact

次の一覧はstub schedulerでsweepを展開し、各行の環境から実際のrecipeを読み、配置検査と
`build_command` を実行して生成した。scheduler stubの `123.pbs1` は実job IDではない。
認証情報を読まないためmanifest生成時だけW&B offlineを使った。onlineと学習CLIは同じで、
manifest内の運用設定はonlineとして記録する。

- [当初の全120設定 CSV](../../outputs/validation/20261017-zero-reward-study/initial-120-settings.csv)
- [当初の全設定・学習CLI・scheduler引数 JSON](../../outputs/validation/20261017-zero-reward-study/initial-120-settings.json)
- [当初のsweep展開記録](../../outputs/validation/20261017-zero-reward-study/initial-120-sweep-preview.txt)

下表の各行を6条件すべてで実行する。数値はCSVのID。

| Staleness | Train:rollout nodes | GPU配分 | 4B TIS | 4B TIS+SC | 8B TIS | 8B TIS+SC | 30B TIS | 30B TIS+SC |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 1:7 | 8:56 | 1 | 21 | 41 | 61 | 81 | 101 |
| 12 | 2:6 | 16:48 | 2 | 22 | 42 | 62 | 82 | 102 |
| 12 | 3:5 | 24:40 | 3 | 23 | 43 | 63 | 83 | 103 |
| 12 | 4:4 | 32:32 | 4 | 24 | 44 | 64 | 84 | 104 |
| 16 | 1:7 | 8:56 | 5 | 25 | 45 | 65 | 85 | 105 |
| 16 | 2:6 | 16:48 | 6 | 26 | 46 | 66 | 86 | 106 |
| 16 | 3:5 | 24:40 | 7 | 27 | 47 | 67 | 87 | 107 |
| 16 | 4:4 | 32:32 | 8 | 28 | 48 | 68 | 88 | 108 |
| 20 | 1:7 | 8:56 | 9 | 29 | 49 | 69 | 89 | 109 |
| 20 | 2:6 | 16:48 | 10 | 30 | 50 | 70 | 90 | 110 |
| 20 | 3:5 | 24:40 | 11 | 31 | 51 | 71 | 91 | 111 |
| 20 | 4:4 | 32:32 | 12 | 32 | 52 | 72 | 92 | 112 |
| 24 | 1:7 | 8:56 | 13 | 33 | 53 | 73 | 93 | 113 |
| 24 | 2:6 | 16:48 | 14 | 34 | 54 | 74 | 94 | 114 |
| 24 | 3:5 | 24:40 | 15 | 35 | 55 | 75 | 95 | 115 |
| 24 | 4:4 | 32:32 | 16 | 36 | 56 | 76 | 96 | 116 |
| 28 | 1:7 | 8:56 | 17 | 37 | 57 | 77 | 97 | 117 |
| 28 | 2:6 | 16:48 | 18 | 38 | 58 | 78 | 98 | 118 |
| 28 | 3:5 | 24:40 | 19 | 39 | 59 | 79 | 99 | 119 |
| 28 | 4:4 | 32:32 | 20 | 40 | 60 | 80 | 100 | 120 |

## 検証

- 120設定のmodel/loss/ratio/staleness、run名の一意性、共通LR/batch、R3/Replay、walltimeを検査。
- 120設定のrecipe CLI生成と配置検査が成功。
- 3モデル×2 lossの実引数でcoreのscore-centering設定検査が成功。
  actorの事前forwardを呼ぶ条件式が両lossでfalseになることも設定値から検査した。
  これはCPU上の設定検査であり、GPUのactor実行ではない。
- 起動script・truncation関連のCPUテストは161件成功。
- 追加で試した既存Megatron actor lifecycleテスト51件は、login nodeに `libcuda.so.1` がなく、
  Transformer Engineのimport時点でsetup error。学習処理の失敗やGPUでの検証結果とは扱わない。
- GPU学習、SCの実モデル生成、300 step完走、Replay resumeは未実施。

検証log: [script・rewardテスト](../../outputs/validation/20261017-zero-reward-study/launch-and-reward-tests.log)、
[core設定検査](../../outputs/validation/20261017-zero-reward-study/core-args-validation.log)。

## TIS pre/post 統計と常時 logging

通常TISとTIS＋SCの双方に `train/tis/pre/{mean,std,min,max}`、
`train/tis/post/{mean,std,min,max}`、`train/tis/clip_fraction` を追加した。
`staleness_math` のzero_reward/zero_loss両recipeはpolicy lagとTIS統計を常に有効にする。
全120件のCSV/JSONにも両logging設定とCLIを反映した。
定義・母集団・非有限値の扱いは [logging](../agents/logging.md#tis-の-pre--post)。

検証は公開済みv0.1.2 SIFにworkspaceをbindしてCPUで実行した。

- 初回の関連テスト：233 passed、CUDAが必要な6件はskip。
- 最終変更後のTIS/policy-lag/SC pipeline/launch検査：172 passed。
- 3モデル×2 lossの実CLIでTIS・SC検査と常時loggingを確認。
- loss/gradientの不変性、mask=0の除外、advantage=0の包含、非有限値、空partition、
  不均一なmicrobatch/rankの件数・極値・母標準偏差をCPU/mockで検査した。
- shell構文、Black、isort、Ruff、diffの空白検査に成功。

計算済みのpost weightを再利用し、追加のmodel forwardはない。
global min/maxは4 scalarを1回の追加MAX通信にまとめる（集約groupごと・stepごと）。
GPU上での通信検証・学習全体のoverhead測定は未実施。実験jobは投入していない。

検証log: [初回](../../outputs/validation/20261017-zero-reward-study/tis-metrics-tests.log)、
[最終検査](../../outputs/validation/20261017-zero-reward-study/tis-metrics-final-tests.log)、
[実CLI](../../outputs/validation/20261017-zero-reward-study/tis-core-args-validation.log)。
