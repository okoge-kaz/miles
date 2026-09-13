# Lightning: durable replay buffer + rollout routing replay

## 実装範囲

`--use-replay-buffer`と`--use-rollout-routing-replay`の併用を、refactoredな
built-in single-turn生成で許可する。`rollout`と`inflight`の両buffer形式が対象。
trainer側だけのrouting replayとindexer replayは引き続き対象外。

生成済みサンプルの`rollout_routed_experts`は既存Sample codecでlosslessに保存・復元する。
ready queue、prefetch済みtraining batch、inflight内の生成完了サンプルも同じ経路を通る。
inflightの途中生成では、保存済みtokensとrollout log-probに対応するroutingを維持する。
SGLangが継続requestのprefixを再prefillして返したroutingで、旧routingを上書きしない。

T tokenのprefixはT−1行のroutingを持つ。prefixの最終tokenをforwardするのは
継続requestなので、旧T−1行と新しい応答のT−1行目以降を連結する。
新tokenがないabortでは旧routingを保持する。途中prefixや新規tokenのroutingが
欠ける場合、およびshape/dtypeが一致しない場合は明示的に失敗する。
R3の有効状態、layer数、top-k、expert数をdataset/config fingerprintに含め、
異なる構成のbufferをそのまま再開することを拒否する。

queue admission、stalenessの境界、reward、Dynamic Filteringは変更していない。
KV cacheは保存しない。再開時には保存済みtoken prefixからKV cacheを再計算する。
R3はexpertの選択をreplayするものであり、router parameterのfreezeではない。

## 4ノード検証

`tests/manual/run_oci_async_smoke.sbatch`に
`OCI_SMOKE_MODEL=nemotron-3.5-lightning`を追加した。
W&Bは`OCI_SMOKE_WANDB_MODE=online`、projectは`async-rl-nemotron-lightning`。
T:R=2:2、train TP=2/CP=1/EP=4、GBS=32（4 prompts × 8 responses）、
max staleness=8、buffer capacity=6000 groups、同時生成64 samples。
Dynamic Filtering/TIS/R3/fused actor log-probsはON、sequence filterはOFF。
response上限8192、context上限10240。通常実験のGBS3360/response16384とは異なる
保存・再開の機能検証であり、32ノード性能の検証ではない。

全体horizonを6 updateに固定し、3 updateで停止後、同じ`SMOKE_TAG`で残り3 updateを実行する。
Megatronとbufferを毎update、HFを3 updateごとに保存する。evalはrecipeどおりOFF。

確認方法:

- 回帰テストはbuffer全体、single-turn継続、queue/staleness、trainingのR3転送を含む。
- `tests/manual/oci_validation/check_replay_routing.py --capture`で、停止時buffer内の
  生成済み・途中生成サンプルのtokens、log-prob、routingのSHA256を保存する。
- 再開後の`dump/rollout_data/*.pt`と照合し、両種類のサンプルが元のprefixを保ったまま
  trainingへ渡されたことを確認する。MoE layerのexpert ID範囲とDynamic Filteringも検査する。
- `OCI_R3_VALIDATION=1`は検証専用のMegatron init hookを有効にする。
  `replay_routing_probe.py`は、全trainer rankでR3配列が実際のforward/backward再計算へ
  同一値で渡り、全microbatchが消費されたことをassertする。
  device同期を追加するため、通常実験では無効。
- gradient normが正で有限、lossが有限であることを確認する。

検証ログは`experiments/outputs/nemotron-3.5-lightning/math/train-rollout-ratio/`、
照合用データは検証checkpointの`dump/r3-validation/`と`dump/replay_routing_probe/`。

## 検証記録（2026-09-13 UTC）

- 学習image内の回帰テスト413件と追加の4ケースが成功（合計417件）。
  `replay-r3-regression-7114990.log`と`replay-r3-final-unit-7114990.log`を参照。
- allocation 7114990でupdate 0–2を実行。lossは0.26739, 0.18284, 0.13423、
  gradient normは0.49925, 0.43097, 0.34320。Megatron/bufferの0,1,2とHFの2を保存。
- 最終bufferはprepared batch 1個とinflight 8 groups（190606 response tokens）。
  生成済み34 samplesと途中生成62 samplesのrouting shape/ID範囲を検証し、prefixを記録した。
- Ray training本体は成功したが、この実行中に検証フックをlauncherへ追記したため、
  shellが更新後ファイルの誤った位置から読み直し、終了時に`--seed: command not found`
  （exit 127）が発生した。モデル・buffer・HF保存後のlauncher失敗である。
  実行中のlauncherは以降変更せず、確定したscriptを別ジョブで検証する。
- 別の4ノードjob 7115209はcheckpoint 2から再開してupdate 3–5を完了し、
  Slurm `COMPLETED / 0:0`、elapsed `00:14:43`となった。
  lossは0.16071, 0.18897, 0.14346、gradient normは0.40070, 0.35435, 0.32007。
- 保存時に生成済みだった34 samples、途中生成だった22 samplesが、再開後に
  実際に完了したoptimizer updateで使用された。旧prefixのtokens、log-prob、
  routingが全て保存時と完全一致した。残りの途中生成samplesはこの6 updateでの
  学習済み判定には含めない（Dynamic Filteringまたはbuffer内の残り）。
- 再開後の全3 updateで、全8 trainer rank × 23 MoE streamsについて、
  読み込んだroutingがforwardとbackward再計算に同一値で渡り、全microbatchで
  消費された。全6 updateでDynamic Filtering後のgroupが8 responsesかつreward
  `{0, 1}`を含むこと、lossが有限、gradient normが正で有限であることも確認した。
- HF export 2と5は`.complete`と全6243 tensorsのinventoryが整合し、MTPを含まない。
  16 expert matricesの各64×64要素を抽出して比較すると、初期SFT→2で5292要素、
  初期SFT→5で7742要素、2→5で5569要素が変化し、全抽出値が有限だった。
- 元の32ノードjob 7114153は実装開始時に一時holdした後、Slurm上で
  `CANCELLED by 158580`となった。自動で再投入していない。
  依存していたコンテナ移動job 7114760は成功し、canonical pathは通常ファイルとなった。

検証checkpointは
`/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/training/math/dapo-math-17k/Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000/lightning-r3-inflight-20260913-7114990`。
照合結果は同ディレクトリの`dump/r3-validation/verification.json`。
再開ログは`lightning-r3-resume-7115209.log`。
W&Bは[初回](https://wandb.ai/ai-horizons/async-rl-nemotron-lightning/runs/gv2oo3z7)と
[再開](https://wandb.ai/ai-horizons/async-rl-nemotron-lightning/runs/ev47izks)。

## Replay bufferの容量と保存時間

上記4ノード・GBS32検証の実測。MB/GBは十進表記で、モデル・optimizer・HFの
checkpoint容量と保存時間は含まない。

| update | buffer bytes | capture秒 | write秒 | 合計秒 |
|---|---:|---:|---:|---:|
| 0 | 511432400 | 1.328 | 0.823 | 2.151 |
| 1 | 318192295 | 0.708 | 0.588 | 1.296 |
| 2 | 474416871 | 1.201 | 0.833 | 2.034 |
| 3 | 617434656 | 1.708 | 0.964 | 2.672 |
| 4 | 513563395 | 0.888 | 0.973 | 1.861 |
| 5 | 632417041 | 0.854 | 1.068 | 1.922 |

checkpoint 2のbuffer復元はread 0.656秒、restore 0.411秒、合計1.068秒。
capture後、ディスクwrite完了前に生成を再開するため、上表の合計を全engineの
停止時間とは扱わない。R3 OFFとの同条件比較は行っていないので、R3追加分の
step時間やthroughput低下率は未計測。

保存されたSample codec内のrouting ndarrayを集計すると、checkpoint 2は
96配列・468657696 bytes（buffer全体の98.786%）、checkpoint 5は
136配列・625025856 bytes（98.831%）だった。現在のrouting表現は
`[total_tokens - 1, 52 layers, 6 experts]`のint32で、1行あたり
`52 × 6 × 4 = 1248 bytes`。実際のMoEは23層だが、保存時は全52層分を保持する。
常駐routingとsnapshotのコピーはCPU memoryにも影響する。

容量6000 groupsが8 samples/group、平均総系列長8192 tokensで満杯になると、
routing配列だけで`6000 × 8 × (8192 - 1) × 1248 = 490673664000 bytes`
（約491 GB）になる。これは容量設定からの試算であり、実際のqueue占有量や
本番実測値ではない。inflight/prepared分、コピー、その他metadataは別途増える。
4ノードで保存が数秒だった結果を、そのまま本番のbuffer容量に外挿できない。
