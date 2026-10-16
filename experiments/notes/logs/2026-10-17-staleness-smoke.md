# staleness_math の8 node・10 step GPU検証

## 条件と判定

基準revisionはv0.1.2 ABCI merge `d278456cd358346505dc65f18fca2623f2990f28`。
未commitの実験script・TIS pre/post統計追加を含む3145 fileを固定snapshotにコピーした。
[source manifest](../../outputs/validation/20261017-staleness-smoke/source-manifest.json) に各fileのSHA256を記録する。

各jobは8 node / 64 GPU、max weight staleness=16、10 optimizer step。
4Bはtrain:rollout=2:6、30B-A3Bは3:5。
当初は各モデルのTIS成功後にTIS＋score centeringを投入する計画だった。
その後ユーザー指示によりSCを除外し、TISの2条件を成功条件の対象とした。
4BのTIS成功後、ユーザー指定により30BのTISと並行して4BのSCを開始した。
各条件を別のcheckpoint pathとし、SFT iter4000から開始する。

LR=1e-6、global batch=3072、rollout batch=192×16、生成長上限16384、context上限32768。
その他の設定は [本実験の契約](../agents/zero-reward-study.md) に従う。
4BはR3 OFF / ReplayDataBuffer ON、30B-A3BはR3 ON / ReplayDataBuffer OFF。
TIS pre/postとpolicy lagは常時記録。HFとresume checkpointは10 step間隔の設定を維持する。
検証jobのwalltimeは各8時間、queue=`R9920261300`、resource=`rt_HF`。
SIFは `miles-v0.1.2-20261016-235248-2668746.sif` に固定する。

成功条件は10 stepの学習、loss/grad normと診断値の確認、最終保存、jobの正常終了。
SCでは候補確率の収集とSCの診断値も確認する。この短時間検証では収束や300 step完走を判定しない。

## 実行記録

| Model | Loss | Train:rollout | PBS job | 結果 |
| --- | --- | --- | --- | --- |
| 4B | TIS | 2:6 | 2669721.pbs1 | 10 step・最終保存完了、PBS exit 0 |
| 30B-A3B | TIS | 3:5 | 2669723.pbs1 | 10 step・最終保存完了、PBS exit 0 |
| 4B | TIS＋SC | 2:6 | 2670239.pbs1 | 最初のstep到達前、ユーザーのSC除外指示により停止 |
| 30B-A3B | TIS＋SC | 3:5 | 未投入 | ユーザー指示により対象外 |

[実行設定・jobとlog path](../../outputs/validation/20261017-staleness-smoke/runs.json)、
[最新状態](../../outputs/validation/20261017-staleness-smoke/status.json)。

W&Bは `okoge/miles-v0.1.2-abci-math` にonlineで記録する。
[4B TIS](https://wandb.ai/okoge/miles-v0.1.2-abci-math/runs/eom25sqf)、
[30B-A3B TIS](https://wandb.ai/okoge/miles-v0.1.2-abci-math/runs/xtqp3f9a)、
[4B TIS＋SC](https://wandb.ai/okoge/miles-v0.1.2-abci-math/runs/pufe7mjr)。
PBS stdoutが共有出力先に現れる前も、snapshotの `wandb/run-*/files/output_*.log` で
driverとRay workerの進捗を確認できる。

## Checkpoint の保存と保持

4B TISは02:01 JSTに10 stepを完了し、`iter_0000009`、`hf/9`、
`rollout/replay_0000009` とdataset stateを保存した。
resume側に16個のdistributed shard、HF側に16個のsafetensors shardと `.complete`、
Replay側に `COMMITTED.json` が存在する。02:02 JSTにPBS exit 0で終了した。
30B TISも02:33–02:35 JSTに最終保存を完了し、PBS exit 0。
`iter_0000009` に24個のdistributed shard（約427.6 GB）、optimizerのchained stateとRNGの
metadataが存在する。HFは115 shard、indexと `.complete` を確認し、cursorも保存された。
30Bは指定どおりReplay directoryなし。両モデルの完全なcheckpoint auditは検証artifactに保存した。
実際の再開学習はこの10 step検証には含めない。

現在のsnapshotは `SAVE_INTERVAL=10`、`SAVE_RETAIN_INTERVAL=100`、
`SAVE_HF=1`、`HF_SAVE_INTERVAL` 未指定（resumeと同間隔）。
ただしSIF内Megatronの保持条件は `prev_iteration % save_retain_interval == 0` であり、
Milesの0始まりの保存番号9,19,...,99とは一致しない。
したがって定期保存では前のresume世代が次回保存時に削除され、100 stepごとの保持にならない。
HFとReplayはこの削除処理の対象外。今回の10 stepでは保存が一度だけで、過去世代の削除はない。
`SAVE_RETAIN_INTERVAL=none` は引数を渡さず、Megatronの既定Noneにより全世代を残す。
ユーザー指定により、今後は最新＋50 stepごとの保持へ変更する。
staleness_mathの両recipeに50を設定し、SIFのMegatron判定を `(prev_iteration + 1) % interval` に修正した。
build job 2670420.pbs1はexit 0で完了した。新SIFは
`miles-v0.1.2-20261017-021524-2670420.sif`。固定snapshot/SIFには遡及適用しない。
固定Megatron sourceにpatchを適用して、実際の保存完了callbackの保持動作をCPU検証した。
モデル保存番号0も削除対象として正しく判定し、Noneの全量保持とsymlink保護を維持する。

## v0.1.1との性能比較

ユーザーから追加loggingによる速度低下の確認依頼があり、旧版の実測を読み取り専用で参照した。
集計scriptは `outputs/validation/20261017-staleness-smoke/compare_perf.py`、
[更新される測定値](../../outputs/validation/20261017-staleness-smoke/performance-comparison.json) に
step別timerと比較区間を保存する。GPUあたりtoken throughputは、全体のtoken throughputを
学習GPU数で除算する。平均はactor train時間で加重する。

- 4B旧版：`20261012-qwen3-4b-step-time-audit/metrics.json` のReplay ON、job 2545462。
  train:rollout=1:7、p10–90データ、標準報酬。今回の2:6、nonzero-stdデータ、zero rewardとは異なる。
- 30B旧版：`20261016-staleness-tuning` の `sm-30b-rollout-r2e1-t1c1e8-n2-1016a`、job 2654405。
  train:rollout=2:6に対し今回は3:5。モデル・batch・R3・train/rollout並列度は同じ。
  旧版でもpolicy lagを有効にしていた。

TISの両モデルは10 stepと最終保存を完了し、PBS exit 0。全10 stepでTIS pre/post・policy lag・
loss/grad normを含む73個の診断値が有限値であることを完全なPBS stdoutから確認した。
初回と最終stepを除くindex 1–8の比較は次のとおり。

| Model | Train時間 旧/新 (s) | 待ちを含むstep時間 旧/新 (s) | 学習token/s/GPU 旧/新 | 変化 |
| --- | --- | --- | --- | --- |
| 4B | 233.08 / 109.90 | 393.50 / 126.18 | 8780.37 / 8755.72 | −0.28% |
| 30B-A3B | 152.25 / 118.60 | 288.62 / 278.55 | 6335.38 / 6321.57 | −0.22% |

index 5–8に限るとtoken/s/GPUは4Bで−0.52%、30Bで−3.61%。配置、入力長、依存versionが
一致しないため、logging単独のoverheadや差がないことを証明する比較ではない。
最終checkpointの保存時間はこの比較に含めない。

## Score centering の検証中止

4B SCは02:10:16 JSTにrolloutを開始したが、02:32時点でoptimizer stepは0だった。
02:23の48 engine集計では4,370回答・21,738,661 tokensを生成済みで、生成中3、待ちqueue 0。
Ray Reporterのstack取得では4個のRewardWorkerは待機、RolloutExecutorはHTTP response bodyの受信中だった。
採点計算や学習lossが主因と断定できる証拠はなく、候補128個の応答転送・処理が疑われる。
15秒のCPU profile取得はRPC timeoutで失敗したため、割合の定量評価はできていない。
ユーザーの指示により4B SCをqdelし、30B SCも対象外とした。今回のsweepはTISのみ60設定。
診断artifactは `outputs/validation/20261017-staleness-smoke/4b-sc-{ray-actors,ray-nodes,tracebacks,sglang-metrics}.json`。

## 今回の保持・sweep変更の検査

60設定のstub scheduler展開と学習CLI生成が成功。起動・保持・Replay・executorのCPUテストは
177件成功、SC exportを必須としていた古い期待値1件を更新し、その再実行も成功した（合計178件）。
新SIF内のMegatron保存完了callbackについて、同期・非同期、0番、50間隔、最終保存、
全世代保持、symlink保護を検査した。Replay/cursorはモデル保存完了と対応するsidecarを確認し、
保存失敗時・未来世代・未完了世代・残存モデル・symlink・HFを保護するCPUテストを実施。
GPUでの実resumeと、50 step到達時のGPU保持動作は今回の10 step検証には含まない。
