# Logging

`--log-category-staleness` (`LOG_CATEGORY_STALENESS=1`) で有効化する。
provenance には patch 済み SIF が必要。

## CPU 採点 worker

`RM_NUM_WORKERS` を `--rm-num-workers` に渡す。Math・multi-env recipe の既定は4、core CLI の既定は0。
0は従来のinline採点、正の値はRolloutExecutorごとの共通Ray CPU actor数。
deepscaler・gemma_math・dapo・math・f1・gpqa・ifbenchを扱い、remote/custom rewardとrandom系は従来経路を使う。
同時RPC数をworker数に制限し、sampleのcancel後も採点完了まで使用枠を保持する。

| Metric | 範囲・意味 |
| --- | --- |
| `rollout/reward/completed_samples` | 前回の metric 回収後に共通採点 executor で完了したbuilt-in採点の回答数 |
| `perf/reward_queue_time_mean` | 完了した採点が worker 使用枠を待った平均秒数 |
| `perf/reward_rpc_time_mean` | worker RPC の開始から結果を event loop が受け取るまでの平均秒数 |
| `perf/reward_compute_time_mean` | worker 内で既存の採点関数を実行した平均秒数 |

採点 metrics の母集団は選択された training batch ではなく、その期間に完了した採点。
共通poolを使う複数scorerの完了分を合算する。IFBenchの初回importも計算/RPC時間に含む。

## Event loop の遅延

`--log-rollout-event-loop`（Math recipeでは `LOG_ROLLOUT_EVENT_LOOP=1`）で有効化し、既定はOFF。
100 ms sleepの予定時刻から再開までの超過秒数を観測し、直近最大8,192件から
`perf/rollout_event_loop_lag_{mean,p95,max}` をfully asyncのmetric回収時に出す。
`RM_NUM_WORKERS=0` でも計測でき、この場合の採点queue時間は0、RPC時間はinline実行時間となる。
遅延には採点・JSON処理・CPU競合・GC等が含まれ、原因を特定する指標ではない。

## Batch 取得後の処理

`--log-rollout-postprocess` で、`perf/rollout_postprocess/<stage>/` に
`wall_time`（経過秒）、`cpu_time`（実行threadのCPU秒）、`gc_time`（同threadの循環GC秒）、
`gc_collections`（同threadのGC回数）を出す。CPU/GC時間はwall時間の内数で、加算しない。
同期区間のみを測り、GCを強制実行・無効化せず、callbackは区間終了時に解除する。
参照countによるオブジェクト解放はCPU時間に含まれるが、GC callbackの時間には含まれない。

| Stage | 範囲 |
| --- | --- |
| `debug_save` | debug rollout保存。保存無効時はほぼ空の処理 |
| `category_metrics` | category/staleness集計。有効な場合のみ |
| `rollout_logging` | 既存rollout metricの集計とtrackingへの送信 |
| `sample_conversion` | Sampleから学習用dictへの変換、reward後処理、loss mask等の作成 |
| `sample_release` | 元のSampleリストを指すローカル参照の解放。他の参照が残れば実体は解放されない |
| `dp_schedule` | DP/microbatch割付。legacy splitではDP partitionの作成 |
| `shard_packaging` | rankごとのdict作成と事前計算したscheduleの添付 |
| `object_store_put` | object storeへの格納。delay split時は分割前dataを格納 |
| `diagnostic_logging` | 追加診断metric自身のtracking送信。自己計測のためconsoleのみ |

`perf/rollout_postprocess_time` は `_get_rollout_data` 完了から格納後の診断送信直前まで。
`perf/rollout_get_time` はget呼び出しの本体開始から同時点までで、取得待ちも含む。
診断metricは同じrollout stepに別のrecordとして送信し、既存reward/取得metricを上書きしない。
consoleの `postprocess <id>:` には診断送信自身の時間も含める。

`train_async.py` は同フラグでdriverの `get.remote` 待ちを
`rollout_delivery <id>: {'perf/rollout_rpc_time': ...}` としてconsoleへ出す。
get本体時間と診断送信時間を差し引いた残りにはRayでの呼び出し待ち・返却・driver再開等が含まれ、
単純なネットワーク転送時間とは解釈しない。

## Category balance

`rollout/category/<category>/` は選択された batch を集計する。

| Metric | 母集団・意味 |
| --- | --- |
| `group_fraction` | prompt group の割合 |
| `sample_fraction` | sibling を含む sample の割合 |
| `response_token_fraction` | response 区間 token の割合 |
| `response_length/{count,mean,variance,std,p50,p90,p99,max,frac_zero}` | sample 単位の長さ |
| `truncated_fraction` | TRUNCATED sample の割合 |
| `target_group_fraction` | source category と epoch の整数 quota から計算した投入比率 |
| `group_fraction_delta` | 実際の group 比率 − target |
| `rollout/category/balance/total_variation` | category ごとの絶対比率差の和 / 2 |

category は `metadata.category` → `task_source` → `extra_env_info.task_source` →
`agent_ref.name` → `task_name` → `unknown`。key は percent-encode する。
明示 category の非出現 batch は count/fraction=0、未観測の平均は省略する。
response token 総数 0 の token fraction は省略する。row ごとに category が変わる source の target は推定しない。

## Version lag

`rollout/staleness/{total,category/<category>}/` に [各定義](staleness.md) の
mean、母分散、std、p50/p90/p99、max、整数 histogram（0–40、>=41）を出す。
group 単位の coverage と欠損数、response/loss-enabled token 加重 lag も記録する。
欠損を lag 0 にしない。

response 集計は過去の prompt/prefill span を除く。loss-enabled は mask 0 と `remove_sample` を除く。
training 側はさらに TIS/OPSM rejection mask を反映する。
multimodal span の対応を保証できない場合は欠損にする。

## Logprob 差 δ

`δ = log π_train − log μ_inf`。
π は今回の training forward、μ は生成時に engine が返した token logprob。

`train/policy_lag/{total,category/<category>}/{all_response,loss_enabled,truncated,non_truncated}/`
に `delta_mean`、`delta_abs_mean`、`delta_sq_mean`、`delta_std`、token count、有限値 coverage を出す。
DP/CP/microbatch の sum/count/二乗和を集約してから平均を求める。
非有限値を含む母集団の平均は NaN、mask 外の値は loss-enabled に含めない。
診断値は detach し、loss/gradient を変更しない。

δ は KL ではない。現在の通常 sampler は temperature 後、top-p/top-k 再正規化前の logprob を返す。
original-logprob、true-on-policy、sampling-support replay を変更した場合は π/μ の分布を確認する。


`--log-policy-lag-metrics` (`LOG_POLICY_LAG_METRICS=1`) で有効化する。
`staleness_math` と `staleness_multi_env` の recipe は既定で有効。
`LOG_POLICY_LAG_METRICS=0` で無効化できる。core CLI の既定は無効。
