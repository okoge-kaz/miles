# Logging

`--log-category-staleness` (`LOG_CATEGORY_STALENESS=1`) で有効化する。
provenance には patch 済み SIF が必要。

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
