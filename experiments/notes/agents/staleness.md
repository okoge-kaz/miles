# Staleness と DataBuffer 順序

## Version と測定方法

基準 `current` は最後に成功した weight publication version。optimizer step 数とは異なる。
`--async-staleness-metric`（環境変数 `ASYNC_STALENESS_METRIC`）で判定を選ぶ。
core の既定は `oldest`、disaggregate recipe は `first-prefill`。

| Metric | Group の lag |
| --- | --- |
| `oldest` | current − 生成 span の最小 version |
| `newest` | current − 生成 span の最大 version |
| `token-mean` | current − token 数で加重した version 平均 |
| `submission` | current − submit 時の local version の group 内最小値 |
| `first-prefill` | current − engine の初回 prefill version の group 内最小値 |
| `ready` | current − 生成・reward 完了時の local version の group 内最小値 |

lifecycle metric は group の全 sample に値が必要。
`lag <= MAX_WEIGHT_STALENESS` を採用し、`MAX_WEIGHT_STALENESS=none` はこの上限を外す。
非既定 metric の provenance が欠けた状態で上限を指定すると停止する。

## DataBuffer の順序・容量

| 環境変数 | 値・意味 |
| --- | --- |
| `ASYNC_DATA_BUFFER_ORDER` | `fifo`（既定）/ `oldest-policy-first` |
| `ASYNC_DATA_BUFFER_CAPACITY_FACTOR` | 既定 `30`。容量は `floor(factor × rollout_batch_size)` groups |
| `ASYNC_UNUSED_SAMPLES_HANDLER` | `retry` / `drop` |
| `ASYNC_MAX_CONCURRENT_SAMPLES` | 生成 concurrency。完了 buffer 容量とは独立 |

`fifo` は completion 順。`oldest-policy-first` は最古の生成 version が小さい順、
同値は挿入順、未知 version は最後。満杯では producer が待つ。
priority は `ASYNC_STALENESS_METRIC` とは独立し、常に最古生成 version を使う。
`retry` は prompt を再投入する。

version stats は put 時に計算し、FIFO は O(1)、priority は O(log N)。
保存時は容量待ち group も含めて順序を固定する。
実装は `miles/rollout/{fully_async_data_buffer,buffer_ordering,staleness_metrics}.py`。

## Custom 定義

`CUSTOM_STALENESS_FUNCTION_PATH` → `--custom-staleness-function-path`。
関数の契約は `(args, DataBufferInput, current_version) -> lag | None`。
clock の単位と対象 token を決め、有限・非負の値を返す。
新しい定義は保存した provenance から計算し、filter・ordering・logging の母集団を明示する。

## Version の取得

`WeightUpdater` → controller → `RolloutExecutor.set_weight_version` で通知した値を worker が使う。
生成 version は SGLang response、first-prefill は [provenance patch](containers.md) から受け取る。
sample ごとの `/get_weight_version` GET は行わない。

selection の oldest/newest/token-mean は whole-trajectory span を使う。
[logging](logging.md) は最終 response token を切り出すため、multi-turn の値は異なり得る。
