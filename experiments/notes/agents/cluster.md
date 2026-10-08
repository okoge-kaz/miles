# ABCI の起動

## PBS 設定

| 用途 | Queue | Resource | 1 node の割当 | Place |
| --- | --- | --- | --- | --- |
| 学習・変換・検証 | `R9920261300` | `rt_HF` | 192 CPU、8 GPU、1 MPI process | `scatter:excl` |
| Image build | `R9920261300` | `rt_HC` | 32 CPU、1 MPI process | `scatter` |

`PBS_PROJECT` と queue/resource の設定は `env.sh`。投入には `setup.sh` / `submit_training.sh` を使う。
`scatter:excl` は各 chunk を別 host に置き、選択した vnode を専有する。

## Script の担当

| File | 担当 |
| --- | --- |
| 各 recipe の `run.sh` | PBS allocation と共通 launcher の呼び出し |
| `common/pbs.sh` | job ID、node/rank、resource profile、qsub |
| `common/singularity.sh` | container と MPI fanout |
| `common/ray_cluster.sh` | Ray 起動・join・終了 |
| `scripts/cluster.sh` / `src/cluster.py` | image revision、package、GPU、BF16 の検査 |

配置名は `disaggregate` / `colocated`。非同期方式は [学習設定](training-scripts.md) で指定する。

## MPI と Ray

- MPI は `-map-by ppr:1:node -bind-to none`。各 node に1 process を置き、CPU affinity を固定しない。
- node ごとの終了状態を集約し、payload の失敗を job に伝える。
- Ray head IP と終了 flag は共有 `outputs/.ray/` に置く。
- 全 node の join 後に学習を開始する。driver 終了時に各 node の Ray を停止する。
- dashboard は loopback に bind する。

## Log 出力

`qsub -o` で次のファイルを指定し、`#PBS -j oe` で stdout/stderr をまとめる。

| Job | 出力先 |
| --- | --- |
| 学習 | `outputs/train/<project>/<experiment>/<model>_<run>_<timestamp>-p<pid>.ou` |
| 変換 | `outputs/ckpt/<direction>/<model>_<variant>_<timestamp>-p<pid>.ou` |
| 採点 | `outputs/judge/sglang/<JUDGE_RUN_NAME>.ou` |

## 環境変数

`common/job_env.sh` の allowlist で qsub へ渡す変数を限定する。
W&B key は compute 側で解決し、採点 key は owner-only の共有ファイルから読む。
credential を scheduler の引数や Ray runtime metadata に含めない。
