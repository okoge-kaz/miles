# 共通 script の責務

## 環境・実行基盤

cluster、workspace、mount、cache、image の既定値は `env.sh` に置く。

| `common/` の file | 担当 |
| --- | --- |
| `pbs.sh` | PBS context、node/rank、resource profile、submission |
| `singularity.sh` | mount、container 環境、MPI fanout、終了状態 |
| `ray_cluster.sh` | Ray 起動・join・終了 |
| `wait_for_ray_nodes.py` | Ray node 数の待機 |
| `load_job_env.sh` | allowlist 内の dotenv 読み込み |
| `dotenv.disabled` | container の `.env` を無効化する mount 用 file |
| `job_env.sh` | qsub に渡す環境変数の allowlist |

## 学習設定・検証

| File | 担当 |
| --- | --- |
| `training_config.sh` | 共通 fallback、export 変数名、未対応設定の拒否 |
| `training.py` | 設定検証、recipe 引数の取得、公式 model 引数、Ray submission |
| `dataset_args.py` | single/mixture の入力指定と path の検証 |
| `placement.sh` | GPU 割当、TP/CP/PP/EP/DP、batch 整合性 |
| `run_identity.sh` | run 名と checkpoint path |
| `clean_checkpoint.sh` | 指定 run の削除と path 検証 |

モデル固有の設定は `scripts/models/`、dataset は `scripts/datasets/{train,eval}/`。
各 `train.sh` に学習設定と実際の CLI 配列を列挙する。

## 起動

| File | 担当 |
| --- | --- |
| `sweep_training.sh` | 条件展開、run 命名、job chain |
| `run_training.sh` | host 側の設定・asset 検証、NCCL、全 node 起動 |
| `train.sh` | container 側の recipe 読み込み、preflight、Ray、driver |

## 変更時の依存関係

- `job_env.sh` は `training_config.sh` の `TRAINING_ENV_NAMES` を使う。
- Ray worker に必要な変数は `training.py` の runtime env にも追加する。
- `placement.sh` は allocation と学習 topology の両方を参照する。
- judge API key は環境変数に加えず、共有 owner-only file を使う。
