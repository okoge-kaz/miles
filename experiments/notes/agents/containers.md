# Container

## 依存と設定

`env.sh` は `release-lock.json` から OCI image と依存 revision を読む。
build 時に SGLang / Megatron-LM の commit を照合する。

学習用 Python はベース image から引き継ぐ `/opt/sglang` の venv。
`python3` は `/opt/sglang/bin/python3` を指す。ABCI 側の `miles.def` はこの学習 venv を作り直さない。
`INSTALL_NEMO_GYM=1` では別に `/opt/nemo-gym/.venv`（Python 3.13.14）を作成する。
導入済み v0.1.2 SIF の build・検査結果は [統合記録](../logs/2026-10-16-v012-abci-merge.md)。

| 設定 | 用途 |
| --- | --- |
| `PBS_PROJECT` | PBS 投入時の ABCI group。`qsub -P` に渡す |
| `DOCKER_IMAGE` | OCI image。内容を固定する場合は digest を指定 |
| `CONTAINER_IMAGE` | 実行に使う SIF。既定 `$CONTAINER_DIR/miles-v0.1.2.sif` |
| `IMPORT_OUTPUT_IMAGE` | build の出力 SIF |
| `SINGULARITY_LINK` | 検証後に更新する stable symlink |
| `APPLY_SGLANG_PROVENANCE_PATCH=1` | 固定 SGLang pin に provenance patch を適用。省略時は `0` |
| `INSTALL_CODE_SANDBOX=1` | Code sandbox を追加。省略時は `0` |
| `INSTALL_NEMO_GYM=1` | NeMo Gym と CPU 採点・Tau data を独立 venv に導入。省略時は `0` |

## Build

```bash
PBS_PROJECT=gai51740 APPLY_SGLANG_PROVENANCE_PATCH=1 bash experiments/setup.sh container
# 実行する場合は --submit を追加する。
```

`container/import_image.sh` は PBS の CPU job。
node-local scratch で fakeroot build し、通常 UID の mount・依存検査後に
SIF、checksum、provenance を公開する。成功した場合だけ stable symlink を更新する。

`container/miles.def` は EFA plugin を除去し、mlx5/InfiniBand と通常 UID 用の mount 先を用意する。
patch は scratch に staging し、SHA256 と `git apply --check` を検査する。
provenance patch を有効にした build では、SGLang の patch 回帰テストも公開前に実行する。

### Multi-env image

`INSTALL_NEMO_GYM=1 INSTALL_CODE_SANDBOX=1 APPLY_SGLANG_PROVENANCE_PATCH=1` を指定し、
`IMPORT_OUTPUT_IMAGE` と `SINGULARITY_LINK` に multi-env 専用の名前を指定する。
`container/code_sandbox.requirements.json` は Code sandbox の command 契約を持つ。
通常 UID で sandbox の実行を確認する。NeMo Gym も導入する場合は、
Code / MCQA / IF の正誤採点と Tau の模擬対話まで検査してから公開する。
IFEvalG / Tau の旧 source と依存の導入は廃止し、学習 Python の package を変更しない。
NeMo Gym の採点・対話は [専用 CPU runtime](../../src/nemo_gym/README.md) に分離する。
既存 SIF に残る旧 package は利用せず、新規 build から導入を省く。

### NeMo Gym の image 内配置

同じ SIF 内に学習 venv と NeMo Gym venv を配置する。独立した CPU server は用意せず、
確保済み GPU node の CPU で Gym process を起動する。
学習側は `/opt/sglang/bin/python3`、Gym 側は `/opt/nemo-gym/.venv/bin/python` を明示して呼び、
job 全体の PATH を Gym 用に変更しない。

| 内容 | 配置 | 準備するタイミング |
| --- | --- | --- |
| NeMo Gym 対応の Python 本体 | `/opt/python/` | image build。venv の symlink が host や builder の home を指さないようにする |
| NeMo Gym checkout と patch | `/opt/nemo-gym/` | source commit を固定して build 時に配置・適用 |
| NeMo Gym / Tau / IF 等の Python package | `/opt/nemo-gym/.venv/` | build 時に install。学習 venv から独立 |
| Tau 初期DB・domain policy・task / split・参照資料 | `/opt/nemo-gym/responses_api_agents/tau2/tau2_data/` | 固定 Tau runtime source の `data/` を build 時に展開 |
| IFBench source と NLTK data / spaCy model | upstream が参照する source path と `/opt/nemo-gym/nltk_data/`、専用 venv | build 時に用意し、実行時 download を不要にする |
| Miles 学習用 JSONL / Parquet | 既存の `/data/` bind | 学習前の dataset prepare。SIF 更新なしで source / mixture を変更できる |
| Qwen3.8 等の user / judge 重み | 既存の `/ckpt/hf/` bind | 別途 download・固定。SIF には含めない |
| cache、log、endpoint manifest、episode 作業領域 | `/cache/`、job の `--state-dir`、node-local scratch | job ごとに writable 領域を割り当てる |

#### Build 処理への組込み

`src/nemo_gym/runtime/setup.py` の導入手順を `container/miles.def` の build 段階から呼び、
host 用と container 用に採点器の install ロジックを複製しない。
Python の配置先は `UV_PYTHON_INSTALL_DIR=/opt/python` に固定し、通常 UID に読み取り・実行を許可する。

- `sources.lock.json` は `r0.7.0` branch の commit を固定する。正式 `v0.7.0` tag と同一とは扱わない。
- 使用する service の依存だけを専用 venv に入れる。NeMo Gym の全環境の依存を一括 install しない。
  最初の対象は Code / MCQA / IF / Tau2。追加 task は [runtime の service 対応表](../../src/nemo_gym/runtime/serve.py)と依存を確認して選ぶ。
- `runtime/requirements.lock.txt` に Python 3.13 / Linux の依存解決結果を固定する。更新は
  `setup.py --root <writable checkout> --resolve-lock <lock path>`。通常 build は再解決せず `uv pip sync` を使う。
  `miles-runtime.json` に source・lock・patch・asset の checksum と installed package 一覧を残す。
- NLTK / spaCy / IFBench / Tau の必要 asset を build 時に事前取得する。
- 通常 UID で import、CPU service 起動、代表 task の採点、ネットワーク download が発生しないこと、
  read-only asset と writable state の分離を確認してから image を公開する。

#### Tau package・DB・学習行の違い

`tau2` の Python package は環境・tool・評価のコードを提供するが、repository の `data/` は同梱しない。
現行 setup も `ensure_tau2_data_dir()` を別途呼び、runtime DB / policy / task 等を配置している。
container build でもこのデータを含める。helper は banking_knowledge の asset の存在も確認するため、
現在使用する3領域の `db.json` だけを抜き出す構成にはしない。

固定する source は2種類ある。

- `tau_commit`：Tau runtime package と初期DB等の `data/`。`NEMO_GYM_TAU2_BENCH_REF` に渡す。
- `tau_data_commit`：NeMo Gym benchmark prepare が読む task dump の source。
  `NEMO_GYM_TAU2_BENCH_DATA_REF` に渡し、Miles の train / held-out JSONL を学習前に作る。

初期DBは image 内の読み取り専用 asset、対話で更新される状態は episode ごとの状態として扱う。
upstream の `ensure_tau2_data_dir()` はデータ配置済みでも隣接 directory に lock を作るため、
`readonly_assets.patch` で `NEMO_GYM_OFFLINE=1` の事前配置済みデータ読み込みを追加する。
不足時は download せず失敗する。DB の採点処理は変更しない。
`runtime/serve.py` は cache / results を `--state-dir` 以下へ向け、起動時に source・patch・lock を照合する。
通常 UID の `runtime/verify.py` が asset checksum、native service の起動、MCQA/IF/Code 正誤、
Code timeout の未採点化、模擬 policy/user による Tau 複数ターンを検査した場合だけ image を公開する。

現在の mixture で起動しない IFBench service には、upstream checker が `nltk.download()` を毎回呼ぶ箇所が残る。
IFBench の全面的な offline 運用は、追加 patch と制約ごとの検証が必要。現在の IF source は別の instruction_following service を使う。

user simulator のモデルはこの DB と別に起動する固定推論 server。
Qwen3.8 等の重み・GPU memory・対応 SGLang image は [user simulator の起動](../../scripts/nemo_gym/user_simulator/run.sh)で管理する。

## Mount

| Host | Container |
| --- | --- |
| checkout | `/root/miles` |
| RL dataset root | `/data` |
| pre-training / SFT roots | `/data/pre-train`、`/data/sft` |
| checkpoint roots | `/ckpt/hf`、`/ckpt/megatron`、`/ckpt/training` |
| shared cache | `/cache`、`/root/.cache` |
| `common/dotenv.disabled` | `/root/miles/.env:ro` |

実行は `--no-eval --no-home --writable-tmpfs`、home は `/cache/home`。
framework/compiler cache は共有 storage、Ray の node-local 一時ファイルは `/tmp`。

## SGLang provenance

first-prefill を使う学習 job には patch 済み SIF が必要。
recipe は `SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1` と `--sglang-weight-version 0` を渡す。
以降の version は成功した weight publication で進める。sample ごとの version GET は行わない。
beam/speculative generation はこの provenance 実装の対象外。

互換 API として `begin_weight_update(weight_version=...)` を保持し、指定された version は
成功した `end_weight_update` まで保留する。現在の Miles はこの追加引数を使わず、
end → 標準 `update_weight_version` → 生成再開の順で version を公開する。

### 別の SGLang checkout への適用

`release-lock.json` の SGLang commit と一致する clean な checkout で実行する。

```bash
git -C /path/to/sglang apply --check /path/to/miles/experiments/container/sglang-v0.1.2-provenance.patch
git -C /path/to/sglang apply /path/to/miles/experiments/container/sglang-v0.1.2-provenance.patch
```
