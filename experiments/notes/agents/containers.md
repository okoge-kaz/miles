# Container

## 依存と設定

`env.sh` は `release-lock.json` から OCI image と依存 revision を読む。
build 時に SGLang / Megatron-LM の commit を照合する。

| 設定 | 用途 |
| --- | --- |
| `PBS_PROJECT` | PBS 投入時の ABCI group。`qsub -P` に渡す |
| `DOCKER_IMAGE` | OCI image。内容を固定する場合は digest を指定 |
| `CONTAINER_IMAGE` | 実行に使う SIF。既定 `$CONTAINER_DIR/miles-v0.1.1.sif` |
| `IMPORT_OUTPUT_IMAGE` | build の出力 SIF |
| `SINGULARITY_LINK` | 検証後に更新する stable symlink |
| `APPLY_SGLANG_PROVENANCE_PATCH=1` | 固定 SGLang pin に provenance patch を適用。省略時は `0` |

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

### 別の SGLang checkout への適用

`release-lock.json` の SGLang commit と一致する clean な checkout で実行する。

```bash
git -C /path/to/sglang apply --check /path/to/miles/experiments/container/sglang-v0.1.1-provenance.patch
git -C /path/to/sglang apply /path/to/miles/experiments/container/sglang-v0.1.1-provenance.patch
```
