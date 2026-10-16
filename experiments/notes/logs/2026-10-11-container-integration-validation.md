# Multi-env container の導入・公開検査

## 対象と既存 build

`INSTALL_NEMO_GYM=1`、`INSTALL_CODE_SANDBOX=1`、`APPLY_SGLANG_PROVENANCE_PATCH=1` を使う
container 差分を、`c4c9aaab65` に重ねる範囲で確認した。今回の commit 整理では再 build や PBS job 投入は行っていない。

既存の CPU build job は `2562709.pbs1`。公開済み image は
`/groups/gai51740/kazuki_fujii/containers/miles-v0.1.1-multi-env-gym-r070-20261011.sif`。
`miles-v0.1.1-multi-env.sif` はこの image を指す。
build log は `experiments/outputs/setup/container_20261011-163427-p2433563.ou`、
build 時の採点検査は `experiments/outputs/validation/nemo-gym-image-2562709/verification.json` にある。
これらの出力 artifact は Git 管理対象外。

## 公開済み image との照合

- 現在の `miles.def` の SHA256 は sidecar provenance と一致した。
  `23740f5c7d3cd530a667ffb0a35ba29f7027ef535445c45f9a51692939dd9951`。
- sandbox の requirements、prepare helper、verify helper は image 内のファイルと一致した。
- 通常 UID で `runtime.serve.check_install()` を実行し、source revision、依存 lock、6個の patch と適用済みファイルを照合した。
  NeMo Gym は `r0.7.0` の commit `1f8afc71d45fac14872d86fe300374afa2001e7a`、
  Python は専用 venv の 3.13.14。asset manifest の記録は1049件だが、今回この全 asset の再検査は行っていない。

既存 build 時の検査 artifact では MCQA / IF / Code が正解 `1.0`、不正解 `0.0`。
Code の時間切れは `execution_time_limit` として未採点にする検査に成功している。
Tau2 は模擬 user / policy endpoint による対話で、user 呼び出し2回、policy 呼び出し1回、reward `1.0`。
これは接続と採点経路の検査であり、実モデルの能力評価ではない。

## 今回の CPU 検査

無関係な未 commit 差分を除いた作業用 snapshot を既存 SIF に bind して検査した。

- `tests/fast/experiments/container/test_container_publication.py`：9件成功。
  base / sandbox / NeMo Gym の検査失敗時に既存 SIF と stable symlink が維持され、
  未完了の公開ファイルが残らないことを確認。PBS preview への導入フラグ引き渡しも確認した。
- `tests/fast/experiments/common/test_pbs_helpers.py`：10件成功。
- container helper と変更した test の Ruff、`import_image.sh` の `bash -n` を実施。

公開処理の test は Singularity を模擬している。実 GPU 学習、実モデルを使う Tau 対話、今回の差分からの再 build は未実施。
