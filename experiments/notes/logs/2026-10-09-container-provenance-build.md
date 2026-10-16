# SGLang provenance 有効 container の CPU build

## 条件

- 実施日：2026-10-09 JST。
- checkout：`2884a9e922` と、その後に commit する `experiments/container/` の実装。
- OCI image：`radixark/miles:v0.1.1`。
- SGLang：`880e3d2453eb7ef1738350e8c35ba2b956cc93a9`。
- Megatron-LM：`f148a32b4385b758b66a77c9c3ad1641f1295d4b`。
- `PBS_PROJECT=gai51740 APPLY_SGLANG_PROVENANCE_PATCH=1 WANDB_MODE=disabled bash experiments/setup.sh container --submit`。
- CPU 32 cores、memory 320 GB、GPU 0、walltime 上限30分。

## 結果と生成物

- build job `2529508.pbs1`：exit 0、13分40秒。
- SIF：`/groups/gai51740/kazuki_fujii/containers/miles-v0.1.1-20261009-014517-2529508.sif`。
- stable link：`/groups/gai51740/kazuki_fujii/containers/miles-v0.1.1.sif`。
- SIF SHA256：`20cc25243d175f3ac04eb6028c5bba641d545d9ca63ffdec9ff3afd945488471`。
- patch SHA256：`8497eb219bdd67fc37d09ccc56c46478cb7d28394cf53584e23a42d9cf19c974`。
- definition SHA256：`361710e56cb0e0ca1961359a01882d35b5c8668d664c6d3fc1e88547348f460a`。
- 通常 UID の mount・書き込み先・依存 import・固定 commit 検査、および公開時の checksum 照合に成功。
- SIF と同じ path に `.sha256` と `.provenance.env` を保存。
- build log：`experiments/outputs/setup/container_20261009-014449-p731834.ou`。

## 検証

- CPU 検証 job `2529616.pbs1`：exit 0、37秒。
- SIF 内で `git apply --reverse --check` に成功し、配布 patch の適用を確認。
- patch 同梱の `test_policy_weight_version.py`：12 passed。
- log と JUnit：`experiments/outputs/validation/20261009-container-provenance/{verify.ou,pytest.xml}`。
- 共通 launcher の commit 候補に対する既存の container publication、PBS helper、ABCI launch、checkpoint conversion テスト：49 passed。外部採点への依存部分を除いた commit 候補で実行。
- 今回は CPU build・CPU/mock 検証。GPU 学習・multi-node Ray・通信性能は測定していない。
