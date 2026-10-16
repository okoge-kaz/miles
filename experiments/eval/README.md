# 評価

学習中 eval は OFF。保存した checkpoint の独立評価をここで管理する。
学習データの source・混合比率は `scripts/<project>/train.sh` または
`scripts/<project>/<experiment>/train.sh` に置く。

## 現在の実装

- NeMo Evaluator / Evaluator Launcher を使用する方針。
  Math500 / GPQA / LiveCodeBench / IFBench の独自変換と、Miles 向け評価 YAML は置かない。
  benchmark のデータ準備・prompt・採点は採用する評価 backend が担当する。
- NeMo Evaluator / Evaluator Launcher の version lock、依存環境の準備、評価 suite、
  ABCI PBS の評価 job、結果回収 script は未実装。
  `scripts/nemo_gym/llm_judge` は reward 用 judge の推論 server で、checkpoint 評価 job ではない。

## NeMo Gym と NeMo Evaluator

NeMo Gym の `gym eval run` は、環境・agent・verifier を使った rollout と reward の集計を行う。
Tau2 など、RL と同じ環境の held-out データでの検証に利用できる。
現在の Miles 接続の実装は学習向けであり、独立した Gym 評価 job の検証とは区別する。

NeMo Evaluator は複数 benchmark の実行・集計・比較を担当する。
採用予定は NeMo Evaluator / Evaluator Launcher。
現行 nightly には Gym 環境への接続もあるが、導入時に Evaluator の revision を固定し、
この repository が固定する Gym API との互換性を確認する。

ABCI では PBS allocation 内で local 実行する構成を候補とし、推論は SGLang の API に接続する。
必要な sandbox はタスクごとに選ぶ。Docker / Slurm 前提の Launcher 例をそのまま使わない。
model、dataset revision/split、prompt、sampling、長さ、repeat 数、scorer を評価成果物に保存する。

- [Gym evaluation](https://github.com/NVIDIA-NeMo/Gym#run-evaluation)
- [Evaluator architecture (nightly)](https://docs.nvidia.com/nemo/evaluator/nightly/architecture)
- [Evaluator / Gym integration (nightly)](https://docs.nvidia.com/nemo/evaluator/nightly/tutorials/gym-integration)
