# TIS・truncation・staleness 解析

Snapshot: 2026-09-06。ローカル `metrics.jsonl` を run 内の記録順で latest-write-wins にし、
Train:Rollout=1:7、max response length=16,384、step<=300 に限定した。太線は trailing-5 mean。

## 結論

- `zero-loss-on-truncated` は、この設定で S=16/20 の catastrophic collapse を止めるには十分だった。
  ただし、これだけで『staleness の問題を一般に解決した』または『truncated sample が唯一の原因』とは言えない。
- collapse arm では `train/tis_abs` の trailing-5 mean が 0.05 を越えるのが reward collapse より
  22--41 update 早い。したがって policy/rollout mismatch は単なる末期指標ではなく、collapse loop の一部である。
- 一方、truncation 50% 超は reward collapse 後に現れる。末期の truncated-objective dominance は
  collapse の増幅・固定化を示すが、発火点を単独では同定しない。観測される順序は
  `TIS mismatch 上昇 -> 短文相 -> reward collapse -> overlong/truncated 相` である。
- staleness-aware loss は独立な反証ではない。高 staleness では truncated gradient を 4--12% 程度まで
  落とす soft zero-loss として働き、同じ経路を遮断している。
- 最も妥当な解釈は、staleness 自体が一定値を越えて即座に壊すのではなく、
  長文・truncation・group-normalized negative advantage が作る遅延フィードバックを staleness が増幅する、である。

## Collapse の時間順序

| treatment | S | |TIS-1|>=0.05 | reward<0.30 | lead | mean len<=2K | trunc>=50% | collapse-window TIS abs | clip | token ESS |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| none | 16 | 125 | 150 | 25 | 145 | 180 | 0.211 | 1.29% | 0.645 |
| zero reward | 16 | 237 | 259 | 22 | 254 | 287 | 0.148 | 1.04% | 0.718 |
| zero reward | 20 | 119 | 147 | 28 | 143 | 181 | 0.278 | 1.47% | 0.569 |
| zero reward | 24 | 233 | 274 | 41 | 270 | 287 | 0.875 | 0.49% | 0.114 |
| zero reward | 28 | 135 | 169 | 34 | 167 | 204 | 0.433 | 1.71% | 0.305 |

`tis_clipfrac` は `[0, 2]` の上限 2 を越えた token だけを数える。ratio<1 は clip されないため、
S=24 のように mean TIS が下側へ崩れる局面では clip fraction だけを見ると mismatch を過小評価する。

## Collapse しなかった観測窓

| treatment | S | last step | realized stale | reward | trunc | TIS abs | TIS clip | policy KL | token ESS |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| zero-reward | 12 | 300 | 11.80 | 0.868 | 3.5% | 0.017 | 0.0022% | 0.0012 | 0.9977 |
| none | 8 | 300 | 7.66 | 0.864 | 3.3% | 0.015 | 0.0009% | 0.0009 | 0.9981 |
| zero-loss | 8 | 300 | 7.77 | 0.836 | 8.0% | 0.015 | 0.0011% | 0.0009 | 0.9981 |
| zero-loss | 16 | 300 | 15.78 | 0.829 | 8.0% | 0.017 | 0.0011% | 0.0011 | 0.9978 |
| zero-loss | 20 | 300 | 19.72 | 0.795 | 15.1% | 0.019 | 0.0022% | 0.0014 | 0.9972 |
| staleness-aware | 12 | 300 | 11.57 | 0.828 | 10.1% | 0.017 | 0.0012% | 0.0012 | 0.9977 |
| staleness-aware | 16 | 290 | 15.76 | 0.848 | 8.6% | 0.020 | 0.0030% | 0.0016 | 0.9968 |
| staleness-aware | 20 | 264 | 19.64 | 0.827 | 5.6% | 0.023 | 0.0044% | 0.0019 | 0.9963 |
| staleness-aware | 24 | 252 | 23.87 | 0.844 | 5.9% | 0.022 | 0.0055% | 0.0018 | 0.9964 |
| staleness-aware | 28 | 224 | 27.48 | 0.764 | 18.9% | 0.022 | 0.0049% | 0.0018 | 0.9965 |

同じ realized staleness 16--24 でも、安定 arm の TIS abs は約0.017--0.022、clip率は約0.001--0.006%で、
collapse-window の TIS abs 0.15--0.87、clip率0.5--1.7%より桁違いに小さい。
よって lag は mismatch の十分統計ではない。実際の mismatch は lag 中に積み上がった update の大きさと方向にも依存する。

## Staleness-aware loss は何をしているか

| S | truncated token share | mean scale | truncated objective pre -> post | total objective retained | late reward |
|--:|--:|--:|--:|--:|--:|
| 12 | 19.8% | 0.117 | 40.4% -> 7.4% | 64.4% | 0.828 |
| 16 | 18.0% | 0.078 | 40.6% -> 5.1% | 62.6% | 0.848 |
| 20 | 13.3% | 0.060 | 28.3% -> 2.3% | 73.4% | 0.827 |
| 24 | 14.4% | 0.048 | 32.0% -> 2.2% | 69.5% | 0.844 |
| 28 | 35.0% | 0.041 | 50.6% -> 4.0% | 51.5% | 0.764 |

S=20 aware は truncated objective share を late-20 で 28.3% から 2.3% へ落としている。
これは『高 staleness のまま通常の truncated loss を保持しても安全』という結果ではなく、
その loss をほぼ消すと安全、という結果である。

## Age-bin で見えるもの

- zero-reward S=20 の transition window (step 128--147) では、staleness=20 が pre-loss token mass の 90.4% を占め、mean absolute log-ratio は 0.514、upper-clip率は 1.47% だった。
- zero-loss S=20 late では、staleness=20 bin の response token の 35.5% が initial loss mask で消えている。全 age 合計でも 29.4% の response-token mass が masked である。
- したがって zero-loss の低い global TIS/clip は、安定化の結果に加えて、問題になり得る truncated token を
  TIS 集計の母集団から外した censoring を含む。zero-loss の TIS 図だけでは masked token の counterfactual mismatch は分からない。

## No-treatment が zero-reward と似る理由

no-treatment S=16 の collapse 前 step 1--149 では、truncated sample 31,629 件の平均 raw reward は 0.000854、reward=1 は 27 件 (0.085%) だけだった。
したがって verifier に通常どおり truncated text を採点させても、ほぼ全件が自然に0点になる。
このデータでは no-treatment と explicit zero-reward は実効的に同じ負 feedback を持つため、
両者の類似挙動は truncation-gradient 仮説への反証にはならない。

## なぜ『staleness 耐性』より別の現象が見えるのか

1. **staleness は delay、truncation は高 gain の非線形 feedback**: truncated response は長いため token mass が大きく、
   binary reward と group normalization により強い負 advantage を持ちやすい。その更新が lag 分遅れて到着する。
2. **lag x policy velocity が mismatch**: policy が緩やかなら realized lag 20 でも ratio は狭い。長さ方策が動き始めると、
   同じ lag でも queued trajectory が急速に off-policy になり、TIS/KL が跳ねる。
3. **TIS は stale advantage を直さない**: token ratio を `[0,2]` にするだけで、古い policy が生成した trajectory の
   reward/advantage、state distribution、length censoring の意味は current policy 向けに再計算されない。
4. **one-sided cap**: 下限0は実質 no-op なので、ratio<<1 の token は小さくなるだけで除外されない。
   upper-clip率が小さくても大きな directional mismatch や stale negative feedback は残り得る。
5. **二つの attractor**: 実測は最初に短文・低 reward 相へ移り、その後 max-length 相へ反跳する。
   これは単純な『overlong率が徐々に上がって collapse』より、遅延系の振動・overshoot に近い。

## VCPO など既報との関係

- [VCPO paper](https://arxiv.org/abs/2602.17616) が論じる heavy-tailed importance ratio / high-variance 問題と矛盾しない。
  今回も collapse arm では reward 崩壊より22--41 update 先に TIS mismatch が急増している。
- 一方、[official MATH recipe](https://github.com/mit-han-lab/vcpo/blob/main/recipe/fully_async_policy/shell/vcpo/math/synchronous.sh) は
  max response 2K、sequence-level IS、threshold 8 であり、今回の16K、token-level TIS `[0,2]`、truncation ablation とは estimator と censoring regime が異なる。
- したがって VCPO の結果を無効とはいえないが、そこでの configured lag を今回の realized lag と同一視して、
  Long-CoT の length/truncation feedback まで解決済みとみなす外的妥当性もない。二つは異なる failure channel を観測し得る。

## 何がまだ未証明か

- 各 treatment は同じ生成 batch に対する paired counterfactual ではなく、seed repeat もない。介入後は prompt/sample selection 自体が変わる。
- zero-loss は direct gradient を消すが、truncated sample の reward は group baseline/advantage に残るため、完全な sample filtering ではない。
- staleness-aware は total absolute objective を約27--49%落とす。truncation-specific 効果と generic effective-step-size reduction が混ざる。
- zero-loss は S=20 まで300 stepと late AIMEで確認できるが、aware S=20/24/28 の late downstream eval はまだ十分でない。
- 1 model、1 math dataset、16K、LR=1e-6、300 update の結果であり、math RL 全般や長期 horizon へは外挿できない。

## 決着をつける control

1. S=16/20 で zero-reward、zero-loss、aware を複数 seed 反復する。
2. aware と同じ total objective norm / update norm になる global attenuation または random-sample attenuation を置く。
3. zero-loss と、truncated sample を group baseline からも除く full filtering を分ける。
4. 同じ saved batch に対して通常 loss / zero-loss / aware の gradient cosine・norm を offline replay で比較する。
5. truncated/non-truncated x staleness x advantage sign ごとの post-TIS objective と gradient proxy を保存する。

## 出力

- [Cross-treatment TIS diagnostics](figures/tis-cross-treatment-t1r7-maxlen16384.svg)
- [Collapse time alignment](figures/tis-collapse-time-alignment-t1r7-maxlen16384.svg)
- [Staleness-aware exact objective attenuation](figures/staleness-aware-post-tis-objective-t1r7-maxlen16384.svg)
- [S=20 staleness-bin profiles](figures/tis-staleness-bin-profiles-s20-t1r7-maxlen16384.svg)
- [Per-arm summary](tis-diagnostics-summary.csv)
- [Staleness-bin summary](tis-staleness-bin-summary.csv)
- [No-treatment truncated-reward summary](no-treatment-truncated-reward-summary.csv)
