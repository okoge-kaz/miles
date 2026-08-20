# math/async / dapo-math-p10-90

DAPO-Math-17K を step-4000 Qwen3-4B policy で難易度フィルタした 10,778 問の
上での fully-async GRPO。**dynamic sampling は行わない**（§4.0）。学習中の
評価は既定で無効にし、報告値は 10 rollout step ごとの HF checkpoint に対する
オフライン評価から作る（§2）。

フィルタなしの `dapo-math` レシピは削除済み — 全実験をこのデータセットで行う。

`<task>/<dataset>/` 配下の README は共通の型を持つ — 先に公表値、次に本レシピとの
差分、最後に探索範囲。[contract](../../README.md#dataset-directories) を参照。

> **このファイルだけ日本語。** 他の全 README は英語のまま。contract 側は本ファイルを
> 型の worked example として名指ししているので、型（3部構成）は保っている。
> 対になる `math/sync/dapo-math-p10-90/README.md` は未作成（§10-1）。

---

## 0. この実験が答える問い

> **現実的な off-policy step における recipe を作る。** `MAX_WEIGHT_STALENESS` を
> 指定したとき **実際にどれだけの off-policy ness が発生するのか**、そしてその
> recipe は on-policy と比べて **同じ downstream 性能にどれだけ速く到達するのか**。
> さらにその利得が **task 難易度と model 規模に対してどう動くのか** を測り、
> より難しい task・より大きな model で取るべき recipe を導く。

**これは「単一手法を他手法と比較する」論文ではない。** 先行研究（M2PO, GAC, ESTR）は
提案手法が注入 staleness に耐えることを示す形をとる。本実験は逆で、**staleness は
注入しない** — 実際のモデル開発と同じく上限だけを指定し、そこで発生した lag を測る。
出力は単一の「2.5× 高速」ではなく、**off-policy step を軸にした scaling の面**である。

したがって成果物は 3 つ:

1. **実現 lag の地図** — `MAX_WEIGHT_STALENESS` を指定したとき `P(L)` が何になるか。
   task・model・応答長・throughput 比の関数として
2. **利得の地図** — `S_m(p)`（同一 downstream 性能への wall-clock 比）が、その実現 lag
   でどれだけ出るか
3. **recipe** — 1 と 2 から、より難しい task・より大きな model で取るべき設定

「速い」の定義は事前登録済みで、測定コードは
[`experiments/src/offpolicy_acceleration`](../../src/offpolicy_acceleration/README.md)
にある。主要指標は 2 つ:

| 記号 | 意味 |
|---|---|
| `τ_m(δ)` | on-policy plateau `Q_on*` から δ 以内に入る最初の時刻（非劣性時刻） |
| `S_m(p)` | `τ_on(q_p) / τ_m(q_p)` — 中間目標 `q_p` ごとの**速度向上プロファイル** |

時間軸の第一義は**総 GPU 数を固定した wall-clock**、第二軸が GPU-hours。
「カレンダー時間の短縮」と「計算効率の改善」を混同しないため。

変数は 3 階層に分かれ、README の残りはこの区別で並ぶ。

| 階層 | 何か | 扱い |
|---|---|---|
| **A. 主軸** | off-policy 度合いと、それが測られる条件 | §5.1。full grid。staleness × 応答長 × pause × algorithm × LR |
| **B. 制御変数** | 主軸が意味を持つための前提 | §5.2。先に決めて全 arm で凍結 |
| **C. スループット専用** | 学習内容を変えず速度だけ変える | §5.3。モデルごとに 1 回調整して凍結 |

C を主軸と同時に動かすと wall-clock 軸が壊れ、それが本実験の主要指標なので致命的。
B のうち凍結するものを動かすと品質軸が壊れる。ladder（§7）はこの順に進む。

**まず full grid を回す。** 他のモデルサイズに広げる段階で削る。実験数と
node-hours は §5.4 に実測ベースで置いてあるので、削るなら数字を見て削る。

---

## 1. 先行研究

| system | base model | 報告値 | 本実験での位置づけ | reference |
|---|---|---|---|---|
| DAPO | Qwen2.5-32B | **AIME 2024 = 50** | アルゴリズムの土台。decoupled clip を既定で採用。ただし **dynamic sampling は採用しない**（§4.0） | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| Dr. GRPO | 7B base | **AIME 2024 = 43.3** | length / std 正規化への反論。`--grpo-std-normalization` off が該当 | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| AReaL | — | 同一 GPU 数で同期比 **2.77× の学習速度** | **本タスクが検証する対象**。staleness 上限つき PPO = `--max-weight-staleness` | [arXiv:2505.24298](https://arxiv.org/abs/2505.24298) |
| ROLL Flash | Qwen3-8B Base/Think | RLVR で **2.24×**、agentic で 2.72×、同一 GPU 予算 | **速度の測り方がここと違う。** 報告は「50 step あたり wall-clock」= **固定 step 数でのスループット**で、"time-to-" という語は本文に一度も現れない。off-policy が同じ品質に**より多くの step を要する**なら、step あたり 2.24× は品質到達時間では 2.24× にならない。この差が `τ_m(δ)` / `S_m(p)` を定義する理由 | [arXiv:2510.11345](https://arxiv.org/pdf/2510.11345) |
| PipelineRL 系（`in_place` 継続） | — | — | 重み更新をまたいで KV cache 上で生成を継続する方式。`PAUSE_GENERATION_MODE` の 1 水準として **grid の外で**比較する（§5.5） | — |
| **Prosperity before Collapse** | — | stale データでどこまで到達できるか。**M2PO**（importance weight の 2 次モーメントを制約） | **最も近い先行研究。** 不安定な run では KL/勾配爆発の直前に **ESS ratio が崩壊**すると報告。本実験の `rollout_ess_ratio` はこの量。positioning で必ず突き合わせる。M2PO は algorithm 軸の候補 | [arXiv:2510.01161](https://arxiv.org/abs/2510.01161) |
| Entropy-Scaled Trust Regions | — | importance ratio の自然なスケールが **token entropy と系統的に変わる**。低 entropy では train-inference 乖離がノイズに増幅、高 entropy では in-flight 更新が探索的逸脱を生む | **entropy × staleness の交互作用は既に着手されている。** ただし扱っているのは token entropy であって learning rate 感受性ではない（§5.1） | [arXiv:2607.22186](https://arxiv.org/html/2607.22186v1) |
| GAC | — | stale 方向への勾配射影で非同期 RL を安定化 | algorithm 軸の候補 | [arXiv:2603.01501](https://arxiv.org/html/2603.01501) |
| **Nemotron 3 Nano** | MoE hybrid | **同期 GRPO**、128 prompt/step × **16 generation**、batch 2048、"making our updates on-policy" | **小さいモデルでは非同期を使っていない。** async は規模の関数として採用されている — 本実験の model 軸そのもの。train-inference 不整合には **masked importance sampling** を使用（miles の `tis_mode=mask` と同族） | [tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) |
| **Nemotron 3 Super** | MoE hybrid (120B-A12B) | **非同期**、数千 GPU、21 環境。256 prompt/step × **16 responses**、batch 4096、**1 rollout = 1 gradient update**。**最大生成長 49K → 64K** | **応答長の決定的な対比。** アルゴリズム論文が 4k–16k で検証している一方、実運用の RL は **49K–64K** で回る。IS の積の項数が 3–16 倍違う領域で検証されていない（§5.1） | [tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) |
| Nemotron 3 Ultra | MoE hybrid | **one-step off-policy asynchronous RL** | 非同期構成の実例として引く。**`s=1` を「大きな lag は不要」の根拠には使わない** — 社内で検証しきれていないという運用判断であり、測定結果ではない。有用なのは別の観測: **rollout が律速で、その時間は少数の straggler generation に支配される**（＝ lag は応答長分布の裾が作る、§5.1） | [tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) |
| GRPO (DeepSeekMath) | DeepSeekMath 7B | MATH 51.7（self-consistency@64 で 60.9） | GRPO の原典。group size 64 の出所 | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| Qwen3 | 0.6B–235B, dense + MoE | thinking / non-thinking を 1 モデルに | ベースモデル系列 | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |

### 1.1 Positioning — 論文に書く形

**本文を読んで検証した上での文言:**

**本実験の立てる問いは、先行研究の問いと形が違う。** 先行研究は
「*人為的に課した* lag に手法は耐えるか」を問い、本実験は「*上限を指定したとき*
lag はいくつ発生し、それが同一性能への時間をどれだけ縮めるか、そしてその関係は
task と model でどう動くか」を問う。前者は手法の比較、後者は **recipe の scaling**。

> Prior work on off-policy LLM RL splits into two groups that do not meet.
> **Algorithm papers impose staleness by construction** — M2PO's *Stale-k*
> training deliberately holds rollouts for `k` model updates
> (`k ∈ {0,32,64,128,256}`), GAC fixes `s ∈ {4,8,16,32}` — and plot accuracy
> against **training steps**, reporting no wall-clock at all. **Systems papers
> report throughput speedup at a fixed step count** (AReaL 2.77×, ROLL Flash
> 2.24×, ESTR 2.6×). Neither group answers **when** an off-policy run becomes
> statistically non-inferior to a **converged** on-policy reference, and neither
> measures **what staleness an asynchronous pipeline actually produces**. All of
> them fix the learning rate at 1e-6 and fix the response-length budget.

**検証結果（全て PDF 本文を取得して全文検索）:**

| 論文 | staleness の扱い | 速度指標 | LR | 応答長 | 品質の基準 |
|---|---|---|---|---|---|
| M2PO | `s ∈ {0,32,64,128,256}` を sweep | **なし**（`wall-clock` 0 件、`time-to` 0 件） | 1e-6 固定 | 4k/16k 固定 | 1000 step 中の**ベスト checkpoint** |
| GAC | `s ∈ {4,8,16,32}` を sweep | **なし**（`speedup` 0 件、`time-to` 0 件）。横軸は Training Steps | 1e-6 固定 | 16k 固定 | step 対 accuracy の目視 |
| ESTR | intra `{1,5,7,9}` × inter `{1,5,15,20,30}` を sweep | throughput と **s/step**、2.6×（`time-to` 0 件） | 記載なし | 記載なし | 固定 step で sync と同等 |
| ROLL Flash | 体系的 sweep なし | **"Wall-clock per 50 Steps"**、2.24× | — | — | 固定 step |
| AReaL | — | throughput 2.77× | — | — | — |

**先に書いていた 2 つの主張は誤りだった。取り下げる。**

* ~~「staleness は選ばれた 1 点」~~ → **M2PO・GAC・ESTR はいずれも sweep している。**
  M2PO の 256 は sweep の上端であって単一点ではない。
* ~~「モデル規模を振っていない」~~ → M2PO は 1.7B–32B の 6 モデル、GAC も
  Qwen3 1.7B/4B/8B + Llama-3.2-3B。

**最大の差分: 先行研究の staleness は「注入されたもの」であって「発生したもの」ではない。**

M2PO は *Stale-k RL training* を導入し、**k 回の model update 分だけデータを意図的に
保持して**学習する。本文にこうある — 「最初の k 更新の間はまだ stale model が存在
しないので、base model が生成したデータで学習する」。つまり `s=256` は
**設計して作った条件**であって、非同期パイプラインが自然に生む lag ではない。GAC も
「他を固定して」`S-0/8/16/32` を構成している。

したがって「アルゴリズム A は s=256 に耐える」という主張は、**s=256 が実際に起きるか
どうかについて何も言っていない**。本レシピの監査 run では `weight_version/max − min = 2` だった
（実測、job 15113756）。**実運用の lag がこの程度なら、`s=256` への頑健性は誰も
持っていない問いへの答えになる。**

ただし「フロンティアの採用値」を根拠にはしない。Nemotron 3 が `s=1` なのは
社内で検証しきれていないという運用上の判断であって、**大きな lag が発生しない
という測定結果ではない**（内部事情、公開文献からは読み取れない）。したがって
「実運用の lag は小さい」は**本実験が測って示すべき仮説**であり、他所の採用値を
引いて既定事実として書いてはいけない。

本実験では staleness は **設定する独立変数ではなく、train/rollout の速度比から
発生する従属変数**であり、`--max-weight-staleness` はその上限に過ぎない。だから
測る（`staleness/rollout/{mean,p50,p90,p99,max}`、
`staleness/bound_exceeded_sample_frac`、`mixed_version_ratio`）。逆向きのリスクも引き受けることになる — **自然な lag が
小さければ staleness 軸が動かない**。§5.1 の事前確認はそのためにある。

両者は競合ではなく**別の問いを測っている**: 先行研究は「人為的に課した lag に
アルゴリズムは耐えるか」、本実験は「lag は実際にいくつ発生し、それに耐えることが
時間の節約になるか」。

**残る差分は 4 つで、いずれも全文検索で確認済み。**

1. **時間を測っていない。** staleness を最も丁寧に調べた 2 本（M2PO, GAC）は
   **速度指標を一切報告していない** — 横軸は training step。速度を報告する側
   （AReaL, ROLL Flash, ESTR）は**固定 step 数あたりのスループット**。
   「staleness に耐えられること」が「時間の節約」になるかは、**どちらの group も
   答えていない**。`τ_m(δ)` / `S_m(p)` はここを埋める。
2. **収束した参照がない。** M2PO は 1000 step 中のベスト checkpoint を報告する。
   事前登録した収束判定も、非劣性の信頼区間もない。
3. **LR と応答長が固定。** 全て LR=1e-6、応答長は 4k/16k 固定。off-policy 耐性が
   これらの関数としてどう動くかは測られていない。
4. **realized lag が報告されていない。** 全て設定値 `s` のみ。そもそも `s` は
   注入値なので、報告すべき「実現値」が存在しない設計になっている。

**ESTR の staleness 分解は取り込む価値がある。** ESTR は staleness を
**intra-trajectory**（1 本の生成が重み更新をまたぐ）と **inter-trajectory**
（バッチが古い重みで作られた）に分けている。miles ではこれが
`rollout/weight_version/mixed_version_ratio`（intra）と
`staleness/total/mean`（inter）に対応する。**両方すでに記録している**ので、この
分解に沿った報告ができる。

> **Provenance.** M2PO / ESTR / GAC / ROLL Flash / Nemotron 3 Ultra は
> **PDF 本文を取得して全文検索済み**。上表の「0 件」は実際に検索した結果。
> DAPO / Dr. GRPO / GRPO / Qwen3 / AReaL の代表値は abstract から取っており
> **本文で再検証していない**。§3 のハイパラ比較表も同様。

## 2. 評価の二層構造

**学習中の評価と、報告する評価を分離する。** 理由は本実験が wall-clock を主要指標に
するから — 学習中の eval が重ければ、その overhead がそのまま throughput の議論に
混入し、arm 間の比較が eval 予算の比較になってしまう。

| | 学習中 eval | オフライン eval（報告値） |
|---|---|---|
| 目的 | opt-in の監視、崩壊検出（既定では無効） | `Q(t)`、`τ_m(δ)`、`S_m(p)` |
| ベンチマーク | opt-in 時は **AIME-2025**（30 問） | **AIME-2023 / 24 / 25 / 26**（各 30 問） |
| n / prompt | 8（opt-in 時） | 16 |
| 生成長 | 16384（opt-in 時、学習と同じ） | 32768 |
| sampling | T=1.0 / top_p=1.0 | 比較に必要な設定（既定は学習と同じ 1.0 / 1.0） |
| 実行 | `train.sh` の `--eval-prompt-data` | [`src/offline_eval/run_eval.sbatch`](../../src/offline_eval/run_eval.sbatch) |
| コスト | 学習の critical path 上 | **別ジョブ。学習の wall-clock に一切入らない** |

これで解決すること:

* **検出力。** 30 問 × 8 では bootstrap 半値幅が δ より大きく、
  offpolicy_acceleration の `analyze.py` は `UNDERPOWERED` を出す。4 年 × 30 問
  × n=16 なら 120 問で標準誤差 ≈ 4.5 点、5 年なら 150 問。
* **打ち切りバイアス。** 現在のレシピは 16,384 token で打ち切られた応答を
  opt-in 設定により 0 点にする。オフライン側の予算を変える場合は、学習時と
  異なる評価量になることを明示する。
* **overhead。** 学習中の eval を既定で無効にすることで、arm 間の wall-clock
  差が eval 予算の差で汚染されない。

### 時間分解能は 10 rollout step

オフライン eval は `--save-hf` が書く HF snapshot を読む。retention 設計
（コミット `82b86040`）により:

```
torch_dist  10 step ごとに書き、100 step ごとの milestone + latest を保持
HF          10 step ごとに書き、offline eval 用に保持
```

HF は `--save` の外に落ちるので `--save-retain-interval` の刈り込み対象外。
したがって **`Q(t)` の時間解像度は `HF_SAVE_INTERVAL` = 10 rollout step**。
`τ_m(δ)` の分解能はこれで決まるので、`SAVE_INTERVAL` は eval 設定の一部として
全 arm で共通に固定する（§6.1）。

---

### 2.1 汚染 — AIME 2023 は評価に使えない

学習セットと評価セットを**逐語照合した結果**（正規化して完全一致を数えた）:

| 学習セット | aime23 | aime24 | aime25 | aime26 |
|---|---|---|---|---|
| `dapo-math-17k`（フィルタ前） | **23/30** | 0 | 0 | **1/30** |
| **`dapo-math-p10-90`（現行の学習セット）** | **11/30** | 0 | 0 | 0 |
| `deepscaler-preview` | **26/30** | 0 | 0 | 0 |

**現行の学習セットは AIME 2023 の 15 問（50%）を含んでいる。** DeepScaleR を
足すと 26/30 になる（DeepScaleR は AIME 2023 以前の過去問から構築されているため）。
**DeepScaleR は採用しない** — フィルタ通過率が 9.7%（実測 846 問での推定）と低く、
追加できるのは約 3,900 問にとどまる一方、データ源が増えて実験の解釈が濁るため。

**したがって:**

* **`aime23` はオフライン評価から外す。** `run_eval.sbatch` の既定は 4 年なので
  3 年（24/25/26）に直す必要がある。プール後の held-out は 120 問ではなく **90 問**
  になり、§5.4 の検出力の議論もその分下がる
* `aime24` / `aime25` はすべての学習セットに対して**清潔**
* `aime26` は `dapo-math-17k` に 1 問だけ混入している（同一本文・同一ラベル 277 の
  逐語一致を目視確認）。窓を 0.1–0.9 に広げた後も **`p10-90` では 0/30 のまま**
  （再照合済み）。ただし
  **フィルタ窓を広げるとこの 1 問が入る**。窓を触るときは再照合すること
* DeepScaleR をプールに足す場合、汚染は aime23 に閉じるので **aime23 を落とせば
  追加して安全**

照合はプロンプト本文を正規化した完全一致で、言い換えられた問題は捕まらない。
数値だけ変えた変種も同様。**これは下限**であって、真の汚染はこれ以上ありうる。

## 3. 先行研究のハイパラ vs 本レシピ

| knob | DAPO | 本レシピ | 備考 |
|---|---|---|---|
| group size (`N_SAMPLES_PER_PROMPT`) | 16 | 8 | DeepSeekMath は 64 |
| prompts per rollout (`ROLLOUT_BATCH_SIZE`) | 512 | 32 | **最大の乖離** |
| learning rate (`LR`) | 1e-6, constant | 1e-6, constant | 一致 |
| clip low / high | 0.2 / 0.28 | 0.2 / 0.28 | 一致 |
| KL coefficient | 0 | 0 | 一致 |
| max response (`MAX_RESPONSE_LEN`) | 20480 (16384 + 4096 overlong buffer) | 24576 | — |
| **dynamic sampling** | **あり** | **なし（事前フィルタで代替）** | §4.0 |

---

## 4. 難易度フィルタ — dynamic sampling を置き換える

### 4.0 なぜ置き換えるのか

GRPO の advantage は group 標準偏差に比例し、二値報酬では pass rate `p` に対して
ちょうど `sqrt(p(1-p))`。全員正解／全員不正解の group は**勾配に何も寄与せず、
生成コストだけ払う**。本クラスタの実測（Qwen3-4B thinking, DAPO-Math）:

```
raw_reward       0.84, 46 step 通じて横ばい
zero_std         32 group 中 21.6 が全問正解, 2.4 が全問不正解  (~76% が無駄)
grad_norm        0.04
AIME24 avg@16    0.706 (step 0) -> 0.692 (step 19)
```

これに対する手当ては 2 通りあり、本実験は**オフライン側を採る**。

| | online (`--dynamic-sampling-filter-path`) | offline（本実験の方式） |
|---|---|---|
| コストを払うタイミング | 毎 rollout、永久に | (dataset, policy) ごとに 1 回 |
| 実測コスト | rollout time 253 s → 761 s（**3×**） | 推論ジョブ 1 本 |
| ポリシー改善への追従 | する | しない — 陳腐化したら再測定 |
| step 時間の決定性 | **不定**（何 group 落ちるかで変わる） | **決定的** |

最後の行が本実験にとって決定的に重要。**wall-clock が主要指標である以上、
step 時間が学習内容と無関係な要因で揺れるのは許容できない。** dynamic sampling は
1 step あたりの生成量を run ごとに変えてしまうので、arm 間の wall-clock 比較に
非制御の分散を持ち込む。3× の rollout time 増もそのまま主要指標を汚す。

副次的な効果として、`GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE × N / NUM_STEPS_PER_ROLLOUT`
の不変条件が厳密に成立し、optimizer step 数が rollout 数から決定的に決まる。

### 4.1 使うプロンプト集合

現行の `dapo-math-p10-90`:

| 項目 | 値 |
|---|---|
| 元データ | DAPO-Math-17K, 17398 問 |
| dataset | `dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000` |
| 残った問題数 | **10,778 問（61.9%）** |
| window | pass rate **0.1–0.9** |
| 測定ポリシー | **Qwen3-4B-Base-LR2e-5-Step4000** |
| 測定条件 | n=16, T=1.0, max_new_tokens=16384, max_context_length=32768, `rm_type=deepscaler`, truncation reward 0 |

生成手順は [`tools/difficulty_filter`](../../../../tools/difficulty_filter/README.md)。
測定（GPU 1 ジョブ）と window 選択（CPU 数秒）が分離されているので、window を
変えるだけなら再測定は要らない。

### 4.2 フィルタはポリシーに紐づく — モデルを変えたら測り直す

difficulty は *(prompt, policy, sampling-params)* の三つ組の性質であって、
プロンプト単体の性質ではない。**現在の policy-specific dataset は
Qwen3-4B-Base-LR2e-5-Step4000 にとっての 0.1–0.9 であり、他のモデルに
とってはそうではない。**

これは §8 のモデル拡張計画に直接効く制約:

* 1.7B に対しては、4B にとって適度だった問題の多くが「全問不正解」になる
* 30B-A3B に対しては、同じ問題の多くが「全問正解」になる
* dynamic sampling を外している以上、**その group は落とされずにそのまま学習に入る**
  — つまりフィルタが陳腐化したとき、online フィルタのような安全網がない

したがってモデルを 1 つ追加するたびに `measure_pass_rate.py` を回し、
`<dataset>-p10-90-<policy>` として**別 dataset ディレクトリ**を切る
（`PROMPT_DATA` は `<task>/<dataset>/` で固定される設計なので、これが唯一の作法）。

同時に **verifier をポリシーに合わせる**。`deepscaler` は応答に `</think>` が
無いと 0 を返すので、non-thinking checkpoint では reward ≡ 0 になり、
「学習しないモデル」に見える設定ミスが無言で成立する。SFT 済み Base モデルは
SFT のフォーマット次第なので、`measure_pass_rate.py` の `verifier_preflight`
（GPU 時間ゼロ、verifier 呼び出し 3 回）を毎回通す。

---

## 5. 動かす parameter

**基本方針: full grid。** 計算予算は制約でないので、`LR × MAX_WEIGHT_STALENESS ×
NUM_STEPS_PER_ROLLOUT × PAUSE_GENERATION_MODE × …` は入れ子探索ではなく全格子で
回す。最適 LR が off-policy 度合いとともに動くなら、その交互作用自体が結果だから。

### 5.1 主軸 — full grid（これが結果）

| knob | 水準 | 数 | 何を測るか |
|---|---|---|---|
| `MAX_WEIGHT_STALENESS` | **1, 2, 4, 8, 16, 32** | 6 | `queue-recycle` の主軸。**math では自然には大きな off-policy ness が発生しない**という仮説の検証も兼ねる。設定値ではなく実現値で判定する。on-policy 端点は別の colocated baseline とする |
| `MAX_RESPONSE_LEN` | **4k, 8k, 16k, 32k** | 4 | **制御変数から主軸に昇格。** 短い出力長で検証している先行研究への示唆が目的で、**下端の 4k は VCPO 系の設定域に合わせてある**（VCPO 自体は 2k だが MATH での検証なので 4k を採る）。短い予算は 1 サンプルが跨ぐ weight sync 数を機械的に減らすので、off-policy が実力以上に良く見える。§5.4 の警告を読んでから回すこと |
| RL algorithm | DAPO(GRPO), VCPO, CISPO, … | **A（未定）** | off-policy に有効と報告のあるものを一通り。**clip low/high と `--use-tis` はアルゴリズムの一部**として、ここで一緒に決める |
| `LR` | **1e-7, 1e-6, 5e-6** | **3** | 低 / コンセンサス / 上限付近。1e-6 は DAPO と Nemotron 系 GRPO の既定値。5e-6 は公開されたチューニング範囲 {5e-6, 1e-6, 5e-5} の上側で、**試されて 1e-6 に負けた値**なので「発散する」ではなく「訓練はできるが劣る」ことが分かっている高値。最適 LR が off-policy 度合いとともに動くなら、その交互作用が結果 |

**応答長軸は「公平性」ではなく、IS 補正が壊れる機構そのものを測る軸である。**
系列レベルの importance weight は per-token 比の積なので、log 重みは T 個の
per-token log 比の**和**になる。per-token の標準偏差を σ とすれば
`std(log w) ≈ σ√T`、`Var(log w) ∝ T`、そして ESS はおおよそ `exp(-Var(log w))` で
落ちる — **系列長に対して指数的**。

したがって短い出力長で検証された IS 系手法（TIS, CISPO, …）は、**補正が容易な領域
でしか検証されていない**可能性がある。4k と 32k は 8 倍の長さ差なので `Var(log w)`
は 8 倍、ESS は桁で落ちうる。「4k では成立するが 32k では成立しない手法」という
主張は、**algorithm 軸 × 応答長軸の交互作用**として本実験の中心的な結果になる。

**この対比は実在する。** off-policy アルゴリズム論文の検証長は M2PO が 4k/16k、
GAC が 16k 固定。一方 **Nemotron 3 Super の RLVR は最大生成長 49K で始めて 64K
まで上げる**（tech report 本文で確認）。**実運用は検証環境より 3–16 倍長い。**
IS 重みの項数がその倍率で効くので、「アルゴリズムが検証された領域」と
「アルゴリズムが使われる領域」が一桁ずれている。本実験が 32k まで振るのは
この隙間を埋めるため。

測る量は `train/rollout_ess_ratio`（`π_train/π_rollout` の ESS、§6.3 で追加）。
`train/ess_ratio` は別物なので使わない。実測の per-token 乖離は
`train/tis_abs = 0.0100`（staleness ~2）なので、この σ と系列長から ESS の予測が
立ち、実測と突き合わせられる。

**4k では「劣化」ではなく「学習しない」になる可能性がある。** 平均応答は実測
**6,880 token**（中央値 6,701）。4096 上限では **8 割近くが打ち切られる**。打ち切り
サンプルは全 rule-based verifier で 0 点なので reward が潰れ、group が全滅
（zero variance）して **advantage が恒等的に 0 = 勾配が立たない**。プロンプト集合が
`max_new_tokens=16384` で pass rate 0.1–0.9 に絞ってある（§4.1）ことも、4k では
window が意味を失う方向に効く。grid の 1/4 を投じる前に 1 本で確認する
（`truncated_ratio`, `zero_std_group_frac`, `rollout/raw_reward`）。

**staleness の両端には確認が要る。**

* `1` — `queue-recycle` の最小値。dequeue 判定は `D-F < M` なので `M=0` は
  非負の gap を 1 件も admit できず、設定時点で拒否される。on-policy 端点には
  colocated baseline を使う。fully-async の staleness-0 を別途測る場合は
  equality を admit する `queue-max M=0` と明記し、queue type の差を混同しない。
* `32` — 自然発生しないなら軸を削る。判定は
  `staleness/rollout/max < 32` かつ
  `staleness/bound_exceeded_sample_frac = 0` であることと、
  `staleness/rollout/mean`。上限に届かなければ、その水準は下位の水準と同じ実験を
  別名で回しているだけ。

**algorithm 軸は実装から。** miles が持つのは `grpo`, `gspo`,
`reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `ppo` の 5 つだけ
(`arguments.py:1243`)。**CISPO / VCPO / CTPO は未実装**（grep して 0 件）。
`--custom-tis-function-path` は pg_loss に重みを掛ける形なので、CISPO のように
PPO clip 自体を clipped-IS に置き換える目的関数はこのフックでは表現できない。
損失側に手を入れる必要がある。A は総実験数の乗数なので、ここが決まるまで規模は
確定しない。

### 5.2 先に決めて凍結

grid の軸ではない。主軸を回す前に決め、以降全 arm で共通に固定する。

| knob | 決め方 | 現状 |
|---|---|---|
| `N_SAMPLES_PER_PROMPT` | **Nemotron3 の値を踏襲**。8 か 16 になる見込み | 先行研究調査中。**コストが線形に効く**（§5.4） |
| `N_SAMPLES_PER_PROMPT` | **16** | Nemotron 3 Super と同じ値で、難易度フィルタも n=16 で測定済み（§4.1）。window が訓練時の group をそのまま記述し、n=8 より degenerate group を減らす。|
| `ROLLOUT_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` | **64 prompt / 512 サンプル**（`NUM_STEPS_PER_ROLLOUT=1`）| **バッチ形状はコストを変えないが、主軸である実現 lag を変える**（下記）。64 なら 10 epoch = 619 step で、刻み 20 でも 31 評価点。degenerate 8.4% を引いて実効 59 group |

**バッチ形状は主軸と結合している — これが選択理由。** 実現 lag は
**weight version 単位**で測る量なので、

```
実現 lag ≈ (1 group の生成時間) / (1 optimizer step の時間)
```

分子は応答長と engine が決めるので `R` に依存しないが、分母は `R` に比例する。
したがって **実現 lag ∝ 1/R**。監査 run（`R=32`）で `weight_version/max − min = 2`
だったことを錨にすると:

| R | global batch | 予測される実現 lag |
|---|---|---|
| 32 | 256 | 2.0（実測） |
| **64** | **512** | **1.0** |
| 128 | 1024 | 0.5 |
| 512 | 4096 | 0.12 |

**大きなバッチは、測ろうとしている現象そのものを消す。** `R=128` だと lag が
0.5 版程度になり、staleness 軸 `{0,1,2,4,8,16,32}` はほぼ全水準が同じ実験になる
——§6.2 が警告している状態そのもの。`R=64` を採るのは評価点の数のためではなく、
**主軸が動く余地を残すため**。

lag を作る他の手段は `ASYNC_MAX_CONCURRENT_SAMPLES`（生成の並列度を上げて
キューを深くする）と actor/rollout 分割で、こちらは学習側の統計を触らないぶん
より清潔。**どのレバーで lag を作るかは §7 Stage 1 で実測して決める** — この表は
予測であって測定ではない。
| KL coefficient | **使わない。** reference model を持たないため | `KL_LOSS_COEF=0.00` |
| dynamic sampling | **off。** 難易度フィルタで置換（§4.0） | フィルタ済みプロンプト集合 |
| `--rollout-temperature` | 1.0 固定。フィルタが T=1.0 で測られている | 1.0 |
| 難易度フィルタの測定条件 | `max_new_tokens=16384`, total context 32768, n=16 で測定済み | **2k / 8k の水準では window が意味を失う**（§5.1）。フィルタを測り直すかは 2k の pilot 結果を見て決める |
| deterministic kernel | **sweep 軸にしない**（§5.5） | 未使用 |

### 5.3 スループット専用 — モデルごとに 1 回調整して凍結

学習内容を変えないので `batch_short` レーンで振り、以降固定する。

| knob | default | 探索範囲 | 理由 |
|---|---|---|---|
| actor / rollout 分割 | 1 node + 1 node | **1+1, 1+3, 2+2, 3+1** | 実測で **train step 時間 = rollout step 時間** になる分割を選び、全 off-policy step で共通に使う。colocated 参照 arm は **train + rollout の合計ノード数**で回す。cell 内で総 GPU 数が一致していることが wall-clock 比較の成立条件（§5.4） |
| `MAX_TOKENS_PER_GPU` | model 別 | OOM まで上げる | step あたり microbatch 数が減る。`MAX_TOKENS_PER_GPU × cp ≥ ROLLOUT_MAX_CONTEXT_LEN` は必須 |
| `TENSOR_PARALLEL_SIZE` / `CONTEXT_PARALLEL_SIZE` | model 別 | 積が学習 GPU 数を割る 2 冪 | 残りが `dp` になり `GLOBAL_BATCH_SIZE` を割り切る必要がある。長い応答が載るかは CP 次第 |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | model 別 | 1, 2, 4, 8 | 小さい engine = 数が増えて並列度↑。重みが載らなくなったら大きくする |
| `SGLANG_MEM_FRACTION` | 0.7 | **0.6–0.85** | KV cache サイズ |
| `SGLANG_MAX_RUNNING_REQUESTS`, `SGLANG_CUDA_GRAPH_MAX_BS` | unset | 実測ピークから | 到達しない並列度で CUDA graph を capture すると起動時間とメモリの無駄 |
| `EXPERT_PARALLEL_SIZE` | model 別 | MoE のみ | `qwen3-30b-a3b` 専用 |

### 5.4 実験数と GPU node-hours

コスト模型は実測値のみで組む。すべて qwen3-4b-instruct-2507 / この
データセット / 2 ノード（train 8 + rollout 8）での測定。

| 実測量 | 値 | 出所 |
|---|---|---|
| rollout step 速度 | **24.4 step/h** | 11 step / 0.45 active h |
| 平均応答長 | **6,880 token**（中央値 6,701） | `rollout/response_len/{mean,median}`、24576 上限で |
| 打ち切り率 | **4.0%** | `rollout/truncated_ratio` |
| 生成スループット | **2,019 token/GPU/s** | `perf/tokens_per_gpu_per_sec` |
| rollout time | 129 s/step | `perf/rollout_time` |
| 起動時間 | 約 13.5 分 | engine 起動 → 最初の metric |

**コストは optimizer step 数で決まる。** Nemotron 3 Super の形状（n=16、global
batch 4096 サンプル）だと **1 optimizer step = 256 prompt**。実測 2,019 token/GPU/s
から:

| 量 | 値 |
|---|---|
| 1 step のトークン | 4096 × 6,880 = **28.2 M** |
| 生成のみ | 0.48 h/step（rollout 8 GPU） |
| 実測オーバーヘッド込み（+35%） | **1.31 node-h / optimizer step**（24k, 2 node） |
| 1 epoch | 3962 / 256 = **15.5 optimizer step** |

**上限別コストは `E[min(X, cap)] / E[X]` で決まる。** 平均 6,880 / 中央値 6,701 /
24576 で 4.0% 打ち切り、という実測分布から:

| 上限 | 4k | 8k | 16k | 24k | 32k |
|---|---|---|---|---|---|
| 打ち切り率（推定） | ~78% | ~38% | ~10% | 4%（実測） | ~0% |
| 相対コスト | **0.50** | **0.72** | **0.94** | 1.00 | **1.00** |

採用する {4k, 8k, 16k, 32k} の合計は **3.17**。上端はほとんど差がない — 平均応答が
6.9k である以上、24k と 32k は同じコスト。**効くのは下端だけ**で、4k が 0.50 まで
落ちるのは 8 割が打ち切られているからで、計算が安いのではなく**タスクが別物に
なっている**。打ち切り率は推定値なので、初回 run の `response_len/p90` `p99`
（今回追加）で確定させる。

セル数:

```
off-policy   staleness(7) × length(4) × A × L = 28·A·L
reference    length(4) × A × L                =  4·A·L   (colocated は staleness 軸を持たない)
合計                                             32·A·L
```

pause mode と deterministic kernel は grid の外（§5.5）で、それぞれ数 run。

`node-hours = 1.31 × S × 3.17 × 8 × A × L`（S = 収束までの optimizer step 数）。
A=3, L=3 なら **288 run**:

| S | epoch 換算 | node-hours | 14 日で必要なノード数 |
|---|---|---|---|
| 77 | 5.0 | 23,000 | 68 |
| 150 | 9.7 | 44,700 | 133 |
| **300** | **19.4** | **89,400** | **266** |
| 600 | 38.8 | 178,800 | 532 |

**S が最大の未知数になった。** バッチが 16 倍になったので、同じ epoch 数なら
optimizer step は 1/16 になる。5 epoch = 77 step は RLVR の収束としてはおそらく
足りない。**S を決めるのは on-policy 参照 run 1 本**（§7 Stage 3）で、それが
grid 全体のコストを決める。

**評価と保存の刻みは `S` から逆算する。** 必要なのは「刻みの値」ではなく
**評価点の数**で、収束判定（plateau window 5 点）+ 連続 3 点の交差則 +
trailing mean 3 点を考えると **20–30 点**欲しい。

| R | 10 epoch での S | 刻み 10 | 刻み 20 |
|---|---|---|---|
| 64 | 619 | 62 点 | **31 点** |
| 128 | 310 | **31 点** | 15 点 |
| 512 | 77 | 8 点 | 4 点 |

コストは制約にならない。学習中 eval（aime25, n=8, 30 問）は **1 step の 24%**
なので、刻み 10 で 2.4%、刻み 20 で 1.2% のオーバーヘッドにしかならない
（実測 2,019 tok/GPU/s から）。効くのは **HF checkpoint の保存量**で、
1 個 8 GB。刻み 20・S=619 なら 31 個/run、288 run で 71 TB がピーク
（オフライン eval 後に削除する前提の同時保有量。89 ノード運用なら同時実行
44 run で約 11 TB）。オフライン eval 自体は 1 checkpoint あたり 0.23 node-h、
全体で 2,000 node-h（学習予算の 7%）。

`N_SAMPLES_PER_PROMPT` が 8 なら全体が半分になる。生成量が prompt-visit × n に
比例するため、これは依然として単一で最大のレバー。

削減の効き方（S=300, A=3, L=3 の 89,400 node-h から）:

| 削減 | 後 | 削減率 |
|---|---|---|
| LR を 3 → 2 水準 | 59,600 | −33% |
| algorithm を 3 → 2 | 59,600 | −33% |
| staleness を 7 → 5（16, 32 を落とす） | 63,900 | −29% |
| length を 4 → 3（8k か 16k を落とす） | 68,300 〜 68,900 | −23% |
| n を 16 → 8 | 44,700 | −50% |

4h の `batch` allocation では 1 run が複数 allocation に分かれる。再開時の起動
（13.5 分）は wall-clock 軸から除外されるが GPU 時間としては実在する。
`SAVE_INTERVAL` で再開できるので正しさの問題ではない。

### 5.5 grid から外して単独で測るもの

どちらも「off-policy 度合いを上げたときの速度」という主軸とは別の問いで、
全交差に載せると乗数になるわりに交互作用が薄い。1 本の軸（staleness）上の
数点で測る。

#### `PAUSE_GENERATION_MODE`

重み更新時の in-flight 生成の扱い。`in_place` は PipelineRL 型の継続（旧重みで
作った KV cache 上で再開、re-prefill なし → 速いが mismatch 源が増える）、
`retract` は KV 再計算、`abort` は破棄して再生成。**コストの差が出るのは
interruption が頻繁に起きるときだけ**なので、staleness が実際に binding して
いる水準を 1 つ選び、そこで 3 モードを比べる。読む metric は
`aborted_groups_recycled`, `aborted_tokens`, `wasted_token_frac`,
`perf/rollout_time`, `train/tis_abs`。

#### バッチ形状と自然発生 lag

実現 lag は **weight version 単位**で測るので、`(1 group の生成時間) /
(1 optimizer step の時間)` に支配される。分母は `ROLLOUT_BATCH_SIZE` に比例するので、
バッチを変えると発生する lag が変わる。

**これを主 grid に入れない。** 理由は 2 つ。(1) lag を作るために非標準の
バッチ形状を採ると、先行研究の Stale-k と同じく「作為的に構成した設定」になり、
「現実的な recipe」という本実験の枠組みが壊れる。(2) LR やアルゴリズムを固定した
上で単独に振るほうが、交絡なく係数が取れる。

したがって **`ROLLOUT_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` は先行研究の標準値に固定し**
（§5.2）、「バッチ形状を動かすと自然 lag がどう動くか」は deterministic kernel と
同じく**別軸の単独検証**とする。

**lag は定常状態を測る。** 起動直後はキューが空なので lag は 0 から始まり、生成が
学習に先行するにつれて積み上がって定常値に落ち着く。したがって報告すべきは
「1 点の lag」ではなく**時系列と、その定常部分の分布**である。
`staleness/total/{mean,p50,p90,p99}` は **rollout step ごとに**記録される
ので、立ち上がりと定常部分は事後に分離できる（`analyze.py` の plateau 判定と同じ
考え方を lag に適用すればよい）。

**現実的な設定で大きな lag が発生しないなら、それが結果である。** 「発生させる」
ために設定を歪める必要はない。

#### deterministic kernel

`--true-on-policy-mode` は品質軸の control であって速度軸の水準ではない。
qwen3_dense contract は Megatron の sequence-parallel を無効化し、
`--transformer-impl local` と `--use-cpu-initialization` を強制し、
batch-invariant mode と deterministic TP allreduce を有効化し、rope / swiglu
fusion を無効化する（`true_on_policy/schema.py:26`, `config.py:60-95`）。
**すべてスループットを下げる**ので、この軸を動かすと測定対象の wall-clock 軸
そのものが動く。

代わりに 2 つの問いに分けて、少数の run で答える。

1. **mismatch を消すと収束は良くなるか** — step を揃えた品質比較。実測で
   `train/train_rollout_logprob_abs_diff = 0.0100`（staleness 0 時）が実装由来の
   mismatch の床。deterministic ではここが 0 に近づくので、**「off-policy による
   劣化」と「数値 mismatch による劣化」を分離できる**。査読で必ず突かれる交絡
2. **スループット低下は割に合うか** — 同一 staleness 内の比較なので全交差は不要

**制約: 完全な parity は colocated でしか取れない。** contract の
`logprob_contract` は `sglang_prefill` のみ（`schema.py:10`）で、`--fully-async` は
`--recompute-logprobs-via-prefill` を明示的に拒否する（`arguments.py:55-57`）。
「rollout/training mismatch が完全に消えた状態」を見たいなら colocated 側の実験になる。

---

## 6. 固定するもの、観測するもの、塞ぐべき穴

### 6.1 固定する変数とその理由

| 変数 | 値 | 理由 |
|---|---|---|
| dynamic sampling | **off**（`--dynamic-sampling-filter-path` を渡さない） | §4.0。step 時間を決定的にし、wall-clock 軸を守るため |
| プロンプト集合 | ポリシーごとに測定した難易度フィルタ済み集合 | §4.1–4.2 |
| KL coefficient | 0 | DAPO 準拠。参照モデルも読まない |
| `RM_TYPE` | checkpoint ごと | 探索対象ではなく正しさの設定（§9） |
| R3 (`--use-rollout-routing-replay`) | MoE では常時 on | routing mismatch の除去。MoE RL は無しでは崩壊することが知られている |
| `ROLLOUT_MAX_CONTEXT_LEN` | 32768 | 全モデル共通 |
| 評価設定（学習中 / オフライン両方） | §2 の表 | arm 間で評価予算が違えば `Q_m(t)` は比較不能 |
| `SAVE_INTERVAL` / `EVAL_INTERVAL` | 20（`R=64`, S≈619 で 31 評価点） | `Q(t)` の時間分解能そのもの（§2, §5.4）|
| 並列度・engine 形状・`MAX_TOKENS_PER_GPU`・`SGLANG_MEM_FRACTION` | model ごとに 1 回調整して凍結 | throughput 専用。動かすと **wall-clock 軸が無効になる** |

### 6.2 現状の動作点 — 主軸がまだ動いていない

監査済み async run（`dapo-math-p10-90` / `qwen3-4b-instruct-2507`, job 15113756、
**dynamic sampling on の時点**）:

| 観測 | 値 | 意味 |
|---|---|---|
| `weight_version/{min,max}` | 全 step で `1 == 1` | — |
| `mixed_version_ratio` | 全 step で `0.0` | **設定した staleness が一度も binding していない** |
| `perf/wait_time_ratio` | 0.83 | 学習側が step の 83% を待機。**rollout が律速** |
| `train/tis_abs` | 0.0100 | staleness ゼロでの mismatch の**数値的な床**。policy lag に帰属させる前に差し引く定数 |
| `train/tis_clipfrac` | 4.9e-6 | 補正はほぼ効いていない |

rollout が律速なので生成は常に最新重みで始まり、lag が発生しない。
**この状態で `MAX_WEIGHT_STALENESS` を振っても全 arm が on-policy として走る。**

dynamic sampling を外すとこの状況は改善する方向に動く：実測で rollout time が
761 s → 253 s（3×）に戻るので、律速が学習側に移り、生成が学習に先行してキューが
埋まる — AReaL が想定する動作点そのもの。ただし**移るかどうかは測って確認する**
必要があり、それが §7 Stage 1 の完了条件（`mixed_version_ratio > 0`）である。
足りなければ `ASYNC_MAX_CONCURRENT_SAMPLES` と actor/rollout 分割で追い込む。

ここが未達のまま出した staleness カーブは「平坦」に見えるが、それは頑健性ではなく
**未実施**である。

### 6.3 観測する metric

| 軸 | metric | 用途 |
|---|---|---|
| 品質（報告値） | オフライン eval の avg@16 × 4–5 年 | `Q(t)`, `τ_m(δ)`, `S_m(p)` |
| 品質（監視） | `eval/aime25` | 学習しているかの確認、崩壊検出 |
| 速度 | `perf/rollout_time`, `train_wait_time`, `wait_time_ratio`, `tokens_per_gpu_per_sec` | arm が速い**理由**。どちらが律速か |
| off-policy 実現度 | `staleness/rollout/{mean,max,p50,p90,p99,frac_zero}`, `staleness/total/*`, **`staleness/bound_exceeded_sample_frac`**, `fully_async/train_weight_version`, `rollout/weight_version/{min,mean,p90,max}`, `rollout/weight_version/mixed_version_ratio` | **設定値ではなく実現値**。ここが動いていない結果は無効（§6.2）。`rollout/max` が上限未満かつ排除率が 0 なら上限は binding していない |
| drift | `train/tis`, `tis_abs`, `tis_clipfrac`, `train_rollout_logprob_abs_diff`, `train_rollout_kl`, **`rollout_ess_ratio`** | 実現した mismatch。0.0100 の数値床を引いてから policy lag に帰属させる。`train/ess_ratio` は**別物** — PPO 内側ループの ESS で `NUM_STEPS_PER_ROLLOUT=1` では恒等的に 1.0。長系列での ESS 崩壊は `rollout_ess_ratio` で見る |
| 崩壊ガード | `rollout/raw_reward`, `truncated_ratio`, `repetition_frac`, `train/grad_norm`, `train/entropy_loss` | 収束定義の前提条件。entropy は `--observe-training-entropy` が要る（§6.4） |
| 無駄 | `stale_groups_recycled`, `aborted_groups_recycled`, **`{stale,aborted,dynamic_filter}_tokens`, `kept_tokens`, `wasted_token_frac`** | off-policy 化のコスト側。個数ではなくトークンで測る |
| 応答長分布 | `rollout/response_len/{mean,median,p90,p99,max}`, `truncated_ratio` | 応答長軸（§5.1）の実態。上限ではなく実際の分布が lag を決める |

### 6.4 計測の穴 — 開始前に埋めるべき差分

[offpolicy_acceleration の監査](../../src/offpolicy_acceleration/README.md#is-the-current-logging-enough)
から、本レシピに直接効くもの。**これらが未修正のまま走らせた run は解析できない。**

| status | where | 変更 | 埋まる穴 |
|---|---|---|---|
| **済** | 全レシピ | `--dynamic-sampling-filter-path` を削除 | §4.0 |
| **済** | 全レシピ | `--observe-training-entropy` を追加 | `train/entropy_loss` が恒等的に 0 で entropy collapse 判定が実行不能だった |
| **済** | datasets | AIME-2023 を配置、`run_eval.sbatch` の既定を 4 年に | §2 の 4 年構成 |
| **済** | datasets | AIME-2024 に instruction wrapper を付与し 4 年を統一 | 年をまたいだ比較可能性（§2） |
| 未 | 全レシピ | `SEED` を `CONFIG_TAG` に、`--seed` / `--rollout-seed` を `train.sh` に | 現状 2 seed が**同じ checkpoint ディレクトリを共有して互いの optimizer state から resume する**。seed 複製が物理的に不可能 |
| **済** | `fully_async_rollout.py` | staleness の percentile / max-staleness 排除 sample 数・率 / `train_weight_version`、router クエリ失敗の warning 昇格 | **設定済みでも staleness metric が 1 件も出ていなかった。** 原因は router `/model_info` の失敗が `logger.debug` に落ちること。`current` が None になると**メトリクスが出ないだけでなく上限自体が黙って無効化される** |
| **済** | `fully_async_rollout.py` | 廃棄生成をトークンで計上 | 個数では「捨てた量」が測れない。サンプル効率の主張に必要 |
| **済** | `loss_hub/losses.py` | `train/rollout_ess_ratio` | 長系列での ESS 崩壊は平均（`tis_abs`）からは判定できない |
| **済** | `metric_utils.py` | `compute_statistics` に p90 / p99 | 応答長の裾。`max` だけでは「1 本だけ長い」と「1 割が長い」を区別できない |
| **済** | `arguments.py` | `--no-dump-policy-loss-debug`、全レシピに追加 | dump の 76%（実測 1.17 GB / 2512 files / 12 step）を占め、本研究では読まない。41 GB → 11 GB/run |
| **済** | `dashboard/args.py` | 研究の因子を `_SNAPSHOT_KEYS` に | run が自己記述的になり、ディレクトリ名に頼らず cell に帰属できる |
| 未 | 全レシピ | `SEED` を `CONFIG_TAG` に、`--seed` / `--rollout-seed` を `train.sh` に | §7 Stage 3 の前提 |
| 未 | レシピ | `ADVANTAGE_ESTIMATOR` / clip / TIS の env 化 + `sweep.py` の未消費 knob 検出 | **現状、未配線の knob を sweep しても同一設定の run が別名で並ぶだけでエラーが出ない**。algorithm 軸を触る前に必須 |
| 未 | `loss_hub/` | CISPO / VCPO の実装 | §5.1。miles は grpo / gspo / reinforce++ / ppo しか持たない |
| 未 | retention | `{dump}/rollout_data` と `{dump}/dashboard`、**全 allocation の job log** を保持 | prompt 単位 eval reward と、resume を跨ぐ wall-clock |

`check_logging.py` を各 arm の初回 job に対して必ず走らせる。`FAIL` で非ゼロ終了
するので、sweep 投入のゲートにできる。

---

## 7. 実行順序（ladder）

計算予算は制約ではないが、**回帰の原因が特定できる順序**であることは制約。
一段で一階層しか動かさない。

| stage | パーティション | 動かすもの | 完了条件 |
|---|---|---|---|
| **0. 計測の穴を塞ぐ** | — | §6.4 のレシピ差分、AIME-2023 の配置 | `check_logging.py` が pass |
| **1. 動作点を作る** | `batch_short` | §5.3（分割・engine 形状・token 予算）+ `ASYNC_MAX_CONCURRENT_SAMPLES` | `wait_time_ratio` が 0.83 から下がり、**`mixed_version_ratio > 0`**。ここを通らない限り主軸は測れない |
| **2. バッチ形状を決める** | `batch_short` → `batch` | `ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT` | `zero_std_group_frac` と `grad_norm` が納得できる水準。以降凍結 |
| **3. on-policy 参照 + seed 複製** | `batch` | seed のみ | `Q_on*` と run 間分散の推定値 |
| **3.5 algorithm を実装** | — | CISPO / VCPO を書き、env 化し、`sweep.py` に未消費 knob 検出を入れる | A が確定 = 総実験数が確定（§5.4） |
| **4. 主軸 full grid** | `batch` | §5.1 の全格子 | `τ_m(δ)`, `S_m(p)` |
| **5. アルゴリズム系アブレーション** | `batch` | `--eps-clip-high`, `--grpo-std-normalization`, MIS の設定面 | 各補正の寄与 |

Stage 3 が Stage 4 より前にあるのは、**run 間分散が未測定のうちはどの差も
「本物」と言えない**から。on-policy（colocated）側で seed 複製を持ち、そこで
得た広がりを grid 全体に適用する。

---

## 8. モデルの拡張計画

| 順 | モデル | 位置づけ | 前提作業 |
|---|---|---|---|
| 1 | **Qwen3-4B** | 主対象。全 grid をここで回す | policy-specific filter 測定済み（n=16, 16k） |
| 2 | Qwen3-1.7B-Base + SFT | パラメータ規模の下端 | 難易度再測定、verifier 確認 |
| 3 | Qwen3-8B-Base + SFT | 同上端（dense） | 同上 |
| 4 | Qwen3-30B-A3B-Base + SFT | MoE。R3 が常時 on | 同上 |
| 5 | Qwen3.5 系 | **世代**が変わったときに結論が保つか | 同上 |

問いは「off-policy の折れ点はパラメータ規模と世代に対して**どう動くか**」——
安定かどうかの yes/no ではなく、**scaling の向きを取ること**が目的。向きが取れれば、
実験していない規模（より大きなモデル）への外挿ができる。
規模を変えると生成／学習の比が変わるので、**律速がどちらかも変わる** — つまり
同じ `MAX_WEIGHT_STALENESS` が実現する lag が変わる。設定値ではなく
**実現 lag を横軸にして比較する**こと（`staleness/total/mean` を必ず並べる）。

各モデルで必要になるのは、§4.2 の通り:

1. `measure_pass_rate.py` によるそのポリシー固有の難易度測定
2. `apply_filter.py` で window を切り、**新しい dataset ディレクトリ**を作る
3. `verifier_preflight` で `RM_TYPE` の妥当性確認（SFT の出力フォーマット依存）
4. §5.3 のスループット調整をそのモデルで 1 回

`qwen3-4b`（hybrid thinking）は §4.0 の通り post-training RLVR の不動点にいるので、
伸びしろのあるモデルではなく**崩壊検出器**として扱う。

---

## 9. ハイパラではないもの

`RM_TYPE` は探索対象ではなく正しさの設定。`deepscaler` は応答に `</think>`
区切りが無いと 0 を返す。現在の `qwen3-4b` レシピは hybrid-thinking checkpoint
を使うため `deepscaler` を既定にしている。SFT 済み Base モデルは SFT の
フォーマット次第なので、毎回 preflight で確認する。

## 10. 未決事項

1. **`math/sync/dapo-math-p10-90/` に README が無い。** `dapo-math` を消した際に
   worked example だった `math/sync/dapo-math/README.md` も消えた。contract は
   「README が無いデータセットは staged ではない」としているので、colocated 側にも
   README が要る。本ファイルから async 固有の節を落とした版になる。
2. **`S_m(p)` の報告形式。** AReaL の 2.77× はスループット比なので、
   「スループット比は N× 、`S_m` は M×」と両方出す形にするか。
3. **フィルタ window (0.1–0.9) を grid に含めるか。** 現状は固定扱いだが、
   dynamic sampling を外した以上これが唯一の難易度制御であり、`ROLLOUT_BATCH_SIZE`
   と同じく「勾配に寄与する group の割合」を決めている。再測定は不要（window の
   切り直しは CPU 数秒）なので、含めるコストは低い。

   > **メモ（未採用）。** これを task 難易度の軸として使う案。別データセットを
   > 並べると難易度・ドメイン・出力フォーマット・verifier が同時に動くので難易度の
   > 効果を分離できないが、window を切り直す（`p20-40` / `p60-80` …）なら
   > **同一データセット・同一 verifier のまま、実測 pass rate という数値尺度の
   > 難易度だけ**が動く。予想される機構は、難易度が上がる → group 内 pass rate が
   > 下がる → `zero_std_group_frac` が上がる → 実効的な勾配信号が薄くなる →
   > stale な勾配の相対的な害が増す、で「難しい task ほど許容 lag が小さい」。
   > 検証は `zero_std_group_frac` × staleness。採否は未定。

## 11. より広い変数空間

off-policy study の完全なカタログ — miles が起動時に拒否する組み合わせ、
algorithm/clip/IS の面、固定するものとその理由、後でサンプル効率を主張するために
記録が必要なもの — は
[`notes/off-policy-variables.md`](../../notes/off-policy-variables.md) にある。
