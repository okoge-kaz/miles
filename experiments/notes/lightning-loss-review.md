# Lightning loss review — 2026-09-12

今回の実験では **leave-one-outを採用しない**。Milesのグループ平均・
グループ標準偏差によるGRPOを維持し、advantage clippingは[-20,20]とする。
`--fuse-one-step-actor-logprobs`だけを指定し、
`--skip-actor-forward-only`は指定しない。SGLang memory fractionは0.70。
TISはYAMLと同じ[0.2,5]。最新の指示でsequence log-prob filterはdefault OFFとし、
`USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER=1`のsweep条件だけ閾値2でONにする。
`--zero-reward-on-truncated`も最新の指示で削除した。truncatedでも通常のmath verifierで
採点し、長さ上限への到達だけではreward・lossを0にしない。Dynamic FilteringはONを維持する。
新規実行は300 updateで終了する（4時間のSlurm上限によって先に終了する場合がある）。

## Leave-one-outをNeMoが使う理由と研究上の範囲

NVIDIAの[Lightning RL説明](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/lightning35/rl.md)
は採用理由をadvantage推定の分散削減としている。
[NeMo API](https://docs.nvidia.com/nemo/rl/latest/apidocs/nemo_rl/nemo_rl.algorithms.utils.html)
は不偏なbaselineとしてRLOO論文を引用している。

同じpromptから独立にK回答を生成したとき、回答iには
`b_i = sum(r_j, j != i)/(K-1)`を使う。自分の回答をbaselineから除くので、
素のon-policy REINFORCEでは、そのbaselineを引いても期待勾配を変えずに
分散を抑えることが狙える。criticを別に学習せず、既に生成した他の回答を使える。
これは標準偏差による正規化・clipping・async IS truncationまで含む推定器が
元の目的に対して不偏であるという保証ではない。

Ahmadian et al., ACL 2024,
[Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://aclanthology.org/2024.acl-long.662/)
は、RLOOを用いたRLHFを対話・要約で評価し、比較対象のPPOなどに対して
良い結果を報告している。RLOO方式全体の実験であり、今回のLightning＋
DAPO-MATHでLOO baseline単独が現在のGRPOより優れることを示すものではない。

平均だけを比較し標準偏差による除算をしない場合は、代数的に
`r_i - b_i = K/(K-1) * (r_i - group_mean)`。
K固定なら主な違いは一定倍率であり、これだけで学習品質の向上を断定できない。
一方、確認したNeMo実装は標準偏差も自分以外の回答から計算するので、
Milesとの差は一般には一定倍率にならない。
NeMoの`GRPOAdvantageEstimator`はstd=0なら除算せず、centered rewardを使う。
このゼロ分散の扱いは、単に`std+epsilon`で割ることとは異なる。

Liu et al.,
[Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
は、promptごとのreward標準偏差による正規化が問題ごとの重みを変える点を
分析し、Dr. GRPOではその正規化を除いている。LOO自体を否定する研究ではなく、
baseline・正規化・loss reductionを区別して判断すべきことを示す関連研究。
今回、これらの方式変更は行わない。

## Sequence filterとTISは併用されている

対象は`pipeline/RLVR/nemotron-3.5-lightning/configs/lightning_rlvr_86n.yaml`。
YAMLは`use_importance_sampling_correction=true`、type=`tis`、bounds=[0.2,5]、
`sequence_level_importance_ratios=false`に加えて
`seq_logprob_error_threshold=2`を指定している。

生成tokenの同一prefixに対する比を`rho_it = pi_train / mu_rollout`とする。
NeMoのsequence filterが計算するのは、回答iの有効な生成token上の算術平均

`E_i = mean_t(exp(abs(log pi_train - log mu_rollout)))`

`    = mean_t(max(rho_it, 1/rho_it))`。

`E_i > 2`なら、その回答の`sample_mask`を0にする。単純な`mean(rho)`でも、
確率比の積でも、幾何平均でもない。例えば全tokenでrho=0.4ならE=2.5となり
除外される。一方、8 tokenのうち1つだけrho=3、残りが1ならE=1.25で残る。
個々のtokenが2を超えるだけで必ず除外されるわけではない。

これは回答全体のloss寄与をなくす処理であり、報酬を0に変更する処理ではない。
配列から回答を削除して代わりを生成するのではなく、学習maskで除外する。
通常のmaskを省略すると、残る回答のtoken重みは概念上
`1[E_i <= 2] * clamp(rho_it, 0.2, 5)`。
TISが重みを有限範囲に切り詰め、sequence filterが回答単位で寄与を0にする。

実装は`nemo_rl/algorithms/grpo.py:compute_and_apply_seq_logprob_error_masking`
と`nemo_rl/algorithms/loss/loss_functions.py:ClippedPGLoss`を確認した。

## 1 updateとlog-prob forward

1 updateではtraining forwardのlog-probをdetachしてactor分母に使える。
同じ値から上記のsequence maskも計算可能であり、filterという数式自体が
追加forwardを要求しているわけではない。

現在のNeMoはlog-prob推論→sequence mask確定→advantage処理→policy training
の順序を取る。そのため`_resolve_logprob_skip_flags`はfilter設定時に
別forwardを残している。training forwardへ統合するなら、maskのdetach、
microbatch/CPをまたぐ回答集計、全体のloss正規化との整合性も扱う必要がある。
Milesに`--use-train-rollout-logprob-sequence-filter`と
`--train-rollout-logprob-sequence-filter-threshold`（default 2.0）を追加した。
Lightningではfilterを明示的にONにした場合だけ両方を指定する。その場合、回答の全tokenに同じmaskを適用し、CPでは
誤差の和とtoken数を合算して判定する。残ったtoken数をMegatronへ返し、
全microbatchとDP/CPの合計を分母として勾配を正規化する。
報酬・GRPO advantageは書き換えず、追加generationで回答を補充しない。
したがって入力は3360回答のままだが、lossに寄与する回答数はfilter率で変わる。
全回答が除外された場合はloss・勾配・有効token数が0となり、分母は安全にclampする。
optimizer step自体は通常どおり進むため、既存のAdam momentumがあれば
パラメータが必ず不変になるという意味ではない。

オプションはMegatron・fused one-step・per-token loss・標準reducerに限定し、
MTP、independent DP、multi-LoRA、custom TIS/reducerとの組合せを拒否する。
これらの経路へ未検証の正規化を流さないための適用範囲である。NeMo本体は変更していない。

FilterをONにした場合、W&Bとlocal dashboardに`train/train_rollout_logprob_sequence_filter/`以下の
`rejected_sequence_fraction`、`rejected_token_fraction`、`rejected_sequences`、
`kept_tokens`を記録する。fractionはmicrobatch平均ではなく全体の件数比。
通常のtrain loss／TIS metricsはsequence filter後の有効token上で集計する。

`SKIP_ACTOR_FORWARD_ONLY`と`FUSE_ONE_STEP_ACTOR_LOGPROBS`は、別actor scoring
forwardを省略する主目的が共通する。現在のGRPO・1 update・KLなしでは
両方ともtraining-forward log-probをdetachしたactor分母を使い、TISは
そのactor確率と生成確率から計算する。fused側にはGRPO専用の設定チェック、
専用metrics、任意のshadow検証がある。通常のshadow検証はOFF。

LightningではfusedだけをONにし、rolloutから事前に読み込むrouting replayを
維持する。既存のactor実装はscoring前にrouteを読み込み、training forwardで
backward用replayを消費する。fusedの互換性チェックをこの経路に限定して許可した。
trainer側だけでrouteを記録するreplay、indexer replay、routing併用時のshadow検証は
引き続き拒否する。fused + R3 + durable inflight bufferの4ノード保存・再開検証は
[専用note](lightning-replay-buffer-r3.md)に記す。sequence filterはその検証ではOFF。

残る設定差分は[比較表](lightning-config-comparison.md)を参照。

## Durable replay bufferとの併用調査（2026-09-13 UTC）

以下は修正前の調査記録。対応実装と4ノードでの検証は
[durable buffer + R3の専用note](lightning-replay-buffer-r3.md)に記す。

`_validate_replay_buffer`はtrainer/rollout側のrouting/indexer replayを一律に拒否する。
これは現実装の対応範囲であり、手法上の排他関係ではない。Sample codecはrouting配列を
保持でき、`test_packed_sample_codec_is_lossless_and_restores_independent_samples`にも
その検証がある。一方、今回のinflight方式は保存時に生成をabortし、途中のSampleを
継続生成へ渡す。`generate_hub/single_turn.py`は既存tokensを次のrequestのprefixに使うが、
`generate_endpoint_utils.update_sample_from_response`は既存log-probには追記する一方で、
`rollout_routed_experts`と`rollout_indexer_topk`を新しい応答の配列で丸ごと置き換える。
そのため、再開前に生成したtokenの元のrouting情報が失われ得る。

実際のupdate関数本体をASTで単離し、syntheticな継続応答を与えるCPU確認では、
既存prefixのrouting `[1,1,1]`が`[9,9,9]`に上書きされ、旧log-probは保持された。
これは関数の代入動作の確認であり、SGLang/GPUの併用検証ではない。
対応には既存prefixのrouting/indexer情報の保持、新規token部分との接続、token境界と
shapeの整合、保存・継続・job再開を通した検証が必要。guardを外すだけでは対応完了にならない。
inflight保存は通常のcheckpoint周期でも発生するため、新規実行にも関係する。
この調査では引数guard、学習設定、jobを変更していない。
