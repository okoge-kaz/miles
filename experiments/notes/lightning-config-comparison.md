# Lightning async configuration differences

Reviewed 2026-09-12. This compares the current Miles recipe, including the user's
SFT/checkpoint/W&B changes, against:

1. `experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/{run.sbatch,train.sh}`.
2. `/lustre/fsw/portfolios/coreai/users/kfujii/src/pipeline/RLVR/nemotron-3.5-lightning/configs/lightning_rlvr_86n.yaml`.

The NeMo column describes the YAML itself, with launcher overrides identified
explicitly. The Lightning entrypoint defaults `MODEL_PATH` to the same
`upsampled-iter6000/hf` SFT source used here; `pipeline/tools/launch.sh` passes it
as `policy.model_name`. Thus the initial policy source is **not a difference
from the launcher defaults**, although the literal YAML names the public model.
The entrypoint now defaults to `lightning_rlvr_86n_sc.yaml`; this document still
compares the specifically requested `lightning_rlvr_86n.yaml`.
The exact reviewed YAML is
copied under `experiments/outputs/lightning-node-ratio/reference/`, together with
its SHA-256. Miles values come from both launch scripts, the Nano model-argument
definition, and relevant Miles loss/model-provider code. Bridge may supply
model-specific defaults, so flags absent from this recipe are explicitly
identified rather than assumed to equal NeMo defaults.

The node-ratio sweep is **not a reproduction of the 86-node RLVR recipe**.
Task distribution, batch, sequence lengths, and several loss settings differ.
After review, LR is 4e-6, top-p is 1.0, advantages are clipped to [-20,20], and
warmup is explicitly disabled. One-step training reuses its gradient-enabled
forward via `--fuse-one-step-actor-logprobs`, preserving rollout routing replay.
The fused guard now allows this preloaded rollout-routing path without shadow
verification; trainer-only routing replay and indexer replay remain rejected.
The user accepted the allocation/batch/staleness differences, disabled MTP,
and retained the Miles math verifier and seeds. The latest instruction removes
forced zero reward on truncation and sets a fresh-run horizon of 300 updates.
TIS remains [0.2,5]. The response-wide train/rollout log-prob filter now defaults
OFF for an ON/OFF sweep; when enabled, its threshold 2 rule matches the YAML.
Leave-one-out was explicitly declined;
the reasoning and research references are retained in `lightning-loss-review.md`.

## Differences from Qwen3-4B async

| Setting | Qwen3-4B async | Lightning async |
|---|---|---|
| Initial model | Qwen3-4B Base LR2e-5 Step4000, `iter_0004000` | Local Lightning SFT `upsampled-iter6000`, policy view |
| HF mount | Step4000-specific parent | User's general HF parent, distinct SFT model directory |
| Model / conversion | Dense Qwen3; standard conversion path | Hybrid Mamba/attention/MoE; NVIDIA Bridge |
| Data | Qwen-specific DAPO p10–90 | Unfiltered DAPO-MATH-17K |
| Dynamic filtering | No dynamic filter passed | Mixed-reward group filter ON |
| Batch | 192 prompts × 16 = 3072 × 1 update | 420 prompts × 8 = 3360 × 1 update |
| Default allocation | 2 nodes; 1 train node, remaining GPUs for rollout | 32 nodes; default 4:28, overridden by the eight-point sweep |
| TP / CP / EP | 2 / 1 / 1 | 2 / 1 / 4 |
| Context / response | 32768 / 16384 | 18432 / 16384 |
| Max tokens per GPU | 32768 | 18432 |
| Queue | Defaults to 1000, promoted to at least 6000 for s8+ | Fixed at 6000 for both s8 and s1 |
| Async admitted samples | Unset; producer derives its default from rollout batch | 3584 at 32/32; 7168 at 64/64 (draft), each fixed across node ratios |
| SGLang memory fraction | 0.70 | 0.70 |
| Max running requests / graph BS | Optional external overrides | 32 / 32 and 64 / 64 serving profiles |
| Generation top-p | 1.0 | 1.0 (aligned after review) |
| Reward | `deepscaler`, configurable | `math`, fixed; thinking template enabled explicitly |
| Truncation | Exposed zero-reward/zero-loss alternatives | No forced zero reward or zero loss |
| Rollout horizon | 300 | 300 (300 updates for a fresh run); four-hour wall limit also applies |
| Optional per-launch stop | `DEBUG_EXIT_AFTER_ROLLOUT` exposed | Now exposed as well; empty by default |
| Checkpoints | Megatron/HF every 10, retain interval 100 | Megatron/HF every 10, retain interval 50 |
| Scheduler on resume | `USE_CHECKPOINT_OPT_PARAM_SCHEDULER=0` default | Always load saved scheduler settings |
| One-step actor log-probs | `--fuse-one-step-actor-logprobs` ON | Same flag ON; preloaded rollout routing replay also supported |
| Learning rate / advantage clip | 1e-6 / disabled | 4e-6 / [-20,20] |
| Routing replay | Not passed by this Qwen recipe | ON; Lightning SGLang compatibility overlay |
| Detailed telemetry | Update diagnostics and policy-lag metrics ON; many debug switches | Core timings, queue/staleness, sample staleness, dashboard; those extra switches are not enabled |
| Importance correction | TIS default with selectable alternatives / denominators | TIS [0.2,5]; optional response-wide log-prob filter at 2, default OFF |
| Run identity | Derived by `common/run_identity.sh`, including algorithm knobs | Unique sweep namespace plus s/T/R; all settings retained in the review/submission manifests |
| Checkpoint cleanup | Sources `clean_checkpoint.sh` | Does not invoke checkpoint cleanup; sweep requires fresh checkpoint paths |
| W&B | Required | Now also required, project `async-rl-nemotron-lightning` |
| Static review | Normal recipe execution | `--print-config` validates actual defaults and placement without Slurm |

Common settings include GRPO, constant LR schedules, PPO clip 0.2/0.28, Adam
betas 0.9/0.999, epsilon 1e-8, weight decay zero, grad clip one, one update per
rollout, queue-recycle/prefill/in-place, inflight replay, and evaluation OFF.
The new Lightning scripts follow Qwen's organization; they intentionally do not
inherit every algorithm variant or optional diagnostic knob.

## Differences from `lightning_rlvr_86n.yaml`

### Allocation, task, and batch

| Setting | NeMo RLVR YAML | Current Miles Lightning | Reason / consequence |
|---|---|---|---|
| Total resources | 86 nodes: 32 train + 32 policy + 22 judge/service nodes | 32 train/rollout nodes per arm | Requested fixed-budget ratio study; no judge-model pools |
| Train/policy split | 32:32 | 4:28, 6:26, 8:24, 10:22, 12:20, 14:18, 16:16 | Requested sweep |
| Train TP / CP / EP / PP | 4 / 4 / 16 / 1 | 2 / 1 / 4 / 1 | Different per-rank workload and communication |
| Dataset / environments | NeMo Gym mixed RLVR tasks, with launcher-supplied dataset paths | DAPO-MATH-17K only | Different task and reward distribution |
| Judges / tools | Safety, GenRM, NL2Bash and other Gym servers | Local math verifier | Different reward pipeline and rollout cost |
| Shuffle | `data.shuffle=false` | `--rollout-shuffle` | Explicitly accepted; retain current ordering behavior |
| Initial weights | YAML names public BF16; entrypoint overrides with SFT `upsampled-iter6000` | Same user-specified SFT source, MTP-free policy view | Policy source aligned with launcher defaults |
| Prompt groups × responses | 512 × 16 | 420 × 8 | Both have one update per batch; Miles GBS is the DP LCM |
| Global batch | 8192 | 3360 | Fixed across all Miles arms; TP=2 confirmed by user |
| Dynamic sampling | `use_dynamic_sampling=false`, ratio 1.0 | Mixed-reward filtering ON, continuous replenishment | Explicit request for unfiltered DAPO |
| Sampling concurrency | vLLM generation batch 64; other admission is NeMo-specific | 32/32: 3584 trajectories; 64/64: 7168 (draft); each admission budget fixed across ratios | These are different controls and must not be equated |
| Data/iteration limit | `max_num_epochs=1`, `max_num_steps=100000` | `NUM_ROLLOUT=300`, no one-epoch cap | User explicitly requested 300 updates for a fresh run and accepted no epoch cap; four-hour allocation limit also applies |

### Learning objective and generation

| Setting | NeMo RLVR YAML | Current Miles Lightning | Consequence |
|---|---|---|---|
| Learning rate | 4e-6 constant | 4e-6 constant | Aligned after review |
| Warmup | 10 iterations, starting at 4e-7 | Explicitly 0 iterations | User requested no warmup |
| TIS bounds | [0.2,5] | [0.2,5] | Aligned after review; token-level clipping |
| PPO ratio | `force_on_policy_ratio=true`: current log-probs detached as denominator | Training-forward log-probs detached as denominator | Both use ratio one in the single update; async TIS still compares actor against generation probabilities |
| Reward baseline | Leave-one-out enabled | Group-mean centering and group standard deviation | User explicitly declined leave-one-out |
| Advantage clipping | [-20,20] | [-20,20], after normalization | Aligned after review |
| Sequence log-prob filter | `seq_logprob_error_threshold=2`, ON | Default OFF; `USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER=1` enables threshold 2 | User requested OFF by default for an ON/OFF sweep. When ON, same response-mean symmetric error rule and post-filter token normalization, using the training forward |
| Async age | `max_trajectory_age_steps=1` | Max weight staleness 8, plus one s1 arm | Different bounds; even s1 uses Miles queue/prefill semantics, not an assumed identical NeMo admission rule |
| Context / generation limit | Both configured as 73728 (shared max length) | Context 18432, response 16384 | Explicitly accepted: DAPO math and mixed RLVR have different task difficulty and generation budgets |
| Temperature / top-p | 1 / 1 | 1 / 1 | Aligned after review |
| MTP | Five repeated/detached layers, loss scale 0.3 | MTP disabled; 270 auxiliary tensors excluded from the policy index | Different trained architecture/objective |
| Router weights | `freeze_moe_router=true` | Trainable | No dedicated existing option; implementation assessment below, freeze not yet implemented |
| Rollout routing replay | No `policy.router_replay` block; helper defaults OFF | `--use-rollout-routing-replay` ON | Explicitly requested; expert indices are replayed, router weights remain trainable |
| Router bias update | 1e-3 | 0 from the Nano model args, propagated to Bridge | Different routing updates |
| Router balancing | `none`, auxiliary coefficient 0 | Model args request `seq_aux_loss`, coefficient 0; Bridge owns unpropagated model defaults | Do not infer the effective Bridge balancing mode from that CLI flag alone |
| Truncation / overlong treatment | `overlong_filtering=false`, reward shaping disabled | No forced zero reward or zero loss for truncated responses | Aligned on truncation alone after the latest request; verifier/penalties still differ |
| Reward penalties | Duplicate reasoning, empty answer, unwanted token, malformed think tag, invalid tool-call penalty | Math correctness; those extra penalties are absent | Different reward definition |
| Thinking template | Gym policy enables thinking and `truncate_history_thinking=false`; reasoning-off service also exists | `enable_thinking=true` on all DAPO prompts | No mixed effort/reasoning-off task distribution |
| Seed | GRPO seed 42 | Train seed 1234, rollout seed 42 | Training RNG differs |

The NeMo ratio behavior is implemented in `nemo_rl/algorithms/loss/loss_functions.py`
(`prev_logprobs = curr_logprobs.detach()` under `force_on_policy_ratio`). Miles
reward normalization is in `miles/ray/rollout/train_data_conversion.py`, and
Bridge runtime propagation is in `miles/backends/megatron_utils/model_provider.py`.
These distinctions cannot be resolved by matching parameter names alone.

### Runtime, memory, saving, and logging

| Setting | NeMo RLVR YAML | Current Miles Lightning |
|---|---|---|
| Policy backend | vLLM, TP=4, EP=1 | SGLang, one GPU per engine |
| MoE inference | `flashinfer_cutlass` | Triton, chosen after the prior Lightning live-weight-update failure with automatic TRTLLM |
| Serving memory fraction | 0.85 | 0.70, explicitly requested |
| Max sequences / CUDA graphs | 64, graph sizes through 64 | 32/32 and 64/64; the latter aligns the numeric limits |
| Mamba SSM cache | Explicit float32 | No explicit cache-dtype override in the Miles recipe |
| Training packing | NeMo sequence packing, modified first-fit decreasing, 73728-token budget | Miles dynamic batching/balancing, 18432 tokens per GPU |
| Log-prob chunk | 2048 | 128 |
| Optimizer implementation | Precision-aware distributed Adam, explicit float32 params | Distributed Adam through Miles/Megatron; no equivalent precision-aware/params-dtype override |
| Gradient communication | FP32 reduction false; overlap parameter gather true | Explicit FP32 gradient accumulation/all-reduce; no overlap parameter-gather flag |
| Gradient accumulation fusion | false | true by the selected image's Megatron CLI default, propagated into Bridge |
| Fusions | Several explicit NeMo compute/MoE fusion settings | Those overrides are not all exposed or propagated by this Bridge recipe |
| Shared expert overlap | false | Not explicitly overridden; Bridge model default applies |
| Checkpoint formats | Safetensors model + optimizer/FT checkpoints | Resumable Megatron `torch_dist` plus HF safetensors exports |
| Checkpoint periods | Model every 10; FT every 1, latest FT 1 | Both Megatron and HF every 10 |
| Retention | `keep_top_k=1000000`, FT latest 1 | Megatron retain interval 50; HF series every 10 |
| Save deadline | `checkpoint_must_save_by=00:23:35:00` | No graceful save deadline; Slurm wall time 04:00:00 (shared NeMo launcher also passes allocation deadline/reserve) |
| Evaluation | Start/end false, period 10000000 | OFF; retained HF available for offline eval |
| W&B | YAML false; shared launcher enables it when `WANDB_API_KEY` is present | Always required, explicitly requested |
| GPU monitoring | 10-second sampling/flush | Miles dashboard defaults: GPU 1 second, engine scrape 2 seconds, flush 5 seconds |

Both use BF16 model computation, sequence parallelism, all-to-all MoE dispatch,
Adam betas 0.9/0.999, epsilon 1e-8, weight decay zero, gradient clip one, PPO
clip 0.2/0.28, and no reference-policy KL penalty. These common values do not
make the objectives or runtime numerically equivalent.

TIS bounds are aligned; the sequence-filter rule matches when explicitly enabled,
but its default OFF differs from the YAML by the user's latest request. Remaining differences
and their explicit authorization status are recorded in the Japanese decision
tables below. Matching the filter does not align the reward/baseline, routing,
or runtime. Memory fraction is 0.70 as requested; the SFT async recipe still
needs GPU qualification.

## Review explanations

For K responses, leave-one-out uses `b_i = sum(r_j for j != i)/(K-1)`.
This NeMo implementation also estimates each response's standard deviation
from the other valid responses. When that standard deviation is zero, the
current NeMo GRPO estimator skips division and keeps the centered reward;
it does not divide by epsilon alone. Miles currently uses the full group's
mean and sample standard deviation. Only clipping was aligned; leave-one-out
was explicitly declined. Research references and the estimator distinction
are recorded in `lightning-loss-review.md`.

The sequence filter computes the response-token mean of
`exp(abs(log p_generation - log p_actor))`, then masks the entire response when
that mean exceeds 2. It is not a reward-difficulty filter. NeMo's
`_resolve_logprob_skip_flags` explicitly keeps the separate actor scoring pass
when this filter is configured, even with `force_on_policy_ratio=true`.
Miles now computes this detached filter in the training forward, then passes
the retained-token count to Megatron for normalization across microbatches and
DP/CP. A separate actor scoring pass remains skipped. NeMo establishes the mask
before advantage processing; Miles retains the already computed GRPO advantages
as requested. The reward-group filter and sequence filter are separate stages. Both TIS
and this sequence filter are enabled by the YAML; they are not alternatives.

In the requested YAML, `overlong_filtering=false`, reward shaping is disabled,
and `stop_properly_penalty_coef=null`: truncation alone neither zeros reward
nor removes the response from loss. Gym/judge scoring still applies. Empty
final answers or malformed think tags caused by truncation can independently
trigger the enabled zero-reward penalties. The latest user request removes
Miles' `--zero-reward-on-truncated` rule. Truncated responses now receive normal
math-verifier scores and are not excluded from loss solely for truncation.

## ユーザーの明示了承を再確認した差分一覧（2026-09-12）

ここで「了承済み」は会話で明示されたものに限る。実装上の初期値や、質問への
説明後に返答がない項目を了承済みとは扱わない。比較は指定された86n YAMLと
現在のMiles script。別のNeMo実行での環境変数やCLI上書きまでは推測しない。

### 明示了承・指定済みなので維持する差分

| 項目 | NeMo YAML → Miles | 会話上の根拠 |
|---|---|---|
| 総ノード数・T:R | 86、32:32＋judge → 32、指定8条件 | 「ノード数は異なって問題ありません」「train:rollout ratioも異なって問題ありません」 |
| Train TP・GBS | TP4・8192 → TP2・3360 | 「batch=3360、TP=2で揃える」 |
| Prompt/response batch構成 | 512×16 → 420×8 | batchの差異は許容、rollout batch × n = GBS × 1を指定。8 responsesという個別値への独立の明言はない |
| データ・reward | 混合Gym RLVR＋各種penalty → DAPO-MATH＋math verifier | DAPO-MATHを指定。最新の「zero-reward-on-truncatedはなし」で旧truncation処理を上書き。truncation単独ではzeroにしない点はYAMLと一致 |
| Sequence log-prob filter | 閾値2でON → default OFF、ON時は閾値2 | 「sweepの中でon/offをしますので、defaultではONにしない」と明示指定 |
| Dynamic filtering | OFF → reward非zero-std groupのみ採用、連続補充 | difficulty filteringなしなのでONを明示指定 |
| Running requests / CUDA graph | 64 / max64 → 32/32と64/64を比較 | 「64, 64も追加してください」と明示指定。各node-ratio/staleness点に両条件を追加 |
| Rollout routing replay | 未設定、NeMo helper default OFF → ON | 「Rollout routing replay (R3) はONにしてください」と明示指定 |
| データ巡回・update上限 | max_num_epochs=1、max_num_steps=100000 → epoch capなし、NUM_ROLLOUT=300 | epoch capなしを了承済み。最新の「新規で300 stepで止めたい」でupdate上限を指定 |
| Warmup | 10 → 0 | 「warmupは特に入りません」 |
| Advantage baseline/std | leave-one-out → 全グループの平均・標準偏差 | 「Leave-one-outは今回は採用しません。しかし、noteには残して」 |
| MTP | 5層・loss 0.3 → OFF | 「MTPはなくして良い」 |
| Staleness | age 1 → max weight staleness 8、追加s1 | sweep条件を明示指定、差異を許容 |
| Context・response制限 | 73728/73728 → 18432/16384 | 今回「タスクの難易度が大きく異なる」ため差異を許容。DAPO mathと混合RLVRの生成予算を一致させない |
| Train/rollout seed | 42 → 1234/42 | 「seedもそのままでよい」 |
| Serving memory | 0.85 → 0.70 | 「memory fractionですが、0.70」 |
| Gradient / parameter gather | FP32 reduce OFF、param gather overlap ON、grad accumulation fusion OFF → FP32 accumulation/reduce ON、param gather overlap OFF、fusion ON | 今回「gradientやparameter gather周りは異なりますがその旨を記すだけで良い」。変更しない |
| Shuffle | false → ON | 今回「shuffleについても同様」 |
| Actor log-prob scoring | NeMoはfilter用の別forward → training forward内のdetached mask/anchor | `--fuse-one-step-actor-logprobs`で省略を明示指定 |
| W&B | YAML false、launcherはAPI key次第 → 必須ON | 「常にON」 |
| 保存周期・保持 | model10、FT1/latest1 → Megatron/HF10、retain interval50 | 「10updateごと」「retrain=50」。50は保存アーカイブ間隔であり保持件数ではない |
| 実験時間 | YAML save deadline23:35、launcherはallocation deadlineも指定 → Slurm4h | 4:00:00を指定。終了前のgraceful saveを実装しないことまで明示了承したものではない |

### 未了承、または実装確認が残る差分

| 項目 | NeMo YAML → Miles | 現状・影響 |
|---|---|---|
| Train CP/EP | 4/16 → 1/4 | なぜ選んだかという質問には、旧public Lightningの動作確認設定の継承と全armの割り切りを説明済み。CP1/EP4の差異を許容する明言はない。TP2だけは確定 |
| Router weight freeze | true → 未凍結 | 既存optionならONという依頼に対し専用optionなし。今回の実装可能性確認は下記。freeze自体はまだ実装していない |
| Router bias update | 1e-3 → 0 | `moe_router_bias_update_rate`はBridgeへ伝播するため確定差分。freezeとは独立、差異を許容する明言なし |
| Inference backend/並列 | vLLM TP4/EP1 → SGLang 1GPU/engine | 同一ノード数でもengine数・通信・kernel・cache動作が異なる。memory fraction以外の個別設定は未了承 |
| Inference kernel | flashinfer_cutlass、FLASH_ATTN → Triton MoE、attention auto | Triton MoEは過去のLightning weight update不具合への対処として選択済み。YAMLとの差は残る |
| Async admission/queue | NeMo実装依存 → 3584/7168 trajectories（serving profile別）、6000 groups | oversamplingの適切な設定を依頼されて選んだ初期値。値そのものへの明示承認やthroughput tuning実績はない。`OVER_SAMPLING_BATCH_SIZE=2520`はfully-asyncの実効admissionではない |
| Packing・padding | modified first-fit decreasing、round64 → Miles length balancing/dynamic microbatch、独自padding | 長さ上限は了承済みだが、packing algorithmやmicrobatch構成まで同等ではない |
| Log-prob chunk | 2048 → 128 | メモリ/計算効率の差。fusedでもtraining lossでlog-probを求めるchunk制御は残る |
| Optimizer precision | precision-aware ON、params_dtype float32 → 同等の指定なし（image CLIのprecision-aware default OFF） | gradient通信設定とは別。master weight・optimizer stateを含む実効dtypeまで同等とは断定しない |
| 保存実装・終了処理 | NeMo async save/FT専用checkpoint/終了前save deadline → Megatron＋HF export、Slurmによる停止 | 10/50の周期指定は了承済み。FT形式、非同期I/O、graceful saveの差まで明示了承されたわけではない |
| 評価・監視 | 実験規模では実質evalなし（period10M、start/end false） → OFF、HFでoffline eval | AIMEが評価用であることは確認済み。評価OFFという独立の明言はない。GPU監視周期10秒→1秒などの差も未指示 |
| 数値的な実行経路 | NeMoのscoring forwardでmask → Miles training forwardでmask | 省略は依頼どおり。kernel/packingやroutingの違いで閾値付近の判定までbitwise一致する保証はない |

### 設定名だけでは一致・不一致を断定しない項目

- Router balancing: YAML `none`/aux=0、Nano CLI `seq_aux_loss`/aux=0。
  `_apply_bridge_runtime_config`はaux係数を伝播するがbalancing種別は伝播しない。
  Bridge providerの実効値を起動時に確認する必要がある。CLIの違いをそのまま
  activeなaux loss差とは扱わない。
- Shared expert overlap、MoE permute fusion、weighted squared ReLU fusion、
  bias/activation fusion、defer-fp32-logits、SSM cache dtypeは、NeMoが明示する値と
  MilesのBridge/SGLang defaultsをすべて実機で比較できていない。YAML同等とは未認定。
  `apply_rope_fusion`はLightningの位置埋込み構成上、flagの差が実計算差を生むとは限らない。
- ネットワーク、checkpoint並列I/O、GPU cache cleanupなどのbackend defaultsは
  scriptで全部固定していない。GPUを起動せず、実行時のresolved configまで一致を保証しない。

今回再確認して**差分ではなかった**もの: 初期SFTはNeMo entrypointのデフォルトも
同じ`upsampled-iter6000/hf`。LR4e-6 constant、top-p1/temperature1、advantage
clip[-20,20]、TIS[0.2,5]、sequence filterのON時の閾値2、PPO ratio one、token-level loss、
PPO clip0.2/0.28、KLなし、Adam betas/eps・weight decay0・grad clip1は一致する。
Activation recomputeもNeMo setupのdefaultをたどるとfull/uniform/1で一致。
PP1、expert TP1、BF16、sequence parallel、alltoallも同じ。

## Router freezeの実装可能性と作業量

**実装可能。新アルゴリズムは不要で、小規模な追加で済む。ただし現在は未実装。**

NeMoの`nemo_rl/models/megatron/setup.py`はpre-wrap hookで
`layer.mlp.router.weight.requires_grad=False`を設定する。MilesのBridge経路では
`miles/backends/megatron_utils/model_provider.py:wrapped_bridge_provider`が
`provider.provide()`した直後、DDP wrapとoptimizer構築より前が対応する位置。
実装するときは専用flag（例`--freeze-moe-router`）を追加し、router moduleを
型で列挙してweightだけを凍結、対象数と名前を記録し、ONなのに対象0なら失敗させる。
HybridBlockやwrapperを含むため、単なる名前の部分一致やGPT専用の固定パスより
module型を基準にした列挙が確実。criticや他の学習パラメータを巻き込まない。

DDP後に凍結するとgrad-ready待ち、optimizer後だと不要なstate保持の問題が出るため、
単にtraining開始直前にrequires_gradを変える実装にはしない。Frozen routerを通じた
hidden-stateへの勾配は残すので、forward全体を`no_grad`にはしない。

Router weightとexpert biasは別物。NeMo YAMLはweightを凍結しながら
`moe_router_bias_update_rate=1e-3`を使う。Megatronの`finalize_model_grads`には
expert bias更新処理があり、weightのrequires_gradをOFFにしても止まらない。
現在Milesはbias rate=0なので、freezeだけ追加してもYAMLとのrouting差は残る。
Rollout routing replayも独立であり、freezeの代わりにはならない。

見積りは**実装とCPU単体テストで半日程度、SFTでの分散save/resumeまで含む検証で
合計0.5〜1日程度**（allocation待ちと予期しない互換性修正を除く）。主な確認点は
routerだけoptimizerから除外されること、router weightがupdate前後で不変なこと、
他の重みとhidden-state gradientは更新されること、biasの独立した挙動、
TP2/EP4・routing replay・full recomputeでのDDPと保存再開である。
GPU検証は小規模な1〜2ノード構成から行う想定。今回は調査のみで、jobは投入していない。


## R3とfreeze、32/32の設定根拠（追記）

ユーザーはepoch capなしとR3 ONを明示了承。scriptはすでに
`--use-rollout-routing-replay`を渡しており、`--num-epoch`を指定していないので、
設定変更は不要。Dynamic Filteringも`--dynamic-sampling-filter-path`でONのまま。
8回答すべてのrewardが同じgroupを除外し、420 groupを満たすまで連続補充する。
training-forwardのsequence log-prob filterとは別段階であり、後者で落ちた回答を
追加生成で埋め直すことはしない。

R3はrouter freezeを必須としない。一方、R3 ONはrouter weightを固定することと
同等ではなく、Lightning SFT asyncでの安定性を保証するものでもない。
`miles/utils/replay_base.py:get_topk_fn`は保存したexpert indicesを再利用し、
weightは現在の`scores.gather(1, top_indices)`から取るためrouterへの勾配は残る。
新たなrolloutではその時点のrouterでexpertを選択する。Freezeはrouter weightの
更新を止める制約であり、hidden statesが変わればfreeze中でもexpert選択は変わり得る。
[R3論文](https://arxiv.org/html/2510.11370v2)はtraining/inference間のrouting不整合を
減らす効果を示すが、このSFT・staleness8条件でfreezeの要否を直接比較していない。
今回R3を使う構成ではfreeze追加を必須とする理由はないと判断するが、
freeze OFFという差分は残す。ユーザーのfreezeに関する発言は確認質問なので、
freeze OFF自体への独立した明示了承とは記録しない。

「32/32」の二つの値は、次の別々の設定を表す。

| 値 | 意味 |
|---|---|
| `SGLANG_MAX_RUNNING_REQUESTS=32` | 各SGLang engineで同時にrunningとなるrequest数の上限。現在1GPU/engine |
| `SGLANG_CUDA_GRAPH_MAX_BS=32` | Decode CUDA graphのcapture対象batch sizeの最大値。request数に追加する枠ではない |

根拠は以前のpublic Lightning・2ノードcolocated bring-upの設定を継承したこと。
`experiments/outputs/lightning-setup/train-7094021-full.log`の実行argvに両方の32があり、
memory fraction0.60、context18432/response16384、GBS256で1 updateとsaveが成功している。
現状の32はDP/GBSから計算した値でも、16/32/64のthroughput比較で選んだ最適値でもない。
CUDA graphの最大値をrunning上限に揃えているが、両者を同じにする必須条件はない。
新しいSFT・async・memory fraction0.70で32や64の性能比較はまだ行っていない。

現在のrollout fleetは64〜112 enginesなので、設定上のrunning枠合計は
2048〜3584 request。実際のrunning数はsequence lengthやcache容量にも依存する。
`ASYNC_MAX_CONCURRENT_SAMPLES=3584`は最大fleetの112×32を基準に全armで固定した
producer admission上限であり、小さいfleetではserving側の待機を含み得る。
`OVER_SAMPLING_BATCH_SIZE=2520`はfully-asyncの実効入場数を決めない。


64/64追加後のreview planは16条件。producer admissionは32/32で3584、
64/64で7168（112×64）の暫定案とした。ユーザーが明示指定したのはrequest／graphの
64/64追加であり、7168という値への独立した回答はまだない。
両条件3584固定にはsweepの`--set ASYNC_MAX_CONCURRENT_SAMPLES=3584`を使用できる。
run名・checkpointにprofileを含め、集計CSVにもrequest、graph、admissionを記録する。
5 epochへのstep概算と採用率の根拠は[実験note](lightning-node-ratio.md)に追記した。
