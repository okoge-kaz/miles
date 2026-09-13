# Lightning 30B-A3B: 32-node train/rollout ratio sweep

The first requested async experiment was submitted as **job 7114153** after the
user authorized T:R=8:24 (details below); it is pending resources, not yet GPU-qualified. The earlier
Lightning qualification covered colocated training, save/resume, and one full
batch-256 update. It does not qualify this separate-node async recipe, batch
3360, the combined Lightning/provenance overlays, or 32-node weight transfers.
GPU training qualification of this SFT async configuration remains pending.

SFT conversion completed in job 7098620 (exit 0, elapsed 00:04:02). The log is
`experiments/outputs/lightning-sft-convert/convert-lightning-sft6000-7098620.log`.
The converted checkpoint is
`/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/megatron/Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000_torch_dist`.
Validation checked 6,243 policy tensors and all 17,398 DAPO prompts; the longest
templated prompt was 1,509 tokens. This conversion check does not qualify the
new SFT async training recipe.

## First submitted experiment (2026-09-12 PDT)

The user explicitly requested one T:R=8:24 experiment before the remaining sweep.
Job **7114153** was accepted under `nemotron_sw_post`, `batch/normal`, 32 nodes,
128 GPUs, `04:00:00`. At the last check it was `PENDING (Resources)`.
Its initial checkpoint is SFT `upsampled-iter6000`; GBS3360/TP2/CP1/EP4,
max weight staleness 8, serving requests/graph 32/32, admission 3584,
sequence filter OFF, dynamic filtering/TIS/R3 ON, no forced zero reward on
truncation, and a fresh-run horizon of 300 updates. Megatron/HF save every 10
updates, with Megatron retain interval 50.

- W&B project: `async-rl-nemotron-lightning` (also the new recipe default).
  A live run URL is pending job startup.
- Run/group: `lightning-sft6000-filter-off-20260913-024232-s8-t8r24-req32-cg32`.
- Plan: `experiments/outputs/lightning-node-ratio/lightning-sft6000-filter-off-20260913-024232-plan.json`.
- Submission manifest: `experiments/outputs/lightning-node-ratio/lightning-sft6000-filter-off-20260913-024232-submitted.jsonl`.
- Slurm log: `experiments/outputs/nemotron-3.5-lightning/math/train-rollout-ratio/lightning-sft6000-filter-off-20260913-024232-s8-t8r24-req32-cg32-7114153.log`.
- Checkpoint: `/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/training/math/dapo-math-17k/Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000/lightning-sft6000-filter-off-20260913-024232-s8-t8r24-req32-cg32`.
  Local dashboard metrics are under its `dump/dashboard/` directory.

Live partition/QoS/account discovery and `sbatch --test-only` passed before
submission. HF/DCP/data/image assets and W&B credential availability were
checked, and the saved plan required a fresh writable checkpoint path.
The user explicitly selected this 32-node, four-hour run; no additional smoke
or tuning allocations or other sweep arms were submitted.

## Additional short allocation (2026-09-12 PDT)

At the user's request, **job 7114396** duplicates job 7114153's learning and
serving settings on `batch/short`, account `nemotron_sw_post`, 32 nodes, with
the QoS limit of **02:00:00**. The 300-update horizon also remains active.
The existing scripts and normal job were unchanged; source hashes match the
original plan. Only allocation QoS/time and run/checkpoint identity differ.
Both were pending at the last check: short `Resources`, normal `Priority`.

- W&B project: `async-rl-nemotron-lightning`.
- New run/group: `lightning-sft6000-short-20260913-031449-s8-t8r24-req32-cg32`.
- Submission specification: `experiments/outputs/lightning-node-ratio/lightning-sft6000-short-20260913-031449-s8-t8r24-req32-cg32-submission.json`.
  This records the actual `sbatch` argv including CLI overrides for QoS/time;
  the unchanged sweep CLI itself still targets four-hour normal allocations.
- Submission manifest: `experiments/outputs/lightning-node-ratio/lightning-sft6000-short-20260913-031449-s8-t8r24-req32-cg32-submission-submitted.jsonl`.
- Slurm log: `experiments/outputs/nemotron-3.5-lightning/math/train-rollout-ratio/lightning-sft6000-short-20260913-031449-s8-t8r24-req32-cg32-7114396.log`.
- Checkpoint: `/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/training/math/dapo-math-17k/Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000/lightning-sft6000-short-20260913-031449-s8-t8r24-req32-cg32`.

Live short limits were rechecked (2 hours, 64 nodes/job and user). Exact
resolved settings, fresh writable checkpoint parent, assets, source hashes,
and `sbatch --test-only` were checked before this single extra submission.
Megatron/HF saving remains every 10 updates, retain interval 50.

## Short-job failure and output organization (2026-09-13 UTC)

The following records the original failure. Durable buffer + R3 support and a
successful four-node save/resume test are now documented in
[the R3 validation note](lightning-replay-buffer-r3.md). Job 7114153 later became
`CANCELLED by 158580`; it has not been resubmitted. Image relocation job 7114760
completed, and the canonical container path is now a regular file.

Job 7114396 failed after 00:03:26, before training. The exception at log line 461
is `ValueError: --use-replay-buffer requires no routing/indexer replay`.
The recipe combines durable inflight replay (`--use-replay-buffer`) with R3
(`--use-rollout-routing-replay`), which `_validate_replay_buffer` rejects.
The earlier fused-logprob compatibility check did not cover this independent
guard. This is an argument-validation failure, not an OOM or a training result.
Pending normal job 7114153 still has the same conflicting settings.

Slurm logs now use `experiments/outputs/nemotron-3.5-lightning/math/train-rollout-ratio/`.
The completed short log was moved there, and Slurm accepted the pending normal
job's `StdOut` update. Learning settings were unchanged by this cleanup.
The recipe's image default and `.env` now point into the user's `container/`.
That path is temporarily an alias while the original image remains available
to job 7114153. CPU relocation job 7114760 waits for the old consumers to end;
see `experiments/maintenance/relocate-runtime-image-20260913.json` and its future
`.completed.json` receipt. The relocation and launcher checks passed 20 CPU tests.

## Files and review

- Sweep: `experiments/staleness/lightning_node_ratio.py`.
- Recipe: `experiments/scripts/math/async/dapo-math-17k/nemotron-3.5-lightning/{run.sbatch,train.sh}`.
- Analysis: `experiments/tools/summarize_lightning_node_ratio.py`.
- Offline verification: `tests/fast/experiments/test_lightning_node_ratio.py`.

`run.sbatch` groups the defaults by role, following the Qwen3-4B async recipe.
Its `--print-config` path resolves and validates the actual defaults without
sourcing `env.sh`, writing assets, or invoking Slurm. `train.sh` contains the
corresponding checkpoint, rollout, telemetry, performance, GRPO, optimizer,
SGLang, and miscellaneous argument arrays.

From the repository root, this command only prints and saves a review plan:

```bash
python -m experiments.staleness.lightning_node_ratio \
  --namespace lightning-sft6000-ratio-review \
  --plan-file experiments/outputs/lightning-node-ratio/review-plan.json
```

Use `--set KEY=VALUE` for shared recipe overrides and repeatable `--point S:T:R`
to select node-ratio/staleness points. Each selected point now runs both serving
profiles: `--serving-profile 32:32` and `--serving-profile 64:64` (requests:graph).
Omitting this option selects both; specifying one selects only that profile.
For example, `--point 1:16:16 --serving-profile 64:64` selects one arm.
The sweep reuses `experiments/sweep.py` batch validation. Eight correlated
node-ratio/staleness points × two serving profiles give 16 arms. The default
`--max-jobs` is 16, and version-1 review plans must be regenerated.
Ambient learning settings are excluded when resolving the plan. All resolved
settings are explicitly exported to the submitted jobs.

Sequence log-prob filtering defaults OFF. Use
`--set USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER=1` for an ON plan (threshold 2),
or `=0` for OFF. Use distinct namespaces, for example
`lightning-sft6000-filter-on` and `lightning-sft6000-filter-off`, and separate
plan files for this comparison so their checkpoint paths remain separate.
This switch does not disable reward-based dynamic filtering or TIS [0.2,5].

The submission interface requires both an explicit `--submit` and a saved
`--plan-file`; it does not accept new setting overrides during submission. It
checks the saved launcher hashes, every placement, fresh checkpoint paths, and
required assets before the first submission. Its manifest records job IDs and
resolved settings and prevents duplicate submission of the same namespace.
Use a new namespace for a new experiment. The launcher hash check covers the
listed scripts and overlays, not an immutable snapshot of all Miles code.
Regenerate the review plan after editing those files.

## Experimental matrix

Every arm uses 32 nodes, four GB200 GPUs per node, `batch / normal`, account
`nemotron_sw_post`, and a single `04:00:00` allocation. There is no dependency
chain, automatic resubmission, checkpoint cleanup, or continuation from an old
run. Four hours includes container startup, initialization, training, and saves.
`NUM_ROLLOUT=300` with one update per rollout stops a fresh run after 300 updates.
Slurm can end the allocation earlier at the four-hour wall limit.
The user explicitly accepted no epoch cap and requested rollout routing replay (R3) ON.
Each row below has a 32/32 and a 64/64 serving variant. If all 16 allocations
use their full limit, the total is 2,048 node-hours (8,192 GPU-hours).
Run names/checkpoints include `-req32-cg32` or `-req64-cg64`; summary.csv and
steps.csv also record both limits and the producer admission setting.

| Max weight staleness | T:R nodes | Train GPUs | Rollout GPUs / engines | Train DP | Batch / DP rank |
|---:|---:|---:|---:|---:|---:|
| 8 | 4:28 | 16 | 112 | 8 | 420 |
| 8 | 6:26 | 24 | 104 | 12 | 280 |
| 8 | 8:24 | 32 | 96 | 16 | 210 |
| 8 | 10:22 | 40 | 88 | 20 | 168 |
| 8 | 12:20 | 48 | 80 | 24 | 140 |
| 8 | 14:18 | 56 | 72 | 28 | 120 |
| 8 | 16:16 | 64 | 64 | 32 | 105 |
| 1 | 16:16 | 64 | 64 | 32 | 105 |

TP=2, PP=1, CP=1, EP=4, expert TP=1 remain fixed. With 4 GPUs/node,
`DP = 4*T / (TP*CP) = 2*T`. The least common multiple of
`{8,12,16,20,24,28,32}` is **3360**. Therefore every arm uses:

```text
rollout_batch_size * n_samples_per_prompt = global_batch_size * num_steps_per_rollout
              420 *                    8 =             3360 *                     1
```

The existing sync batch of 256 does not divide DP=12,20,24,28. The new batch is
the smallest common batch for these fixed parallelism settings. All points
start from the same staged Megatron SFT-upsampled-iter6000 policy release.

These parallelism values inherit the earlier public-Lightning two-node
bring-up (job 7094021), including a 256-sample update at the current 18,432-token
context limit without OOM. TP=2 is now explicitly fixed by the user. CP=1 keeps
context communication out of that tested configuration; EP=4 with expert TP=1
distributes 128 experts over four ranks and divides every requested training
world size. This is a compatible starting point, not a measured throughput
optimum for the new SFT, async mode, or 32-node allocation.

The Qwen async recipe defaults to SGLang memory fraction 0.70. Lightning now
uses 0.70 at the user's request; the earlier public-Lightning qualification
used 0.60. The changed SFT async recipe has not been GPU-qualified at 0.70.

The reviewed LR/top-p/clipping/one-step-forward changes passed 14 launcher,
sweep, summary, and SFT staging tests plus five numerical advantage-clipping
tests on CPU. Eight additional non-kernel log-prob reuse tests passed; six
comparisons requiring Megatron fused cross-entropy could not execute in the
temporary CPU environment because Megatron is absent. No GPU job was allocated
for this follow-up. The changed SFT async training path remains unqualified.

The subsequent fused/memory-fraction review passed 27 CPU loss/recipe tests,
26 isolated argument/forward-selection tests, and three isolated actor/replay
lifecycle checks (including fused + rollout routing). Isolation executes the
checkout's actual function bodies while omitting serving/GPU imports and mocking
model execution; it does not qualify kernels or distributed GPU training.
The decision to retain the group-mean baseline and the LOO research references
are in [the loss review](lightning-loss-review.md).

## Shared training configuration

| Setting | Value |
|---|---|
| Training data | Original, unfiltered DAPO-MATH-17K, 17,398 prompts |
| Algorithm / correction | GRPO, actor denominator, token TIS clipped to [0.2, 5] |
| PPO clipping / entropy / KL | 0.2 low, 0.28 high; entropy 0; no KL loss |
| Optimizer | Adam, LR 4e-6 constant, warmup 0, betas 0.9/0.999, epsilon 1e-8, weight decay 0, grad clip 1 |
| Advantage clipping | [-20,20], after normalization; group-mean baseline retained |
| Train/rollout sequence filter | Default OFF; enable with `USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER=1` for a sweep variant. When ON, threshold 2 uses training forward and post-filter token denominator |
| Actor log-probs | `--fuse-one-step-actor-logprobs`; reuse training forward for the single update, with rollout routing replay; shadow verification OFF |
| Generation | Thinking enabled; temperature 1, top-p 1.0, top-k -1 |
| Response / context limits | 16,384 / 18,432 tokens |
| Reward | `math`; no forced zero reward or zero loss for truncated responses |
| Dynamic filtering | ON, `apply_reward_nonzero_std_filter`; replenish until 420 mixed-reward groups are accepted |
| Sampling setting | `OVER_SAMPLING_BATCH_SIZE=2520` (6 × 420), recorded for consistency with the sync recipe |
| Effective async concurrency | 32/32: 3584 samples (448 groups); 64/64: 7168 (896 groups), each fixed across node ratios |
| Completed queue | 6,000 prompt groups for both s8 and s1 |
| Queue / reference / pause | `queue-recycle` / `prefill` / `in_place` |
| Weight updates / replay | Update each optimizer step; inflight replay saved with training checkpoints |
| SGLang | One GPU/engine, Triton MoE backend, memory fraction 0.70; requests/graph profiles 32/32 and 64/64 |
| Training memory | Full uniform recompute, max tokens/GPU 18,432, routing replay ON |
| Seeds | Training 1234, rollout 42 |
| Saving | Megatron every 10 updates, retain every 50; HF export every 10 |
| In-run evaluation | OFF; AIME remains evaluation-only and can use the retained HF exports |

The async producer does not use `OVER_SAMPLING_BATCH_SIZE` for its admission
budget. It continuously replenishes filtered groups under
`ASYNC_MAX_CONCURRENT_SAMPLES`. The draft default is the largest pool's
112 engines × requests: 3584 at 32/32, 7168 at 64/64. Smaller rollout pools
keep the same admission budget within their profile, so some admitted requests
wait at the serving layer. The 7168 value is a proposed default to exercise the
64-request limit, not an independently confirmed user choice or measured optimum.
The comparison therefore changes serving limits and admission together.
`--set ASYNC_MAX_CONCURRENT_SAMPLES=3584` fixes admission across both profiles
for a comparison of only the two engine limits; this may leave the 64/64 fleet
below its aggregate running capacity. The sixfold oversampling flag is not an
effective sixfold async concurrency multiplier.

The 6,000-group queue exceeds the common experiment's s8 floor and
`ceil(3360*8/8)=3360` groups. Keeping it at 6,000 for s1 makes the extra arm differ
only in its staleness bound. Queue-recycle admits a group when `D-F < M`, where
`F` is its earliest prefill version and `D` is the dequeue version. In the normal
one-update-prefetch pipeline, the trained version is `T=D+1`, giving `T-F <= M`.
Realized staleness is measured rather than assumed to equal the configured bound.

Both overlays run in each writable container before Ray starts: the pinned
OCI prefill/token-provenance patch and Lightning's routing-replay compatibility
patch. The base image is `/lustre/fsw/portfolios/coreai/users/kfujii/container/miles-oci-aarch64-7035384.sqsh`.
Combined async operation on Lightning's hybrid model still needs GPU validation.

## Logging and measurement

Slurm stdout/stderr: `experiments/outputs/lightning-node-ratio/<run-name>-<job-id>.log`.
Each job prints its complete resolved configuration and checkpoint/dump paths.
The submission manifest is `<namespace>-submitted.jsonl` in that same directory.

The checkpoint root is
`/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/training/math/dapo-math-17k/Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000/<namespace>-s<S>-t<T>r<R>-req<requests>-cg<graph>`.
Local telemetry is always written under its `dump/dashboard/`, including hourly
`metrics/*.jsonl`, phase timelines, GPU utilization, and engine metrics. W&B is
mandatory, using project `async-rl-nemotron-lightning` and a separate group per arm.
After `env.sh` resolves `.env` / `.netrc`, the job rejects a missing key before
starting containers. The local CSV analyzer does not require W&B access.

```bash
python -m experiments.tools.summarize_lightning_node_ratio \
  --plan-file experiments/outputs/lightning-node-ratio/review-plan.json \
  --output-dir experiments/outputs/lightning-node-ratio/summary
```

The analysis writes `steps.csv`, `summary.csv`, and `metadata.json`. It joins
separate metric writers by rollout step, excludes the first two completed
training steps by default (`--warmup-steps`), and excludes prefetched rollouts
that never trained. Missing data is blank/`insufficient_data`, never a zero time.

| Readout | Meaning |
|---|---|
| `perf/actor_train_time` | Actor forward/backward/optimizer phase |
| `perf/train_time`, `perf/log_probs_time` | Full trainer call and its scoring phase |
| `perf/train_wait_time`, `perf/wait_time_ratio` | Trainer idle time and local wait fraction |
| `perf/step_time` | Existing trainer-local train + wait boundary |
| `perf/rollout_time` | Async batch drain time, including waiting for accepted groups; not model generation time |
| `throughput/window_seconds` | Pipeline completion-to-completion wall window |
| Generated/useful tokens per second | Sum of the corresponding counts / sum of matching wall windows |
| `perf/update_weights_time`, `perf/save_model_time` | Optional recorded synchronization/save phases |

Timing columns include mean, p50, p90, and sample count. The summary also retains
response length, truncation, realized staleness, filtering/recycling waste, queue
depth, and backpressure. Save and synchronization costs are not subtracted;
compare pipeline windows and useful tokens/sec alongside trainer-local timers.
Generation and training overlap, so their durations must not be added to derive
end-to-end time. Different response lengths and filter acceptance rates can
change observed speed even with fixed training settings.

The train/rollout sequence-filter counters are written to W&B and dashboard
under `train/train_rollout_logprob_sequence_filter/`. The existing
`throughput/window_accepted_loss_tokens` counter measures the accepted rollout
population before the training-forward sequence filter; it is not the final
post-filter token count. Use the new `kept_tokens` and rejection fractions to
interpret loss participation alongside the timing measurements.

Offline checks cover all eight DP/batch shapes, the executed shell argument
builder, dry-run isolation from Slurm, mocked single-allocation submissions,
and multi-writer/time-weighted CSV aggregation. No training measurements exist
for these new arms yet.


## SFT source and termination controls (2026-09-12)

The initial checkpoint is now the user's SFT model at
`/lustre/fsw/portfolios/llmservice/users/venkats/nemo-evaluator-rundirs/nano_v35_sft/conversions/upsampled-iter6000/hf`.
The public BF16 model is no longer the async recipe's default. The separate
conversion script is `experiments/scripts/ckpt-convert/nemotron-3.5-lightning-sft-iter6000.sh`;
its local staging helper is `experiments/src/checkpoints/lightning_sft.py`.
The helper copies policy shards without changing their tensor bytes, translates
legacy hybrid configuration to native HF fields, and excludes auxiliary MTP
heads. The original SFT model remains unchanged. The HF policy and Megatron
release use the distinct `Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000`
identity under the user's checkpoint roots. File SHA-256 provenance is stored
in `staging_manifest.json` (HF) and `source_manifest.json` (Megatron).

`NUM_ROLLOUT=300` is the total rollout/update horizon, not 300 additional updates
after a resume. The sweep requires a fresh checkpoint path, so its runs start
at rollout 0 and finish at rollout 299 (300 updates). The final update aligns
with the 10-update Megatron/HF save cadence. There is no independent training
step cap here: Megatron's `train_iters` is derived from the rollout count and
batch shape. Dynamic filtering's rejected groups do not advance this counter.

`DEBUG_EXIT_AFTER_ROLLOUT=N` exposes the existing Miles CLI in this recipe.
It exits after N completed rollouts in this launch, relative to its resumed
start, while `NUM_ROLLOUT` retains the total scheduler horizon. At one update
per rollout, this is N optimizer steps. It does not force an extra checkpoint:
use N aligned with the 10-update save cadence if the terminal state must be
saved. Leave it empty for the requested fresh 300-update experiment. Slurm's wall
limit can interrupt a step; it is not a graceful step-boundary timer.

The core also supports `--num-epoch`, which is converted to a rollout count
from dataset size. If `--num-rollout` is also supplied, it takes precedence.
The recipe deliberately keeps its explicit rollout horizon for time-limited
comparisons. `--save-trigger-sentinel` requests a save at the next save point;
it does not request termination.

W&B is always enabled. `SAVE_INTERVAL=10`, `HF_SAVE_INTERVAL=10`, and
`SAVE_RETAIN_INTERVAL=50` are the current defaults. The retention value is an
archive interval, not a request to keep exactly 50 checkpoint directories.
Detailed differences against Qwen and the NeMo RLVR YAML are in
[lightning-config-comparison.md](lightning-config-comparison.md).


## 今回のYAML比較で維持する差分

対象YAMLは`lightning_rlvr_86n.yaml`。DAPO mathとNeMo Gymの混合RLVRでは
タスクの難易度が大きく異なるため、**context 18,432 / response 16,384**を維持し、
YAMLの73,728へは変更しない。これはユーザーが明示的に了承した差分。
gradient FP32 accumulation/all-reduce、gradient accumulation fusion、parameter
gatherのoverlap設定、およびshuffle ONも、今回の指示どおり維持する。

全項目の数値比較、ユーザーの明示了承の有無、router freezeの実装見積りは
[Lightning専用の比較note](lightning-config-comparison.md)に記録した。
TISとsequence filterの数式・実装・loggingは
[loss review](lightning-loss-review.md)を参照。Leave-one-out不採用の判断も保持する。


## Sequence-filter実装のCPU検証

2026-09-12: loss/advantage/recipeの通常CPUテスト27件、およびsequence filter、
loss集計、argument guardのCPU検証42件が成功。後者はGPU/runtimeのimport依存を
避けるため、guardと`aggregate_train_losses`の関数本体をASTでそのまま単離して実行。
数値loss・filter・TIS・CP slicingは実際のMiles/PyTorch実装を使用した。
CPのcollectiveとDP集計は単体テスト用に代替しているため、分散GPU検証ではない。
検証対象は対称な算術平均、閾値2の境界、TIS上下限、回答全体のzero gradient、
除外後のtoken分母、全除外microbatch、異なる長さのmicrobatch間のloss/log集計、
THD/BSHDでのCP分割、対応外optionの拒否、全8armの実argvとbatch整合。
証跡は`experiments/outputs/lightning-node-ratio/sequence-filter-validation.json`。
学習jobは未投入。SFT async・32ノード・新filterを含むGPU実行は未確認。


## Dynamic Filtering採用率からの5 epoch概算

5 epochを「元データ17,398問を延べ5周分読む」と定義する。
現設定は420 prompt groups × 8 responses = 3360 samples / optimizer update。
採用率をpとすると、先読み・queue滞留・recycleを除いた定常近似は
`raw prompts/update = 420/p`、`updates for 5 epochs = 5*17398*p/420`。

参考にした旧public Lightningの16K生成・1 updateでは、32 group採用、
reward全1による除外81、全0による除外28で、完了groupの採用率は
`32/(32+81+28)=22.695%`。根拠は
`experiments/outputs/lightning-setup/qualification.json`の`runs.full_size.metrics`。
従って約1850.625 raw prompts/update、5周分には47.006 update、
整数で到達するのは48 stepという概算。採用率20〜30%なら42〜63 step程度。

これはSFT-upsampled-iter6000の実測採用率ではない。旧検証はpublic checkpoint、
LR1e-6/top-p0.95、colocated、GBS256、truncatedはzero rewardで、現在の
SFT・async・LR4e-6/top-p1・truncatedの強制zero rewardなしとは異なる。
旧syncでは打ち切り時の未完了groupもあり、この採用率は全datasetの不偏な推定ではない。
新しいモデルでの採用率変化、async先読み・queue滞留・再生成により、dataset cursorが
5周へ達する時点の完了train stepはこの近似からずれ得る。

一方、「採用したprompt groupの累積数が元データ5周分」という名目上の
学習投入量なら`ceil(5*17398/420)=208 step`。各問題を実際に5回学習する保証ではない。
training-forwardのsequence filterをONにしても補充を伴わないため、420 groupという
投入量を変えず、有効なloss回答数だけを減らす。最新の指示でdefaultはOFFとした。
停止条件は新規300 updateまたは4時間、epoch capなし。truncatedの報酬処理も
変更したため、上記の旧採用率によるepoch見積りは新設定で再測定する必要がある。


## GBSを小さくする候補（設定は未変更）

全T:R条件・train TP2/CP1・4 GPUs/node・同一GBS・DPの倍数という条件では、
DP={8,12,16,20,24,28,32}のLCM=3360が最小。R3、serving側32/64、EPの変更は
このDP計算を変えない。8 responses/promptも維持して計算する。

| 全T:R条件を維持する案 | 最小GBS | rollout prompts × responses |
|---|---:|---:|
| TP2 / CP1（現状） | 3360 | 420 × 8 |
| TP2 / CP2 | 1680 | 210 × 8 |
| TP2 / CP4 | 840 | 105 × 8 |

各行の最小GBSの整数倍も候補。CP2/CP4は配置previewを全8点で通過したが、
Lightning SFTの分散GPU学習・packing・R3・sequence filterの組合せは未確認。
CPを増やすと通信と1回答の計算分割が変わるため、全armで同じCPに固定して比較する。

TP2/CP1を維持してノード比の候補を減らす案:

| 残すtrain node数（rollout=32−train） | 最小GBS | GBS候補例 |
|---|---:|---|
| 4,6,8,10,12,16（14:18を外す） | 480 | 480,960,1440,1920,2400,2880 |
| 4,6,8,12,16（10:22と14:18を外す） | 96 | 384,768,1536 |
| 4,8,16 | 32 | 256,512,1024 |

追加のstaleness1・16:16は同じDPなのでLCMを増やさない。14:18が唯一の素因数7を
持つDP28を作るため、その一点を外す効果が大きい。最初の二つのsubsetも実際の
previewが通過した。最小GBSは算術上の制約であり学習上の推奨値とは区別する。
ユーザーは候補を求めた段階なので、review planのGBS3360/TP2/CP1は維持した。

64/64追加の検証は`tests/fast/experiments/test_lightning_node_ratio.py`の15件が成功。
16条件の組合せ、実train.sh argv、保存先分離、profile不整合拒否、固定admission
上書き、CSV識別列を確認した。Slurmはmockのみ、実job投入なし。

最新の300 update停止・truncatedの強制zero reward削除・sequence filter default OFFへの
変更後、同じlauncherテスト18件が成功。32/32・64/64それぞれでfilter OFF/ONの実argv、
300 rollout、dynamic filtering・TIS・R3・保存引数を確認した。default OFFの16条件
review planを再生成し、`--set USE_TRAIN_ROLLOUT_LOGPROB_SEQUENCE_FILTER=1`の
previewも成功。Bash構文とPython formattingを確認済み。実job投入なし。
証跡は`experiments/outputs/lightning-node-ratio/300-update-validation.json`。
