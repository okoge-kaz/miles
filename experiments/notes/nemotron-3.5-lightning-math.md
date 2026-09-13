# Nemotron 3.5 Lightning: DAPO math RL on OCI

Assets staged on 2026-09-11 under `/lustre/fsw/portfolios/coreai/users/kfujii`.
The download ran on `cpu / cpu-interactive`, job `7094019`, and completed
in 7m13s including extraction of the training container.

| Asset | Workspace-relative path | Provenance |
|---|---|---|
| Full HF BF16 checkpoint | `checkpoints/hf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` |
| RL policy HF view | `checkpoints/hf/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-policy` | Relative links to the full checkpoint; MTP disabled in config/index |
| Megatron policy release | `checkpoints/megatron/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16_torch_dist` | NVIDIA Megatron Bridge; tracker is `release` |
| Training data | `datasets/dapo-math-17k/dapo-math-17k.jsonl` | `zhuzilin/dapo-math-17k`, revision `2e65612930298bde4c5d58fd97b3f23a483aaff9`, 17,398 rows |
| Evaluation | `datasets/aime-2024/aime-2024.jsonl` | `zhuzilin/aime-2024`, revision `1c625e328db94ec7ef7ff169016b097c468d60b9`, 30 rows |

Each staged asset has `staging_manifest.json`. The full download has 14 shards,
6,513 tensors and 65,827,374,264 bytes of shard files. The policy has
6,243 tensors / 31,577,937,344 parameters (63,155,886,464 tensor bytes).
The 270 excluded tensors are auxiliary MTP heads, which this RL run does not
train or use for speculative decoding. The complete original checkpoint is
retained. Excluding MTP from the policy index is also necessary for complete
HF export: this Bridge version only writes a source shard when all of its
indexed tensors have been yielded by the trained model.

The image is the immutable `/lustre/fsw/portfolios/coreai/users/kfujii/container/miles-oci-aarch64-7035384.sqsh`:
PyTorch 2.13.0+cu130, SGLang 0.5.20.dev54+ga8e5c63, Transformers 5.12.1.
The existing Nano 30B model-argument definition supplies the shared dimensions;
NVIDIA Bridge reads Lightning's actual hybrid pattern from its HF config.
The conversion tool now honors `--megatron-to-hf-mode bridge` for weight
loading as well as model construction. The wrapper explicitly sets the
`alltoall` token dispatcher required by shared expert overlap. Before Ray starts,
`experiments/src/math/lightning_sglang_compat.py` applies an idempotent, source-hash-guarded
overlay inside each writable container: Nemotron's `TopK` must receive
`layer_id=layer_idx` for routed-expert capture. The unpatched version passes
`None`, adds an indexing dimension, and fails CUDA graph capture when routing
replay is enabled. The base SQSH is unchanged.

## Recipes

- Checkpoint download: `scripts/download/checkpoints/nemotron_3_5_lightning_30b_a3b.py`.
- Training dataset download: `scripts/download/datasets/training/dapo_math_17k.py`.
- Evaluation dataset download: `scripts/download/datasets/eval/aime_2024.py`.
- CPU interactive download wrapper: `experiments/scripts/download.sbatch`.
- Conversion: `experiments/scripts/ckpt-convert/nemotron-3.5-lightning-30b-a3b.sh`.
- Training: `experiments/scripts/math/sync/dapo-math-17k/nemotron-3.5-lightning/run.sbatch`.
- Asset checks: `tests/manual/check_lightning_assets.py`.
- Update/filter/HF-export checks: `tests/manual/check_lightning_training.py`.

Create `experiments/outputs/lightning-setup` and submit from the repository root.
The Lightning wrappers use the HF parent directory rather than the Qwen
Step4000-specific HF root in `.env`. Override `HF_CKPT_DIR`, `MEGATRON_CKPT_DIR`,
`DATASET_DIR` and `SQSH_IMAGE` as needed.

The three download entrypoints operate independently. The checkpoint entrypoint
also creates the `-policy` view; `--policy-only` rebuilds it from an existing full
download. Dataset commands preserve the original unfiltered JSONL. Run these
Python modules directly in an environment with `huggingface_hub`, `safetensors`
and PyTorch, or use the CPU interactive wrapper with the qualified image:

```bash
mkdir -p experiments/outputs/lightning-setup
sbatch experiments/scripts/download.sbatch \
  scripts.download.checkpoints.nemotron_3_5_lightning_30b_a3b --hf-root /ckpt/hf
sbatch experiments/scripts/download.sbatch \
  scripts.download.datasets.training.dapo_math_17k --dataset-root /data
sbatch experiments/scripts/download.sbatch \
  scripts.download.datasets.eval.aime_2024 --dataset-root /data
```

Once the checkpoint download finishes, convert it with:

```bash
sbatch experiments/scripts/ckpt-convert/nemotron-3.5-lightning-30b-a3b.sh
```

The download directories separate training and evaluation entrypoints; the
staged datasets remain at `datasets/dapo-math-17k` and `datasets/aime-2024`.
Download implementation bodies and the complete conversion script were preserved
in this layout change. The move proof and its before/target snapshots are in
`experiments/outputs/lightning-layout-refactor/`; reproduce with
`tests/manual/reproduce_lightning_download_move.py --before <before> --target <move>`.
The separate CLI entrypoints and argument-forwarding wrapper are covered by
six passing offline tests in `tests/fast/experiments/test_lightning_setup.py`
and `tests/fast/experiments/test_lightning_downloads.py`. Both shell wrappers
pass `bash -n`. Existing model assets and the qualified training recipe keep
their original paths and settings.

CPU validation jobs `7097382` and `7097585` also ran the launcher suites.
The working tree had 499 passed, 7 failed and 4 errors including the six
download tests; an isolated archive of `HEAD` had 493 passed, 7 failed and
4 errors in the launcher suites. All eleven failure/error IDs match: existing
AMD API/missing-snapshot issues and Kimi/Search-R1 snapshot mismatches.
The comparison is recorded in `experiments/outputs/lightning-layout-refactor/validation.json`,
with `focused-tests.log`, `tests.log` and `baseline-tests.log` beside it.

No difficulty filtering is performed. Training enables
`miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter`,
rejecting groups with all-zero or all-one rewards. The rollout target is a
number of **accepted prompt groups**. Oversampling defaults to six times that
number, and the sampler replenishes candidates until the target is full.
The initial 8K-response smoke kept 4 of 22 completed groups (18.2%); 13 groups
were all correct and five were all incorrect. Six times is an initial candidate
batch estimate based on that small sample, not a limit on retries or a multiplier
of the training batch. Recheck acceptance rates at the full response length and
as training progresses. Eight responses per prompt provide within-group comparisons.
At the full 16K response limit, the validation rollout kept 32/141 completed
groups (22.7%); 81 were all correct and 28 all incorrect. The initial 192-group
candidate batch was sufficient, with unfinished candidates cancelled once the
32-group training target was met. Rollout collection took about 8m30s including
aborting pending work; accepted responses averaged 12,438 tokens.
Among accepted responses, 43.36% reached the 16K limit and received zero reward.
This measures the current response-budget constraint; longer-response training
and learning-curve quality remain separate tuning work.

Defaults are 32 accepted groups x 8 responses = global batch 256, one optimizer
step per rollout, oversampling 192 groups, GRPO, learning rate 1e-6,
clip low/high 0.2/0.28, temperature 1, top-p 0.95, thinking enabled,
maximum response 16,384 and context 18,432 tokens. `--rm-type math` checks boxed
answers without requiring a particular reasoning delimiter. Truncated responses
receive zero reward. MoE routing replay is enabled. SGLang uses `--sglang-moe-runner-backend triton`: the image's automatically selected FlashInfer TRTLLM backend repacks expert weights and failed the first live weight update (packed dimension 64 versus HF dimension 2688).

The initial placement is two GB200 nodes (four GPUs each), colocated,
TP=2, PP=1, CP=1, EP=4. Resumable Megatron and HF exports both default to every
five rollouts. Each new job derives a unique checkpoint path. To resume, preserve
`CONFIG_TAG` or explicitly pass the same `CKPT_PATH`; increasing `NUM_ROLLOUT`
sets the total target including previously completed rollouts.
`--use-checkpoint-opt-param-scheduler` preserves the saved constant learning rate
and scheduler progress when extending that target. Without it, Megatron rejects
the changed total sample count (64 versus 96 in the two-to-three-step check).
Model weights, Adam moments and RNG state are loaded normally.

The small correctness check uses the following overrides. Keep them unchanged
for its resume check, set `NUM_ROLLOUT=3`, and set `CONFIG_TAG` to the original
run's tag (the auto-generated tag includes the Slurm job ID).

```bash
mkdir -p experiments/outputs/lightning-setup
NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=8 \
OVER_SAMPLING_BATCH_SIZE=8 MAX_RESPONSE_LEN=8192 MAX_CONTEXT_LEN=10240 \
MAX_TOKENS_PER_GPU=12288 EVAL_INTERVAL=1 SAVE_INTERVAL=1 \
SAVE_RETAIN_INTERVAL=1 HF_SAVE_INTERVAL=1 \
sbatch experiments/scripts/math/sync/dapo-math-17k/nemotron-3.5-lightning/run.sbatch
```

After correctness and resume pass, the full batch/response-length check is:

```bash
NUM_ROLLOUT=1 EVAL_INTERVAL=1 SAVE_INTERVAL=1 SAVE_RETAIN_INTERVAL=1 \
HF_SAVE_INTERVAL=1 \
sbatch experiments/scripts/math/sync/dapo-math-17k/nemotron-3.5-lightning/run.sbatch
```

## Qualification results

GPU interactive allocation `7094021` completed conversion and two-node
training/save/resume checks and was released afterwards. These are bring-up
results; production learning curves have not been measured. Logs and the
machine-readable `qualification.json` are under
`experiments/outputs/lightning-setup/`. `source-sha256.txt` records the exact
preparation, conversion, training and verification files used.

- Full model and dataset download: passed.
- Megatron release conversion: passed (`convert-7094021-retry2.log`).
- Nemotron/reward regression checks: 58 passed.
- Policy-view source-preservation/index test: passed.
- All 17,398 prompts: maximum 1,509 tokens, within the 2,048-token prompt budget.
- Two-node Ray cluster: 2/2 nodes joined.
- Megatron release resharded and loaded at TP=2 / EP=4 on all eight ranks.
- Initial AIME 2024 evaluation: 11/30 correct at 8,192 response tokens.
- Two optimizer updates: finite losses 0.15525 / 0.15935, gradient norms
  0.48909 / 0.42685. The second train phase took 23.8s after cold compilation.
- Both Megatron checkpoints and HF exports verified; each HF export has all
  6,243 policy tensors. Comparing 16 expert matrix slices against the original
  found 1,007 changed BF16 elements after two updates.
- All eight accepted groups contain both reward 0 and reward 1, with eight
  responses each. The second rollout kept 4/14 completed groups; combined
  acceptance across the first two rollouts is 8/36 (22.2%).
- AIME after the second update: 12/30. This tiny stochastic evaluation does
  not establish a learning improvement.
- Resume: passed (`train-7094021-resume-scheduler.log`, Ray job
  `raysubmit_MywmcHWHEZJT1vA3`, outer shell exit 0). Loaded iteration 1,
  optimizer/scheduler/RNG and dataset cursor, then saved iteration 2. The third
  update had loss 0.14817, gradient norm 0.46713, and learning rate 1e-6.
  `training-check-resume.log` verifies all three HF exports and finds 802 changed
  elements between the pre-resume and post-resume expert slices (1,387 versus
  the original policy). All 12 accepted groups passed the mixed-reward check.
  The resumed rollout kept 4/23 completed groups (12/59 across all three).
- AIME after the resumed update: 9/30; the 9–12/30 scores here are smoke
  measurements, not evidence of consistent improvement.
- Production-sized update: passed (`train-7094021-full.log`): 32 groups x 8
  responses, maximum response/context 16,384/18,432, loss 0.14348, gradient norm
  0.23824, 3,225,873 useful training tokens. Log-prob computation took 45.9s and
  actor training took 97.3s. No OOM; GPU memory snapshots were about 115 GiB in
  rollout and 85 GiB in training, out of 185 GiB per GPU.
- Full-size HF export validation: passed (`training-check-full.log`), all 6,243
  tensors present, all 32 prompt groups contain mixed rewards, and 566 sampled
  BF16 weight elements changed from the original policy.
- Full-size AIME evaluation: 15/30 before and 15/30 after the update at 16K.
- Full-size clean shutdown: passed (Ray job `raysubmit_FDztHjR6a3XAmc2H`, outer
  shell exit 0). The saved full-size training checkpoint is
  `checkpoints/training/math/dapo-math-17k/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/grpo-lr1e-6-b32n8-o192-df-tp2ep4-7094021`,
  with resumable iteration `0` and HF export `hf/0`. The small/resume check is
  in the sibling `grpo-lr1e-6-b4n8-o8-df-tp2ep4-7094021` directory,
  with iteration `2` and HF exports `0`, `1`, `2`.

The first successful Ray job (`raysubmit_ePUTJ3VQCgkmL94q`) finished normally.
Its outer shell returned 127 because the launcher file was edited while Bash
was waiting for `srun`; the post-command read resumed at a changed byte offset.
This was a direct Bash invocation inside an existing allocation; normal `sbatch`
submissions already copy the script. Subsequent validation runs use a frozen
launcher copy. This did not affect the saved training states; their independent validation log is
`training-check-step2.log`.

Stage 1 bring-up is complete, including a full-size update. Follow
`miles-run-ladder` for Stage 2 short-QoS tuning and Stage 3 production allocations;
those stages were not run here. The wrapper's default allocation is two hours
of interactive QoS, and `NUM_ROLLOUT` is a total target that can be continued
across allocations by preserving the checkpoint path.
