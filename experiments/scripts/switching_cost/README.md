# Colocated switching-cost measurements

The job measures colocated partial-rollout transitions. The training CLI keeps
all switching-cost telemetry opt-in. This recipe enables the low-overhead phase
timers and byte counters, but keeps hot-path memory logging, the Miles dashboard,
event dumps, and full Ray log forwarding disabled by default. A separate
host/device bandwidth benchmark runs per node before Ray starts. The job does not
save checkpoints or run evaluation.

The relevant controls are:

```text
LOG_COLOCATE_SWITCH_METRICS=1  # driver block timers and rank-local snapshot timer
LOG_COLOCATE_TRANSFER_BYTES=1  # exact snapshot/payload bytes plus physical HBM delta
LOG_MEMORY_USAGE=0             # hot-path HBM/CPU snapshots; diagnostic-only
ENABLE_MILES_DASHBOARD=0       # per-rank Timer event RPCs; diagnostic-only
ENABLE_DUMP_DETAILS=0          # synchronous event JSONL writes; diagnostic-only
RAY_DEDUP_LOGS=1               # keep Ray log deduplication enabled
```

The matched partial-rollout comparison across 4B, 8B, and 30B-A3B has a
dedicated submission wrapper:

```bash
experiments/scripts/switching_cost/submit-partial-rollout-4h.sh       # validate and print commands
experiments/scripts/switching_cost/submit-partial-rollout-4h.sh --submit
```

Submission validates the matching Hugging Face and Megatron checkpoints, but
deliberately does not gate on difficulty-filter completion so the jobs can wait
in the Slurm queue while filtering finishes. The policy-specific filtered file
must be complete before its allocation starts. Each arm uses eight nodes with a
four-hour limit, accepts 192 prompts from an oversampling batch of 256 under
`--partial-rollout`, and logs online to the `colcoated-switching-cost` W&B
project.

Submit all three four-hour production measurements after their policy-specific
difficulty datasets are ready:

```bash
experiments/scripts/switching_cost/submit-colocated-4h.sh all
```

The wrapper submits 8-node jobs to `batch` with an explicit `04:00:00` limit.
It forces switch timing and transfer-byte telemetry on, while leaving intrusive
memory snapshots, dashboard RPCs, and event dumps off. A single model can be
selected with `qwen3-4b`, `qwen3-8b`, or `qwen3-30b-a3b`. It refuses to submit a
model whose filtered prompt file is not yet present.

`actor_weight_snapshot_bytes` is the exact number of tensor bytes copied from
each trainer rank to its pinned CPU backup. `weight_update_payload_bytes` is the
exact logical tensor payload produced by each trainer rank for the colocated
weight update. `trainer_offload_hbm_released_bytes` is the CUDA driver free-HBM
increase around `torch_memory_saver.pause`, not an exact CPU/GPU transfer-byte
counter. The installed memory-saver API and shared library do not expose such a
counter. SGLang currently does not return exact weight/KV bytes
from its release/resume endpoints, so the rollout-side phases have timings but
not direct byte counters.

For a short two-node 4B offline validation:

```bash
MODEL_VARIANT=qwen3-4b WANDB_MODE=offline NUM_ROLLOUT=2 \
ROLLOUT_BATCH_SIZE=8 OVER_SAMPLING_BATCH_SIZE=12 N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=16 MAX_RESPONSE_LEN=1024 MAX_TOKENS_PER_GPU=8192 \
BANDWIDTH_BENCHMARK=0 \
sbatch -A coreai_horizon_dilations -p batch --qos=interactive -N 2 --time=01:00:00 \
  experiments/scripts/switching_cost/run-colocated.sbatch
```

The RL job now defaults to the two SFT HuggingFace checkpoints supplied for the
measurement. On the cluster they are expected below:

```text
/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface/
├── Qwen3-8B-Base/LR1.5e-5-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000
└── Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000
```

SGLang can read those exports directly, but Megatron cannot. After each transfer
has completed, run the dedicated conversion job so the rollout and trainer start
from identical weights:

```bash
MODEL_VARIANT=qwen3-8b \
sbatch experiments/scripts/switching_cost/convert-sft-checkpoint.sbatch

MODEL_VARIANT=qwen3-30b-a3b \
sbatch experiments/scripts/switching_cost/convert-sft-checkpoint.sbatch
```

The conversion wrapper runs on the `interactive` partition and validates every
weight shard listed in `model.safetensors.index.json` before using the shared
conversion implementation. A partially transferred checkpoint therefore fails
before `torchrun` starts. The outputs are
`Qwen3-8B-Base-LR1.5e-5-Step4000_torch_dist` and
`Qwen3-30B-A3B-Base-LR2.0e-5-Step4000_torch_dist` below
`MEGATRON_CKPT_DIR`.

Both RL jobs fail before launching the bandwidth benchmark or Ray if their SFT
HF export is not readable or the matching `torch_dist` conversion is
absent/incomplete. After the conversions finish, submit 8B with:

```bash
MODEL_VARIANT=qwen3-8b sbatch experiments/scripts/switching_cost/run-colocated.sbatch
```

Submit 30B-A3B with:

```bash
MODEL_VARIANT=qwen3-30b-a3b sbatch experiments/scripts/switching_cost/run-colocated.sbatch
```

The defaults are training TP4/EP1 and rollout TP4 for 8B, and training TP4/EP8
and rollout TP8 for 30B-A3B. Override the corresponding environment variables
at submission only after a production-token HBM smoke. These values are not a
parameter-count scaling rule: 8B TP4 has approximately the same per-rank weight
footprint as the measured 4B TP2 run, while 30B uses EP8 independently of dense
TP. The rollout group must also cover every source TP/EP shard during the
colocated weight update, so lowering the inference group requires a live weight
equality check as well as an HBM check. Keep `SGLANG_MEM_FRACTION=0.65` for the
primary comparison; a later fraction sweep is useful for isolating KV
allocation cost.

The `TP2/CP2/EP8` strings in the source SFT directory names describe the SFT
run. The HF export is topology-independent; the colocated RL topology is set by
this job after conversion to distributed-checkpoint format.

## Lower-TP HBM probes

Do not raise TP from an OOM-free lower setting. Before the full measurements,
probe train TP2 at the production 32k token budget while keeping the rollout
groups large enough to cover all source shards:

```bash
MODEL_VARIANT=qwen3-8b TENSOR_PARALLEL_SIZE=2 \
CHECK_WEIGHT_UPDATE_EQUAL=1 NUM_ROLLOUT=2 BANDWIDTH_BENCHMARK=0 \
RUN_NAME=qwen3-8b-train-tp2-hbm-probe \
sbatch experiments/scripts/switching_cost/run-colocated.sbatch

MODEL_VARIANT=qwen3-30b-a3b TENSOR_PARALLEL_SIZE=2 \
CHECK_WEIGHT_UPDATE_EQUAL=1 NUM_ROLLOUT=2 BANDWIDTH_BENCHMARK=0 \
RUN_NAME=qwen3-30b-a3b-train-tp2-hbm-probe \
sbatch experiments/scripts/switching_cost/run-colocated.sbatch
```

These probes retain rollout TP4 for 8B and rollout TP8 for 30B-A3B. A completed
two-step run establishes both that the lower trainer TP survives a real dynamic
batch and that the colocated source-to-rollout mapping starts from equal
weights. If TP2 passes with repeatable peak-HBM margin, use TP2 for the full
measurement; TP4 is only the fallback.

## Policy-specific difficulty filters

The 4B p10-90 dataset is complete. The 8B and 30B-A3B SFT policies require
separate measurements because difficulty belongs to the prompt/policy/sampling
triple. Submit their resumable pipelines with:

```bash
experiments/scripts/switching_cost/submit-difficulty-filters.sh all
```

For each model the wrapper submits a 32-prompt smoke followed by two four-hour
`batch`/`interactive`-QoS shards. A dependent `cpu`/`cpu-interactive`-QoS
coordinator resubmits an incomplete shard for up to eight rounds and then merges
the measurements and materializes the p10-90 dataset. Sampling matches the
switching workload:
`n=16`, response length 16384, context length 32768, temperature 1, DeepScaler
reward, and zero reward for truncation. The 8B inference sweep uses TP1/DP8;
30B-A3B uses TP2/DP4 to leave a practical KV budget without consuming more than
one 8-GPU node per shard.
