#!/bin/bash
# Replay-buffer screening cohort for the three selected async DAPO-MATH arms:
#
#   s2-t2r6  stable async anchor
#   s2-t3r5  phase-boundary arm
#   s4-t2r6  matched-ratio terminal-collapse arm
#
# The run namespace is part of CONFIG_TAG, RUN_NAME, Slurm job name, log name,
# and checkpoint path. These runs therefore cannot silently resume the legacy
# s*-t*r* checkpoints, which have no replay buffer.
#
# Usage:
#   experiments/replay_buffer_staleness_screen.sh             # dry-run / preflight
#   experiments/replay_buffer_staleness_screen.sh --submit    # fresh 3-arm chains
#   experiments/replay_buffer_staleness_screen.sh --check     # newest segment summary
#   experiments/replay_buffer_staleness_screen.sh --resume    # extend completed chains
#
# Override a new cohort name instead of deleting or reusing an existing cohort:
#   REPLAY_BUFFER_SCREEN_NAMESPACE=replay-buffer-v2 \
#       experiments/replay_buffer_staleness_screen.sh --submit

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SWEEP="${REPO_ROOT}/experiments/staleness_ratio_sweep.sh"

case "${1:-}" in
    ""|--submit|--resume|--check) ;;
    --help|-h)
        sed -n '2,20s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
        exit 0
        ;;
    *)
        echo "usage: $0 [--submit|--resume|--check]" >&2
        exit 1
        ;;
esac
(( $# <= 1 )) || {
    echo "usage: $0 [--submit|--resume|--check]" >&2
    exit 1
}

# A dated semantic cohort name prevents legacy checkpoint or W&B-group reuse.
# Change the override for a genuinely new cohort; never toggle the replay buffer within it.
: "${REPLAY_BUFFER_SCREEN_NAMESPACE:=replay-buffer-v1-20260815}"
: "${WANDB_PROJECT:=async-rl-dapo-math-replay-buffer-screen}"
: "${NUM_ROLLOUT:=300}"
: "${CHAIN_JOBS:=10}"
: "${WALL:=04:00:00}"
: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=10}"
: "${HF_SAVE_INTERVAL:=10}"

RUN_NAMESPACE="${REPLAY_BUFFER_SCREEN_NAMESPACE}"
TOTAL_NODES=8
RATIOS="1:7 2:6 3:5 4:4"
STALENESS_LEVELS="1 2 4 8"

# Pin the legacy sweep's training conditions. The only semantic interventions
# in this cohort are replay-buffer resume and the added diagnostics below.
ADVANTAGE_ESTIMATOR=grpo
ENTROPY_COEF=0.00
KL_LOSS_COEF=0.00
EPS_CLIP=0.2
EPS_CLIP_HIGH=0.28
EPS_CLIP_C=
RATIO_DENOMINATOR=actor
IS_CORRECTION=tis
TIS_CLIP=2.0
TIS_CLIP_LOW=0
USE_OPSM=0
LR=1e-6
MAX_RESPONSE_LEN=32768
TRAIN_SEED=1234
ROLLOUT_SEED=42
ROLLOUT_BATCH_SIZE=192
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=3072
NUM_STEPS_PER_ROLLOUT=1
QUEUE_POLICY=queue-recycle
QUEUE_FACTOR=1
USE_REPLAY_BUFFER=1
REPLAY_BUFFER_TYPE=rollout
ACTOR_GPUS_PER_NODE=8
TENSOR_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1
EXPERT_PARALLEL_SIZE=1
MAX_TOKENS_PER_GPU=32768
RECOMPUTE_GRANULARITY=full
OVERLAP_COMM=0
ROLLOUT_NUM_GPUS_PER_ENGINE=1
SGLANG_MEM_FRACTION=0.70
RM_TYPE=math
SAVE_HF=1
FUSE_ONE_STEP_ACTOR_LOGPROBS=1
VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS=0
unset ASYNC_MAX_CONCURRENT_SAMPLES SGLANG_MAX_RUNNING_REQUESTS SGLANG_CUDA_GRAPH_MAX_BS

# All currently implemented telemetry that has small measured runtime/HBM cost.
# Standard sequence ESS and current-policy-vs-rollout ESS are always emitted by
# the loss path and need no flag or extra model forward.
LOG_STALENESS_GRADIENT_METRICS=1
LOG_STALENESS_GRADIENT_RATIO_HISTOGRAM=1
SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1

# Keep the multi-GB per-microbatch debug dump disabled. --dump-details remains
# enabled by the recipe, so trajectory/token and disposition data are retained.
DUMP_POLICY_LOSS_DEBUG=0
DUMP_TRAIN_DATA=0

export RUN_NAMESPACE WANDB_PROJECT NUM_ROLLOUT CHAIN_JOBS WALL TOTAL_NODES
export RATIOS STALENESS_LEVELS
export ADVANTAGE_ESTIMATOR ENTROPY_COEF KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
export RATIO_DENOMINATOR IS_CORRECTION TIS_CLIP TIS_CLIP_LOW USE_OPSM
export LR MAX_RESPONSE_LEN TRAIN_SEED ROLLOUT_SEED
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_STEPS_PER_ROLLOUT
export QUEUE_POLICY QUEUE_FACTOR USE_REPLAY_BUFFER REPLAY_BUFFER_TYPE
export ACTOR_GPUS_PER_NODE TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE
export MAX_TOKENS_PER_GPU RECOMPUTE_GRANULARITY OVERLAP_COMM
export ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION RM_TYPE
export SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL
export FUSE_ONE_STEP_ACTOR_LOGPROBS VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS
export LOG_STALENESS_GRADIENT_METRICS LOG_STALENESS_GRADIENT_RATIO_HISTOGRAM
export SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS
export DUMP_POLICY_LOSS_DEBUG DUMP_TRAIN_DATA

printf 'Replay-buffer screening cohort: %s\n' "${RUN_NAMESPACE}"
printf 'Selected arms: s2-t2r6, s2-t3r5, s4-t2r6 (no colocated rerun)\n'
printf 'Natural Slurm resume timing is retained; every resume uses the replay buffer.\n'
printf 'Telemetry: staleness-gradient bins+histograms, standard ESS families, '
printf 'response weight-version segments, disposition/trajectory dumps.\n\n'

exec "${SWEEP}" \
    --point 2:2:6 \
    --point 2:3:5 \
    --point 4:2:6 \
    "$@"
