#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --job-name=miles-oci-async-smoke
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/%x-%j.log

# Disposable async qualification: 2 training nodes + 2 rollout nodes.
# Keep the six-update horizon fixed; run three updates, then resubmit the same
# SMOKE_TAG for the remaining three. MCore/replay every update, HF every three.
set -euo pipefail
: "${SMOKE_TAG:?set a unique validation tag; reuse only to verify resume}"
[[ "${SLURM_JOB_NUM_NODES}" == 4 ]] || { echo "requires exactly 4 nodes" >&2; exit 1; }
REPO_ROOT="${SLURM_SUBMIT_DIR}"
export WANDB_MODE=offline
export WANDB_API_KEY=offline-oci-validation
source "${REPO_ROOT}/experiments/env.sh"
export OCI_SGLANG_PROVENANCE_PATCH=1
export CONFIG_TAG="${SMOKE_TAG}"
export RUN_NAME="${SMOKE_TAG}"
export RAY_TEMP_DIR="/root/.cache/ray/oa-${SLURM_JOB_ID}"
export ACTOR_NUM_NODES=2 ACTOR_GPUS_PER_NODE=4 ROLLOUT_NUM_GPUS=8
export TENSOR_PARALLEL_SIZE=2 CONTEXT_PARALLEL_SIZE=1 ROLLOUT_NUM_GPUS_PER_ENGINE=1
export NUM_ROLLOUT=6 DEBUG_EXIT_AFTER_ROLLOUT=3
export ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=16 NUM_STEPS_PER_ROLLOUT=1
export MAX_RESPONSE_LEN=8192 ROLLOUT_MAX_CONTEXT_LEN=32768 MAX_TOKENS_PER_GPU=32768
export ASYNC_MAX_CONCURRENT_SAMPLES=32
export SGLANG_MAX_RUNNING_REQUESTS=4 SGLANG_CUDA_GRAPH_MAX_BS=4
export QUEUE_TYPE=queue-recycle MAX_WEIGHT_STALENESS=2 TRAINING_BUFFER_QUEUE_SIZE=16
export STALENESS_REFERENCE=prefill PAUSE_GENERATION_MODE=in_place
export USE_REPLAY_BUFFER=1 REPLAY_BUFFER_TYPE=inflight
export SAVE_INTERVAL=1 SAVE_RETAIN_INTERVAL=1 SAVE_HF=1 HF_SAVE_INTERVAL=3
export USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1
export DUMP_TRAIN_DATA=1 LOG_SAMPLE_STALENESS_METRICS=1 LOG_UPDATE_DIAGNOSTICS=1
export LOG_POLICY_LAG_METRICS=1 SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1
export EVAL_INTERVAL=3 EVAL_DATASET_NAME=dapo-smoke
export EVAL_PROMPT_DATA='/data/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl@[0:2]'
export N_SAMPLES_PER_EVAL_PROMPT=1 EVAL_MAX_RESPONSE_LEN=8192 SKIP_EVAL_BEFORE_TRAIN=1
exec bash "${REPO_ROOT}/experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch"
