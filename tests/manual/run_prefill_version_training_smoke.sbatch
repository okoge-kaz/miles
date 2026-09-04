#!/bin/bash
#SBATCH --job-name=miles-prefill-training-smoke
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/training/%x-%j.log

# End-to-end validation with real DAPO math data. Four H100s train Qwen3-4B
# while four independent SGLang engines keep fully-async generation in flight.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SGLANG_COMMIT="f994b9aedfd0b1465dbb8f4e2a02eb789fc76dce"

export SQSH_IMAGE="${SQSH_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/container/miles-staleness-weight-boundaries-${SGLANG_COMMIT:0:9}.sqsh}"
export CONFIG_TAG="prefill-version-smoke-${SGLANG_COMMIT:0:9}-${SLURM_JOB_ID}"
export RUN_NAME="prefill-version-smoke-${SGLANG_COMMIT:0:9}-${SLURM_JOB_ID}"
# Ray's Unix-domain socket path is limited to 107 bytes. /root/.cache is a
# persistent host mount, so this short path preserves logs without violating it.
export RAY_TEMP_DIR="/root/.cache/ray/pv-${SLURM_JOB_ID}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
export ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-8}"

export ACTOR_NUM_NODES=1
export ACTOR_GPUS_PER_NODE=4
export ROLLOUT_NUM_GPUS=4
export TENSOR_PARALLEL_SIZE=2
export CONTEXT_PARALLEL_SIZE=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=1

export MAX_WEIGHT_STALENESS=2
export STALENESS_REFERENCE=prefill
export PAUSE_GENERATION_MODE=in_place

export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-8}"
export SGLANG_MEM_FRACTION=0.70
export EVAL_INTERVAL=0
export SAVE_INTERVAL=1000
export SAVE_RETAIN_INTERVAL=1000
export SAVE_HF=0
export DUMP_TRAIN_DATA=1
export DUMP_POLICY_LOSS_DEBUG=1
export LOG_SAMPLE_STALENESS_METRICS=1
export LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM=0
export SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1

# Keep this validation self-contained and avoid creating an external run.
export WANDB_MODE=offline
export WANDB_API_KEY="offline-prefill-version-smoke"
export WANDB_PROJECT="miles-prefill-version-smoke"

test -f "${SQSH_IMAGE}"
test -f "/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/datasets/dapo-math-p10-90/dapo-math-p10-90.jsonl"

bash "${REPO_ROOT}/experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch"
