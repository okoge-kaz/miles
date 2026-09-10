#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --job-name=miles-oci-step4000-smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/%x-%j.log

# Disposable sync bring-up: one rollout, then resubmit the same SMOKE_TAG with
# SMOKE_NUM_ROLLOUT=2 to test resume. Retains both MCore and HF snapshots.
set -euo pipefail
: "${SMOKE_TAG:?set a unique validation tag, and reuse it only for its resume check}"
REPO_ROOT="${SLURM_SUBMIT_DIR}"
export WANDB_MODE=offline
export WANDB_API_KEY=offline-oci-validation
source "${REPO_ROOT}/experiments/env.sh"

export CONFIG_TAG="${SMOKE_TAG}"
export NUM_ROLLOUT="${SMOKE_NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE=4
export N_SAMPLES_PER_PROMPT=4
export GLOBAL_BATCH_SIZE=16
export NUM_STEPS_PER_ROLLOUT=1
export MAX_RESPONSE_LEN="${SMOKE_MAX_RESPONSE_LEN:-8192}"
export EVAL_INTERVAL=1
export EVAL_DATASET_NAME=dapo-smoke
export EVAL_PROMPT_DATA='/data/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl@[0:2]'
export N_SAMPLES_PER_EVAL_PROMPT=1
export EVAL_MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN}"
export SKIP_EVAL_BEFORE_TRAIN=1
export SAVE_INTERVAL=1
export SAVE_RETAIN_INTERVAL=1
export SAVE_HF=1
export HF_SAVE_INTERVAL=1
export USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1
export DUMP_TRAIN_DATA=1
export SGLANG_MAX_RUNNING_REQUESTS=4
export SGLANG_CUDA_GRAPH_MAX_BS=4

# Check the bring-up fixes before starting the GPU workers. The port tests also
# exercise core defaults, so only this CPU-only subprocess unsets OCI overrides.
srun -N 1 -n 1 --cpus-per-task=4 --mem=16G \
    --container-image="${SQSH_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir=/root/miles --no-container-mount-home --export=ALL \
    bash -lc '
        set -e
        export PYTHONPATH=/root/miles:/root/Megatron-LM HF_HUB_OFFLINE=1
        unset MILES_WORKER_PORT_START MILES_WORKER_PORT_END
        python -m pytest -q --tb=short \
            tests/fast/experiments \
            tests/fast/utils/workers/test_addr_allocator.py \
            tests/fast/ray/test_train_actor.py \
            tests/fast/backends/sglang_utils/test_sglang_router_api_client.py \
            tests/fast/ray/rollout/test_inference_controller.py \
            tests/fast/ray/rollout/test_server_cell_state_machine.py
    '

exec bash "${REPO_ROOT}/experiments/scripts/math/sync/dapo-math-p10-90/qwen3-4b/run.sbatch"
