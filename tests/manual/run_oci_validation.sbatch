#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --job-name=miles-oci-validation
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/%x-%j.log

# Disposable infrastructure/regression validation. No training, model download,
# checkpoint writes, or HF exports; this does not qualify a production recipe.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
source "${REPO_ROOT}/experiments/env.sh"
test -r "${SQSH_IMAGE}"
mapfile -t validation_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export VALIDATION_MASTER_ADDR="${validation_nodes[0]}"
export VALIDATION_MASTER_PORT="${VALIDATION_MASTER_PORT:-29693}"

container_args=(
    --container-image="${SQSH_IMAGE}"
    --container-name="miles-validation-${SLURM_JOB_ID}"
    --container-mounts="${REPO_ROOT}:/root/miles"
    --container-workdir=/root/miles
    --no-container-mount-home
)

srun "${container_args[@]}" bash -lc '
    set -euo pipefail
    torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node="$GPUS_PER_NODE" \
        --node-rank="$SLURM_PROCID" --master-addr="$VALIDATION_MASTER_ADDR" \
        --master-port="$VALIDATION_MASTER_PORT" tests/manual/oci_nccl_smoke.py
'

srun -N1 -n1 -w "${validation_nodes[0]}" "${container_args[@]}" bash -lc '
    set -euo pipefail
    export PYTHONPATH=/root/miles:/root/Megatron-LM HF_HUB_OFFLINE=1
    python -m compileall -q miles train.py train_async.py
    # These Ray tests advertise 8 logical GPU resources but run no GPU workers.
    # Keep this setting local to the mock tests, never the NCCL/training step.
    unset CUDA_VISIBLE_DEVICES
    python -m pytest -q tests/fast/experiments \
        tests/fast/rollout/test_fully_async_rollout.py \
        tests/fast/rollout/test_experimental_fully_async_rollout.py \
        tests/fast/rollout/test_replay_buffer.py \
        tests/fast/rollout/test_queue_policy.py \
        tests/fast/ray/rollout/real_ray/test_rollout_executor.py \
        tests/fast/ray/train/test_switch_timing.py \
        tests/fast/test_train_async.py tests/fast/test_train_async_replay_buffer.py
'
