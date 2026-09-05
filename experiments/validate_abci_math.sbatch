#!/bin/bash
#PBS -N miles-v01-math
#PBS -q R9920261300
#PBS -l select=8:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -j oe

# Use the already staged public Qwen3-4B and DAPO-Math on ABCI. Override the
# allocation to two nodes for bring-up; the maintained validation uses eight.
set -euo pipefail
cd "${PBS_O_WORKDIR:?submit from the Miles checkout}"
mkdir -p experiments/outputs/abci-validation
exec > >(tee -a "experiments/outputs/abci-validation/${PBS_JOBID}.live.log") 2>&1
export WANDB_MODE=offline
export MODEL_NAME=Qwen3-4B HF_MODEL_NAME=Qwen3-4B MODEL_PROFILE=qwen3-4B
export DATASET_TAG=dapo-math-17k
export PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
export VALIDATION_NAMESPACE="${VALIDATION_NAMESPACE:-v01-abci-math-${PBS_JOBID}}"
export TRAIN_CKPT_DIR="${PWD}/experiments/outputs/abci-validation/checkpoints"
source experiments/env.sh

export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-$((MILES_JOB_NUM_NODES / 2))}"
export ACTOR_GPUS_PER_NODE=8
export ROLLOUT_NUM_GPUS="$(((MILES_JOB_NUM_NODES - ACTOR_NUM_NODES) * GPUS_PER_NODE))"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export GLOBAL_BATCH_SIZE="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
export NUM_STEPS_PER_ROLLOUT=1
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-30}"
export ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-${GLOBAL_BATCH_SIZE}}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2}" SAVE_RETAIN_INTERVAL=2
export SAVE_HF="${SAVE_HF:-1}" HF_SAVE_INTERVAL=2
export USE_REPLAY_BUFFER=1 REPLAY_BUFFER_TYPE=inflight
export SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1
export LOG_SAMPLE_STALENESS_METRICS=1 LOG_UPDATE_DIAGNOSTICS=1
export RUN_NAME="${VALIDATION_NAMESPACE}"
export CONFIG_TAG="${VALIDATION_NAMESPACE}"
export CLEAN_CHECKPOINT=0
test -r "${HF_CKPT_DIR}/${HF_MODEL_NAME}/config.json"
test -r "${MEGATRON_CKPT_DIR}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt"
test -r "${DATASET_DIR}/${PROMPT_DATA#/data/}"
if [[ "${CHECKPOINT_SELECTION_PREFLIGHT:-0}" == 1 ]]; then
    miles_srun --nodes=1 --image="${CONTAINER_IMAGE}" --bind="${CONTAINER_MOUNTS}" \
        --no-home --export=ALL bash -lc '
        cd /root/miles
        PYTHONPATH=/root/miles python3 -m pytest -q \
            tests/fast/backends/megatron_utils/test_checkpoint_step_selection.py \
            tests/fast/backends/megatron_utils/test_lora_checkpoint_helpers.py \
            tests/fast/backends/megatron_utils/test_in_memory_checkpoint.py
        '
fi
printf 'ABCI validation: revision=%s nodes=%s namespace=%s\n' \
    "$(git rev-parse HEAD)" "${MILES_JOB_NUM_NODES}" "${VALIDATION_NAMESPACE}"
exec bash experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch
