#!/bin/bash
#PBS -N miles-v01-multi-env
#PBS -q R9920261300
#PBS -l select=4:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:?submit from the Miles checkout}"
mkdir -p experiments/outputs/abci-validation
exec > >(tee -a "experiments/outputs/abci-validation/${PBS_JOBID}.live.log") 2>&1
export WANDB_MODE=offline
export MODEL_NAME=Qwen3-4B HF_MODEL_NAME=Qwen3-4B MODEL_PROFILE=qwen3-4B
export VALIDATION_NAMESPACE="${VALIDATION_NAMESPACE:-v01-abci-multi-${PBS_JOBID}}"
export TRAIN_CKPT_DIR="${PWD}/experiments/outputs/abci-validation/checkpoints"
source experiments/env.sh

export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
export NUM_ROLLOUT=30
export DEBUG_EXIT_AFTER_ROLLOUT="${DEBUG_EXIT_AFTER_ROLLOUT:-3}"
export ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-256}"
export SAVE_INTERVAL=2 SAVE_RETAIN_INTERVAL=2 SAVE_HF=1 HF_SAVE_INTERVAL=2
export CODE_EXEC_SANDBOX=bubblewrap CODE_EXEC_CONCURRENCY=16
export USE_REPLAY_BUFFER=1 REPLAY_BUFFER_TYPE=inflight
export SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1
export LOG_SAMPLE_STALENESS_METRICS=1 LOG_UPDATE_DIAGNOSTICS=1
export RUN_NAME="${VALIDATION_NAMESPACE}" CONFIG_TAG="${VALIDATION_NAMESPACE}"
export CLEAN_CHECKPOINT=0
test -r "${DATASET_DIR}/nemotron-performance-transfer/math-code-stem-balanced-audit.json"
test -r "${DATASET_DIR}/reasoning-gym-deps/.miles-reasoning-gym-0.1.25"
miles_srun --nodes=1 --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},/usr/bin/bwrap:/usr/local/bin/bwrap" --no-home --export=ALL \
    bash -lc 'cd /root/miles; PYTHONPATH=/root/miles python3 -m experiments.tools.check_multi_env_rewards \
        --output /root/miles/experiments/outputs/abci-validation/multi-env-reward-preflight.json'
printf 'ABCI validation: revision=%s nodes=%s namespace=%s\n' \
    "$(git rev-parse HEAD)" "${MILES_JOB_NUM_NODES}" "${VALIDATION_NAMESPACE}"
exec bash experiments/scripts/multi_env/async/math-code-stem/qwen3-4b/run.sbatch
