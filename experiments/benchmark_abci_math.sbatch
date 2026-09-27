#!/bin/bash
#PBS -N miles-math-throughput
#PBS -q R9920261300
#PBS -l select=8:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=04:00:00
#PBS -j oe

# Filtered Qwen3-4B SFT validation and production runs. The submit wrapper
# selects conversion, bring-up/resume, fit, 20 updates, or all 300 updates.
set -euo pipefail
export MILES_REPO="${MILES_REPO:-${PBS_O_WORKDIR:?}}"
cd "${MILES_REPO}"
source experiments/env.sh
: "${BENCHMARK_STAGE:?}"
: "${BENCHMARK_ID:?}"
: "${BENCHMARK_ACTOR_NODES:=2}"

export MODEL_NAME=Qwen3-4B-SFT-iter4000-main-20260927
export HF_MODEL_NAME=training/reasoing-sft/Qwen3-4B-Base/LR2.0e-5-MINLR2.0e-6-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1/iter_0004000
export MODEL_PROFILE=qwen3-4B MEGATRON_MODEL_TYPE=qwen3-4B
export DATASET_TAG=dapo-math-p10-90-qwen3-4b-sft-iter4000
export PROMPT_DATA="/data/${DATASET_TAG}/${DATASET_TAG}.jsonl"
export WANDB_TEAM=okoge WANDB_PROJECT=async-rl-staleness-dapo-math-miles WANDB_MODE=online
export ACTOR_GPUS_PER_NODE=8 TENSOR_PARALLEL_SIZE=2 CONTEXT_PARALLEL_SIZE=1 EXPERT_PARALLEL_SIZE=1
export ROLLOUT_BATCH_SIZE=192 N_SAMPLES_PER_PROMPT=16 GLOBAL_BATCH_SIZE=3072 NUM_STEPS_PER_ROLLOUT=1
export MAX_RESPONSE_LEN=16384 ROLLOUT_MAX_CONTEXT_LEN=32768 MAX_TOKENS_PER_GPU=32768
export QUEUE_TYPE=queue-recycle STALENESS_REFERENCE=prefill
export MAX_WEIGHT_STALENESS="${MAX_WEIGHT_STALENESS:-16}"
[[ "${MAX_WEIGHT_STALENESS}" =~ ^[1-9][0-9]*$ ]] || {
    echo 'MAX_WEIGHT_STALENESS must be a positive integer' >&2; exit 2;
}
export PAUSE_GENERATION_MODE=in_place
export ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-3072}"
export USE_REPLAY_BUFFER=1 REPLAY_BUFFER_TYPE=inflight SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1
export TRAIN_SEED=1234 ROLLOUT_SEED=42 LR=1e-6
export FUSE_ONE_STEP_ACTOR_LOGPROBS=1 ZERO_REWARD_ON_TRUNCATED=1 ZERO_LOSS_ON_TRUNCATED=0
export ROLLOUT_NUM_GPUS_PER_ENGINE=1 SGLANG_MEM_FRACTION=0.70
export SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-cache_aware}"
export NUM_ROLLOUT=300 SAVE_INTERVAL=10 SAVE_RETAIN_INTERVAL=100 SAVE_HF=1 HF_SAVE_INTERVAL=10
export EVAL_INTERVAL=0 SKIP_EVAL_BEFORE_TRAIN=1 DUMP_TRAIN_DATA=0 DUMP_POLICY_LOSS_DEBUG=0
export LOG_SAMPLE_STALENESS_METRICS=1 LOG_UPDATE_DIAGNOSTICS=1 LOG_POLICY_LAG_METRICS=1
export DUMP_ROLLOUT_DATA="${DUMP_ROLLOUT_DATA:-1}" LOG_ROLLOUT_REPETITION="${LOG_ROLLOUT_REPETITION:-1}"
export LOG_TOKEN_LAG_METRICS="${LOG_TOKEN_LAG_METRICS:-1}" LOG_ROLLOUT_TIMING="${LOG_ROLLOUT_TIMING:-0}"
export MILES_NCCL_TRANSPORT=system CLEAN_CHECKPOINT=0

test -r "${HF_CKPT_DIR}/${HF_MODEL_NAME}/config.json"
test -r "${DATASET_DIR}/${PROMPT_DATA#/data/}"
test -r "${CONTAINER_IMAGE}"
if [[ "${BENCHMARK_STAGE}" == convert ]]; then
    mkdir -p "${MILES_REPO}/experiments/outputs/abci-throughput/logs"
    exec > >(tee -a "${MILES_REPO}/experiments/outputs/abci-throughput/logs/${BENCHMARK_ID}-convert-${PBS_JOBID}.log") 2>&1
    exec bash experiments/setup/models/convert_checkpoint.sbatch
fi
test -r "${MEGATRON_CKPT_DIR}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt"

case "${BENCHMARK_STAGE}" in
    smoke|resume)
        [[ "${MILES_JOB_NUM_NODES}" == 2 ]]
        export ACTOR_NUM_NODES=1 ROLLOUT_NUM_GPUS=8
        export ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=8
        export ASYNC_MAX_CONCURRENT_SAMPLES=32 MAX_RESPONSE_LEN=512 ROLLOUT_MAX_CONTEXT_LEN=2048
        export SAVE_INTERVAL=1 SAVE_RETAIN_INTERVAL=1 SAVE_HF=1 HF_SAVE_INTERVAL=1
        export DEBUG_EXIT_AFTER_ROLLOUT=2
        [[ "${BENCHMARK_STAGE}" != resume ]] || export DEBUG_EXIT_AFTER_ROLLOUT=1
        export RUN_NAME="${BENCHMARK_ID}-smoke"
        ;;
    fit|throughput|production)
        [[ "${MILES_JOB_NUM_NODES}" == 8 ]]
        [[ "${BENCHMARK_ACTOR_NODES}" =~ ^[123]$ ]]
        export ACTOR_NUM_NODES="${BENCHMARK_ACTOR_NODES}"
        export ROLLOUT_NUM_GPUS="$(((8 - ACTOR_NUM_NODES) * 8))"
        export DEBUG_EXIT_AFTER_ROLLOUT=20
        [[ "${BENCHMARK_STAGE}" != fit ]] || export DEBUG_EXIT_AFTER_ROLLOUT=1
        [[ "${BENCHMARK_STAGE}" != production ]] || export DEBUG_EXIT_AFTER_ROLLOUT=""
        export RUN_NAME="${BENCHMARK_ID}-${BENCHMARK_STAGE}-t${ACTOR_NUM_NODES}r$((8 - ACTOR_NUM_NODES))"
        ;;
    *) echo "unknown BENCHMARK_STAGE=${BENCHMARK_STAGE}" >&2; exit 2 ;;
esac
export CONFIG_TAG="${RUN_NAME}"
export TRAIN_CKPT_DIR="${MILES_REPO}/experiments/outputs/abci-throughput/checkpoints"
mkdir -p "${MILES_REPO}/experiments/outputs/abci-throughput/logs"
exec > >(tee -a "${MILES_REPO}/experiments/outputs/abci-throughput/logs/${RUN_NAME}-${PBS_JOBID}.log") 2>&1
printf 'ABCI benchmark: revision=%s stage=%s train_nodes=%s rollout_gpus=%s image=%s\n' \
    "$(git rev-parse HEAD)" "${BENCHMARK_STAGE}" "${ACTOR_NUM_NODES}" "${ROLLOUT_NUM_GPUS}" "${CONTAINER_IMAGE}"
printf 'ABCI rollout: router_policy=%s concurrent_samples=%s\n' \
    "${SGLANG_ROUTER_POLICY}" "${ASYNC_MAX_CONCURRENT_SAMPLES}"
printf 'ABCI schedule: max_weight_staleness=%s num_rollout=%s debug_exit_after_rollout=%s\n' \
    "${MAX_WEIGHT_STALENESS}" "${NUM_ROLLOUT}" "${DEBUG_EXIT_AFTER_ROLLOUT:-none}"
if [[ "${BENCHMARK_STAGE}" == smoke ]]; then
    # Transformer Engine needs libcuda even for these path-selection tests.
    # Validate the image's Megatron on the allocated GPU host before training.
    miles_srun --nodes=1 --ntasks=1 --image="${CONTAINER_IMAGE}" \
        --bind="${CONTAINER_MOUNTS}" --writable-tmpfs --no-home --export=ALL \
        bash --noprofile --norc -c 'cd /root/miles; PYTHONPATH=/root/Megatron-LM:/root/miles OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests/fast/backends/megatron_utils/test_checkpoint_step_selection.py -q -o addopts="" --tb=short'
fi
exec bash experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch
