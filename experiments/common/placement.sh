#!/bin/bash
# Needs GPUS_PER_NODE, ACTOR_NUM_NODES, ACTOR_GPUS_PER_NODE, ROLLOUT_NUM_GPUS
# (0 when colocated), ROLLOUT_NUM_GPUS_PER_ENGINE, TENSOR_PARALLEL_SIZE,
# CONTEXT_PARALLEL_SIZE, GLOBAL_BATCH_SIZE. Sets TRAIN_WORLD, DATA_PARALLEL_SIZE.
#
# Sourced by run.sbatch before srun, so a bad shape fails in seconds instead of
# after every container has started.

TRAIN_WORLD=$(( ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE ))
_alloc=$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))
_engine_pool=$(( ROLLOUT_NUM_GPUS > 0 ? ROLLOUT_NUM_GPUS : TRAIN_WORLD ))

if (( TRAIN_WORLD + ROLLOUT_NUM_GPUS != _alloc )); then
    echo "placement does not use the allocation:" \
         "${ACTOR_NUM_NODES}x${ACTOR_GPUS_PER_NODE} train + ${ROLLOUT_NUM_GPUS} rollout" \
         "!= ${SLURM_JOB_NUM_NODES}x${GPUS_PER_NODE}" >&2
    exit 1
fi

if (( TRAIN_WORLD % (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE) != 0 )); then
    echo "tp${TENSOR_PARALLEL_SIZE} * cp${CONTEXT_PARALLEL_SIZE} does not divide" \
         "${TRAIN_WORLD} training GPUs" >&2
    exit 1
fi

DATA_PARALLEL_SIZE=$(( TRAIN_WORLD / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE) ))

if (( GLOBAL_BATCH_SIZE % DATA_PARALLEL_SIZE != 0 )); then
    echo "global_batch_size ${GLOBAL_BATCH_SIZE} is not divisible by dp ${DATA_PARALLEL_SIZE}" >&2
    exit 1
fi

if (( _engine_pool % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
    echo "${_engine_pool} rollout GPUs is not divisible by" \
         "--rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}" >&2
    exit 1
fi

echo "placement ${ACTOR_NUM_NODES}x${ACTOR_GPUS_PER_NODE} train (${TRAIN_WORLD} GPU)" \
     "+ ${ROLLOUT_NUM_GPUS} rollout, tp${TENSOR_PARALLEL_SIZE} cp${CONTEXT_PARALLEL_SIZE}" \
     "-> dp${DATA_PARALLEL_SIZE}"

export TRAIN_WORLD DATA_PARALLEL_SIZE
