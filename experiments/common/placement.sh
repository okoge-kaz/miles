#!/bin/bash
# Validate the complete exclusive-node allocation before starting containers.

for _miles_dimension in GPUS_PER_NODE MILES_JOB_NUM_NODES ACTOR_NUM_NODES \
    ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS_PER_ENGINE TENSOR_PARALLEL_SIZE \
    CONTEXT_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE EXPERT_PARALLEL_SIZE \
    EXPERT_TENSOR_PARALLEL_SIZE GLOBAL_BATCH_SIZE; do
    [[ "${!_miles_dimension}" =~ ^[1-9][0-9]*$ ]] || {
        echo "${_miles_dimension} must be a positive integer" >&2
        return 2
    }
done
unset _miles_dimension
[[ "${ROLLOUT_NUM_GPUS}" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "ROLLOUT_NUM_GPUS must be a nonnegative integer" >&2
    return 2
}
TRAIN_WORLD=$(( ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE ))
_miles_alloc=$(( MILES_JOB_NUM_NODES * GPUS_PER_NODE ))
(( ACTOR_GPUS_PER_NODE == GPUS_PER_NODE && TRAIN_WORLD + ROLLOUT_NUM_GPUS == _miles_alloc )) || {
    echo "actor + rollout must use the entire ${MILES_JOB_NUM_NODES}x${GPUS_PER_NODE} GPU allocation" >&2
    return 2
}
if [[ "${PLACEMENT}" == disaggregate ]]; then
    (( ROLLOUT_NUM_GPUS > 0 )) || { echo "disaggregate requires separate rollout GPUs" >&2; return 2; }
    _miles_engine_pool=${ROLLOUT_NUM_GPUS}
elif [[ "${PLACEMENT}" == colocated ]]; then
    (( ROLLOUT_NUM_GPUS == 0 )) || { echo "colocated requires ROLLOUT_NUM_GPUS=0" >&2; return 2; }
    _miles_engine_pool=${TRAIN_WORLD}
else
    echo "PLACEMENT must be disaggregate or colocated" >&2
    return 2
fi
(( TRAIN_WORLD % (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE) == 0 )) || {
    echo "TP * CP * PP must divide the actor GPU count" >&2; return 2
}
(( TRAIN_WORLD % (EXPERT_TENSOR_PARALLEL_SIZE * EXPERT_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE) == 0 )) || {
    echo "expert TP * EP * PP must divide the actor GPU count" >&2; return 2
}
DATA_PARALLEL_SIZE=$(( TRAIN_WORLD / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE) ))
(( GLOBAL_BATCH_SIZE % DATA_PARALLEL_SIZE == 0 )) || {
    echo "GLOBAL_BATCH_SIZE must be divisible by data parallel size ${DATA_PARALLEL_SIZE}" >&2; return 2
}
(( _miles_engine_pool % ROLLOUT_NUM_GPUS_PER_ENGINE == 0 )) || {
    echo "rollout GPU count must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE" >&2; return 2
}
unset _miles_alloc _miles_engine_pool
export TRAIN_WORLD DATA_PARALLEL_SIZE
echo "placement=${PLACEMENT} actor_gpus=${TRAIN_WORLD} rollout_gpus=${ROLLOUT_NUM_GPUS} dp=${DATA_PARALLEL_SIZE}"
