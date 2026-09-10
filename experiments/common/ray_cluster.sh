#!/bin/bash
# Needs RAY_HEAD_IP, RAY_DONE_FLAG, GPUS_PER_NODE. Sets NNODES, NODEID.
# Worker nodes exit from here; only node 0 returns to the caller.

: "${RAY_HEAD_IP:?}"
: "${RAY_DONE_FLAG:?}"
: "${GPUS_PER_NODE:?}"

RAY_PORT=6379
RAY_JOIN_TIMEOUT=600
NNODES="${SLURM_JOB_NUM_NODES}"
NODEID="${SLURM_NODEID:-0}"

RAY_TEMP_ARGS=()
if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
    NODE_RAY_TEMP_DIR="${RAY_TEMP_DIR}/node-${NODEID}"
    mkdir -p "${NODE_RAY_TEMP_DIR}"
    RAY_TEMP_ARGS=(--temp-dir "${NODE_RAY_TEMP_DIR}")
fi

ray stop --force || true

if [[ "${NODEID}" -ne 0 ]]; then
    # Both loops below run under the caller's `set -x` and both poll for minutes
    # to hours, so traced they would bury the driver's output in the shared log:
    # the idle loop alone prints two lines every 15 s for the whole job.
    set +x
    _waited=0
    until timeout 5 bash -c "cat < /dev/null > /dev/tcp/${RAY_HEAD_IP}/${RAY_PORT}" 2>/dev/null; do
        (( _waited % 60 == 0 )) && echo "worker ${NODEID}: waiting for ray head at ${RAY_HEAD_IP}:${RAY_PORT} (${_waited}s)"
        sleep 5
        _waited=$(( _waited + 5 ))
    done
    set -x
    ray start --address="${RAY_HEAD_IP}:${RAY_PORT}" --num-gpus "${GPUS_PER_NODE}" \
        --disable-usage-stats "${RAY_TEMP_ARGS[@]}"
    set +x
    echo "worker ${NODEID}: joined, idling until the driver signals completion"
    while [[ ! -f "${RAY_DONE_FLAG}" ]]; do sleep 15; done
    set -x
    echo "worker ${NODEID}: driver finished, shutting down"
    ray stop --force || true
    exit 0
fi

trap 'touch "${RAY_DONE_FLAG}" 2>/dev/null || true' EXIT

ray start --head --node-ip-address "${RAY_HEAD_IP}" --port "${RAY_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 "${RAY_TEMP_ARGS[@]}"

python3 /root/miles/experiments/common/wait_for_ray_nodes.py \
    "${RAY_HEAD_IP}:${RAY_PORT}" "${NNODES}" "${RAY_JOIN_TIMEOUT}"
