#!/bin/bash
# Bring up the Ray cluster for one Slurm allocation. Sourced by every train.sh,
# once per node, after RUN_NAME is set.
#
# Node 0 is the head and the driver and returns to its caller. Every other node
# joins, idles until the driver signals completion, and EXITS from here — so
# nothing after the `source` line runs on a worker.
#
# The driver signals through RAY_DONE_FLAG on the shared filesystem, set from an
# EXIT trap, so a crashed driver releases the workers instead of leaving them to
# burn the walltime. run.sbatch clears a stale flag before the job starts.
#
# Sets: NNODES, NODEID, GPUS_PER_NODE, RAY_PORT, RAY_HEAD_IP, RAY_DONE_FLAG.

NNODES="${SLURM_JOB_NUM_NODES:-1}"
NODEID="${SLURM_NODEID:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
RAY_DONE_FLAG="${RAY_DONE_FLAG:-/root/miles/experiments/outputs/.ray/${RUN_NAME}.done}"

ray stop --force || true

if [[ "${NODEID}" -ne 0 ]]; then
    until timeout 5 bash -c "cat < /dev/null > /dev/tcp/${RAY_HEAD_IP}/${RAY_PORT}" 2>/dev/null; do
        echo "worker ${NODEID}: waiting for ray head at ${RAY_HEAD_IP}:${RAY_PORT}"
        sleep 5
    done
    ray start --address="${RAY_HEAD_IP}:${RAY_PORT}" --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats
    while [[ ! -f "${RAY_DONE_FLAG}" ]]; do sleep 15; done
    echo "worker ${NODEID}: driver finished, shutting down"
    ray stop --force || true
    exit 0
fi

trap 'touch "${RAY_DONE_FLAG}" 2>/dev/null || true' EXIT

ray start --head --node-ip-address "${RAY_HEAD_IP}" --port "${RAY_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

python3 /root/miles/experiments/common/wait_for_ray_nodes.py \
    "${RAY_HEAD_IP}:${RAY_PORT}" "${NNODES}" "${RAY_JOIN_TIMEOUT:-600}"
