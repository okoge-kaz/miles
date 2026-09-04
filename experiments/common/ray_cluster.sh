#!/bin/bash
# Needs RAY_DONE_FLAG and GPUS_PER_NODE. Sets NNODES, NODEID, RAY_HEAD_IP.
# Worker nodes exit from here; only node 0 returns to the caller.

: "${RAY_DONE_FLAG:?}"
: "${GPUS_PER_NODE:?}"

RAY_PORT=6379
RAY_JOIN_TIMEOUT=600
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
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
    if [[ -n "${RAY_HEAD_IP_FILE:-}" ]]; then
        _waited=0
        until [[ -s "${RAY_HEAD_IP_FILE}" ]]; do
            (( _waited % 30 == 0 )) \
                && echo "worker ${NODEID}: waiting for node 0 to publish its ray head IP (${_waited}s)"
            sleep 1
            _waited=$(( _waited + 1 ))
            if (( _waited >= 120 )); then
                echo "worker ${NODEID}: ray head IP file was not published: ${RAY_HEAD_IP_FILE}" >&2
                exit 1
            fi
        done
        RAY_HEAD_IP="$(< "${RAY_HEAD_IP_FILE}")"
    fi
    : "${RAY_HEAD_IP:?set RAY_HEAD_IP or RAY_HEAD_IP_FILE}"
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

RAY_HEAD_IP="$(getent hosts "$(hostname)" 2>/dev/null | awk '{print $1; exit}' || true)"
if [[ -z "${RAY_HEAD_IP}" || "${RAY_HEAD_IP}" == 169.254.* ]]; then
    RAY_HEAD_IP="$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' || true)"
fi
[[ -n "${RAY_HEAD_IP}" && "${RAY_HEAD_IP}" != 169.254.* ]] || {
    echo "no routable address for the ray head on node 0 ($(hostname))" >&2
    exit 1
}
export RAY_HEAD_IP
if [[ -n "${RAY_HEAD_IP_FILE:-}" ]]; then
    _head_ip_tmp="${RAY_HEAD_IP_FILE}.node0.$$"
    printf '%s\n' "${RAY_HEAD_IP}" > "${_head_ip_tmp}"
    mv -f "${_head_ip_tmp}" "${RAY_HEAD_IP_FILE}"
    echo "node 0 ($(hostname)) published ray head ${RAY_HEAD_IP}"
fi

trap 'touch "${RAY_DONE_FLAG}" 2>/dev/null || true' EXIT

ray start --head --node-ip-address "${RAY_HEAD_IP}" --port "${RAY_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats \
    --dashboard-host="${RAY_DASHBOARD_HOST}" --dashboard-port=8265 "${RAY_TEMP_ARGS[@]}"

python3 /root/miles/experiments/common/wait_for_ray_nodes.py \
    "${RAY_HEAD_IP}:${RAY_PORT}" "${NNODES}" "${RAY_JOIN_TIMEOUT}"
