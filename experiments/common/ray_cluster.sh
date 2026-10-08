#!/bin/bash
# Needs RAY_DONE_FLAG and GPUS_PER_NODE. Sets NNODES, NODEID, RAY_HEAD_IP.
# Worker nodes exit from here; only node 0 returns to the caller.

: "${RAY_DONE_FLAG:?}"
: "${GPUS_PER_NODE:?}"

_MILES_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if ! declare -F _miles_pbs_refresh_context >/dev/null; then
    # shellcheck source=experiments/common/pbs.sh
    source "${_MILES_COMMON_DIR}/pbs.sh"
fi
unset _MILES_COMMON_DIR

MILES_JOB_NUM_NODES="${MILES_JOB_NUM_NODES:?PBS node count is not initialized}"
MILES_NODE_RANK="${MILES_NODE_RANK:?PBS node rank is not initialized}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_JOIN_TIMEOUT="${RAY_JOIN_TIMEOUT:-600}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-127.0.0.1}"
NNODES="${MILES_JOB_NUM_NODES}"
NODEID="${MILES_NODE_RANK}"

RAY_TEMP_ARGS=()
if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
    NODE_RAY_TEMP_DIR="${RAY_TEMP_DIR}/node-${NODEID}"
    mkdir -p "${NODE_RAY_TEMP_DIR}"
    RAY_TEMP_ARGS=(--temp-dir "${NODE_RAY_TEMP_DIR}")
fi

ray stop --force || true

_miles_stop_ray() {
    if [[ "${NODEID}" == 0 ]]; then
        touch "${RAY_DONE_FLAG}" 2>/dev/null || true
    fi
    ray stop --force || true
}
trap _miles_stop_ray EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ "${NODEID}" -ne 0 ]]; then
    if [[ -n "${RAY_HEAD_IP_FILE:-}" ]]; then
        _waited=0
        until [[ -s "${RAY_HEAD_IP_FILE}" ]]; do
            (( _waited % 30 == 0 )) \
                && echo "worker ${NODEID}: waiting for node 0 to publish its ray head IP (${_waited}s)"
            sleep 1
            _waited=$(( _waited + 1 ))
            if [[ -f "${RAY_DONE_FLAG}" ]] || (( _waited >= RAY_JOIN_TIMEOUT )); then
                echo "worker ${NODEID}: ray head IP file was not published: ${RAY_HEAD_IP_FILE}" >&2
                exit 1
            fi
        done
        RAY_HEAD_IP="$(< "${RAY_HEAD_IP_FILE}")"
    fi
    : "${RAY_HEAD_IP:?set RAY_HEAD_IP or RAY_HEAD_IP_FILE}"
    _waited=0
    until timeout 5 bash -c "cat < /dev/null > /dev/tcp/${RAY_HEAD_IP}/${RAY_PORT}" 2>/dev/null; do
        (( _waited % 60 == 0 )) && echo "worker ${NODEID}: waiting for ray head at ${RAY_HEAD_IP}:${RAY_PORT} (${_waited}s)"
        sleep 5
        _waited=$(( _waited + 5 ))
        if [[ -f "${RAY_DONE_FLAG}" ]] || (( _waited >= RAY_JOIN_TIMEOUT )); then
            echo "worker ${NODEID}: ray head did not become ready" >&2
            exit 1
        fi
    done
    ray start --address="${RAY_HEAD_IP}:${RAY_PORT}" --num-gpus "${GPUS_PER_NODE}" \
        --disable-usage-stats "${RAY_TEMP_ARGS[@]}"
    echo "worker ${NODEID}: joined, idling until the driver signals completion"
    while [[ ! -f "${RAY_DONE_FLAG}" ]]; do sleep 15; done
    echo "worker ${NODEID}: driver finished, shutting down"
    ray stop --force || true
    exit 0
fi

RAY_HEAD_IP="${RAY_HEAD_IP:-$(getent ahostsv4 "$(hostname)" 2>/dev/null | awk '{print $1; exit}' || true)}"
if [[ -z "${RAY_HEAD_IP}" || "${RAY_HEAD_IP}" == 169.254.* ]]; then
    RAY_HEAD_IP="$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' || true)"
fi
[[ "${RAY_HEAD_IP}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ && "${RAY_HEAD_IP}" != 169.254.* ]] || {
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

ray start --head --node-ip-address "${RAY_HEAD_IP}" --port "${RAY_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats \
    --dashboard-host="${RAY_DASHBOARD_HOST}" --dashboard-port=8265 "${RAY_TEMP_ARGS[@]}"

python3 /root/miles/experiments/common/wait_for_ray_nodes.py \
    "${RAY_HEAD_IP}:${RAY_PORT}" "${NNODES}" "${RAY_JOIN_TIMEOUT}"
export MASTER_ADDR="${RAY_HEAD_IP}"
export RAY_ADDRESS="http://127.0.0.1:8265"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,${RAY_HEAD_IP}"
