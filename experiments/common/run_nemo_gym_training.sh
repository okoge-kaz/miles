#!/bin/bash

set -euo pipefail
train_script="${1:?usage: run_nemo_gym_training.sh <train.sh> [NeMo Gym options]}"
shift
REPO_ROOT="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ -n "${NEMO_GYM_ENDPOINTS:-}" || "${MILES_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
    exec bash "${REPO_ROOT}/experiments/common/run_training.sh" \
        "${train_script}"
fi

source "${REPO_ROOT}/experiments/env.sh"
state_dir="${OUTPUT_DIR}/nemo_gym/${MILES_JOB_ID_SHORT}"
[[ ! -e "${state_dir}" ]] || { echo "NeMo Gym state already exists: ${state_dir}" >&2; exit 2; }
mkdir -p "${state_dir}"
container_state="/root/miles/experiments/outputs/nemo_gym/${MILES_JOB_ID_SHORT}"
export NEMO_GYM_ENDPOINTS="${container_state}/endpoints.json"
host_ip="$(getent ahostsv4 "$(hostname)" | awk 'NR == 1 {print $1}')"
[[ -n "${host_ip}" && "${host_ip}" != 127.* ]] || { echo "no reachable Gym host address" >&2; exit 2; }

gym_pid="" training_pid=""
cleanup() {
    local pid
    for pid in "${training_pid}" "${gym_pid}"; do
        [[ -z "${pid}" ]] || kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    for pid in "${training_pid}" "${gym_pid}"; do
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
_miles_build_container_command --image "${CONTAINER_IMAGE}" --bind "${CONTAINER_MOUNTS}" \
    --cwd /root/miles --no-nv --cleanenv \
    env PYTHONPATH=/root/miles:/opt/nemo-gym OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    bash /root/miles/experiments/scripts/nemo_gym/serve.sh --state-dir "${container_state}" \
    --host "${host_ip}" "$@"
setsid "${MILES_CONTAINER_COMMAND[@]}" > "${state_dir}/runtime.log" 2>&1 &
gym_pid=$!
for (( waited=0; waited<360; waited++ )); do
    kill -0 "${gym_pid}" 2>/dev/null || { echo "NeMo Gym failed; see ${state_dir}" >&2; exit 1; }
    [[ ! -s "${state_dir}/endpoints.json" ]] || break
    sleep 1
done
[[ -s "${state_dir}/endpoints.json" ]] || { echo "NeMo Gym readiness timeout" >&2; exit 1; }
setsid bash "${REPO_ROOT}/experiments/common/run_training.sh" \
    "${train_script}" &
training_pid=$!
status=0
wait -n "${gym_pid}" "${training_pid}" || status=$?
if ! kill -0 "${gym_pid}" 2>/dev/null; then
    echo "NeMo Gym exited while training was running; see ${state_dir}" >&2
    exit 1
fi
exit "${status}"
