#!/bin/bash
#PBS -N tau-user-simulator
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${MILES_REPO:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}/containers/miles-v0.1.2-multi-env.sif}"

source "${REPO_ROOT}/experiments/env.sh"
[[ "${MILES_JOB_NUM_NODES}" == 1 ]] || { echo "user simulator requires one node" >&2; exit 2; }
: "${TAU_SERVER_RUN_NAME:?set a distinct user-simulator run name}"
[[ "${TAU_SERVER_RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
if [[ "${MILES_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec bash "${REPO_ROOT}/experiments/scripts/nemo_gym/user_simulator/serve.sh" --dry-run
fi
[[ -n "${PBS_JOBID:-}" ]] || { echo "start the simulator inside a PBS allocation" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}/multi_env/user_simulator"
exec > >(tee "${OUTPUT_DIR}/multi_env/user_simulator/${TAU_SERVER_RUN_NAME}.log") 2>&1

endpoint_file="${OUTPUT_DIR}/multi_env/user_simulator/${TAU_SERVER_RUN_NAME}.json"
[[ ! -e "${endpoint_file}" ]] || { echo "endpoint already exists: ${endpoint_file}" >&2; exit 2; }
server_pid="" publisher_pid=""
cleanup() {
    rm -f -- "${endpoint_file}" "${endpoint_file%.json}.tmp"
    for pid in "${publisher_pid}" "${server_pid}"; do
        [[ -z "${pid}" ]] || kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    for pid in "${publisher_pid}" "${server_pid}"; do
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
_miles_build_container_command --nv \
    --image "${CONTAINER_IMAGE}" \
    --bind "${CONTAINER_MOUNTS}" \
    --cwd /root/miles \
    --export=TAU_SERVER_MODEL_PATH,TAU_SERVER_MODEL_REVISION,TAU_SERVER_MODEL_NAME,TAU_SERVER_PORT,TAU_SERVER_TP_SIZE,TAU_SERVER_CONTEXT_LENGTH,TAU_SERVER_MEM_FRACTION,TAU_SERVER_MAX_RUNNING_REQUESTS,TAU_SERVER_TOOL_CALL_PARSER,TAU_SERVER_REASONING_PARSER \
    env PYTHONPATH=/root/miles:/root/Megatron-LM bash /root/miles/experiments/scripts/nemo_gym/user_simulator/serve.sh
setsid "${MILES_CONTAINER_COMMAND[@]}" &
server_pid=$!
host_ip="$(getent ahostsv4 "$(hostname)" | awk 'NR == 1 {print $1}')"
_miles_load_python_runtime
setsid python3 "${REPO_ROOT}/experiments/scripts/nemo_gym/user_simulator/publish_endpoint.py" \
    --base-url "http://${host_ip}:${TAU_SERVER_PORT:-30180}/v1" \
    --model "${TAU_SERVER_MODEL_NAME:-qwen3-4b-sft-user}" --revision "${TAU_SERVER_MODEL_REVISION}" \
    --output "${endpoint_file}" &
publisher_pid=$!
wait -n "${server_pid}" "${publisher_pid}"
[[ -s "${endpoint_file}" ]] || exit 1
kill -0 "${server_pid}" 2>/dev/null || exit 1
publisher_pid=""
echo "TAU_USER_MODEL_CONFIG=/root/miles/experiments/outputs/multi_env/user_simulator/${TAU_SERVER_RUN_NAME}.json"
wait "${server_pid}"
