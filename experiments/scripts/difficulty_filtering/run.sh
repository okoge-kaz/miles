#!/bin/bash
#PBS -N difficulty-filtering
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${MILES_REPO:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
export WANDB_MODE=disabled
source "${REPO_ROOT}/experiments/env.sh"
[[ -n "${PBS_JOBID:-}" && "${MILES_JOB_NUM_NODES}" == 1 ]] || {
    echo 'Run inside a one-node PBS allocation' >&2; exit 2;
}
config_path="${1:?configuration path is required}"
mode="${2:?smoke, filter, or merge is required}"
shard_index="${3:?shard index is required}"
[[ "${config_path}" == /root/miles/experiments/outputs/difficulty_filtering/*/config.json ]] || {
    echo 'Expected a configuration snapshot under experiments/outputs/difficulty_filtering' >&2; exit 2;
}
host_config_path="${REPO_ROOT}/${config_path#/root/miles/}"
source_root="${config_path%/*}/source"
[[ -d "${host_config_path%/*}/source/experiments/src/difficulty_filtering" ]] || {
    echo 'Filtering source snapshot is missing' >&2; exit 2;
}
exec > >(tee -a "${host_config_path%/*}/${mode}-${shard_index}.log") 2>&1

container_options=(--cleanenv)
case "${mode}" in
    smoke|filter)
        container_options+=(--nv)
        payload=(python3 -m experiments.src.difficulty_filtering.core.measure --config "${config_path}" --shard-index "${shard_index}")
        [[ "${mode}" != smoke ]] || payload+=(--limit "${4:?smoke prompt count is required}")
        ;;
    merge)
        container_options+=(--no-nv)
        payload=(python3 -m experiments.src.difficulty_filtering.core.apply_filter --config "${config_path}")
        ;;
    *) echo "Unknown mode: ${mode}" >&2; exit 2 ;;
esac
gym_pid="" measure_pid=""
cleanup() {
    local pid
    for pid in "${measure_pid}" "${gym_pid}"; do
        [[ -z "${pid}" ]] || kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    for pid in "${measure_pid}" "${gym_pid}"; do
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

verifier_env=()
if [[ "${mode}" != merge ]]; then
    mapfile -t verifier_settings < <(python3 - "${host_config_path}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
print(config["reward_backend"])
print(config["startup_timeout"])
PY
    )
    if [[ "${verifier_settings[0]}" != deepscaler ]]; then
        # Singularity --nv makes nested /proc creation fail in the Code sandbox.
        # Keep the verifier in a separate CPU container on this same PBS node.
        state_name="gym-${MILES_JOB_ID_SHORT}-${shard_index}"
        host_state="${host_config_path%/*}/${state_name}"
        container_state="${config_path%/*}/${state_name}"
        [[ ! -e "${host_state}" ]] || { echo "NeMo Gym state already exists: ${host_state}" >&2; exit 2; }
        mkdir -p "${host_state}"
        _miles_build_container_command --image "${CONTAINER_IMAGE}" --bind "${CONTAINER_MOUNTS}" \
            --cwd "${source_root}" --no-nv --cleanenv \
            env PYTHONPATH="${source_root}:/root/miles" PYTHONDONTWRITEBYTECODE=1 \
            OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            /opt/nemo-gym/.venv/bin/python -m experiments.src.difficulty_filtering.core.gym \
            --config "${config_path}" --state-dir "${container_state}"
        setsid "${MILES_CONTAINER_COMMAND[@]}" > "${host_state}/runtime.log" 2>&1 &
        gym_pid=$!
        for (( waited=0; waited<verifier_settings[1]; waited++ )); do
            kill -0 "${gym_pid}" 2>/dev/null || { echo "NeMo Gym failed; see ${host_state}" >&2; exit 1; }
            [[ ! -s "${host_state}/endpoints.json" ]] || break
            sleep 1
        done
        [[ -s "${host_state}/endpoints.json" ]] || { echo 'NeMo Gym readiness timeout' >&2; exit 1; }
        verifier_env+=("NEMO_GYM_ENDPOINTS=${container_state}/endpoints.json")
    fi
fi

_miles_build_container_command --image "${CONTAINER_IMAGE}" --bind "${CONTAINER_MOUNTS}" \
    --cwd "${source_root}" "${container_options[@]}" \
    env PYTHONPATH="${source_root}:/root/miles:/root/Megatron-LM" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false SGLANG_TIMEOUT_KEEP_ALIVE=120 \
    "${verifier_env[@]}" "${payload[@]}"
setsid "${MILES_CONTAINER_COMMAND[@]}" &
measure_pid=$!
status=0
wait -n "${measure_pid}" ${gym_pid:+"${gym_pid}"} || status=$?
if [[ -n "${gym_pid}" ]] && ! kill -0 "${gym_pid}" 2>/dev/null; then
    echo 'NeMo Gym exited while filtering was running' >&2
    exit 1
fi
exit "${status}"
