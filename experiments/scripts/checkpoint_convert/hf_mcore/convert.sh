#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
: "${MODEL_NAME:?set MODEL_NAME, or use a models/*.sh preset}"
: "${MODEL_PROFILE:?set MODEL_PROFILE to an official scripts/models/*.py definition}"
: "${GPUS_PER_NODE:=8}"
: "${CONVERT_BACKEND:=raw}"

source_dir="${CONVERT_INPUT_DIR:-/ckpt/hf/${HF_MODEL_NAME:?set HF_MODEL_NAME or CONVERT_INPUT_DIR}}"
destination="${CONVERT_OUTPUT_DIR:-/ckpt/megatron/${MODEL_NAME}_torch_dist}"
destination="${destination%/}"

[[ "${MODEL_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && "${MODEL_PROFILE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid MODEL_NAME, MODEL_PROFILE, or GPUS_PER_NODE" >&2; exit 2
}
[[ "${source_dir}" == /* && "${destination}" == /* && "${destination}" != / ]] || {
    echo "conversion paths must be absolute container paths" >&2; exit 2
}
[[ -r "${source_dir}/config.json" ]] || { echo "missing HF config: ${source_dir}/config.json" >&2; exit 2; }
case "${CONVERT_BACKEND}" in
    raw) converter=(tools/convert_hf_to_torch_dist.py) ;;
    bridge) converter=(experiments/src/checkpoint_convert/bridge.py hf_mcore) ;;
    *) echo "CONVERT_BACKEND must be raw or bridge" >&2; exit 2 ;;
esac
parallel_args=()
for parallelism in TENSOR PIPELINE EXPERT; do
    variable="CONVERT_${parallelism}_PARALLEL_SIZE"
    [[ -n "${!variable:-}" ]] || continue
    [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "${variable} must be a positive integer" >&2; exit 2; }
    parallel_args+=("--${parallelism,,}-model-parallel-size" "${!variable}")
done
[[ "${CONVERT_TENSOR_PARALLEL_SIZE:-1}" == 1 ]] || parallel_args+=(--sequence-parallel)

export PYTHONPATH="${REPO_ROOT}:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$(( GPUS_PER_NODE - 1 ))")"
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p -- "$(dirname -- "${destination}")"
exec 9> "${destination}.lock"
flock --nonblock 9 || { echo "another conversion is active" >&2; exit 5; }
if [[ -f "${destination}/latest_checkpointed_iteration.txt" ]]; then
    [[ "$(< "${destination}/latest_checkpointed_iteration.txt")" == release ]] || {
        echo "existing destination is not a release checkpoint: ${destination}" >&2; exit 2
    }
    echo "converted checkpoint already exists: ${destination}"
    exit 0
fi
model_args_line="$(python3 miles/utils/external_utils/model_args_utils.py "${MODEL_PROFILE}")"
read -r -a model_args <<< "${model_args_line}"
torchrun --nproc-per-node "${GPUS_PER_NODE}" "${converter[@]}" \
    "${model_args[@]}" "${parallel_args[@]}" --bf16 --megatron-to-hf-mode "${CONVERT_BACKEND}" \
    --hf-checkpoint "${source_dir}" --save "${destination}"
[[ -r "${destination}/latest_checkpointed_iteration.txt" \
    && "$(< "${destination}/latest_checkpointed_iteration.txt")" == release ]] || {
    echo "conversion did not produce a release tracker: ${destination}" >&2; exit 2
}
