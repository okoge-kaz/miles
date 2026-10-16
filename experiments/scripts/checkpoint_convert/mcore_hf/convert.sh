#!/bin/bash

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
: "${MODEL_NAME:?set MODEL_NAME, or use a models/*.sh preset}"
: "${CONVERT_BACKEND:=raw}"
[[ "${MODEL_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid MODEL_NAME" >&2; exit 2; }
source_dir="${CONVERT_INPUT_DIR:-/ckpt/megatron/${MODEL_NAME}_torch_dist}"
destination="${CONVERT_OUTPUT_DIR:-/ckpt/hf/${MODEL_NAME}_from_mcore}"
destination="${destination%/}"
origin_hf_dir="${CONVERT_ORIGIN_HF_DIR:-/ckpt/hf/${HF_MODEL_NAME:?set HF_MODEL_NAME or CONVERT_ORIGIN_HF_DIR}}"
[[ "${source_dir}" == /* && "${destination}" == /* && "${destination}" != / \
    && "${origin_hf_dir}" == /* ]] || { echo "conversion paths must be absolute container paths" >&2; exit 2; }
[[ -r "${origin_hf_dir}/config.json" ]] || { echo "missing original HF config: ${origin_hf_dir}/config.json" >&2; exit 2; }
# The official reverse converter takes an iteration directory, not its parent.
if [[ ! -r "${source_dir}/.metadata" ]]; then
    tracker="${source_dir}/latest_checkpointed_iteration.txt"
    [[ -r "${tracker}" ]] || { echo "missing checkpoint or tracker: ${source_dir}" >&2; exit 2; }
    iteration="$(< "${tracker}")"
    if [[ "${iteration}" == release ]]; then
        source_dir="${source_dir}/release"
    elif [[ "${iteration}" =~ ^[0-9]{1,10}$ ]]; then
        printf -v iteration '%07d' "$(( 10#${iteration} ))"
        source_dir="${source_dir}/iter_${iteration}"
    else
        echo "invalid checkpoint tracker: ${tracker}" >&2; exit 2
    fi
fi
[[ -r "${source_dir}/.metadata" ]] || {
    echo "missing torch_dist .metadata: ${source_dir}" >&2; exit 2
}
mkdir -p -- "$(dirname -- "${destination}")"
exec 9> "${destination}.lock"
flock --nonblock 9 || { echo "another conversion is active" >&2; exit 5; }
[[ ! -e "${destination}" && ! -L "${destination}" ]] || {
    echo "HF output already exists; choose another CONVERT_OUTPUT_DIR: ${destination}" >&2; exit 2
}
extra_args=()
case "${CONVERT_BACKEND}" in
    raw)
        [[ -z "${CONVERT_CHUNK_SIZE:-}" ]] || extra_args+=(--chunk-size "${CONVERT_CHUNK_SIZE}")
        [[ -z "${CONVERT_VOCAB_SIZE:-}" ]] || extra_args+=(--vocab-size "${CONVERT_VOCAB_SIZE}")
        ;;
    bridge)
        : "${MODEL_PROFILE:?bridge export requires an official MODEL_PROFILE}"
        : "${GPUS_PER_NODE:=8}"
        [[ "${MODEL_PROFILE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
            && "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid MODEL_PROFILE or GPUS_PER_NODE" >&2; exit 2; }
        [[ -z "${CONVERT_CHUNK_SIZE:-}${CONVERT_VOCAB_SIZE:-}" ]] || {
            echo "CONVERT_CHUNK_SIZE and CONVERT_VOCAB_SIZE are only supported by the raw converter" >&2; exit 2
        }
        model_args_line="$(python3 miles/utils/external_utils/model_args_utils.py "${MODEL_PROFILE}")"
        read -r -a model_args <<< "${model_args_line}"
        for parallelism in TENSOR PIPELINE EXPERT; do
            variable="CONVERT_${parallelism}_PARALLEL_SIZE"
            [[ -n "${!variable:-}" ]] || continue
            [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "${variable} must be a positive integer" >&2; exit 2; }
            extra_args+=("--${parallelism,,}-model-parallel-size" "${!variable}")
        done
        [[ "${CONVERT_TENSOR_PARALLEL_SIZE:-1}" == 1 ]] || extra_args+=(--sequence-parallel)
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$(( GPUS_PER_NODE - 1 ))")"
        ;;
    *) echo "CONVERT_BACKEND must be raw or bridge" >&2; exit 2 ;;
esac
export PYTHONPATH="${REPO_ROOT}:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
# Publish only after all weights and HF assets are saved successfully.
working_dir="$(mktemp -d "${destination}.tmp.XXXXXX")"
trap 'rm -rf -- "${working_dir}"' EXIT
if [[ "${CONVERT_BACKEND}" == raw ]]; then
    python3 tools/convert_torch_dist_to_hf.py \
        --input-dir "${source_dir}" --output-dir "${working_dir}/hf" \
        --origin-hf-dir "${origin_hf_dir}" "${extra_args[@]}"
else
    # Offline conversion through the official Megatron Bridge and Miles plugins.
    #
    # The official mbridge offline converter has no Nemotron-H mapping. This adapter
    # reuses its argument parsing, Megatron initialization and checkpoint format,
    # and uses the same Bridge weight APIs as Miles' training backend. Run through
    # checkpoint_convert/{hf_mcore,mcore_hf}/convert.sh with CONVERT_BACKEND=bridge.
    torchrun --nproc-per-node "${GPUS_PER_NODE}" experiments/src/checkpoint_convert/bridge.py mcore_hf \
        "${model_args[@]}" "${extra_args[@]}" --bf16 --megatron-to-hf-mode bridge \
        --hf-checkpoint "${origin_hf_dir}" --load "${source_dir}" --save "${working_dir}/hf"
fi
shopt -s nullglob
weight_files=("${working_dir}/hf/"*.safetensors)
[[ -r "${working_dir}/hf/config.json" && ${#weight_files[@]} -gt 0 ]] || {
    echo "conversion did not produce HF config and weights" >&2; exit 2
}
if ! mv -T --no-clobber -- "${working_dir}/hf" "${destination}"; then
    echo "HF output could not be published without replacement: ${destination}" >&2
    exit 2
fi
[[ ! -d "${working_dir}/hf" ]] || { echo "HF output appeared during conversion: ${destination}" >&2; exit 2; }
echo "HF checkpoint saved: ${destination}"
