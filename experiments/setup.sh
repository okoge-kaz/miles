#!/bin/bash
# Prepare shared infrastructure. Jobs preview by default; --submit calls qsub.
set -euo pipefail
action="${1:-status}"
(( $# == 0 )) || shift
submit=0
options=()
model_preset=""
case "${action}" in
    convert|hf_mcore|mcore_hf)
        model_preset="${CONVERT_MODEL_PRESET:-}"
        if [[ $# -gt 0 && "$1" != -* ]]; then model_preset="$1"; shift; fi
        ;;
esac
while (( $# > 0 )); do
    if [[ "$1" == --submit ]]; then submit=1; else options+=("$1"); fi
    shift
done
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MILES_REPO="${REPO_ROOT}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/job_env.sh"
case "${action}" in
    init) echo "workspace=${MILES_WORKSPACE_ROOT} repo=${MILES_REPO}"; exit 0 ;;
    status)
        status_preset="${MODEL_PRESET:-qwen3-4b}"
        [[ "${status_preset}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid MODEL_PRESET" >&2; exit 2; }
        source "${REPO_ROOT}/experiments/scripts/models/${status_preset}.sh"
        for path in "${CONTAINER_IMAGE}" "${HF_CKPT_DIR}/${HF_MODEL_NAME}/config.json" \
            "${MEGATRON_CKPT_DIR}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt"; do
            if [[ -r "${path}" ]]; then printf 'ready   %s\n' "${path}"; else printf 'missing %s\n' "${path}"; fi
        done
        exit 0 ;;
    container) job="experiments/container/import_image.sh"; profile=cpu ;;
    convert|hf_mcore) job="experiments/scripts/checkpoint_convert/hf_mcore/run.sh"; profile=gpu ;;
    mcore_hf) job="experiments/scripts/checkpoint_convert/mcore_hf/run.sh"; profile=gpu ;;
    validate) job="experiments/scripts/cluster.sh"; profile=gpu ;;
    *) echo "usage: setup.sh {init|status|container|convert|hf_mcore|mcore_hf|validate} [model_preset] [--submit] [PBS options]" >&2; exit 2 ;;
esac
if [[ -n "${model_preset}" ]]; then
    [[ "${model_preset}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "invalid model preset" >&2; exit 2; }
    preset="${REPO_ROOT}/experiments/scripts/checkpoint_convert/models/${model_preset}.sh"
    [[ -r "${preset}" ]] || { echo "unknown model preset: ${preset}" >&2; exit 2; }
    source "${preset}"
    export CONVERT_MODEL_PRESET="${model_preset}"
fi
case "${action}" in
    convert|hf_mcore|mcore_hf)
        : "${MODEL_NAME:?select a model preset or set MODEL_NAME}"
        [[ "${MODEL_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid MODEL_NAME" >&2; exit 2; }
        if [[ "${action}" != mcore_hf ]]; then
            : "${MODEL_PROFILE:?select a model preset or set MODEL_PROFILE}"
            : "${CONVERT_INPUT_DIR:=${HF_MODEL_NAME:+/ckpt/hf/${HF_MODEL_NAME}}}"
            : "${CONVERT_INPUT_DIR:?set HF_MODEL_NAME or CONVERT_INPUT_DIR}"
            export CONVERT_INPUT_DIR
        else
            : "${CONVERT_ORIGIN_HF_DIR:=${HF_MODEL_NAME:+/ckpt/hf/${HF_MODEL_NAME}}}"
            : "${CONVERT_ORIGIN_HF_DIR:?set HF_MODEL_NAME or CONVERT_ORIGIN_HF_DIR}"
            export CONVERT_ORIGIN_HF_DIR
        fi
        ;;
esac
log_dir="${OUTPUT_DIR}/setup"
log_prefix="${action}"
case "${action}" in
    convert|hf_mcore|mcore_hf)
        log_dir="${OUTPUT_DIR}/ckpt/$(basename -- "$(dirname -- "${job}")")"
        log_prefix="${CONVERT_MODEL_PRESET:-${MODEL_NAME}}_${MODEL_VARIANT:-custom}"
        ;;
esac
log_prefix="${log_prefix//[-.]/_}"
log_file="${log_dir}/${log_prefix}_$(date +%Y%m%d-%H%M%S)-p$$.ou"
(( submit != 0 )) || options+=(--test-only)
pbs_submit --profile="${profile}" --chdir="${REPO_ROOT}" \
    --output="${log_file}" --export="$(miles_job_export_spec)" \
    "${options[@]}" "${REPO_ROOT}/${job}"
