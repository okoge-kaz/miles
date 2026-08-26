#!/bin/bash
# Convert the three existing SFT Hugging Face checkpoints to Miles/Megatron
# torch_dist checkpoints. No download is performed: the source locations are
# recorded in sft_checkpoints.txt.
#
# Dry-run and validate inputs:
#   experiments/setup/models/stage_sft_checkpoints.sh
# Submit a serialized conversion chain on the interactive QoS:
#   experiments/setup/models/stage_sft_checkpoints.sh --submit

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh

SUBMIT=0
[[ "${1:-}" == --submit ]] && SUBMIT=1
[[ $# -eq 0 || ( $# -eq 1 && "${1}" == --submit ) ]] || {
    echo "usage: $0 [--submit]" >&2
    exit 2
}

MANIFEST=experiments/setup/manifests/sft_checkpoints.txt
mkdir -p "${OUTPUT_DIR}/convert"
previous_job=""
jobs_manifest="${OUTPUT_DIR}/convert/sft-checkpoints.jobs"
if (( SUBMIT != 0 )); then
    printf 'SUBMITTED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${jobs_manifest}"
fi

while IFS='|' read -r model_name hf_root hf_model_name model_type; do
    model_name="$(echo "${model_name}" | xargs)"
    [[ -z "${model_name}" || "${model_name}" == \#* ]] && continue
    hf_root="$(echo "${hf_root}" | xargs)"
    hf_model_name="$(echo "${hf_model_name}" | xargs)"
    model_type="$(echo "${model_type}" | xargs)"

    source_config="${hf_root}/${hf_model_name}/config.json"
    tracker="${MEGATRON_CKPT_DIR}/${model_name}_torch_dist/latest_checkpointed_iteration.txt"
    [[ -f "${source_config}" ]] || {
        echo "missing ${source_config}" >&2
        exit 1
    }
    [[ -f "scripts/models/${model_type}.sh" ]] || {
        echo "missing scripts/models/${model_type}.sh" >&2
        exit 1
    }

    if [[ -f "${tracker}" && "$(cat "${tracker}")" == release ]]; then
        printf 'ready  %-45s %s\n' "${model_name}" "${tracker}"
        continue
    fi

    if (( SUBMIT == 0 )); then
        printf 'would convert %-37s %s/%s\n' \
            "${model_name}" "${hf_root}" "${hf_model_name}"
        continue
    fi

    dependency=()
    [[ -n "${previous_job}" ]] && dependency=(--dependency="afterany:${previous_job}")
    raw_job_id="$(sbatch --parsable \
        -A "${SLURM_ACCOUNT_NAME}" \
        -p "${GPU_PARTITION}" --qos="${INTERACTIVE_GPU_QOS}" \
        --time=04:00:00 \
        --job-name="cv-${model_name}" \
        "${dependency[@]}" \
        --export="ALL,HF_CKPT_DIR=${hf_root},MODEL_NAME=${model_name},HF_MODEL_NAME=${hf_model_name},MEGATRON_MODEL_TYPE=${model_type}" \
        experiments/setup/models/convert_checkpoint.sbatch)"
    previous_job="${raw_job_id%%;*}"
    printf 'submitted %-41s job=%s dependency=%s\n' \
        "${model_name}" "${previous_job}" "${dependency[*]:-none}"
    job_key="$(echo "${model_name}" | tr '[:lower:].-' '[:upper:]__')"
    printf '%s_JOB=%q\n' "${job_key}" "${previous_job}" >> "${jobs_manifest}"
done < "${MANIFEST}"

if (( SUBMIT == 0 )); then
    echo "dry-run only; pass --submit to enqueue conversions"
else
    echo "job manifest: ${jobs_manifest}"
fi
