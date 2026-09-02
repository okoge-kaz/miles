#!/bin/bash
# Convert the three existing SFT Hugging Face checkpoints to Miles/Megatron
# torch_dist checkpoints. No download is performed: the source locations are
# recorded in sft_checkpoints.txt.
#
# Dry-run and validate inputs:
#   experiments/setup/models/stage_sft_checkpoints.sh
# Preview conversions before external SFT exports have been copied into place:
#   experiments/setup/models/stage_sft_checkpoints.sh --allow-missing-sources
# Submit a serialized conversion chain to the PBS GPU queue:
#   experiments/setup/models/stage_sft_checkpoints.sh --submit

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh
source experiments/common/pbs.sh

SUBMIT=0
ALLOW_MISSING_SOURCES=0
while (( $# > 0 )); do
    case "$1" in
        --submit) SUBMIT=1 ;;
        --allow-missing-sources) ALLOW_MISSING_SOURCES=1 ;;
        *)
            echo "usage: $0 [--submit | --allow-missing-sources]" >&2
            exit 2
            ;;
    esac
    shift
done
if (( SUBMIT != 0 && ALLOW_MISSING_SOURCES != 0 )); then
    echo "--allow-missing-sources is only valid for a dry-run" >&2
    exit 2
fi

MANIFEST=experiments/setup/manifests/sft_checkpoints.txt
mkdir -p "${OUTPUT_DIR}/convert"
previous_job=""
SETUP_AFTEROK="${SETUP_AFTEROK:-}"
SETUP_CONVERT_WALLTIME="${SETUP_CONVERT_WALLTIME:-${PBS_PREP_WALLTIME:-08:00:00}}"
SETUP_PATH_EXPORTS="MILES_WORKSPACE_ROOT,MILES_REPO,CHECKPOINT_ROOT,HF_CKPT_DIR,MEGATRON_CKPT_DIR,TRAIN_CKPT_DIR,DATASET_ROOT,PRETRAIN_DATASET_DIR,RL_DATASET_DIR,SFT_DATASET_DIR,DATASET_DIR,CONTAINER_DIR,CACHE_DIR,CONTAINER_IMAGE"
jobs_manifest="${OUTPUT_DIR}/convert/sft-checkpoints.jobs"

validate_relative_path() {
    local field_name="$1"
    local value="$2"
    [[ -n "${value}" && "${value}" != /* ]] || {
        echo "${field_name} must be a non-empty relative path: ${value}" >&2
        exit 1
    }
    case "/${value}/" in
        */../*)
            echo "${field_name} must not contain '..': ${value}" >&2
            exit 1
            ;;
    esac
}

validate_all_inputs() {
    local model_name hf_relative_dir hf_model_name model_type
    local hf_model_path source_config
    local missing_sources=0

    while IFS='|' read -r model_name hf_relative_dir hf_model_name model_type; do
        model_name="$(echo "${model_name}" | xargs)"
        [[ -z "${model_name}" || "${model_name}" == \#* ]] && continue
        hf_relative_dir="$(echo "${hf_relative_dir}" | xargs)"
        hf_model_name="$(echo "${hf_model_name}" | xargs)"
        model_type="$(echo "${model_type}" | xargs)"

        validate_relative_path HF_RELATIVE_DIR "${hf_relative_dir}"
        validate_relative_path HF_MODEL_NAME "${hf_model_name}"
        hf_model_path="${hf_relative_dir%/}/${hf_model_name}"
        source_config="${HF_CKPT_DIR}/${hf_model_path}/config.json"
        if [[ ! -f "${source_config}" ]]; then
            echo "missing ${source_config}" >&2
            missing_sources=$((missing_sources + 1))
        fi
        if [[ ! -f "scripts/models/${model_type}.sh" ]]; then
            echo "missing scripts/models/${model_type}.sh" >&2
            return 1
        fi
    done < "${MANIFEST}"

    if (( missing_sources != 0 && ALLOW_MISSING_SOURCES == 0 )); then
        return 1
    fi
    if (( missing_sources != 0 )); then
        echo "preview continues with ${missing_sources} missing external SFT source file(s)" >&2
    fi
}

validate_all_inputs
if (( SUBMIT != 0 )); then
    printf 'SUBMITTED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${jobs_manifest}"
fi

while IFS='|' read -r model_name hf_relative_dir hf_model_name model_type; do
    model_name="$(echo "${model_name}" | xargs)"
    [[ -z "${model_name}" || "${model_name}" == \#* ]] && continue
    hf_relative_dir="$(echo "${hf_relative_dir}" | xargs)"
    hf_model_name="$(echo "${hf_model_name}" | xargs)"
    model_type="$(echo "${model_type}" | xargs)"

    validate_relative_path HF_RELATIVE_DIR "${hf_relative_dir}"
    validate_relative_path HF_MODEL_NAME "${hf_model_name}"
    hf_model_path="${hf_relative_dir%/}/${hf_model_name}"
    tracker="${MEGATRON_CKPT_DIR}/${model_name}_torch_dist/latest_checkpointed_iteration.txt"

    if [[ -f "${tracker}" && "$(cat "${tracker}")" == release ]]; then
        printf 'ready  %-45s %s\n' "${model_name}" "${tracker}"
        continue
    fi

    if (( SUBMIT == 0 )); then
        printf 'would convert %-37s %s/%s\n' \
            "${model_name}" "${HF_CKPT_DIR}" "${hf_model_path}"
        continue
    fi

    dependency=()
    if [[ -n "${previous_job}" ]]; then
        dependency=(--dependency="afterany:${previous_job}")
    elif [[ -n "${SETUP_AFTEROK}" ]]; then
        dependency=(--dependency="afterok:${SETUP_AFTEROK}")
    fi
    raw_job_id="$(pbs_submit --parsable --profile=gpu \
        --time="${SETUP_CONVERT_WALLTIME}" \
        --job-name="cv-${model_name}" \
        "${dependency[@]}" \
        --export="${SETUP_PATH_EXPORTS},USER,WANDB_MODE=disabled,MODEL_NAME=${model_name},HF_MODEL_NAME=${hf_model_path},MEGATRON_MODEL_TYPE=${model_type}" \
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
