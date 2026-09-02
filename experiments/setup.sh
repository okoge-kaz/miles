#!/bin/bash
# Unified entrypoint for persistent experiment assets.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh
source experiments/common/pbs.sh

ACTION=""
SUBMIT=0

usage() {
    cat <<'EOF'
Usage: experiments/setup.sh [--submit] <action>

Actions:
  init        Create the configured workspace directory layout.
  status      Show configured paths and staged asset counts.
  list        List asset groups and their delegated entrypoints.
  container   Build the configured OCI image as a verified Singularity image.
  models      Download and convert models from manifests/models.txt.
  datasets    Download datasets from manifests/datasets.txt.
  sft         Validate or convert checkpoints from manifests/sft_checkpoints.txt.
  all         Run container, models, datasets, and sft in that order.

Asset actions are dry-run by default. Add --submit to enqueue PBS jobs.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --submit)
            SUBMIT=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -* )
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            [[ -z "${ACTION}" ]] || {
                echo "only one action may be specified" >&2
                usage >&2
                exit 2
            }
            ACTION="$1"
            ;;
    esac
    shift
done
ACTION="${ACTION:-list}"
export SETUP_CONTAINER_WALLTIME="${SETUP_CONTAINER_WALLTIME:-${PBS_CONTAINER_WALLTIME:-00:30:00}}"
export SETUP_DOWNLOAD_WALLTIME="${SETUP_DOWNLOAD_WALLTIME:-${PBS_DOWNLOAD_WALLTIME:-24:00:00}}"
export SETUP_PREP_WALLTIME="${SETUP_PREP_WALLTIME:-${PBS_PREP_WALLTIME:-08:00:00}}"
export SETUP_CONVERT_WALLTIME="${SETUP_CONVERT_WALLTIME:-${PBS_PREP_WALLTIME:-08:00:00}}"

DATASET_ROOT="${DATASET_ROOT:-${MILES_WORKSPACE_ROOT:?source experiments/env.sh first}/datasets}"
PRETRAIN_DATASET_DIR="${PRETRAIN_DATASET_DIR:-${DATASET_ROOT}/pre-train}"
RL_DATASET_DIR="${RL_DATASET_DIR:-${DATASET_DIR}}"
SFT_DATASET_DIR="${SFT_DATASET_DIR:-${DATASET_ROOT}/sft}"
: "${CONTAINER_IMAGE:?source experiments/env.sh first}"

print_command() {
    printf 'dry-run:'
    printf ' %q' "$@"
    printf '\n'
}

run_delegate() {
    if (( SUBMIT != 0 )); then
        "$@"
    else
        print_command "$@"
    fi
}

manifest_rows() {
    awk '
        {
            line = $0
            sub(/#.*/, "", line)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
            if (line != "") count++
        }
        END { print count + 0 }
    ' "$1"
}

manifest_content() {
    sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$1" \
        | grep -v '^$'
}

dataset_complete() {
    local target="$1"
    [[ -s "${target}/MILES_SOURCE_PROVENANCE" ]] || return 1
    find "${target}" -type f \
        ! -path '*/.cache/*' ! -name '*.md' ! -name .gitattributes \
        ! -name 'MILES_SOURCE_PROVENANCE*' \
        -print -quit | grep -q .
}

validate_container_image() {
    local image checksum provenance expected_definition_sha oci_image

    [[ -r "${CONTAINER_IMAGE}" ]] || return 1
    command -v singularity >/dev/null || return 1
    image="$(readlink -f -- "${CONTAINER_IMAGE}")"
    [[ -n "${image}" ]] || return 1
    checksum="${image}.sha256"
    provenance="${image}.provenance.env"
    [[ -f "${checksum}" && -f "${provenance}" ]] || return 1
    (cd "$(dirname -- "${image}")" && \
        sha256sum --check "$(basename -- "${checksum}")" >/dev/null) || return 1
    oci_image="${DOCKER_IMAGE#docker://}"
    grep -Fqx "oci_image=${oci_image}" "${provenance}" || return 1
    grep -Fqx "sglang_repo=${SGLANG_REPO}" "${provenance}" || return 1
    grep -Fqx "sglang_branch=${SGLANG_BRANCH}" "${provenance}" || return 1
    grep -Fqx "sglang_commit=${SGLANG_COMMIT}" "${provenance}" || return 1
    expected_definition_sha="$(sha256sum experiments/container/miles.def)"
    expected_definition_sha="${expected_definition_sha%% *}"
    grep -Fqx "definition_sha256=${expected_definition_sha}" "${provenance}" || return 1
    singularity inspect "${image}" >/dev/null 2>&1
}

init_layout() {
    mkdir -p \
        "${HF_CKPT_DIR}" \
        "${MEGATRON_CKPT_DIR}" \
        "${TRAIN_CKPT_DIR}" \
        "${CONTAINER_DIR}" \
        "${PRETRAIN_DATASET_DIR}" \
        "${RL_DATASET_DIR}" \
        "${SFT_DATASET_DIR}" \
        "${CACHE_DIR}" \
        "${MILES_WORKSPACE_ROOT}/src" \
        "${OUTPUT_DIR}/download" \
        "${OUTPUT_DIR}/convert"
    echo "workspace initialized: ${MILES_WORKSPACE_ROOT}"
}

show_status() {
    local model_total model_hf_ready=0 model_megatron_ready=0
    local dataset_total dataset_ready=0 sft_total sft_source_ready=0 sft_megatron_ready=0
    local name repo model_type extra nodes relative_dir hf_model_name

    printf '%-24s %s\n' workspace "${MILES_WORKSPACE_ROOT}"
    printf '%-24s %s\n' hf-checkpoints "${HF_CKPT_DIR}"
    printf '%-24s %s\n' megatron-checkpoints "${MEGATRON_CKPT_DIR}"
    printf '%-24s %s\n' training-checkpoints "${TRAIN_CKPT_DIR}"
    printf '%-24s %s\n' containers "${CONTAINER_DIR}"
    printf '%-24s %s\n' pretrain-datasets "${PRETRAIN_DATASET_DIR}"
    printf '%-24s %s\n' rl-datasets "${RL_DATASET_DIR}"
    printf '%-24s %s\n' sft-datasets "${SFT_DATASET_DIR}"
    printf '%-24s %s\n' cache "${CACHE_DIR}"
    if validate_container_image; then
        printf '%-24s verified (%s)\n' container "${CONTAINER_IMAGE}"
    elif [[ -e "${CONTAINER_IMAGE}" || -L "${CONTAINER_IMAGE}" ]]; then
        printf '%-24s stale/unverified (%s)\n' container "${CONTAINER_IMAGE}"
    else
        printf '%-24s missing (%s)\n' container "${CONTAINER_IMAGE}"
    fi

    model_total="$(manifest_rows experiments/setup/manifests/models.txt)"
    while IFS='|' read -r name repo model_type extra nodes; do
        name="$(echo "${name}" | xargs)"
        [[ -z "${name}" ]] && continue
        [[ -s "${HF_CKPT_DIR}/${name}/.download_complete" ]] && (( model_hf_ready += 1 ))
        if [[ "$(cat "${MEGATRON_CKPT_DIR}/${name}_torch_dist/latest_checkpointed_iteration.txt" 2>/dev/null || true)" == release ]]; then
            (( model_megatron_ready += 1 ))
        fi
    done < <(manifest_content experiments/setup/manifests/models.txt)
    printf '%-24s hf=%d/%d megatron=%d/%d\n' models \
        "${model_hf_ready}" "${model_total}" "${model_megatron_ready}" "${model_total}"

    dataset_total="$(manifest_rows experiments/setup/manifests/datasets.txt)"
    while IFS='|' read -r name repo; do
        name="$(echo "${name}" | xargs)"
        [[ -z "${name}" ]] && continue
        dataset_complete "${DATASET_DIR}/${name}" && (( dataset_ready += 1 ))
    done < <(manifest_content experiments/setup/manifests/datasets.txt)
    printf '%-24s complete=%d/%d (%s)\n' datasets-manifest \
        "${dataset_ready}" "${dataset_total}" "${DATASET_DIR}"

    sft_total="$(manifest_rows experiments/setup/manifests/sft_checkpoints.txt)"
    while IFS='|' read -r name relative_dir hf_model_name model_type; do
        name="$(echo "${name}" | xargs)"
        [[ -z "${name}" ]] && continue
        relative_dir="$(echo "${relative_dir}" | xargs)"
        hf_model_name="$(echo "${hf_model_name}" | xargs)"
        [[ -f "${HF_CKPT_DIR}/${relative_dir}/${hf_model_name}/config.json" ]] \
            && (( sft_source_ready += 1 ))
        if [[ "$(cat "${MEGATRON_CKPT_DIR}/${name}_torch_dist/latest_checkpointed_iteration.txt" 2>/dev/null || true)" == release ]]; then
            (( sft_megatron_ready += 1 ))
        fi
    done < <(manifest_content experiments/setup/manifests/sft_checkpoints.txt)
    printf '%-24s source=%d/%d megatron=%d/%d\n' sft \
        "${sft_source_ready}" "${sft_total}" "${sft_megatron_ready}" "${sft_total}"
}

list_groups() {
    cat <<'EOF'
container  experiments/container/import_image.sbatch
models     experiments/setup/download/stage_all.sh models
datasets   experiments/setup/download/stage_all.sh datasets
sft        experiments/setup/models/stage_sft_checkpoints.sh
all        container + models + datasets + sft

Workflow-specific dataset entrypoints remain available for selective staging:
  experiments/setup/download/stage_nemotron_rl_datasets.sh
  experiments/setup/download/stage_swe_tool_rl_datasets.sh
  experiments/setup/download/stage_areal_tau2.sh
EOF
}

LAST_CONTAINER_JOB=""
stage_container() {
    if validate_container_image; then
        echo "skip container: verified (${CONTAINER_IMAGE})"
        return
    fi
    if [[ -e "${CONTAINER_IMAGE}" && ! -L "${CONTAINER_IMAGE}" ]]; then
        echo "container exists but is not a verified managed image: ${CONTAINER_IMAGE}" >&2
        echo "move it aside or set CONTAINER_IMAGE/SINGULARITY_LINK to a managed symlink" >&2
        return 1
    fi
    [[ ! -e "${CONTAINER_IMAGE}" && ! -L "${CONTAINER_IMAGE}" ]] || \
        echo "rebuilding stale or unverified container: ${CONTAINER_IMAGE}"

    local exports
    exports="MILES_WORKSPACE_ROOT=${MILES_WORKSPACE_ROOT},MILES_REPO=${MILES_REPO},CONTAINER_DIR=${CONTAINER_DIR},CACHE_DIR=${CACHE_DIR},CONTAINER_IMAGE=${CONTAINER_IMAGE},DOCKER_IMAGE=${DOCKER_IMAGE},SGLANG_REPO=${SGLANG_REPO},SGLANG_BRANCH=${SGLANG_BRANCH},SGLANG_COMMIT=${SGLANG_COMMIT},SINGULARITY_LINK=${CONTAINER_IMAGE},WANDB_MODE=disabled"
    if [[ -n "${IMPORT_OUTPUT_IMAGE:-}" ]]; then
        exports="${exports},IMPORT_OUTPUT_IMAGE=${IMPORT_OUTPUT_IMAGE}"
    fi

    if (( SUBMIT == 0 )); then
        print_command pbs_submit --parsable --profile=cpu \
            --time="${SETUP_CONTAINER_WALLTIME}" --export="${exports}" \
            experiments/container/import_image.sbatch
        return
    fi

    LAST_CONTAINER_JOB="$(pbs_submit --parsable --profile=cpu \
        --time="${SETUP_CONTAINER_WALLTIME}" \
        --export="${exports}" \
        experiments/container/import_image.sbatch)"
    echo "submitted container job=${LAST_CONTAINER_JOB}"
}

stage_models() {
    run_delegate experiments/setup/download/stage_all.sh models
}

stage_datasets() {
    run_delegate experiments/setup/download/stage_all.sh datasets
}

stage_sft() {
    local preview_option="${1:-}"

    if (( SUBMIT != 0 )); then
        experiments/setup/models/stage_sft_checkpoints.sh --submit
    else
        echo "dry-run: validating SFT checkpoint sources"
        experiments/setup/models/stage_sft_checkpoints.sh ${preview_option:+"${preview_option}"}
    fi
}

stage_all() {
    if (( SUBMIT != 0 )); then
        echo "preflight: validating SFT checkpoint sources before queueing any jobs"
        experiments/setup/models/stage_sft_checkpoints.sh
    fi
    stage_container
    if [[ -n "${LAST_CONTAINER_JOB}" ]]; then
        export SETUP_AFTEROK="${LAST_CONTAINER_JOB}"
        echo "remaining setup jobs will wait for ${SETUP_AFTEROK}"
    fi
    stage_models
    stage_datasets
    if (( SUBMIT != 0 )); then
        stage_sft
    else
        stage_sft --allow-missing-sources
    fi
    unset SETUP_AFTEROK
}

stage_with_container() {
    local action="$1"

    stage_container
    if [[ -n "${LAST_CONTAINER_JOB}" ]]; then
        export SETUP_AFTEROK="${LAST_CONTAINER_JOB}"
        echo "${action} jobs will wait for ${SETUP_AFTEROK}"
    fi
    "stage_${action}"
    unset SETUP_AFTEROK
}

case "${ACTION}" in
    init)      init_layout ;;
    status)    show_status ;;
    list)      list_groups ;;
    container) stage_container ;;
    models)    stage_with_container models ;;
    datasets)  stage_with_container datasets ;;
    sft)
        if (( SUBMIT != 0 )); then
            echo "preflight: validating every SFT checkpoint source"
            experiments/setup/models/stage_sft_checkpoints.sh
        fi
        stage_with_container sft
        ;;
    all)       stage_all ;;
    *)
        echo "unknown action: ${ACTION}" >&2
        usage >&2
        exit 2
        ;;
esac
