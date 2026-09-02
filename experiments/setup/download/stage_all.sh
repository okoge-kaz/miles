#!/bin/bash
# Stage every model and dataset listed in models.txt / datasets.txt.
#
#   experiments/setup/download/stage_all.sh              # models + datasets
#   experiments/setup/download/stage_all.sh models       # models only
#   experiments/setup/download/stage_all.sh datasets     # datasets only
#
# Run from the repo root. Submits, then returns; watch with experiments/status.sh.
#
# Scheduling policy:
#   * downloads explicitly use the PBS CPU profile
#   * each model's conversion depends on its own download AND on the previous
#     conversion, so only one 8-GPU job of this batch runs at a time
#   * dataset downloads run concurrently, capped by MAX_PARALLEL_DL chains

set -euo pipefail

WHAT="${1:-all}"
MAX_PARALLEL_DL="${MAX_PARALLEL_DL:-4}"
SETUP_SUBMIT_DELAY_SECONDS="${SETUP_SUBMIT_DELAY_SECONDS:-1}"
HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-2}"
HF_DOWNLOAD_ATTEMPTS="${HF_DOWNLOAD_ATTEMPTS:-5}"
HF_DOWNLOAD_RETRY_DELAY_SECONDS="${HF_DOWNLOAD_RETRY_DELAY_SECONDS:-60}"
[[ "${MAX_PARALLEL_DL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_PARALLEL_DL must be a positive integer" >&2
    exit 2
}
[[ "${SETUP_SUBMIT_DELAY_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "SETUP_SUBMIT_DELAY_SECONDS must be a non-negative number" >&2
    exit 2
}
[[ "${HF_DOWNLOAD_MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_DOWNLOAD_MAX_WORKERS must be a positive integer" >&2
    exit 2
}
[[ "${HF_DOWNLOAD_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "HF_DOWNLOAD_ATTEMPTS must be a positive integer" >&2
    exit 2
}
[[ "${HF_DOWNLOAD_RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]] || {
    echo "HF_DOWNLOAD_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
}
export HF_DOWNLOAD_MAX_WORKERS HF_DOWNLOAD_ATTEMPTS
export HF_DOWNLOAD_RETRY_DELAY_SECONDS

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh
source experiments/common/pbs.sh
SETUP_AFTEROK="${SETUP_AFTEROK:-}"
SETUP_CONVERT_WALLTIME="${SETUP_CONVERT_WALLTIME:-${PBS_PREP_WALLTIME:-08:00:00}}"
SETUP_DOWNLOAD_WALLTIME="${SETUP_DOWNLOAD_WALLTIME:-${PBS_DOWNLOAD_WALLTIME:-24:00:00}}"
SETUP_PATH_EXPORTS="MILES_WORKSPACE_ROOT,MILES_REPO,CHECKPOINT_ROOT,HF_CKPT_DIR,MEGATRON_CKPT_DIR,TRAIN_CKPT_DIR,DATASET_ROOT,PRETRAIN_DATASET_DIR,RL_DATASET_DIR,SFT_DATASET_DIR,DATASET_DIR,CONTAINER_DIR,CACHE_DIR,CONTAINER_IMAGE"

HF_TOKEN_EXPORT=""
[[ -z "${HF_TOKEN:-}" ]] || HF_TOKEN_EXPORT=",HF_TOKEN"

strip() { sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | grep -v '^$'; }

has_payload() {
    local target="$1"
    [[ -d "${target}" ]] || return 1
    find "${target}" -type f \
        ! -path '*/.cache/*' ! -name '*.md' ! -name .gitattributes \
        ! -name 'MILES_SOURCE_PROVENANCE*' \
        -print -quit 2>/dev/null | grep -q .
}

dataset_complete() {
    local target="$1"
    [[ -s "${target}/MILES_SOURCE_PROVENANCE" ]] && has_payload "${target}"
}

throttle_submissions() {
    [[ "${SETUP_SUBMIT_DELAY_SECONDS}" == 0 ]] || sleep "${SETUP_SUBMIT_DELAY_SECONDS}"
}

stage_models() {
    local prev_convert=""
    while IFS='|' read -r name repo type extra nodes; do
        name=$(echo "$name" | xargs); repo=$(echo "$repo" | xargs); type=$(echo "$type" | xargs)
        extra=$(echo "${extra:-}" | xargs); nodes=$(echo "${nodes:-1}" | xargs); nodes=${nodes:-1}
        [[ -z "$name" ]] && continue

        local dl_dep="" cv_dep=""
        if [[ -s "${HF_CKPT_DIR:?source experiments/env.sh first}/${name}/.download_complete" ]]; then
            echo "skip download ${name} (complete)"
        else
            local -a initial_dependency=()
            [[ -z "${SETUP_AFTEROK}" ]] || initial_dependency=(--dependency="afterok:${SETUP_AFTEROK}")
            dl=$(pbs_submit --parsable --profile=cpu \
                 --job-name="dl-${name}" \
                 --time="${SETUP_DOWNLOAD_WALLTIME}" \
                 "${initial_dependency[@]}" \
                 --export="${SETUP_PATH_EXPORTS},USER,WANDB_MODE=disabled,HF_REPO=${repo},MODEL_NAME=${name},HF_DOWNLOAD_MAX_WORKERS,HF_DOWNLOAD_ATTEMPTS,HF_DOWNLOAD_RETRY_DELAY_SECONDS${HF_TOKEN_EXPORT}" \
                 experiments/setup/download/download_model.sbatch)
            throttle_submissions
            dl_dep="afterok:${dl}"
        fi

        local tracker="${MEGATRON_CKPT_DIR:?source experiments/env.sh first}/${name}_torch_dist/latest_checkpointed_iteration.txt"
        if [[ -z "$dl_dep" && -f "$tracker" && "$(cat "$tracker" 2>/dev/null)" == "release" ]]; then
            echo "skip ${name} (already downloaded and converted)"
            continue
        fi

        cv_dep="${dl_dep}"
        [[ -n "${cv_dep}" || -z "${SETUP_AFTEROK}" ]] || cv_dep="afterok:${SETUP_AFTEROK}"
        [[ -n "$prev_convert" ]] && cv_dep="${cv_dep:+${cv_dep},}afterany:${prev_convert}"

        cv=$(pbs_submit --parsable --profile=gpu \
             --job-name="cv-${name}" \
             ${cv_dep:+--dependency=${cv_dep}} \
             --time="${SETUP_CONVERT_WALLTIME}" --nodes="${nodes}" \
             --export="${SETUP_PATH_EXPORTS},USER,WANDB_MODE=disabled,MODEL_NAME=${name},MEGATRON_MODEL_TYPE=${type},CONVERT_EXTRA_ARGS=${extra}" \
             experiments/setup/models/convert_checkpoint.sbatch)
        throttle_submissions
        prev_convert="$cv"
        printf "%-40s download=%-10s convert=%s\n" "$name" "${dl:-skipped}" "$cv"
        unset dl
    done < <(strip < experiments/setup/manifests/models.txt)
}

stage_datasets() {
    local -a chain_tail=()
    local i=0
    while IFS='|' read -r name repo; do
        name=$(echo "$name" | xargs); repo=$(echo "$repo" | xargs)
        [[ -z "$name" ]] && continue
        local target="${DATASET_DIR}/${name}"

        if dataset_complete "${target}"; then
            printf 'skip %-19s (complete)\n' "${name}"
            continue
        fi

        if [[ "${repo}" == "Idavidrein/gpqa" && -z "${HF_TOKEN:-}" ]]; then
            if has_payload "${target}"; then
                printf 'gated %s: payload is present but completion provenance is missing; HF_TOKEN is unset\n' \
                    "${repo}" >&2
            else
                printf 'gated %s: HF_TOKEN is unset; accept the GPQA terms first\n' \
                    "${repo}" >&2
            fi
            continue
        fi

        local slot=$(( i % MAX_PARALLEL_DL ))
        local dep=""
        local -a dependency=()
        [[ -z "${SETUP_AFTEROK}" ]] || dep="afterok:${SETUP_AFTEROK}"
        if [[ -n "${chain_tail[$slot]:-}" ]]; then
            dep="${dep:+${dep},}afterany:${chain_tail[$slot]}"
        fi
        [[ -z "${dep}" ]] || dependency=(--dependency="${dep}")

        jid=$(pbs_submit --parsable --profile=cpu \
              --job-name="ds-${name}" \
              --time="${SETUP_DOWNLOAD_WALLTIME}" \
              "${dependency[@]}" \
              --export="${SETUP_PATH_EXPORTS},USER,WANDB_MODE=disabled,HF_REPO=${repo},LOCAL_NAME=${name},HF_DOWNLOAD_MAX_WORKERS,HF_DOWNLOAD_ATTEMPTS,HF_DOWNLOAD_RETRY_DELAY_SECONDS${HF_TOKEN_EXPORT}" \
              experiments/setup/download/download_dataset.sbatch)
        throttle_submissions
        chain_tail[$slot]="$jid"
        printf "%-24s %-44s job=%s\n" "$name" "$repo" "$jid"
        i=$(( i + 1 ))
    done < <(strip < experiments/setup/manifests/datasets.txt)
}

case "$WHAT" in
    models)   echo "=== models ===";   stage_models ;;
    datasets) echo "=== datasets ==="; stage_datasets ;;
    all)      echo "=== models ===";   stage_models
              echo; echo "=== datasets ==="; stage_datasets ;;
    *) echo "usage: $0 [all|models|datasets]"; exit 1 ;;
esac
