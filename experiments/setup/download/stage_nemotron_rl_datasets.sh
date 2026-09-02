#!/bin/bash
# Submit reproducible dataset transfers on the dedicated data-mover QoS.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/pbs.sh"

MANIFEST="${REPO_ROOT}/experiments/setup/manifests/nemotron_rl_datasets.tsv"
SETUP_DOWNLOAD_WALLTIME="${SETUP_DOWNLOAD_WALLTIME:-${PBS_DOWNLOAD_WALLTIME:-24:00:00}}"
SETUP_PATH_EXPORTS="MILES_WORKSPACE_ROOT,MILES_REPO,CHECKPOINT_ROOT,HF_CKPT_DIR,MEGATRON_CKPT_DIR,TRAIN_CKPT_DIR,DATASET_ROOT,PRETRAIN_DATASET_DIR,RL_DATASET_DIR,SFT_DATASET_DIR,DATASET_DIR,CONTAINER_DIR,CACHE_DIR,CONTAINER_IMAGE"
export HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-2}"
export HF_DOWNLOAD_ATTEMPTS="${HF_DOWNLOAD_ATTEMPTS:-5}"
export HF_DOWNLOAD_RETRY_DELAY_SECONDS="${HF_DOWNLOAD_RETRY_DELAY_SECONDS:-60}"
HF_TOKEN_EXPORT=""
[[ -z "${HF_TOKEN:-}" ]] || HF_TOKEN_EXPORT=",HF_TOKEN"

dataset_complete() {
    local target="$1"
    [[ -s "${target}/MILES_SOURCE_PROVENANCE" ]] || return 1
    find "${target}" -type f \
        ! -path '*/.cache/*' ! -name README.md ! -name .gitattributes \
        ! -name 'MILES_SOURCE_PROVENANCE*' \
        -print -quit 2>/dev/null | grep -q .
}

while IFS=$'\t' read -r hf_repo local_name role revision; do
    [[ -z "${hf_repo}" || "${hf_repo}" == \#* ]] && continue
    if [[ "${role}" == selective_swe_* ]]; then
        printf 'selective\t%s\tuse download_swe_datasets.sbatch\t%s\n' \
            "${role}" "${hf_repo}"
        continue
    fi
    target="${DATASET_DIR}/${local_name}"
    if dataset_complete "${target}"; then
        printf 'present\t%s\t%s\n' "${role}" "${target}"
        continue
    fi
    if [[ "${hf_repo}" == "Idavidrein/gpqa" && -z "${HF_TOKEN:-}" ]]; then
        printf 'gated\t%s\tHF_TOKEN is unset; accept the GPQA terms first\n' "${hf_repo}" >&2
        continue
    fi
    HF_REVISION="${revision}"
    export HF_REVISION
    job_id=$(pbs_submit --parsable --profile=cpu \
        --time="${SETUP_DOWNLOAD_WALLTIME}" \
        --job-name="dl-${local_name:0:28}" \
        --export="${SETUP_PATH_EXPORTS},USER,WANDB_MODE=disabled,HF_REPO=${hf_repo},LOCAL_NAME=${local_name},HF_REVISION,HF_DOWNLOAD_MAX_WORKERS,HF_DOWNLOAD_ATTEMPTS,HF_DOWNLOAD_RETRY_DELAY_SECONDS${HF_TOKEN_EXPORT}" \
        "${REPO_ROOT}/experiments/setup/download/download_dataset.sbatch")
    printf 'submitted\t%s\t%s\t%s\n' "${job_id}" "${role}" "${hf_repo}"
done < "${MANIFEST}"
