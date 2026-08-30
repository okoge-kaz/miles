#!/bin/bash
# Submit reproducible dataset transfers on the dedicated data-mover QoS.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

MANIFEST="${REPO_ROOT}/experiments/setup/manifests/nemotron_rl_datasets.tsv"
ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
HF_TOKEN_EXPORT=""
[[ -z "${HF_TOKEN:-}" ]] || HF_TOKEN_EXPORT=",HF_TOKEN"

has_payload() {
    local target="$1"
    find "${target}" -maxdepth 2 -type f \
        ! -path '*/.cache/*' ! -name README.md ! -name .gitattributes \
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
    if has_payload "${target}"; then
        printf 'present\t%s\t%s\n' "${role}" "${target}"
        continue
    fi
    if [[ "${hf_repo}" == "Idavidrein/gpqa" && -z "${HF_TOKEN:-}" ]]; then
        printf 'gated\t%s\tHF_TOKEN is unset; accept the GPQA terms first\n' "${hf_repo}" >&2
        continue
    fi
    HF_REVISION="${revision}"
    export HF_REVISION
    job_id=$(sbatch --parsable \
        -A "${ACCOUNT}" \
        -p cpu_datamover \
        --qos=cpu-datamover \
        --job-name="dl-${local_name:0:28}" \
        --export="USER,WANDB_MODE=disabled,HF_REPO=${hf_repo},LOCAL_NAME=${local_name},HF_REVISION${HF_TOKEN_EXPORT}" \
        "${REPO_ROOT}/experiments/setup/download/download_dataset.sbatch")
    printf 'submitted\t%s\t%s\t%s\n' "${job_id}" "${role}" "${hf_repo}"
done < "${MANIFEST}"
