#!/bin/bash

# Submit SWE normalization with only the public source selector in scope.

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../..")"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/pbs.sh"
: "${SWE_SOURCE:?set SWE_SOURCE to a supported prepared-source selector}"

MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="MILES_WORKSPACE_ROOT,MILES_REPO,CHECKPOINT_ROOT,HF_CKPT_DIR,MEGATRON_CKPT_DIR,TRAIN_CKPT_DIR,DATASET_ROOT,PRETRAIN_DATASET_DIR,RL_DATASET_DIR,SFT_DATASET_DIR,DATASET_DIR,CONTAINER_DIR,CACHE_DIR,CONTAINER_IMAGE,MILES_SWE_FIXED_EXPORTS,SWE_SOURCE"
readonly SETUP_PREP_WALLTIME="${SETUP_PREP_WALLTIME:-${PBS_PREP_WALLTIME:-08:00:00}}"

submission="$(
    pbs_submit \
        --parsable \
        --profile=cpu \
        --time="${SETUP_PREP_WALLTIME}" \
        --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/prepare_swe_rl.sbatch"
)"
job_id="${submission%%;*}"
[[ "${job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || {
    echo "PBS returned an invalid SWE preparation job ID" >&2
    exit 2
}
printf 'swe_prepare_job_id=%s\n' "${job_id}"
