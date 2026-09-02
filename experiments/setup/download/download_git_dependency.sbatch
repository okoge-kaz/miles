#!/bin/bash
#PBS -N miles-download-git-dependency
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -j oe

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="${USER:-$(id -un)}"
WANDB_MODE=disabled
export PATH USER WANDB_MODE

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${GIT_REPO:?set GIT_REPO}"
: "${LOCAL_NAME:?set LOCAL_NAME}"

target="${DATASET_DIR}/${LOCAL_NAME}"
if [[ -d "${target}/.git" ]]; then
    current_commit="$(git -C "${target}" rev-parse HEAD)"
    if [[ -n "${GIT_REF:-}" ]]; then
        requested_commit="$(git -C "${target}" rev-parse --verify "${GIT_REF}^{commit}" 2>/dev/null)" || {
            echo "ERROR: ${GIT_REF} is not present in the existing checkout ${target}" >&2
            echo "Remove the checkout and rerun to fetch the requested pin." >&2
            exit 1
        }
        [[ "${current_commit}" == "${requested_commit}" ]] || {
            echo "ERROR: ${target} is at ${current_commit}, expected ${requested_commit} (${GIT_REF})" >&2
            exit 1
        }
    fi
    printf 'present %s at %s\n' "${current_commit}" "${target}"
    exit 0
fi
if [[ -e "${target}" ]]; then
    echo "ERROR: ${target} exists but is not a git checkout" >&2
    exit 1
fi

git clone --filter=blob:none "${GIT_REPO}" "${target}"
if [[ -n "${GIT_REF:-}" ]]; then
    if ! git -C "${target}" rev-parse --verify "${GIT_REF}^{commit}" >/dev/null 2>&1; then
        git -C "${target}" fetch --depth=1 origin "${GIT_REF}"
    fi
    git -C "${target}" checkout --detach "${GIT_REF}"
fi
git -C "${target}" rev-parse HEAD
